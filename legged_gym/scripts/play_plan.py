# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR


# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
# xla_flags += ' --xla_gpu_force_compilation_parallelism=1'
os.environ['XLA_FLAGS'] = xla_flags

# Mem fraction
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['CUDA_VISIBLE_DEVICES']="1"
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import jax
from dreamer.jax_mpc import JaxMPC

import matplotlib.pyplot as plt


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_restitution = False
    # env_cfg.commands.heading_command = True

    env_cfg.domain_rand.friction_range = [1.0, 1.0]
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]
    env_cfg.domain_rand.added_mass_range = [0., 0.]  # kg
    env_cfg.domain_rand.com_x_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_y_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_z_pos_range = [-0.0, 0.0]

    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = True
    # env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    # env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_motor_strength = False

    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]


    # env_cfg.terrain.mesh_type = 'plane'
    if(env_cfg.terrain.mesh_type == 'plane'):
        env_cfg.rewards.scales.feet_edge = 0
        env_cfg.rewards.scales.feet_stumble = 0


    if(args.terrain not in ['slope', 'stair', 'gap', 'climb', 'crawl', 'tilt']):
        print('terrain should be one of slope, stair, gap, climb, crawl, and tilt, set to climb as default')
        args.terrain = 'climb'
    env_cfg.terrain.terrain_proportions = {
        'slope': [0, 1.0, 0.0, 0, 0, 0, 0, 0, 0],
        'stair': [0, 0, 1.0, 0, 0, 0, 0, 0, 0],
        'gap': [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
        'climb': [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
        'tilt': [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
        'crawl': [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
     }[args.terrain]

    env_cfg.commands.ranges.lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0, 0]

    env_cfg.commands.ranges.flat_lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.flat_lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.flat_ang_vel_yaw = [0.0, 0.0]

    env_cfg.depth.use_camera = True
    env_cfg.viewer.lookat_id = 8

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, _, infos = env.reset()

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = 'WMP'


    train_cfg.runner.checkpoint = -1
    ppo_runner, train_cfg = task_registry.make_wmp_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    frames_record_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames_' + str(args.terrain))
    data_record_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'data_' + str(args.terrain))

    robot_index = env_cfg.viewer.lookat_id # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = env.max_episode_length + 1 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    history_length = 5
    trajectory_history = torch.zeros(size=(env.num_envs, history_length, env.num_obs -
                                            env.privileged_dim - env.height_dim - 3), device = env.device)
    obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                        obs[:, env.privileged_dim + 9:-env.height_dim]), dim=1)
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

    wm_update_interval = env.cfg.depth.update_interval

    world_model = ppo_runner._world_model.to(env.device)
    world_model.eval()

    torch_wm_model = world_model.get_export_model()
    torch_expert_actor = ppo_runner.alg.actor_critic.get_export_actor()
    jax_mpc = JaxMPC(world_model=torch_wm_model, expert_actor=torch_expert_actor, 
                        wm_horizon=world_model._config.horizon, wm_freq=wm_update_interval)   
    rng = jax.random.PRNGKey(0)

    wm_state = None
    wm_is_first = torch.ones(env.num_envs, device=env.device)
    wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)
    wm_obs = {
            "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
            "is_first": wm_is_first,
        }

    if (env.cfg.depth.use_camera):
        wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)), device=world_model.device)
    
    mpc_state = {'step': 0, 
                'cmd': obs[:, env.privileged_dim + 6:env.privileged_dim + 9], 
                'prev_mean': torch.zeros((env.num_envs, jax_mpc.planning_horizon, jax_mpc.act_dim), device=world_model.device),
                'wm_obs': wm_obs, 
                "wm_state": jax_mpc.init_wm_state(wm_obs), 
                "wm_feature": wm_feature,
                'trajectory_history': trajectory_history}


    total_reward = 0
    not_dones = torch.ones((env.num_envs,), device=env.device)

    # CoM Plot
    if not args.headless:
        # Set up figure and axis outside the loop
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()  # Turn on interactive mode

    for i in range(1*int(env.max_episode_length) + 3):
        with torch.inference_mode():
            # start_time = time.time()
            rng, rng_plan = jax.random.split(rng)
            actions, mpc_state = jax_mpc.plan(rng_plan, mpc_state)
            actions = actions.to(env.device)
            # print(f"Planning took {1/(time.time() - start_time):.2f} Hz")

            wm_is_first[:] = 0

        obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())

        wm_obs["prop"] = obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim].to(world_model.device)
        wm_obs["is_first"] = wm_is_first

        if (env.global_counter % wm_update_interval == 0):
            if (env.cfg.depth.use_camera):
                wm_obs["image"] = infos["depth"].unsqueeze(-1).to(world_model.device)

        # update world model input
        reset_env_ids = reset_env_ids.cpu().numpy()
        if (len(reset_env_ids) > 0):
            wm_is_first[reset_env_ids] = 1

        # process trajectory history
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                            obs[:, env.privileged_dim + 9:-env.height_dim]),
                                           dim=1)
        trajectory_history = torch.concat(
            (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)


        mpc_state['cmd'] = obs[:, env.privileged_dim + 6:env.privileged_dim + 9]
        mpc_state['wm_obs'] = wm_obs
        mpc_state['trajectory_history'] = trajectory_history



        not_dones *= (~dones)
        total_reward += torch.mean(rews * not_dones)

        if RECORD_FRAMES:
            if i % 2:
                record_filename = os.path.join(frames_record_path, f"{img_idx}.png")

                if not os.path.exists(os.path.dirname(frames_record_path)):
                    os.makedirs(os.path.dirname(frames_record_path))

                env.gym.write_viewer_image_to_file(env.viewer, record_filename)
                img_idx += 1 
        if MOVE_CAMERA:
            lootat = env.root_states[env_cfg.viewer.lookat_id, :3]
            camara_position = lootat.detach().cpu().numpy() + [1, -2, 0.5]
            # camara_position = lootat.detach().cpu().numpy() + [0, 1, 0]
            env.set_camera(camara_position, lootat)

        if i < stop_state_log:
            ref_pos = env.root_states[:, :3].detach().cpu().numpy()

            pred_com_pos = mpc_state['data_com_pos'].detach().cpu().numpy()
            pred_com_pos = pred_com_pos.mean(axis=1) # Average
            pred_com_pos += ref_pos.reshape(-1, 1, 3)

            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    
                    'real_com_x': ref_pos[robot_index, 0].item(),
                    'real_com_y': ref_pos[robot_index, 1].item(),
                    'real_com_z': ref_pos[robot_index, 2].item(),
                    'pred_com_x': pred_com_pos[robot_index, :, 0].tolist(),
                    'pred_com_y': pred_com_pos[robot_index, :, 1].tolist(),
                    'pred_com_z': pred_com_pos[robot_index, :, 2].tolist(),
                }
            )
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

            os.makedirs(data_record_path, exist_ok=True)  
            data_log_path = os.path.join(data_record_path, 'log_plan.pkl')
            logger.save(data_log_path)


        if not args.headless:
            # Plot CoM positions (B, H, 3)
            live_plot_com_single_env(fig, ax,
                mpc_state['data_com_pos'][env_cfg.viewer.lookat_id].detach().cpu().numpy(),
                env.root_states[env_cfg.viewer.lookat_id, :3].detach().cpu().numpy())


    print('total reward:', total_reward)


def live_plot_com_single_env(fig, ax, data, base_pos):
    """
    Live-updating plot of CoM trajectories.
    """
    ax.cla()  # Clear the previous frame (but keep figure alive)

    num_traj, horizon, _ = data.shape

    # Compute mean trajectory
    mean_traj = base_pos + data.mean(axis=0)/10 # (horizon, 3)
    std_traj = data.std(axis=0) # (horizon, 3)
    xs_mean, ys_mean, zs_mean = mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2]
    xs_std, ys_std, zs_std = std_traj[:, 0], std_traj[:, 1], std_traj[:, 2]

    # Keep consistent axis limits and view
    ax.set_xlim(base_pos[0] - 0.5, base_pos[0] + 0.5)
    ax.set_ylim(base_pos[1] - 0.5, base_pos[1] + 0.5)
    ax.set_zlim(base_pos[2] - 0.1, base_pos[2] + 0.5)

    # # for 1st point
    # ax.plot([base_pos[0], xs_mean[0]], [base_pos[1], ys_mean[0]], [base_pos[2], zs_mean[0]],
    #         color='blue', alpha=1.0, linewidth=2)

    for i in range(1, horizon):
        # Plot mean trajectory line with fading alpha
        alpha = 1.0 - i / horizon  # Fades out over time
        ax.plot(xs_mean[i-1:i+1], ys_mean[i-1:i+1], zs_mean[i-1:i+1],
                color='blue', alpha=alpha, linewidth=2)

        # Plot std deviation as tubes around the mean trajectory
        u = np.linspace(0, 2 * np.pi, 15)
        for j in range(15):
            x_std1 = xs_mean[i-1] + xs_std[i-1] * np.ones_like(u[j])
            y_std1 = ys_mean[i-1] + ys_std[i-1] * np.sin(u[j])
            z_std1 = zs_mean[i-1] + zs_std[i-1] * np.cos(u[j])

            x_std2 = xs_mean[i] + xs_std[i] * np.ones_like(u[j])
            y_std2 = ys_mean[i] + ys_std[i] * np.sin(u[j])
            z_std2 = zs_mean[i] + zs_std[i] * np.cos(u[j])

            ax.plot([x_std1, x_std2], [y_std1, y_std2], [z_std1, z_std2],
                    color='red', alpha=alpha/2, linewidth=1, linestyle='--')

    ax.view_init(elev=45, azim=-45)  # Lock camera angle

    ax.set_title('3D CoM Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.legend()

    plt.pause(0.01)  # Allow the plot to update


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    args.rl_device = args.sim_device

    play(args)
