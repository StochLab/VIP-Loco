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

import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

MOTION_FILES = glob.glob('datasets/mocap_motions/*')


class A1AMPCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        prop_dim = 33 # proprioception
        action_dim = 12
        privileged_dim = 24 + 26 + 3  # privileged_obs[:,:privileged_dim] is the privileged information in privileged_obs, include 3-dim base linear vel
        height_dim = 187  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
        forward_height_dim = 525 # for depth image prediction
        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES

    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        # 525 dim, for depth image prediction
        measured_forward_points_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                                     2.0]  # 1mx1.6m rectangle (without center line)
        measured_forward_points_y = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.,
                                     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [wave, rough slope, stairs up, stairs down, discrete, gap, pit, tilt, crawl, rough_flat]
        terrain_proportions = [0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.0,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.0,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4


    class depth:
        use_camera = True
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        y_angle = [-5, 5]  # positive pitch down
        z_angle = [0, 0]
        x_angle = [0, 0]

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        # terminate_after_contacts_on = [
        #     "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
        #     "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        # self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.0]

        randomize_base_mass = True
        added_mass_range = [0., 3.]  # kg
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = True
        com_x_pos_range = [-0.05, 0.05]
        com_y_pos_range = [-0.05, 0.05]
        com_z_pos_range = [-0.05, 0.05]

        push_robots = True
        push_interval_s = 15
        min_push_interval_s = 15
        max_push_vel_xy = 1.0

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_action_latency = True
        latency_range = [0.00, 0.005]

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            # privileged
            height_measurements = 5.0
            contact_force = 0.005
            com_pos = 20
            pd_gains = 5


        clip_observations = 100.
        clip_actions = 6.0

        base_height = 0.3 # base height of A1, used to normalize measured height


    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0  # set lin_vel as privileged information
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0  # only for critic

    class rewards(LeggedRobotCfg.rewards):
        reward_curriculum = True
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]

        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        foot_height_target = 0.15
        tracking_sigma = 0.15  # tracking reward = exp(-error^2/sigma)
        lin_vel_clip = 0.1

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            torques = -0.0001
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 0.5
            collision = -1.0
            feet_stumble = -0.1
            action_rate = -0.03

            feet_edge = -1.0
            dof_error = -0.04

            lin_vel_z = -1.0
            cheat = -1
            stuck = -1


    class commands:
        curriculum = False
        max_lin_vel_forward_x_curriculum = 1.0
        max_lin_vel_backward_x_curriculum = 0.0
        max_lin_vel_y_curriculum = 0.0
        max_ang_vel_yaw_curriculum = 1.0

        max_flat_lin_vel_forward_x_curriculum = 1.0
        max_flat_lin_vel_backward_x_curriculum = 0.0
        max_flat_lin_vel_y_curriculum = 0.0
        max_flat_ang_vel_yaw_curriculum = 1.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [0.0, 0.8]  # min max [m/s]
            lin_vel_y = [-0., 0.]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-0., 0.]

            flat_lin_vel_x = [-0.0, 0.8]  # min max [m/s]
            flat_lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            flat_ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            flat_heading = [-3.14 / 4, 3.14 / 4]


class A1AMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'WMPRunner'

    class policy:
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        wm_latent_dim = 32
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        vel_predict_coef = 1.0
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'flat_push1'
        experiment_name = 'a1_amp_example'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 20000  # number of policy updates
        save_interval = 1000

        amp_reward_coef = 0.5 * 0.02  # set to 0 means not use amp reward
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4

    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100

