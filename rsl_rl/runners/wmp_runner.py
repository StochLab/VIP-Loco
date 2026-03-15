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

import time
import os
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import torch

from rsl_rl.algorithms import AMPPPO, PPO
from rsl_rl.modules import ActorCritic, ActorCriticWMP, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer

from dreamer.model_com import *
import ruamel.yaml as yaml
import argparse
import pathlib
import sys
import collections
from dreamer import tools
import datetime
import uuid


class WMPRunner:

    def __init__(self,
                 env,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 history_length=5,
                 ):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]

        self.device = device
        self.env = env
        self.history_length = history_length
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        if self.env.include_history_steps is not None:
            num_actor_obs = self.env.num_obs * self.env.include_history_steps
        else:
            num_actor_obs = self.env.num_obs



        # build world model
        self._build_world_model()

        self.history_dim = history_length * (self.env.num_obs - self.env.privileged_dim - self.env.height_dim-3) #exclude command
        actor_critic = ActorCriticWMP(num_actor_obs=num_actor_obs,
                                          num_critic_obs=num_critic_obs,
                                          num_actions=self.env.num_actions,
                                          height_dim=self.env.height_dim,
                                          privileged_dim=self.env.privileged_dim,
                                          history_dim=self.history_dim,
                                          wm_feature_dim=self.wm_feature_dim,
                                          **self.policy_cfg).to(self.device)

        if train_cfg['runner']['use_amp']:
            amp_data = AMPLoader(
                device, time_between_frames=self.env.dt, preload_transitions=True,
                num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
                motion_files=self.cfg["amp_motion_files"])
            amp_normalizer = Normalizer(amp_data.observation_dim)
            discriminator = AMPDiscriminator(
                amp_data.observation_dim * 2,
                train_cfg['runner']['amp_reward_coef'],
                train_cfg['runner']['amp_discr_hidden_dims'], device,
                train_cfg['runner']['amp_task_reward_lerp']).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        
        if self.wm_config.task == 'trona1_w':
            # since [3,7] is a wheel joint, cannot use joint_pos_limits there
            dof_limits = torch.concat(
                [self.env.dof_pos_limits[[0, 1, 2], 1] - self.env.dof_pos_limits[[0, 1, 2], 1],
                 self.env.dof_vel_limits[3:4] * 2,
                 self.env.dof_pos_limits[[4, 5, 6], 1] - self.env.dof_pos_limits[[4, 5, 6], 0],
                 self.env.dof_vel_limits[7:] * 2
                ], dim=-1)
        else:
            dof_limits = self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0]

        min_std = (
                torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
                (torch.abs(dof_limits)))

        if train_cfg['runner']['use_amp']:
            self.alg = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device,
                                  min_std=min_std, **self.alg_cfg)
        else:
            self.alg = alg_class(actor_critic, device=self.device, min_std=min_std, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs],
                              [self.env.num_privileged_obs], [self.env.num_actions], self.history_dim, 
                              self.wm_feature_dim)

        # Log
        self.log_dir = log_dir
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _, _ = self.env.reset()


    def _build_world_model(self):
        # world model
        print('Begin construct world model')
        yml = yaml.YAML(typ='safe')
        configs = yml.load(
            (pathlib.Path(sys.argv[0]).parent.parent.parent / "dreamer/configs.yaml").read_text()
        )

        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value

        name_list = ["defaults"]
        defaults = {}
        for name in name_list:
            recursive_update(defaults, configs[name])
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true", default=False)
        parser.add_argument("--task", default='go1_amp')
        parser.add_argument("--sim_device", default='cuda:0')
        parser.add_argument("--wm_device", default='None')
        parser.add_argument("--terrain", default='climb')
        parser.add_argument("--checkpoint", default=-1)
        for key, value in sorted(defaults.items(), key=lambda x: x[0]):
            arg_type = tools.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        self.wm_config = parser.parse_args()
        # allow world model and rl env on different device
        if (self.wm_config.wm_device != 'None'):
            self.wm_config.device = self.wm_config.wm_device

        prop_dim = self.env.num_obs - self.env.privileged_dim - self.env.height_dim - self.env.num_actions
        image_shape = self.env.cfg.depth.resized + (1,)
        obs_shape = {'prop': (prop_dim,), 'image': image_shape,}

        self._world_model = KinoDynamicComWorldModel(self.wm_config, obs_shape)
        self._world_model = self._world_model.to(self._world_model.device)
        print('Finish construct world model')
        self.wm_feature_dim = self._world_model.future_preds_dim + self.wm_config.dyn_deter


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        obs, privileged_obs, infos = self.env.reset()
        amp_obs = self.env.get_amp_observations()

        critic_obs = privileged_obs if privileged_obs is not None else obs

        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)

        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        if self.cfg['use_amp']: self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # process trajectory history
        self.trajectory_history = torch.zeros(size=(self.env.num_envs, self.history_length, self.env.num_obs -
                                                    self.env.privileged_dim - self.env.height_dim - 3),
                                              device=self.device)
        obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                                            obs[:, self.env.privileged_dim + 9:-self.env.height_dim]), dim=1)
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)),
                                               dim=1)

        # init world model input
        sum_wm_dataset_size = 0
        wm_state = wm_action = None
        wm_is_first = torch.ones(self.env.num_envs, device=self._world_model.device)
        wm_obs = {
            "prop": obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device),
            "is_first": wm_is_first,
            "wm_state": wm_state,
        }

        if(self.env.cfg.depth.use_camera):
            wm_obs["image"] = torch.zeros(((self.env.num_envs,) + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)

        wm_metrics = None
        self.wm_update_interval = self.env.cfg.depth.update_interval
        wm_reward = torch.zeros(self.env.num_envs, device=self._world_model.device)
        wm_value = torch.zeros(self.env.num_envs, device=self._world_model.device)
        wm_feature = torch.zeros((self.env.num_envs, self.wm_feature_dim), device=self._world_model.device)

        wm_com_state = torch.cat(
                [infos["base_pos"].to(self._world_model.device),
                infos["base_ang_pos"].to(self._world_model.device),
                infos["base_lin_vel"].to(self._world_model.device),
                infos["base_ang_vel"].to(self._world_model.device)], dim=-1)
        wm_joint_state = torch.cat(
                [infos["joint_pos"].to(self._world_model.device),
                 infos["joint_vel"].to(self._world_model.device)], dim=-1)

        self.init_wm_dataset()


        for it in range(self.current_learning_iteration, tot_iter):
            if (self.env.cfg.rewards.reward_curriculum):
                self.env.update_reward_curriculum(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        wm_future_preds, wm_hidden_state, wm_state = self._world_model(wm_obs)
                        wm_feature = torch.cat(
                            [wm_hidden_state, wm_future_preds], dim=-1).to(self._world_model.device)
                        wm_obs["wm_state"] = wm_state

                        wm_is_first[:] = 0

                    history = self.trajectory_history.flatten(1).to(self.device)
                    if self.cfg['use_amp']:
                        actions, critic_values = self.alg.act(obs, critic_obs, amp_obs, history, wm_feature)
                    else:
                        actions, critic_values = self.alg.act(obs, critic_obs, history, wm_feature)

                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(
                        actions)
                    next_amp_obs = self.env.get_amp_observations()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(
                        self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    wm_obs["prop"] = obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device)
                    wm_obs["is_first"] = wm_is_first

                    wm_com_state = torch.cat(
                        [infos["base_pos"].to(self._world_model.device),
                         infos["base_ang_pos"].to(self._world_model.device),
                         infos["base_lin_vel"].to(self._world_model.device),
                         infos["base_ang_vel"].to(self._world_model.device)], dim=-1)
                    wm_joint_state = torch.cat(
                        [infos["joint_pos"].to(self._world_model.device),
                         infos["joint_vel"].to(self._world_model.device)], dim=-1)

                    # store the data in buffer into the dataset before reset
                    reset_env_ids = reset_env_ids.cpu().numpy()
                    if (len(reset_env_ids) > 0):
                        for k, v in self.wm_dataset.items():
                            v[reset_env_ids, :] = self.wm_buffer[k][reset_env_ids].to(self._world_model.device)

                        self.wm_dataset_size[reset_env_ids] = self.wm_buffer_index[reset_env_ids]
                        self.wm_buffer_index[reset_env_ids] = 0
                        sum_wm_dataset_size = np.sum(self.wm_dataset_size)

                        wm_is_first[reset_env_ids] = 1

                    wm_action = actions.to(self._world_model.device)
                    wm_reward = rewards.unsqueeze(-1).to(self._world_model.device)
                    wm_value = critic_values.to(self._world_model.device)
                    
                    # store current step into buffer
                    if (self.env.global_counter % self.wm_update_interval == 0): # because after step
                        if (self.env.cfg.depth.use_camera):
                            wm_obs["image"] = infos["depth"].unsqueeze(-1).to(self._world_model.device)

                        # not_reset_env_ids = (~dones).nonzero(as_tuple=False).flatten().cpu().numpy()
                        not_reset_env_ids = (1 - wm_is_first).nonzero(as_tuple=False).flatten().cpu().numpy()
                        if (len(not_reset_env_ids) > 0):
                            for k, v in wm_obs.items():
                                if (k != "is_first") and (k != "wm_state"):
                                    self.wm_buffer[k][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], ...] = v[not_reset_env_ids].to('cpu')

                            self.wm_buffer["action"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                wm_action[not_reset_env_ids, :].to('cpu')
                            self.wm_buffer["reward"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                wm_reward[not_reset_env_ids].to('cpu')
                            self.wm_buffer["value"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                wm_value[not_reset_env_ids].to('cpu')

                            self.wm_buffer["com_state"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                wm_com_state[not_reset_env_ids].to('cpu')
                            self.wm_buffer["joint_state"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                wm_joint_state[not_reset_env_ids].to('cpu')

                            self.wm_buffer_index[not_reset_env_ids] += 1

                    # Account for terminal states.
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    if self.cfg['use_amp']:
                        rewards = self.alg.discriminator.predict_amp_reward(
                            amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
                        self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
                    else:
                        self.alg.process_env_step(rewards, dones, infos)

                    amp_obs = torch.clone(next_amp_obs)

                    # process trajectory history
                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.trajectory_history[env_ids] = 0
                    obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                                                        obs[:, self.env.privileged_dim + 9:-self.env.height_dim]),
                                                       dim=1)
                    self.trajectory_history = torch.concat(
                        (self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    
                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, wm_feature.to(self.env.device))

            if self.cfg['use_amp']:
                mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            else:
                mean_value_loss, mean_surrogate_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0 and self.log_dir is not None:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()


            start_time = time.time()
            if (sum_wm_dataset_size > self.wm_config.train_start_steps):
                # Train World Model
                wm_metrics = self.train_world_model()
                for name, values in wm_metrics.items():
                    self.writer.add_scalar('World_model/' + name, float(np.mean(values)), it)
            print('training world model time:', time.time() - start_time)

            # copy the config file
            if(it == 0) and self.log_dir is not None:
                if self.wm_config.task == 'go1_amp':
                    os.system("cp ./legged_gym/envs/go1/go1_amp_config.py " + self.log_dir + "/")
                elif self.wm_config.task == 'cassie':
                    os.system("cp ./legged_gym/envs/cassie/cassie_config.py " + self.log_dir + "/")
                elif self.wm_config.task == 'trona1_w':
                    os.system("cp ./legged_gym/envs/trona1_w/trona1_w_config.py " + self.log_dir + "/")
                else:
                    raise ValueError("Unknown task: {}".format(self.wm_config.task))

                os.system("cp ./dreamer/configs.yaml " + self.log_dir + "/")

        self.current_learning_iteration += num_learning_iterations

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def init_wm_dataset(self):
        self.wm_dataset = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.prop_dim),
                                device=self._world_model.device),

            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions), device=self._world_model.device),

            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, 1),
                                  device=self._world_model.device),
            "value": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, 1),
                                  device=self._world_model.device),

            "com_state": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, 12), 
                                    device=self._world_model.device),

            "joint_state": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, 
                                        16 if self.wm_config.task == 'trona1_w' else 24), 
                                    device=self._world_model.device),

        }
        if(self.env.cfg.depth.use_camera):
            self.wm_dataset["image"] = torch.zeros(((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)

        self.wm_dataset_size = np.zeros(self.env.num_envs)

        self.wm_buffer = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.prop_dim),
                                device='cpu'),

            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions), device='cpu'),

            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, 1),
                                  device='cpu'),
            "value": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, 1),
                                  device='cpu'),

            "com_state": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, 12), 
                                    device='cpu'),

            "joint_state": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, 
                                        16 if self.wm_config.task == 'trona1_w' else 24), 
                                    device='cpu'),
        }
        
        if(self.env.cfg.depth.use_camera):
            self.wm_buffer["image"] = torch.zeros(((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device='cpu')

        self.wm_buffer_index = np.zeros(self.env.num_envs)

    def train_world_model(self):
        wm_metrics = {}
        mets = {}
        for i in range(self.wm_config.train_steps_per_iter):
            p = self.wm_dataset_size / np.sum(self.wm_dataset_size)
            batch_idx = np.random.choice(range(self.env.num_envs), self.wm_config.batch_size, replace=True,
                                         p=p)
            traj_length = min(int(self.wm_dataset_size[batch_idx].min()), self.wm_config.max_traj_length)
            if (traj_length <= 1):
                continue  # an error occur about the predict loss if traj_length < 1
            
            traj_end_idx = np.random.randint(low=traj_length, high=self.wm_dataset_size[batch_idx] + 1, size=(len(batch_idx),))
            batch_data = {}
            for k, v in self.wm_dataset.items():
                value = []
                for idx, end_idx in zip(batch_idx, traj_end_idx):
                    value.append(v[idx, end_idx - traj_length: end_idx])
                value = torch.stack(value)
                batch_data[k] = value


                # # Create a range of indices for slicing
                # traj_indices = np.arange(traj_length+1).reshape(1, -1)  # Shape: (1, traj_length)
                # traj_start_indices = traj_end_idx[:, None] - traj_length  # Shape: (batch_size, 1)
                # batch_traj_indices = traj_start_indices + traj_indices  # Shape: (batch_size, traj_length)

                # # Gather the data for all environments in the batch
                # batch_data[k] = v[batch_idx[:, None], batch_traj_indices]

            is_first = torch.zeros((self.wm_config.batch_size, traj_length))
            is_first[:, 0] = 1
            batch_data["is_first"] = is_first

            _, _, mets = self._world_model._train(batch_data)
        wm_metrics.update(mets)
        return wm_metrics

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        if self.cfg['use_amp']:
            self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
            self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
            self.writer.add_scalar('Loss/AMP_mean_policy_pred', locs['mean_policy_pred'], locs['it'])
            self.writer.add_scalar('Loss/AMP_mean_expert_pred', locs['mean_expert_pred'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'world_model_dict': self._world_model.state_dict(),
            'wm_optimizer_state_dict': self._world_model._model_opt._opt.state_dict(),
            # 'wm_optimizer_state_dict': self._world_model._model_opt.state_dict(),
            # 'discriminator_state_dict': self.alg.discriminator.state_dict(),
            # 'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True, load_wm_optimizer = False):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)
        self._world_model.load_state_dict(loaded_dict['world_model_dict'], strict=False)
        if(load_wm_optimizer):
            self._world_model._model_opt._opt.load_state_dict(loaded_dict['wm_optimizer_state_dict'])
            # self._world_model._model_opt.load_state_dict(loaded_dict['wm_optimizer_state_dict'])
        # self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'], strict=False)
        # self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
