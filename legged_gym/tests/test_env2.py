# # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: BSD-3-Clause
# # 
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# # list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# # this list of conditions and the following disclaimer in the documentation
# # and/or other materials provided with the distribution.
# #
# # 3. Neither the name of the copyright holder nor the names of its
# # contributors may be used to endorse or promote products derived from
# # this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #
# # Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from datetime import datetime
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger

import torch

import time

def test_env(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs =  min(env_cfg.env.num_envs, 4096)

    # env_cfg.terrain.num_rows = 3
    # env_cfg.terrain.num_cols = 8

    # env_cfg.terrain.max_init_terrain_level = 9
    # env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.disturbance = False
    # env_cfg.domain_rand.randomize_payload_mass = False
    # env_cfg.commands.heading_command = False
    # env_cfg.terrain.terrain_proportions = [0,0,1,0,0]

    env_cfg.terrain.curriculum = True
    env_cfg.terrain.selected = False
    env_cfg.depth.use_camera = True
    #1: wave
    # env_cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    # #2: rough slope 
    # env_cfg.terrain.terrain_proportions = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # #3: stairs up.
    # env_cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # #4: stairs down.
    # env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # #5: discrete 
    # env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # #6: gap 
    # env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # #7: pit(not shown) 
    # env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    # #8: tilt
    # env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    # #9: crawl 
    # env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # #10: rough flat
    # env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # generic.
    env_cfg.terrain.terrain_proportions = [0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # policy = ppo_runner.get_inference_policy(device=env.device)


    _ = env.reset()
    vision_obs = env.get_observations()

    for i in range(int(10*env.max_episode_length)):
        with torch.inference_mode():
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

            # actions = policy(obs, vision_obs)

            start = time.time()
            obs, vision_obs, priv_obs, rews, dones, infos, _ = env.step(actions)
            end = time.time()
            # print(f"Time taken for step (ms): {(end-start)*1000}")
            # print(vision_obs['depth'].dtype)
            # print(env.warp_camera_sensor.robot_orientation)
            # print(env.warp_camera_sensor.robot_position)
            # camera_position = env.root_states[0, :3].cpu().numpy() + np.array([3.5, -3.5, 3.0])
            # env_cfg.viewer.lookat = env.root_states[0, :3].cpu().numpy()
            # env.set_camera(camera_position, env_cfg.viewer.lookat)

    print("Done")

if __name__ == '__main__':
    args = get_args()
    test_env(args)