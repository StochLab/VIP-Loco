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
import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

MOTION_FILES = glob.glob('datasets/mocap_motions/*')


class CassieCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        # num_observations = 169
        num_contacts = 2
        prop_dim = 33 # proprioception
        action_dim = 12
        include_history_steps = None  # Number of steps of history to include.
        privileged_dim = 24 + 16 + 3  # privileged_obs[:,:privileged_dim] is the privileged information in privileged_obs, include 3-dim base linear vel
        height_dim = 187  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
        forward_height_dim = 525 # for depth image prediction
        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        
    class terrain( LeggedRobotCfg.terrain):
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
        terrain_proportions = [0.0, 0.05, 0.15, 0.15, 0.10, 0.20, 0.20, 0.05, 0.05, 0.05]
        # terrain_proportions = [0.0, 0.2, 0.2, 0.1, 0.0, 0., 0.2, 0., 0.0, 0.3]
        
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'hip_abduction_left': 0.1,
            'hip_rotation_left': 0.,
            'hip_flexion_left': 1.,
            'thigh_joint_left': -1.8,
            'ankle_joint_left': 1.57,
            'toe_joint_left': -1.57,

            'hip_abduction_right': -0.1,
            'hip_rotation_right': 0.,
            'hip_flexion_right': 1.,
            'thigh_joint_right': -1.8,
            'ankle_joint_right': 1.57,
            'toe_joint_right': -1.57
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

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {   'hip_abduction': 100.0, 'hip_rotation': 100.0,
                        'hip_flexion': 200., 'thigh_joint': 200., 'ankle_joint': 200.,
                        'toe_joint': 40.}  # [N*m/rad]
        damping = { 'hip_abduction': 3.0, 'hip_rotation': 3.0,
                    'hip_flexion': 6., 'thigh_joint': 6., 'ankle_joint': 6.,
                    'toe_joint': 1.}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class depth:
        use_camera = True
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.20, 0, 0.03]  # front camera (to be changed)
        euler_deg = [0., 30, 0.]
        y_angle = [-5, 5]  # positive pitch down
        z_angle = [0, 0]
        x_angle = [0, 0]

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0
        far_clip = 4
        dis_noise = 0.0

        scale = 1
        invert = True
    
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf'
        name = "cassie"
        foot_name = 'toe'
        penalize_contacts_on = ['tarsus', 'shin']
        # terminate_after_contacts_on = []
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    
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

        base_height = 0.5 # base height of A1, used to normalize measured height
    
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
            
    class rewards( LeggedRobotCfg.rewards ):
        reward_curriculum = True
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]
        
        soft_dof_pos_limit = 0.95
        base_height_target = 0.65
        foot_height_target = 0.15
        tracking_sigma = 0.15
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        lin_vel_clip = 0.1
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 2.5 # 1.0
            tracking_ang_vel = 0.5 # 1.0
            termination = -200.
            torques = -0.00001 #-5.e-6
            dof_acc = -2.5e-7 #-2.e-7
            base_height = -0.
            feet_air_time = 0.05 # 0.5
            collision = -1.0
            feet_stumble = -0.1
            action_rate = -0.03

            # feet_edge = -1.2 # TODO: why comment?
            dof_error = -0.04 # NEW
            flat_foot = -.1

            dof_pos_limits = -0.1
            no_fly = 0.025 # 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.
            
            lin_vel_z = -1.0
            cheat = -1
            stuck = -1

    class commands:
        curriculum = False
        max_lin_vel_forward_x_curriculum = 2.0
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
            lin_vel_x = [0.0, 2.8]  # min max [m/s]
            lin_vel_y = [-0., 0.]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-0., 0.]

            flat_lin_vel_x = [-0.0, 0.8]  # min max [m/s]
            flat_lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            flat_ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            flat_heading = [-3.14 / 4, 3.14 / 4]


class CassieCfgPPO(LeggedRobotCfgPPO ):
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
        num_learning_epochs = 5
        num_mini_batches = 4
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'parkour'
        experiment_name = 'cassie_viploco_warp_v2'
        algorithm_class_name = 'PPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 10000  # number of policy updates
        save_interval = 250

        min_normalized_std = [0.05, 0.02, 0.05] * 4

        use_amp = False

