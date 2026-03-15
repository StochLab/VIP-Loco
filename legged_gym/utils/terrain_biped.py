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

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.utils import trimesh
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy.ndimage import binary_dilation


class TerrainBiped:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.added_trimesh = None
        if cfg.curriculum:
            self.curriculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.evaluated_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                             self.cfg.horizontal_scale,
                                                                                             self.cfg.vertical_scale,
                                                                                             self.cfg.slope_treshold)

            if self.added_trimesh is not None:
                self.vertices, self.triangles = trimesh.combine_trimeshes(
                    (self.vertices, self.triangles),
                    self.added_trimesh,
                )

            half_edge_width = int(1)
            structure = np.ones((half_edge_width * 2 + 1, 1))
            self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def evaluated_terrain(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty, i, j)
                self.add_terrain_to_map(terrain, i, j)

    def curriculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty, i, j)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=self.width_per_env_pixels,
                                               length=self.width_per_env_pixels,
                                               vertical_scale=self.vertical_scale,
                                               horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty, i = 0, j = 0):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        amplitude = 0.1 + 0.2 * difficulty
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 0.6 * difficulty
        tilt_width = 0.42 - 0.04 * difficulty
        stair_step_width = 0.30 + random.random() * 0.04
        if choice < self.proportions[0]:
            terrain_utils.wave_terrain(terrain, num_waves=5, amplitude=amplitude)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[1]:
            if choice < (self.proportions[0] + self.proportions[1]) / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=stair_step_width, step_height=step_height,
                                                 platform_size=3.)
            # terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
            #                                      downsampled_scale=0.2)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[5]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=4.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[6]:
            climb_terrain(terrain, depth=pit_depth, platform_size=4.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)

        elif choice < self.proportions[7]:
            center_y = terrain.width // 2
            # width = 1 + 1.0 * np.random.random()
            width = 5
            half_width = width / 2
            y1 = int(center_y - half_width / terrain.horizontal_scale)
            y2 = int(center_y + half_width / terrain.horizontal_scale)
            
            terrain.height_field_raw[:,:] = -1000
            terrain.height_field_raw[: , y1: y2] = 0
            
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            box_z = 1
            box_x = 0.4 + 0.4 * np.random.random()
            tilt_front_left_trimesh = trimesh.box_trimesh(
                np.array([
                    box_x,
                    (self.env_width - tilt_width) / 2,
                    box_z
                ], dtype=np.float32),
                np.array([
                    env_origin_x +  self.cfg.border_size + 2 + box_x / 2,
                    env_origin_y + self.cfg.border_size - tilt_width / 2 - (self.env_width - tilt_width) / 4,# / self.cfg.horizontal_scale,
                    box_z/2,
                ], dtype=np.float32),
            )
            if self.added_trimesh is None:
                self.added_trimesh = tilt_front_left_trimesh
            else:
                self.added_trimesh = trimesh.combine_trimeshes(
                self.added_trimesh,
                tilt_front_left_trimesh,
            )

            tilt_front_right_trimesh = trimesh.box_trimesh(
                np.array([
                    box_x,
                    (self.env_width - tilt_width) / 2,
                    box_z
                ], dtype=np.float32),
                np.array([
                    env_origin_x +  self.cfg.border_size + 2 + box_x / 2,
                    env_origin_y + self.cfg.border_size + tilt_width / 2 + (self.env_width - tilt_width) / 4,# / self.cfg.horizontal_scale,
                    box_z/2,
                ], dtype=np.float32),
            )
            if self.added_trimesh is None:
                self.added_trimesh = tilt_front_right_trimesh
            else:
                self.added_trimesh = trimesh.combine_trimeshes(
                self.added_trimesh,
                tilt_front_right_trimesh,
            )

            tilt_back_left_trimesh = trimesh.box_trimesh(
                np.array([
                    box_x,
                    (self.env_width - tilt_width) / 2,
                    box_z
                ], dtype=np.float32),
                np.array([
                    env_origin_x +  self.cfg.border_size - 2 - box_x / 2,
                    env_origin_y + self.cfg.border_size - tilt_width / 2 - (self.env_width - tilt_width) / 4,# / self.cfg.horizontal_scale,
                    box_z/2,
                ], dtype=np.float32),
            )
            if self.added_trimesh is None:
                self.added_trimesh = tilt_back_left_trimesh
            else:
                self.added_trimesh = trimesh.combine_trimeshes(
                self.added_trimesh,
                tilt_back_left_trimesh,
            )

            tilt_back_right_trimesh = trimesh.box_trimesh(
                np.array([
                    box_x,
                    (self.env_width - tilt_width) / 2,
                    box_z
                ], dtype=np.float32),
                np.array([
                    env_origin_x +  self.cfg.border_size - 2 - box_x / 2,
                    env_origin_y + self.cfg.border_size + tilt_width / 2 + (self.env_width - tilt_width) / 4,# / self.cfg.horizontal_scale,
                    box_z/2,
                ], dtype=np.float32),
            )
            if self.added_trimesh is None:
                self.added_trimesh = tilt_back_right_trimesh
            else:
                self.added_trimesh = trimesh.combine_trimeshes(
                self.added_trimesh,
                tilt_back_right_trimesh,
            )

        elif choice < self.proportions[8]:
            
            center_y = terrain.width // 2
            # width = 1 + 1.0 * np.random.random()
            width = 5
            half_width = width / 2
            y1 = int(center_y - half_width / terrain.horizontal_scale)
            y2 = int(center_y + half_width / terrain.horizontal_scale)
            
            terrain.height_field_raw[:,:] = -1000
            terrain.height_field_raw[: , y1: y2] = 0
            
            crawl_height = 0.75 - 0.15 * difficulty #changed the crawl height from 0.35 -> 0.45
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            box_x = 0.2 + 0.2 * np.random.random()
            box_z = 1.2
            front_upper_bar_trimesh = trimesh.box_trimesh(
                np.array([
                    box_x,
                    self.env_width,
                    box_z
                ], dtype=np.float32),
                np.array([
                    env_origin_x +  self.cfg.border_size + 2 + box_x / 2,
                    env_origin_y + self.cfg.border_size,
                    crawl_height + box_z/2,
                ], dtype=np.float32),
            )
            if self.added_trimesh is None:
                self.added_trimesh = front_upper_bar_trimesh
            else:
                self.added_trimesh = trimesh.combine_trimeshes(
                self.added_trimesh,
                front_upper_bar_trimesh,
            )
            back_upper_bar_trimesh = trimesh.box_trimesh(
                np.array([
                    box_x,
                    self.env_width,
                    box_z
                ], dtype=np.float32),
                np.array([
                    env_origin_x +  self.cfg.border_size + 2 + box_x / 2,
                    env_origin_y + self.cfg.border_size,
                    crawl_height + box_z/2,
                ], dtype=np.float32),
            )
            if self.added_trimesh is None:
                self.added_trimesh = back_upper_bar_trimesh
            else:
                self.added_trimesh = trimesh.combine_trimeshes(
                self.added_trimesh,
                back_upper_bar_trimesh,
            )
        elif choice < self.proportions[9]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.00, max_height=0.000, step=0.005,
                                                 downsampled_scale=0.2)
        return terrain


    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = int(center_x - 1 / terrain.horizontal_scale)
    x2 = int(center_x + 2 / terrain.horizontal_scale)
    x3 = x1 - gap_size
    x4 = x2 + gap_size
    width = 1 + 1.0 * np.random.random()
    half_width = width / 2
    y1 = int(center_y - half_width / terrain.horizontal_scale)
    y2 = int(center_y + half_width / terrain.horizontal_scale)
    x5 = gap_size

    terrain.height_field_raw[:,:] = -1000
    terrain.height_field_raw[x5:x3, y1: y2] = 0
    terrain.height_field_raw[x1: x2, y1: y2] = 0
    terrain.height_field_raw[x4:, y1: y2] = 0

def climb_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    x1 = int(1 / terrain.horizontal_scale)
    length = 1.0 + 0.2 * np.random.random()
    x2 = int((1 + length) / terrain.horizontal_scale)
    x3 = int(6 / terrain.horizontal_scale)
    length = 1.0 + 0.2 * np.random.random()
    x4 = int((6 + length) / terrain.horizontal_scale)
    
    center_y = terrain.width // 2
    width = 5
    half_width = width / 2
    y1 = int(center_y - half_width / terrain.horizontal_scale)
    y2 = int(center_y + half_width / terrain.horizontal_scale)
    
    terrain.height_field_raw[:,:] = -1000
    terrain.height_field_raw[ : x1 , y1: y2] = 0
    terrain.height_field_raw[x2 : , y1: y2] = 0
    # terrain.height_field_raw[: , y1: y2] = 0
    terrain.height_field_raw[x1:x2, y1: y2] = depth
    terrain.height_field_raw[x3:x4, y1: y2] = depth
    # terrain.height_field_raw[ : x3 , y1: y2] = 0
    # terrain.height_field_raw[x4 : , y1: y2] = 0
    
    
def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows - 1, :] += (hf[1:num_rows, :] - hf[:num_rows - 1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols - 1] += (hf[:, 1:num_cols] - hf[:, :num_cols - 1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols - 1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows - 1, :num_cols - 1] += (
                    hf[1:num_rows, 1:num_cols] - hf[:num_rows - 1, :num_cols - 1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (
                    hf[:num_rows - 1, :num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1:stop:2, 0] = ind0
        triangles[start + 1:stop:2, 1] = ind2
        triangles[start + 1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0