from typing import Tuple
import warp as wp
import numpy as np
import math
import torch

from legged_gym.utils.math import (
    quat_from_euler_xyz_tensor, 
    torch_rand_float_tensor, 
    quat_from_euler_xyz,
    tf_apply,
    quat_mul,
)

# Debugging
# wp.config.mode = "debug"
# wp.config.verify_cuda = True

# intialize warp
wp.init()

NO_HIT_RAY_VAL = wp.constant(1000.0)

@wp.kernel
def draw_optimized_kernel_depth_range(
        mesh_ids: wp.array(dtype=wp.uint64), # type: ignore
        cam_poss: wp.array(dtype=wp.vec3, ndim=1), # type: ignore
        cam_quats: wp.array(dtype=wp.quat, ndim=1), # type: ignore
        K_inv: wp.mat44,
        pinhole_quat: wp.quat,
        far_plane: float,
        pixels: wp.array(dtype=float, ndim=3), # type: ignore
        collision_ray: wp.array(dtype=float, ndim=3 + 1), # type: ignore
        c_x: int,
        c_y: int,
        calculate_depth: bool,
        active_mask: wp.array(dtype=wp.uint8, ndim=1),  # 1 if active, 0 otherwise # type: ignore
    ):

    env_id, pixel_height, pixel_width = wp.tid()  # env_id, height, width from current thread id # type: ignore

    # Check if the current robot is active
    if active_mask[env_id] == 0:
        # Set depth to zero for inactive robots
        pixels[env_id, pixel_height, pixel_width] = 0.0
        collision_ray[env_id, pixel_height, pixel_width, 0] = 0.0
        collision_ray[env_id, pixel_height, pixel_width, 1] = 0.0
        collision_ray[env_id, pixel_height, pixel_width, 2] = 0.0
        return

    # Existing logic for active robots
    mesh = mesh_ids[0]  # TODO: Hardcoded terrain for now
    cam_pos = cam_poss[env_id]
    cam_quat = cam_quats[env_id]

    uv = wp.vec3(
        float(pixel_width), float(pixel_height), 1.
        )  # get the vector of pixel
    uv_principal = wp.vec3(
        float(c_x), float(c_y), 1.
        )  # get the vector of principal axis

    # Get ray direction
    rd = wp.normalize(
            wp.quat_rotate(cam_quat,
                wp.quat_rotate(pinhole_quat, wp.transform_vector(K_inv, uv))
            )
        )
    rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat,
                wp.quat_rotate(pinhole_quat, wp.transform_vector(K_inv, uv_principal))
            )
        )

    ro = cam_pos

    multiplier = 1.0
    if calculate_depth:
        # multiplier to project each ray on principal axis for depth instead of range
        multiplier = wp.dot(rd, rd_principal)
    dist = NO_HIT_RAY_VAL

    # result (bool): Whether a hit is found within the given constraints.
    # sign (float32): A value > 0 if the ray hit in front of the face, returns < 0 otherwise.
    # face (int32): Index of the closest face.
    # t (float32): Distance of the closest hit along the ray.
    # u (float32): Barycentric u coordinate of the closest hit.
    # v (float32): Barycentric v coordinate of the closest hit.
    # normal (vec3f): Face normal.
    mesh_query = wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier)
    if mesh_query.result:
        dist = multiplier * mesh_query.t

    pixels[env_id, pixel_height, pixel_width] = dist
    # ray direction * distance
    collision_ray[env_id, pixel_height, pixel_width, 0] = rd[0] * dist
    collision_ray[env_id, pixel_height, pixel_width, 1] = rd[1] * dist
    collision_ray[env_id, pixel_height, pixel_width, 2] = rd[2] * dist


class WarpDepthCam:
    def __init__(self, num_envs, config, 
                    trimesh_vertices_list, trimesh_triangles_list,
                    device="cuda:0", dtype=torch.bfloat16):
        self.cfg = config
        self.num_envs = num_envs
        self.device = device

        assert len(trimesh_triangles_list) == len(trimesh_vertices_list)
        self.mesh_id_list = []
        self.mesh_wp_list = []
        for i in range(len(trimesh_triangles_list)):
            assert type(trimesh_triangles_list[i]) == np.ndarray
            assert type(trimesh_vertices_list[i]) == np.ndarray

            wp_mesh = wp.Mesh(
                    points=wp.from_numpy(trimesh_vertices_list[i], dtype=wp.vec3, device=self.device),
                    indices=wp.from_numpy(trimesh_triangles_list[i].flatten(), dtype=wp.int32, device=self.device),
                )
            self.mesh_wp_list.append(wp_mesh)
            self.mesh_id_list.append(wp_mesh.id)

        self.mesh_ids_array = wp.array(self.mesh_id_list, dtype=wp.uint64, device=self.device)


        self.height = self.cfg.resized[0]
        self.width = self.cfg.resized[1]

        self.horizontal_fov = math.radians(self.cfg.horizontal_fov)
        self.far_plane = self.cfg.far_clip
        self.calculate_depth = self.cfg.use_camera

        self.active_mask = None
        self.camera_pos_wrt_world = None
        self.camera_quat_wrt_world = None
        self.graph = None

        self.torch_depth_dtype = dtype

        self.initialize_buffers()

    def initialize_buffers(self):
        # Calculate camera params
        H = self.height
        W = self.width
        (u_0, v_0) = (W / 2, H / 2)
        f = (W / 2) * (1 / math.tan(self.horizontal_fov / 2))

        vertical_fov = 2 * math.atan(H / (2 * f))
        alpha_u = u_0 / math.tan(self.horizontal_fov / 2)
        alpha_v = v_0 / math.tan(vertical_fov / 2)

        # simple pinhole model
        self.K = wp.mat44(
            alpha_u, 0.0,     u_0, 0.0,
            0.0,     alpha_v, v_0, 0.0,
            0.0,     0.0,     1.0, 0.0, 
            0.0,     0.0,     0.0, 1.0,
        )
        self.K_inv = wp.inverse(self.K)

        # Rot_pinhole since camera frame is with z-axis pointing outwards
        torch_quat = quat_from_euler_xyz_tensor(
                torch.tensor([-np.pi/2, 0.0, -np.pi/2], device=self.device)
            )
        self.pinhole_quat = wp.quat(torch_quat[0], torch_quat[1], torch_quat[2], torch_quat[3])

        self.c_x = int(u_0)
        self.c_y = int(v_0)

        ##############################
        # Output tensors
        self.pixels = wp.zeros(
                (self.num_envs, self.height, self.width),
                device=self.device,
                requires_grad=False,
                dtype=wp.float32,
            )

        self.collision_rays = wp.zeros(
                (self.num_envs, self.height, self.width, 3),
                dtype=wp.float32, device=self.device
            )

    def create_render_graph_depth_range(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)

        # with wp.ScopedTimer("render"):
        wp.launch(
            kernel=draw_optimized_kernel_depth_range,
            dim=(self.num_envs, self.height, self.width),
            inputs=[
                self.mesh_ids_array,
                self.camera_pos_wrt_world,
                self.camera_quat_wrt_world,
                self.K_inv,
                self.pinhole_quat,
                self.far_plane,
                self.pixels, # Output
                self.collision_rays, # Output
                self.c_x,
                self.c_y,
                self.calculate_depth,
                self.active_mask, # 1 if active, 0 otherwise
            ],
            device=self.device,
        )

        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)

    def capture(self, active_mask, positions, orientations, debug=False):

        if self.camera_pos_wrt_world is None or self.camera_quat_wrt_world is None:
            self.active_mask = wp.from_torch(active_mask, dtype=wp.uint8)
            self.camera_pos_wrt_world = wp.from_torch(positions, dtype=wp.vec3)
            self.camera_quat_wrt_world = wp.from_torch(orientations, dtype=wp.quat)

        if self.graph is None:
            self.create_render_graph_depth_range(debug=debug)

        if self.graph is not None:
            wp.capture_launch(self.graph)

        # if debug:
        # print(self.collision_rays[0, self.height//2, self.width//2])
        # print('Min:', torch.min(wp.to_torch(self.pixels)), 
        #       ' Max:', torch.max(wp.to_torch(self.pixels)))

        return wp.to_torch(self.pixels).to(dtype=self.torch_depth_dtype)


class DepthCamSensor:
    def __init__(self, sensor_config, num_envs, device, 
                    # List of trimesh objects eg. terrain, obstacles, ...
                    trimesh_vertices_list: Tuple[np.ndarray],
                    trimesh_triangles_list: Tuple[np.ndarray], 
                    depth_dtype
                ):
        # Multi sensor support is not implemented yet but can be done
        print("\033[93m" + "DepthCamSensor" + \
            "will only use static terrain trimesh, if you want to add obstacles " + \
            "please use `trimesh` python library to concatenate it. " + \
            "Further if the terrains or these obstacles change during training " + \
            "then make sure to reload this class." + "\033[0m")
        
        self.cfg = sensor_config
        self.device = device
        self.num_envs = num_envs
        self.depth_dtype = depth_dtype

        # cannot use dtype cuz c++ funcs are only implemented for float32
        self.camera = WarpDepthCam(
                num_envs=self.num_envs,
                config=self.cfg,
                trimesh_vertices_list=trimesh_vertices_list,
                trimesh_triangles_list=trimesh_triangles_list,
                device=self.device,
                dtype=depth_dtype
            )

        # # Debug trimesh
        # import trimesh
        # for i in range(len(trimesh_triangles_list)):
        #     trimesh_obj = trimesh.Trimesh(vertices=trimesh_vertices_list[i], faces=trimesh_triangles_list[i])
        #     trimesh_obj.show()

    def init_tensors(self, env_origin_trimesh):
        self.env_origin_trimesh = env_origin_trimesh

        # Sensor in world frame
        # TODO: Include angle variation
        offset_euler_deg = torch.deg2rad(
                torch.tensor(self.cfg.euler_deg, device=self.device, requires_grad=False)
            )
        offset_quat = quat_from_euler_xyz_tensor(offset_euler_deg)
        self.offset_quat = offset_quat.expand(self.num_envs, -1)
        self.sensor_quat_wrt_robot = self.offset_quat.clone()

        self.offset_pos = torch.tensor(
                self.cfg.position, device=self.device, requires_grad=False
            ).expand(self.num_envs, -1)
        self.sensor_pos_wrt_robot = self.offset_pos.clone()


        # For Randomization
        # TODO: Add support for translation randomization
        # self.sensor_min_translation = torch.tensor(
        #         [self.cfg.min_translation], device=self.device, requires_grad=False
        #     ).expand(self.num_envs, -1)
        # self.sensor_max_translation = torch.tensor(
        #         [self.cfg.max_translation], device=self.device, requires_grad=False
        #     ).expand(self.num_envs, -1)

        self.sensor_min_euler = torch.deg2rad(
                torch.tensor((self.cfg.x_angle[0], self.cfg.y_angle[0], self.cfg.z_angle[0]), 
                            device=self.device, requires_grad=False)
            ).expand(self.num_envs, -1)
        self.sensor_max_euler = torch.deg2rad(
                torch.tensor((self.cfg.x_angle[1], self.cfg.y_angle[1], self.cfg.z_angle[1]), 
                            device=self.device, requires_grad=False)
            ).expand(self.num_envs, -1)

        self.sensor_pos_wrt_world = self.sensor_pos_wrt_robot.clone()
        self.sensor_quat_wrt_world = self.sensor_quat_wrt_robot.clone()
        
        self.depth_active_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.uint8)

        self.reset()

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        # sample local position from min and max translations
        # self.sensor_pos_wrt_robot[env_ids] = torch_rand_float_tensor(
        #         self.sensor_min_translation[env_ids],
        #         self.sensor_max_translation[env_ids],
        #     ) + self.offset_pos[env_ids]

        self.sensor_pos_wrt_robot[env_ids] = self.offset_pos[env_ids]


        # sample local orientation from min and max rotations
        noise_euler = torch_rand_float_tensor(
                self.sensor_min_euler[env_ids], self.sensor_max_euler[env_ids]
            )
        noise_quat = quat_from_euler_xyz(
                noise_euler[..., 0],
                noise_euler[..., 1],
                noise_euler[..., 2],
            )
        self.sensor_quat_wrt_robot[env_ids] = quat_mul(
                self.offset_quat[env_ids], noise_quat
            )

    def update(self, depth_active_mask, robot_position, robot_orientation):
        self.depth_active_mask[:] = depth_active_mask

        # update the sensor position and orientation in world frame
        self.sensor_pos_wrt_world[:] = self.sensor_pos_wrt_robot + robot_position + self.env_origin_trimesh
        self.sensor_quat_wrt_world[:] = quat_mul(
            self.sensor_quat_wrt_robot, robot_orientation)

        # capture the sensor data
        depth_image = self.camera.capture(
                self.depth_active_mask, self.sensor_pos_wrt_world, self.sensor_quat_wrt_world, debug=False
            )

        depth_image = torch.normal(
                mean=depth_image,
                std=self.cfg.dis_noise * torch.ones_like(depth_image),
            )

        # Drop out with probability self.cfg.pixel_dropout_prob
        # TODO: Implement this
        # mask = torch.rand_like(depth_image) > self.cfg.pixel_dropout_prob
        # depth_image = depth_image * mask

        # Apply Range Limits
        depth_image[depth_image > self.cfg.far_clip] = self.cfg.far_clip
        depth_image[depth_image < self.cfg.near_clip] = self.cfg.near_clip

        # normalize the range values to be between -0.5 and 0.5
        depth_image = -0.5 + ((depth_image - self.cfg.near_clip) / (self.cfg.far_clip - self.cfg.near_clip))

        return depth_image.clone().reshape(self.num_envs, 
                    1, # For time dimension 
                    self.cfg.resized[0], self.cfg.resized[1])