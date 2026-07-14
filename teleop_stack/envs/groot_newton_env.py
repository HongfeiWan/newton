"""Headless, batched Newton environment for the dual Nero + L10 scene.

The steady-state interface keeps simulation state, actions, camera images, rewards,
and episode flags on the Warp device. Use :meth:`GrootNewtonEnv.observation_torch`
for zero-copy Torch views when the trainer uses CUDA tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

import newton
from debug import import_dual_nero_linker_l10 as scene_runtime
from newton.sensors import SensorTiledCamera
from newton.viewer import ViewerNull

ARM_JOINT_NAMES = tuple(f"right_joint{index}" for index in range(1, 8))
HAND_JOINT_NAMES = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
    "index_mcp_roll",
    "ring_mcp_roll",
    "pinky_mcp_roll",
    "thumb_cmc_roll",
)
ACTION_SIZE = len(ARM_JOINT_NAMES) + len(HAND_JOINT_NAMES)

_HAND_COMMAND_LIMITS = (
    (0.0, 0.5146),
    (0.0, 1.9189),
    (0.0, 1.3607),
    (0.0, 1.3607),
    (0.0, 1.3607),
    (0.0, 1.3607),
    (0.0, 0.2181),
    (0.0, 0.2181),
    (0.0, 0.3489),
    (0.0, 1.1339),
)
_HAND_SDK_LIMITS = (
    (0.0, 0.75),
    (0.0, 1.43),
    (0.0, 1.62),
    (0.0, 1.62),
    (0.0, 1.62),
    (0.0, 1.62),
    (-0.26, 0.21),
    (0.0, 0.21),
    (0.0, 0.34),
    (-0.52, 1.01),
)
_HAND_RAW_REVERSED = (True, True, True, True, True, True, False, False, False, True)
_HAND_OBSERVATION_LOWER = (0.0, 0.0, 0.0053360784, 0.0053360784, 0.0053360784, 0.0053360784, 0.0, 0.0, 0.0, 0.0)

_NERO_MDH = (
    (0.138, 0.0, 0.0, 0.0),
    (0.0, 0.0, math.pi / 2.0, math.pi),
    (0.31, 0.0, math.pi / 2.0, math.pi),
    (0.0, 0.0, math.pi / 2.0, math.pi),
    (0.27001, 0.0, math.pi / 2.0, math.pi / 2.0),
    (0.0, 0.0, math.pi / 2.0, math.pi / 2.0),
    (0.0235, 0.0, math.pi / 2.0, 0.0),
)


@dataclass(frozen=True)
class GrootNewtonEnvConfig:
    """Configuration for :class:`GrootNewtonEnv`.

    The default timing matches the existing simulator: a policy action is held
    for six 60 Hz frames and each frame contains sixteen physics substeps.
    """

    num_envs: int = 1
    device: str = "cuda:0"
    control_hz: int = 10
    simulation_hz: int = 60
    substeps_per_frame: int = 16
    max_episode_steps: int = 0
    capture_graph: bool = True
    render_images: bool = True
    camera_textures: bool = True
    load_scene_visuals: bool = True
    hydroelastic_contacts: bool = True
    ego_width: int = 320
    ego_height: int = 180
    wrist_width: int = 640
    wrist_height: int = 480
    rigid_contacts_per_env: int = 1024
    mujoco_njmax: int = 2048
    mujoco_nconmax: int = 1024

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.control_hz < 1 or self.simulation_hz < 1:
            raise ValueError("control_hz and simulation_hz must be positive")
        if self.simulation_hz % self.control_hz != 0:
            raise ValueError("simulation_hz must be an integer multiple of control_hz")
        if self.substeps_per_frame < 1:
            raise ValueError("substeps_per_frame must be positive")
        if self.capture_graph and self.substeps_per_frame % 2 != 0:
            raise ValueError("capture_graph requires an even substeps_per_frame so state buffers do not alias")
        if min(self.ego_width, self.ego_height, self.wrist_width, self.wrist_height) < 1:
            raise ValueError("camera dimensions must be positive")


@wp.func
def _reported_hand_position(
    command: wp.float32,
    command_lower: wp.float32,
    command_upper: wp.float32,
    sdk_lower: wp.float32,
    sdk_upper: wp.float32,
    raw_reversed: wp.int32,
    observation_lower: wp.float32,
) -> wp.float32:
    value = wp.clamp(command, command_lower, command_upper)
    ratio = (value - command_lower) / (command_upper - command_lower)
    sdk_value = sdk_lower + ratio * (sdk_upper - sdk_lower)
    sdk_ratio = (sdk_value - sdk_lower) / (sdk_upper - sdk_lower)
    raw_float = sdk_ratio * 255.0
    if raw_reversed != 0:
        raw_float = (1.0 - sdk_ratio) * 255.0
    raw = wp.floor(wp.clamp(raw_float, 0.0, 255.0) + 0.5)
    reported_sdk_ratio = raw / 255.0
    if raw_reversed != 0:
        reported_sdk_ratio = (255.0 - raw) / 255.0
    reported = command_lower + reported_sdk_ratio * (command_upper - command_lower)
    return wp.max(observation_lower, wp.clamp(reported, command_lower, command_upper))


@wp.kernel(enable_backward=False)
def _apply_joint_targets(
    action: wp.array2d[wp.float32],
    joint_coord_world_start: wp.array[wp.int32],
    joint_dof_world_start: wp.array[wp.int32],
    local_q_indices: wp.array[wp.int32],
    local_qd_indices: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
):
    world, slot = wp.tid()
    q_index = joint_coord_world_start[world] + local_q_indices[slot]
    qd_index = joint_dof_world_start[world] + local_qd_indices[slot]
    target_q[q_index] = wp.clamp(action[world, slot], joint_limit_lower[qd_index], joint_limit_upper[qd_index])
    target_qd[qd_index] = 0.0


@wp.kernel(enable_backward=False)
def _gather_joint_position_action(
    joint_q: wp.array[wp.float32],
    joint_coord_world_start: wp.array[wp.int32],
    local_q_indices: wp.array[wp.int32],
    action: wp.array2d[wp.float32],
):
    world, slot = wp.tid()
    action[world, slot] = joint_q[joint_coord_world_start[world] + local_q_indices[slot]]


@wp.kernel(enable_backward=False)
def _extract_state(
    joint_q: wp.array[wp.float32],
    joint_coord_world_start: wp.array[wp.int32],
    arm_local_q_indices: wp.array[wp.int32],
    hand_local_q_indices: wp.array[wp.int32],
    hand_command_limits: wp.array2d[wp.float32],
    hand_sdk_limits: wp.array2d[wp.float32],
    hand_raw_reversed: wp.array[wp.int32],
    hand_observation_lower: wp.array[wp.float32],
    mdh: wp.array2d[wp.float32],
    arm_out: wp.array2d[wp.float32],
    hand_out: wp.array2d[wp.float32],
    eef_out: wp.array2d[wp.float32],
):
    world = wp.tid()
    q_start = joint_coord_world_start[world]

    rotation = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    position = wp.vec3(0.0)
    for joint in range(7):
        q = joint_q[q_start + arm_local_q_indices[joint]]
        arm_out[world, joint] = q
        d_i = mdh[joint, 0]
        a_i = mdh[joint, 1]
        alpha_i = mdh[joint, 2]
        theta = q + mdh[joint, 3]
        ca = wp.cos(alpha_i)
        sa = wp.sin(alpha_i)
        ct = wp.cos(theta)
        st = wp.sin(theta)
        link_rotation = wp.mat33(ct, -st, 0.0, ca * st, ca * ct, -sa, sa * st, sa * ct, ca)
        link_position = wp.vec3(a_i, -sa * d_i, ca * d_i)
        position = position + rotation * link_position
        rotation = rotation * link_rotation

    eef_out[world, 0] = position[0]
    eef_out[world, 1] = position[1]
    eef_out[world, 2] = position[2]
    eef_out[world, 3] = rotation[0, 0]
    eef_out[world, 4] = rotation[0, 1]
    eef_out[world, 5] = rotation[0, 2]
    eef_out[world, 6] = rotation[1, 0]
    eef_out[world, 7] = rotation[1, 1]
    eef_out[world, 8] = rotation[1, 2]

    for joint in range(10):
        command = joint_q[q_start + hand_local_q_indices[joint]]
        hand_out[world, joint] = _reported_hand_position(
            command,
            hand_command_limits[joint, 0],
            hand_command_limits[joint, 1],
            hand_sdk_limits[joint, 0],
            hand_sdk_limits[joint, 1],
            hand_raw_reversed[joint],
            hand_observation_lower[joint],
        )


@wp.kernel(enable_backward=False)
def _compute_camera_transforms(
    body_q: wp.array[wp.transform],
    body_world_start: wp.array[wp.int32],
    local_body_index: wp.int32,
    local_camera_transform: wp.transform,
    camera_transforms: wp.array2d[wp.transform],
):
    world = wp.tid()
    body_transform = body_q[body_world_start[world] + local_body_index]
    camera_transforms[0, world] = wp.transform_multiply(body_transform, local_camera_transform)


@wp.kernel(enable_backward=False)
def _compute_roi_camera_rays(
    native_width: wp.int32,
    native_height: wp.int32,
    crop_x: wp.int32,
    crop_y: wp.int32,
    crop_width: wp.int32,
    crop_height: wp.int32,
    output_width: wp.int32,
    output_height: wp.int32,
    vertical_fov: wp.float32,
    rays: wp.array4d[wp.vec3f],
):
    y, x = wp.tid()
    source_x = float(crop_x) + (float(x) + 0.5) * float(crop_width) / float(output_width)
    source_y = float(crop_y) + (float(y) + 0.5) * float(crop_height) / float(output_height)
    u = source_x / float(native_width) - 0.5
    v = source_y / float(native_height) - 0.5
    h = wp.tan(vertical_fov * 0.5)
    aspect = float(native_width) / float(native_height)
    direction = wp.normalize(wp.vec3(u * 2.0 * h * aspect, -v * 2.0 * h, -1.0))
    rays[0, y, x, 0] = wp.vec3(0.0)
    rays[0, y, x, 1] = direction


@wp.kernel(enable_backward=False)
def _unpack_rgb(packed: wp.array4d[wp.uint32], rgb: wp.array4d[wp.uint8]):
    world, y, x = wp.tid()
    color = packed[world, 0, y, x]
    rgb[world, y, x, 0] = wp.uint8(color & wp.uint32(0xFF))
    rgb[world, y, x, 1] = wp.uint8((color >> wp.uint32(8)) & wp.uint32(0xFF))
    rgb[world, y, x, 2] = wp.uint8((color >> wp.uint32(16)) & wp.uint32(0xFF))


@wp.kernel(enable_backward=False)
def _advance_episode(
    episode_step: wp.array[wp.int32],
    reward: wp.array[wp.float32],
    terminated: wp.array[wp.bool],
    truncated: wp.array[wp.bool],
    max_episode_steps: wp.int32,
):
    world = wp.tid()
    episode_step[world] = episode_step[world] + 1
    reward[world] = 0.0
    terminated[world] = False
    truncated[world] = max_episode_steps > 0 and episode_step[world] >= max_episode_steps


@wp.kernel(enable_backward=False)
def _reset_episode_arrays(
    world_mask: wp.array[wp.bool],
    episode_step: wp.array[wp.int32],
    reward: wp.array[wp.float32],
    terminated: wp.array[wp.bool],
    truncated: wp.array[wp.bool],
):
    world = wp.tid()
    if not world_mask or world_mask[world]:
        episode_step[world] = 0
        reward[world] = 0.0
        terminated[world] = False
        truncated[world] = False


@wp.kernel(enable_backward=False)
def _reset_control_targets(
    world_mask: wp.array[wp.bool],
    joint_coord_world_start: wp.array[wp.int32],
    joint_dof_world_start: wp.array[wp.int32],
    default_target_q: wp.array[wp.float32],
    default_target_qd: wp.array[wp.float32],
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
    coords_per_world: wp.int32,
    dofs_per_world: wp.int32,
):
    world, index = wp.tid()
    if not world_mask or world_mask[world]:
        if index < coords_per_world:
            q_index = joint_coord_world_start[world] + index
            target_q[q_index] = default_target_q[q_index]
        if index < dofs_per_world:
            qd_index = joint_dof_world_start[world] + index
            target_qd[qd_index] = default_target_qd[qd_index]


@wp.kernel(enable_backward=False)
def _clear_reset_qfrc(
    world_mask: wp.array[wp.bool],
    qfrc: wp.array[wp.float32],
    joint_dof_world_start: wp.array[wp.int32],
    dofs_per_world: wp.int32,
):
    world, dof = wp.tid()
    if (not world_mask or world_mask[world]) and dof < dofs_per_world:
        qfrc[joint_dof_world_start[world] + dof] = 0.0


class GrootNewtonEnv:
    """Batched, headless RL environment with GPU-resident observations.

    Actions have shape ``[num_envs, 17]`` in the order
    ``right_joint1..7`` followed by :data:`HAND_JOINT_NAMES`. Values are
    position targets in simulator joint coordinates and are clipped to the
    imported URDF limits.
    """

    def __init__(self, config: GrootNewtonEnvConfig | None = None):
        self.config = config or GrootNewtonEnvConfig()
        self.num_envs = self.config.num_envs
        self.device = wp.get_device(self.config.device)
        self.frames_per_action = self.config.simulation_hz // self.config.control_hz

        args = scene_runtime.Example.create_parser().parse_args([])
        args.device = self.config.device
        args.viewer = "null"
        args.headless = True
        args.fps = float(self.config.simulation_hz)
        args.substeps = self.config.substeps_per_frame
        args.capture_graph = self.config.capture_graph
        args.world_count = self.num_envs
        args.replicate_worlds = True
        args.request_qfrc_actuator = True
        args.quest_teleop = False
        args.d455_preview = False
        args.d405_preview = False
        if not self.config.render_images or not self.config.load_scene_visuals:
            args.scene_glb = scene_runtime.REPO_ROOT / "__headless_visuals_disabled__.glb"
        if not self.config.render_images:
            args.d405_body_visual = False
        args.d455_opencv_window = False
        args.d405_opencv_window = False
        args.hydroelastic_contacts = self.config.hydroelastic_contacts
        args.viewer_contacts = False
        args.viewer_hydro_contact_surface = False
        args.l10_bottle_contact_log = False
        args.l10_bottle_contact_stop = False
        args.enforce_bottle_above_scene_collision = False
        args.initial_right_arm_q = (
            0.2724284429,
            1.6012174157,
            1.4535451076,
            1.2643514167,
            0.2993937799,
            -0.0534419817,
            0.1828232391,
        )
        args.d405_fov = 72.0
        args.d405_connector_rel_euler = (89.483, -1.020, -2.995)
        args.hydroelastic_rigid_contact_max = self.config.rigid_contacts_per_env * self.num_envs
        args.mujoco_njmax = self.config.mujoco_njmax
        args.mujoco_nconmax = self.config.mujoco_nconmax

        self._scene = scene_runtime.Example(ViewerNull(num_frames=2**31 - 1), args)
        self.model = self._scene.model
        self.state_0 = self._scene.state_0
        self.state_1 = self._scene.state_1
        self.control = self._scene.control
        self.solver = self._scene.solver
        if self.model.world_count != self.num_envs:
            raise RuntimeError(f"Expected {self.num_envs} worlds, built {self.model.world_count}")
        if self.solver is None:
            raise RuntimeError("GrootNewtonEnv requires the MuJoCo solver")

        self.coords_per_world = self.model.joint_coord_count // self.num_envs
        self.dofs_per_world = self.model.joint_dof_count // self.num_envs
        self._setup_joint_indices()
        self._setup_observation_arrays()
        self._setup_episode_arrays()
        self._setup_cameras(args)
        self._refresh_observation()

    def _setup_joint_indices(self) -> None:
        joint_world_start = self.model.joint_world_start.numpy()
        joint_q_start = self.model.joint_q_start.numpy()
        joint_qd_start = self.model.joint_qd_start.numpy()
        first_joint = int(joint_world_start[0])
        last_joint = int(joint_world_start[1])
        q_origin = int(self.model.joint_coord_world_start.numpy()[0])
        qd_origin = int(self.model.joint_dof_world_start.numpy()[0])

        def find_local_indices(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
            q_indices = []
            qd_indices = []
            for name in names:
                suffix = f"/{name}"
                matches = [
                    index
                    for index in range(first_joint, last_joint)
                    if self.model.joint_label[index] == name or self.model.joint_label[index].endswith(suffix)
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one joint named {name!r} in world 0, got {len(matches)}")
                joint = matches[0]
                q_indices.append(int(joint_q_start[joint]) - q_origin)
                qd_indices.append(int(joint_qd_start[joint]) - qd_origin)
            return np.asarray(q_indices, dtype=np.int32), np.asarray(qd_indices, dtype=np.int32)

        arm_q, arm_qd = find_local_indices(ARM_JOINT_NAMES)
        hand_labels = tuple(f"right_l10_{name}" for name in HAND_JOINT_NAMES)
        hand_q, hand_qd = find_local_indices(hand_labels)
        self._arm_local_q = wp.array(arm_q, dtype=wp.int32, device=self.device)
        self._hand_local_q = wp.array(hand_q, dtype=wp.int32, device=self.device)
        self._action_local_q = wp.array(np.concatenate((arm_q, hand_q)), dtype=wp.int32, device=self.device)
        self._action_local_qd = wp.array(np.concatenate((arm_qd, hand_qd)), dtype=wp.int32, device=self.device)

    def _setup_observation_arrays(self) -> None:
        self._action = wp.zeros((self.num_envs, ACTION_SIZE), dtype=wp.float32, device=self.device)
        self._eef_9d = wp.zeros((self.num_envs, 9), dtype=wp.float32, device=self.device)
        self._arm_joint_pos = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=self.device)
        self._hand_joint_pos = wp.zeros((self.num_envs, 10), dtype=wp.float32, device=self.device)
        self._hand_command_limits = wp.array(
            np.asarray(_HAND_COMMAND_LIMITS, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self._hand_sdk_limits = wp.array(
            np.asarray(_HAND_SDK_LIMITS, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self._hand_raw_reversed = wp.array(
            np.asarray(_HAND_RAW_REVERSED, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._hand_observation_lower = wp.array(
            np.asarray(_HAND_OBSERVATION_LOWER, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self._mdh = wp.array(np.asarray(_NERO_MDH, dtype=np.float32), dtype=wp.float32, device=self.device)

    def _setup_episode_arrays(self) -> None:
        self.episode_step = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self.reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.terminated = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self.truncated = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)

    def _find_local_body_index(self, suffix: str) -> int:
        body_world_start = self.model.body_world_start.numpy()
        first_body = int(body_world_start[0])
        last_body = int(body_world_start[1])
        matches = [index for index in range(first_body, last_body) if self.model.body_label[index].endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one body ending with {suffix!r} in world 0, got {len(matches)}")
        return matches[0] - first_body

    def _setup_cameras(self, args: Any) -> None:
        self._camera_sensor = None
        self._ego_rgb = wp.zeros(
            (self.num_envs, self.config.ego_height, self.config.ego_width, 3),
            dtype=wp.uint8,
            device=self.device,
        )
        self._wrist_rgb = wp.zeros(
            (self.num_envs, self.config.wrist_height, self.config.wrist_width, 3),
            dtype=wp.uint8,
            device=self.device,
        )
        if not self.config.render_images:
            return

        render_config = SensorTiledCamera.RenderConfig(
            enable_textures=self.config.camera_textures,
            enable_shadows=False,
        )
        self._camera_sensor = SensorTiledCamera(
            self.model,
            config=render_config,
            load_textures=self.config.camera_textures,
        )
        self._camera_sensor.utils.create_default_light(enable_shadows=False, direction=wp.vec3(0.0, 0.0, -1.0))

        d455_config = scene_runtime._load_d455_config(args.d455_json)
        native_width, native_height = (int(value) for value in d455_config["rgb_res"])
        crop_x, crop_y, crop_width, crop_height = scene_runtime._roi_crop_rect(
            native_width,
            native_height,
            zoom=float(args.d455_roi_zoom),
            center_x=float(args.d455_roi_center_x),
            center_y=float(args.d455_roi_center_y),
        )
        self._ego_rays = wp.empty(
            (1, self.config.ego_height, self.config.ego_width, 2), dtype=wp.vec3, device=self.device
        )
        wp.launch(
            _compute_roi_camera_rays,
            dim=(self.config.ego_height, self.config.ego_width),
            inputs=[
                native_width,
                native_height,
                crop_x,
                crop_y,
                crop_width,
                crop_height,
                self.config.ego_width,
                self.config.ego_height,
                math.radians(float(args.d455_fov or d455_config["rgb_fov"])),
                self._ego_rays,
            ],
            device=self.device,
        )
        self._wrist_rays = self._camera_sensor.utils.compute_pinhole_camera_rays(
            self.config.wrist_width,
            self.config.wrist_height,
            math.radians(float(args.d405_fov)),
        )
        self._ego_packed = self._camera_sensor.utils.create_color_image_output(
            self.config.ego_width, self.config.ego_height
        )
        self._wrist_packed = self._camera_sensor.utils.create_color_image_output(
            self.config.wrist_width, self.config.wrist_height
        )
        self._ego_camera_transforms = wp.empty((1, self.num_envs), dtype=wp.transform, device=self.device)
        self._wrist_camera_transforms = wp.empty((1, self.num_envs), dtype=wp.transform, device=self.device)

        d455_position = np.asarray(
            (float(self._scene.d455_body_size[0]) * 0.5 + float(args.d455_front_clearance), 0.0, 0.0)
        )
        self._d455_local_camera = scene_runtime._camera_transform_from_forward_up(
            d455_position,
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
        )

        mount_rotation = scene_runtime._rotation_from_euler_deg(tuple(self._scene.d405_connector_rel_euler))
        d405_local_position = np.asarray(self._scene.d405_body_size) * np.asarray(
            scene_runtime.D405_CAMERA_LOCAL_POS_RATIO
        )
        d405_local_position[2] += float(args.d405_front_clearance)
        d405_camera_in_body = scene_runtime._camera_transform_from_forward_up(
            d405_local_position,
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((0.0, 1.0, 0.0)),
        )
        d405_mount = wp.transform(
            wp.vec3(*self._scene.d405_connector_rel_pos),
            wp.quat(*scene_runtime._quat_xyzw_from_rotation(mount_rotation)),
        )
        self._d405_local_camera = wp.transform_multiply(d405_mount, d405_camera_in_body)
        self._d455_body_local = self._find_local_body_index(scene_runtime.D455_BODY_LABEL_SUFFIX)
        self._d405_connector_local = self._find_local_body_index("/right_connector")

    def _copy_action(self, action: Any) -> None:
        if isinstance(action, wp.array):
            action_wp = action
        else:
            try:
                action_wp = wp.from_torch(action.contiguous(), dtype=wp.float32, requires_grad=False)
            except (AttributeError, TypeError) as exc:
                raise TypeError("action must be a Warp array or a CUDA Torch tensor") from exc
        if action_wp.device != self.device:
            raise ValueError(f"action must be on {self.device}, got {action_wp.device}")
        if action_wp.shape != self._action.shape:
            raise ValueError(f"action must have shape {self._action.shape}, got {action_wp.shape}")
        if action_wp is self._action:
            return
        wp.copy(self._action, action_wp)

    def _apply_action(self) -> None:
        wp.launch(
            _apply_joint_targets,
            dim=(self.num_envs, ACTION_SIZE),
            inputs=[
                self._action,
                self.model.joint_coord_world_start,
                self.model.joint_dof_world_start,
                self._action_local_q,
                self._action_local_qd,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self.control.joint_target_q,
                self.control.joint_target_qd,
            ],
            device=self.device,
        )

    def _render_cameras(self) -> None:
        if self._camera_sensor is None:
            return
        self.model.bvh_refit_shapes(self.state_0)
        if self.model.particle_count:
            self.model.bvh_refit_particles(self.state_0)
        wp.launch(
            _compute_camera_transforms,
            dim=self.num_envs,
            inputs=[
                self.state_0.body_q,
                self.model.body_world_start,
                self._d455_body_local,
                self._d455_local_camera,
                self._ego_camera_transforms,
            ],
            device=self.device,
        )
        wp.launch(
            _compute_camera_transforms,
            dim=self.num_envs,
            inputs=[
                self.state_0.body_q,
                self.model.body_world_start,
                self._d405_connector_local,
                self._d405_local_camera,
                self._wrist_camera_transforms,
            ],
            device=self.device,
        )
        self._camera_sensor.update(
            self.state_0,
            self._ego_camera_transforms,
            self._ego_rays,
            color_image=self._ego_packed,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        self._camera_sensor.update(
            self.state_0,
            self._wrist_camera_transforms,
            self._wrist_rays,
            color_image=self._wrist_packed,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        wp.launch(
            _unpack_rgb,
            dim=(self.num_envs, self.config.ego_height, self.config.ego_width),
            inputs=[self._ego_packed, self._ego_rgb],
            device=self.device,
        )
        wp.launch(
            _unpack_rgb,
            dim=(self.num_envs, self.config.wrist_height, self.config.wrist_width),
            inputs=[self._wrist_packed, self._wrist_rgb],
            device=self.device,
        )

    def _refresh_observation(self) -> None:
        wp.launch(
            _extract_state,
            dim=self.num_envs,
            inputs=[
                self.state_0.joint_q,
                self.model.joint_coord_world_start,
                self._arm_local_q,
                self._hand_local_q,
                self._hand_command_limits,
                self._hand_sdk_limits,
                self._hand_raw_reversed,
                self._hand_observation_lower,
                self._mdh,
                self._arm_joint_pos,
                self._hand_joint_pos,
                self._eef_9d,
            ],
            device=self.device,
        )
        self._render_cameras()

    def observation(self) -> dict[str, dict[str, wp.array]]:
        """Return device-resident Warp views of the current observation."""
        qfrc = self.state_0.mujoco.qfrc_actuator.reshape((self.num_envs, self.dofs_per_world))
        return {
            "video": {"ego_view": self._ego_rgb, "wrist_view": self._wrist_rgb},
            "state": {
                "eef_9d": self._eef_9d,
                "hand_joint_pos": self._hand_joint_pos,
                "arm_joint_pos": self._arm_joint_pos,
                "mujoco.qfrc_actuator": qfrc,
            },
        }

    def observation_torch(self) -> dict[str, dict[str, Any]]:
        """Return zero-copy Torch views of the current GPU observation."""
        return {
            group: {name: wp.to_torch(value) for name, value in values.items()}
            for group, values in self.observation().items()
        }

    def hold_action(self) -> wp.array:
        """Return a GPU action buffer that holds every controlled joint at its current position."""
        wp.launch(
            _gather_joint_position_action,
            dim=(self.num_envs, ACTION_SIZE),
            inputs=[
                self.state_0.joint_q,
                self.model.joint_coord_world_start,
                self._action_local_q,
                self._action,
            ],
            device=self.device,
        )
        return self._action

    def reset(self, world_mask: Any | None = None) -> tuple[dict[str, dict[str, wp.array]], dict[str, wp.array]]:
        """Reset all worlds or a GPU boolean mask of worlds.

        Args:
            world_mask: Optional boolean Warp array or CUDA Torch tensor with
                shape ``[num_envs]``. ``None`` resets every world.
        """
        mask_wp = self._as_world_mask(world_mask)
        self.solver.reset(self.state_0, world_mask=mask_wp)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        wp.launch(
            _reset_control_targets,
            dim=(self.num_envs, max(self.coords_per_world, self.dofs_per_world)),
            inputs=[
                mask_wp,
                self.model.joint_coord_world_start,
                self.model.joint_dof_world_start,
                self.model.joint_target_q,
                self.model.joint_target_qd,
                self.control.joint_target_q,
                self.control.joint_target_qd,
                self.coords_per_world,
                self.dofs_per_world,
            ],
            device=self.device,
        )
        wp.launch(
            _reset_episode_arrays,
            dim=self.num_envs,
            inputs=[mask_wp, self.episode_step, self.reward, self.terminated, self.truncated],
            device=self.device,
        )
        for state in (self.state_0, self.state_1):
            wp.launch(
                _clear_reset_qfrc,
                dim=(self.num_envs, self.dofs_per_world),
                inputs=[mask_wp, state.mujoco.qfrc_actuator, self.model.joint_dof_world_start, self.dofs_per_world],
                device=self.device,
            )
        self.model.bvh_refit_shapes(self.state_0)
        self._refresh_observation()
        return self.observation(), {"episode_step": self.episode_step}

    def reset_torch(self, world_mask: Any | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Reset worlds and return zero-copy Torch views."""
        self.reset(world_mask)
        return self.observation_torch(), {"episode_step": wp.to_torch(self.episode_step)}

    def _as_world_mask(self, world_mask: Any | None) -> wp.array | None:
        if world_mask is None:
            return None
        if isinstance(world_mask, wp.array):
            mask_wp = world_mask
        else:
            try:
                mask_wp = wp.from_torch(world_mask.contiguous(), dtype=wp.bool, requires_grad=False)
            except (AttributeError, TypeError) as exc:
                raise TypeError("world_mask must be a Warp array or a CUDA Torch tensor") from exc
        if mask_wp.device != self.device:
            raise ValueError(f"world_mask must be on {self.device}, got {mask_wp.device}")
        if mask_wp.dtype != wp.bool or mask_wp.shape != (self.num_envs,):
            raise ValueError(
                f"world_mask must be bool with shape ({self.num_envs},), got {mask_wp.dtype} {mask_wp.shape}"
            )
        return mask_wp

    def step(
        self, action: Any
    ) -> tuple[dict[str, dict[str, wp.array]], wp.array, wp.array, wp.array, dict[str, wp.array]]:
        """Apply one control interval and return the Gymnasium-style tuple."""
        self._copy_action(action)
        self._apply_action()
        for _ in range(self.frames_per_action):
            if self._scene.graph is not None:
                wp.capture_launch(self._scene.graph)
            else:
                self._scene.simulate()
        wp.launch(
            _advance_episode,
            dim=self.num_envs,
            inputs=[
                self.episode_step,
                self.reward,
                self.terminated,
                self.truncated,
                self.config.max_episode_steps,
            ],
            device=self.device,
        )
        self._refresh_observation()
        return (
            self.observation(),
            self.reward,
            self.terminated,
            self.truncated,
            {"episode_step": self.episode_step},
        )

    def step_torch(self, action: Any) -> tuple[dict[str, dict[str, Any]], Any, Any, Any, dict[str, Any]]:
        """Step with a CUDA Torch action and return only zero-copy Torch views."""
        self.step(action)
        return (
            self.observation_torch(),
            wp.to_torch(self.reward),
            wp.to_torch(self.terminated),
            wp.to_torch(self.truncated),
            {"episode_step": wp.to_torch(self.episode_step)},
        )

    def close(self) -> None:
        """Release scene-owned auxiliary resources."""
        self._scene.close_l10_bottle_contact_log()
