"""Headless, batched Newton environment for the dual Nero + L10 scene.

The steady-state interface keeps simulation state, actions, camera images, rewards,
and episode flags on the Warp device. Use :meth:`GrootNewtonEnv.observation_torch`
for zero-copy Torch views when the trainer uses CUDA tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
import warp as wp

try:
    import gymnasium as gym
except ImportError:
    gym = None

import newton
from debug import import_dual_nero_linker_l10 as scene_runtime
from newton.sensors import SensorTiledCamera
from newton.viewer import ViewerNull
from teleop_stack.models import NamedJointValues
from teleop_stack.retargeting.hand_config import load_linker_l10_right_hand_spec

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
JOINT_ACTION_SIZE = len(ARM_JOINT_NAMES) + len(HAND_JOINT_NAMES)
STATE_SIZE = len(ARM_JOINT_NAMES) + 9 + len(HAND_JOINT_NAMES)
ACTION_SIZE = 9 + len(HAND_JOINT_NAMES)
POLICY_PROPRIO_SIZE = STATE_SIZE

_CONTROL_MODE_PD_JOINT_POS = 0
_CONTROL_MODE_PD_JOINT_DELTA_POS = 1
_CONTROL_MODE_PD_EEF_POSE_ABS = 2
_CONTROL_MODE_IDS = {
    "pd_joint_pos": _CONTROL_MODE_PD_JOINT_POS,
    "pd_joint_delta_pos": _CONTROL_MODE_PD_JOINT_DELTA_POS,
    "pd_eef_pose_abs": _CONTROL_MODE_PD_EEF_POSE_ABS,
}

_REWARD_MODE_NONE = 0
_REWARD_MODE_SPARSE = 1
_REWARD_MODE_DENSE = 2
_REWARD_MODE_NORMALIZED_DENSE = 3
_REWARD_MODE_IDS = {
    "none": _REWARD_MODE_NONE,
    "sparse": _REWARD_MODE_SPARSE,
    "dense": _REWARD_MODE_DENSE,
    "normalized_dense": _REWARD_MODE_NORMALIZED_DENSE,
}

_OBS_MODES = {"state", "state_dict", "rgb", "state_dict+rgb", "policy"}
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

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
_GROOT_INITIAL_HAND_Q = (
    0.1848468184,
    0.3151794076,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0581703521,
    0.0262242742,
    0.1140341759,
    0.0,
)

_NERO_MDH = (
    (0.138, 0.0, 0.0, 0.0),
    (0.0, 0.0, math.pi / 2.0, math.pi),
    (0.31, 0.0, math.pi / 2.0, math.pi),
    (0.0, 0.0, math.pi / 2.0, math.pi),
    (0.27001, 0.0, math.pi / 2.0, math.pi / 2.0),
    (0.0, 0.0, math.pi / 2.0, math.pi / 2.0),
    (0.0235, 0.0, math.pi / 2.0, 0.0),
)

# Existing data records state in the Nero/CAN frame and absolute EEF actions
# after node0's fixed A * T * B command transform.
_STATE_TO_ACTION_ROTATION = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
_STATE_TO_ACTION_TRANSLATION = (0.0, 0.059, 0.918)
_ACTION_EEF_OFFSET_ROTATION = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
_ACTION_EEF_OFFSET_TRANSLATION = (0.032, 0.0, -0.0235)


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
    max_episode_steps: int = 100
    obs_mode: str = "state_dict+rgb"
    control_mode: str = "pd_eef_pose_abs"
    reward_mode: str = "normalized_dense"
    arm_action_delta: float = 0.1
    hand_action_delta: float = 0.1
    ik_iterations: int = 4
    ik_damping_lambda: float = 0.02
    ik_position_weight: float = 3.0
    ik_orientation_weight: float = 1.0
    ik_max_task_step_m: float = 0.03
    ik_max_rotation_step_rad: float = math.radians(5.0)
    ik_max_joint_step_rad: float = 0.045
    hand_max_joint_step_rad: float = 0.08
    initial_hand_q: tuple[float, ...] = _GROOT_INITIAL_HAND_Q
    bottle_lift_height: float = 0.1
    goal_threshold: float = 0.005
    static_velocity_threshold: float = 0.2
    grasp_finger_count: int = 2
    terminate_on_success: bool = True
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
        if self.max_episode_steps < 0:
            raise ValueError("max_episode_steps cannot be negative")
        if self.obs_mode not in _OBS_MODES:
            raise ValueError(f"Unsupported obs_mode {self.obs_mode!r}; expected one of {sorted(_OBS_MODES)}")
        if self.control_mode not in _CONTROL_MODE_IDS:
            raise ValueError(
                f"Unsupported control_mode {self.control_mode!r}; expected one of {sorted(_CONTROL_MODE_IDS)}"
            )
        if self.reward_mode not in _REWARD_MODE_IDS:
            raise ValueError(
                f"Unsupported reward_mode {self.reward_mode!r}; expected one of {sorted(_REWARD_MODE_IDS)}"
            )
        if self.arm_action_delta <= 0.0 or self.hand_action_delta <= 0.0:
            raise ValueError("action deltas must be positive")
        if self.ik_iterations < 1:
            raise ValueError("ik_iterations must be positive")
        if self.ik_damping_lambda <= 0.0:
            raise ValueError("ik_damping_lambda must be positive")
        if self.ik_position_weight <= 0.0 or self.ik_orientation_weight <= 0.0:
            raise ValueError("IK task weights must be positive")
        if min(self.ik_max_task_step_m, self.ik_max_rotation_step_rad, self.ik_max_joint_step_rad) <= 0.0:
            raise ValueError("IK step limits must be positive")
        if self.hand_max_joint_step_rad <= 0.0:
            raise ValueError("hand_max_joint_step_rad must be positive")
        if len(self.initial_hand_q) != len(HAND_JOINT_NAMES) or not all(
            math.isfinite(value) for value in self.initial_hand_q
        ):
            raise ValueError(f"initial_hand_q must contain {len(HAND_JOINT_NAMES)} finite values")
        if self.bottle_lift_height <= 0.0 or self.goal_threshold <= 0.0:
            raise ValueError("bottle_lift_height and goal_threshold must be positive")
        if self.static_velocity_threshold <= 0.0:
            raise ValueError("static_velocity_threshold must be positive")
        if self.grasp_finger_count < 1 or self.grasp_finger_count > len(_FINGER_NAMES):
            raise ValueError(f"grasp_finger_count must be in [1, {len(_FINGER_NAMES)}]")
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
    joint_q: wp.array[wp.float32],
    joint_coord_world_start: wp.array[wp.int32],
    joint_dof_world_start: wp.array[wp.int32],
    local_q_indices: wp.array[wp.int32],
    local_qd_indices: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    action_scale: wp.array[wp.float32],
    control_mode: wp.int32,
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
):
    world, slot = wp.tid()
    q_index = joint_coord_world_start[world] + local_q_indices[slot]
    qd_index = joint_dof_world_start[world] + local_qd_indices[slot]
    lower = joint_limit_lower[qd_index]
    upper = joint_limit_upper[qd_index]
    normalized = wp.clamp(action[world, slot], -1.0, 1.0)
    target = 0.5 * (normalized + 1.0) * (upper - lower) + lower
    if control_mode == wp.static(_CONTROL_MODE_PD_JOINT_DELTA_POS):
        target = joint_q[q_index] + normalized * action_scale[slot]
    target_q[q_index] = wp.clamp(target, lower, upper)
    target_qd[qd_index] = 0.0


@wp.kernel(enable_backward=False)
def _gather_joint_position_action(
    joint_q: wp.array[wp.float32],
    joint_coord_world_start: wp.array[wp.int32],
    joint_dof_world_start: wp.array[wp.int32],
    local_q_indices: wp.array[wp.int32],
    local_qd_indices: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    control_mode: wp.int32,
    action: wp.array2d[wp.float32],
):
    world, slot = wp.tid()
    if control_mode == wp.static(_CONTROL_MODE_PD_JOINT_DELTA_POS):
        action[world, slot] = 0.0
    else:
        q = joint_q[joint_coord_world_start[world] + local_q_indices[slot]]
        qd_index = joint_dof_world_start[world] + local_qd_indices[slot]
        lower = joint_limit_lower[qd_index]
        upper = joint_limit_upper[qd_index]
        action[world, slot] = wp.clamp(2.0 * (q - lower) / (upper - lower) - 1.0, -1.0, 1.0)


@wp.kernel(enable_backward=False)
def _extract_state(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_coord_world_start: wp.array[wp.int32],
    joint_dof_world_start: wp.array[wp.int32],
    arm_local_q_indices: wp.array[wp.int32],
    arm_local_qd_indices: wp.array[wp.int32],
    hand_local_q_indices: wp.array[wp.int32],
    hand_local_qd_indices: wp.array[wp.int32],
    hand_command_limits: wp.array2d[wp.float32],
    hand_sdk_limits: wp.array2d[wp.float32],
    hand_raw_reversed: wp.array[wp.int32],
    hand_observation_lower: wp.array[wp.float32],
    mdh: wp.array2d[wp.float32],
    arm_out: wp.array2d[wp.float32],
    hand_out: wp.array2d[wp.float32],
    eef_out: wp.array2d[wp.float32],
    qpos_out: wp.array2d[wp.float32],
    qvel_out: wp.array2d[wp.float32],
    policy_state_out: wp.array2d[wp.float32],
):
    world = wp.tid()
    q_start = joint_coord_world_start[world]
    qd_start = joint_dof_world_start[world]

    rotation = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    position = wp.vec3(0.0)
    for joint in range(7):
        q = joint_q[q_start + arm_local_q_indices[joint]]
        arm_out[world, joint] = q
        qpos_out[world, joint] = q
        qvel_out[world, joint] = joint_qd[qd_start + arm_local_qd_indices[joint]]
        policy_state_out[world, joint] = q
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
    eef_out[world, 4] = rotation[1, 0]
    eef_out[world, 5] = rotation[2, 0]
    eef_out[world, 6] = rotation[0, 1]
    eef_out[world, 7] = rotation[1, 1]
    eef_out[world, 8] = rotation[2, 1]
    for index in range(9):
        policy_state_out[world, 7 + index] = eef_out[world, index]

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
        qpos_out[world, 7 + joint] = hand_out[world, joint]
        qvel_out[world, 7 + joint] = joint_qd[qd_start + hand_local_qd_indices[joint]]
        policy_state_out[world, 16 + joint] = hand_out[world, joint]


@wp.kernel(enable_backward=False)
def _initialize_task_goal(
    body_q: wp.array[wp.transform],
    body_world_start: wp.array[wp.int32],
    bottle_local_body: wp.int32,
    lift_height: wp.float32,
    world_mask: wp.array[wp.bool],
    goal_pos: wp.array2d[wp.float32],
):
    world = wp.tid()
    if not world_mask or world_mask[world]:
        position = wp.transform_get_translation(body_q[body_world_start[world] + bottle_local_body])
        goal_pos[world, 0] = position[0]
        goal_pos[world, 1] = position[1]
        goal_pos[world, 2] = position[2] + lift_height


@wp.kernel(enable_backward=False)
def _accumulate_hand_bottle_contacts(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_finger: wp.array[wp.int32],
    shape_is_bottle: wp.array[wp.int32],
    finger_contacts: wp.array2d[wp.int32],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    shape0 = contact_shape0[contact]
    shape1 = contact_shape1[contact]
    if shape0 < 0 or shape1 < 0:
        return

    finger = int(-1)
    world = int(-1)
    if shape_is_bottle[shape0] != 0 and shape_finger[shape1] >= 0:
        finger = shape_finger[shape1]
        world = shape_world[shape0]
    elif shape_is_bottle[shape1] != 0 and shape_finger[shape0] >= 0:
        finger = shape_finger[shape0]
        world = shape_world[shape1]
    if world >= 0 and finger >= 0:
        wp.atomic_add(finger_contacts, world, finger, 1)


@wp.kernel(enable_backward=False)
def _clear_finger_contact_rows(
    world_mask: wp.array[wp.bool],
    finger_contacts: wp.array2d[wp.int32],
):
    world, finger = wp.tid()
    if not world_mask or world_mask[world]:
        finger_contacts[world, finger] = 0


@wp.kernel(enable_backward=False)
def _evaluate_pick_bottle(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_world_start: wp.array[wp.int32],
    bottle_local_body: wp.int32,
    connector_local_body: wp.int32,
    fingertip_local_bodies: wp.array[wp.int32],
    finger_contacts: wp.array2d[wp.int32],
    goal_pos: wp.array2d[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_dof_world_start: wp.array[wp.int32],
    action_local_qd: wp.array[wp.int32],
    goal_threshold: wp.float32,
    static_velocity_threshold: wp.float32,
    grasp_finger_count: wp.int32,
    obj_pose: wp.array2d[wp.float32],
    tcp_pose: wp.array2d[wp.float32],
    tcp_to_obj: wp.array2d[wp.float32],
    obj_to_goal: wp.array2d[wp.float32],
    is_grasped: wp.array[wp.bool],
    is_obj_placed: wp.array[wp.bool],
    is_robot_static: wp.array[wp.bool],
    success: wp.array[wp.bool],
    reaching_reward: wp.array[wp.float32],
    place_reward: wp.array[wp.float32],
    static_reward: wp.array[wp.float32],
):
    world = wp.tid()
    body_start = body_world_start[world]
    bottle_transform = body_q[body_start + bottle_local_body]
    bottle_position = wp.transform_get_translation(bottle_transform)
    bottle_rotation = wp.transform_get_rotation(bottle_transform)
    connector_rotation = wp.transform_get_rotation(body_q[body_start + connector_local_body])

    tcp_position = wp.vec3(0.0)
    for finger in range(5):
        tip_transform = body_q[body_start + fingertip_local_bodies[finger]]
        tcp_position = tcp_position + wp.transform_get_translation(tip_transform)
    tcp_position = tcp_position / 5.0

    obj_pose[world, 0] = bottle_position[0]
    obj_pose[world, 1] = bottle_position[1]
    obj_pose[world, 2] = bottle_position[2]
    obj_pose[world, 3] = bottle_rotation[0]
    obj_pose[world, 4] = bottle_rotation[1]
    obj_pose[world, 5] = bottle_rotation[2]
    obj_pose[world, 6] = bottle_rotation[3]
    tcp_pose[world, 0] = tcp_position[0]
    tcp_pose[world, 1] = tcp_position[1]
    tcp_pose[world, 2] = tcp_position[2]
    tcp_pose[world, 3] = connector_rotation[0]
    tcp_pose[world, 4] = connector_rotation[1]
    tcp_pose[world, 5] = connector_rotation[2]
    tcp_pose[world, 6] = connector_rotation[3]

    tcp_delta = bottle_position - tcp_position
    goal_delta = wp.vec3(
        goal_pos[world, 0] - bottle_position[0],
        goal_pos[world, 1] - bottle_position[1],
        goal_pos[world, 2] - bottle_position[2],
    )
    for axis in range(3):
        tcp_to_obj[world, axis] = tcp_delta[axis]
        obj_to_goal[world, axis] = goal_delta[axis]

    touching_fingers = int(0)
    for finger in range(5):
        if finger_contacts[world, finger] > 0:
            touching_fingers = touching_fingers + 1
    grasped = touching_fingers >= grasp_finger_count
    lift_remaining = wp.max(goal_pos[world, 2] - bottle_position[2], 0.0)
    lifted = lift_remaining <= goal_threshold

    velocity_sq = float(0.0)
    qd_start = joint_dof_world_start[world]
    for joint in range(JOINT_ACTION_SIZE):
        velocity = joint_qd[qd_start + action_local_qd[joint]]
        velocity_sq = velocity_sq + velocity * velocity
    bottle_linear_velocity = wp.spatial_top(body_qd[body_start + bottle_local_body])
    velocity_sq = velocity_sq + wp.dot(bottle_linear_velocity, bottle_linear_velocity)
    static = wp.sqrt(velocity_sq) <= static_velocity_threshold

    is_grasped[world] = grasped
    is_obj_placed[world] = lifted
    is_robot_static[world] = static
    success[world] = lifted
    reaching_reward[world] = 1.0 - wp.tanh(5.0 * wp.length(tcp_delta))
    place_reward[world] = 1.0 - wp.tanh(10.0 * lift_remaining)
    static_reward[world] = 1.0 - wp.tanh(5.0 * wp.sqrt(velocity_sq))


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
    episode_return: wp.array[wp.float32],
    success_once: wp.array[wp.bool],
    reaching_reward: wp.array[wp.float32],
    place_reward: wp.array[wp.float32],
    static_reward: wp.array[wp.float32],
    is_grasped: wp.array[wp.bool],
    is_obj_placed: wp.array[wp.bool],
    success: wp.array[wp.bool],
    dense_reward: wp.array[wp.float32],
    reward: wp.array[wp.float32],
    terminated: wp.array[wp.bool],
    truncated: wp.array[wp.bool],
    max_episode_steps: wp.int32,
    reward_mode: wp.int32,
    terminate_on_success: wp.bool,
):
    world = wp.tid()
    episode_step[world] = episode_step[world] + 1
    dense = reaching_reward[world]
    if is_grasped[world]:
        dense = dense + 1.0 + place_reward[world]
    if is_obj_placed[world]:
        dense = dense + static_reward[world]
    if success[world]:
        dense = 5.0
    dense_reward[world] = dense

    value = float(0.0)
    if reward_mode == wp.static(_REWARD_MODE_SPARSE):
        if success[world]:
            value = 1.0
    elif reward_mode == wp.static(_REWARD_MODE_DENSE):
        value = dense
    elif reward_mode == wp.static(_REWARD_MODE_NORMALIZED_DENSE):
        value = dense / 5.0
    reward[world] = value
    episode_return[world] = episode_return[world] + value
    success_once[world] = success_once[world] or success[world]
    terminated[world] = terminate_on_success and success[world]
    truncated[world] = max_episode_steps > 0 and episode_step[world] >= max_episode_steps


@wp.kernel(enable_backward=False)
def _reset_episode_arrays(
    world_mask: wp.array[wp.bool],
    episode_step: wp.array[wp.int32],
    episode_return: wp.array[wp.float32],
    success_once: wp.array[wp.bool],
    dense_reward: wp.array[wp.float32],
    reward: wp.array[wp.float32],
    terminated: wp.array[wp.bool],
    truncated: wp.array[wp.bool],
):
    world = wp.tid()
    if not world_mask or world_mask[world]:
        episode_step[world] = 0
        episode_return[world] = 0.0
        success_once[world] = False
        dense_reward[world] = 0.0
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
def _mark_world_indices(indices: wp.array[wp.int32], num_worlds: wp.int32, world_mask: wp.array[wp.bool]):
    index = indices[wp.tid()]
    if index >= 0 and index < num_worlds:
        world_mask[index] = True


class GrootNewtonEnv:
    """Batched, headless RL environment with GPU-resident observations.

    The default action has shape ``[num_envs, 19]`` and follows the dataset
    order ``absolute EEF xyz + rotation 6D + absolute L10 targets``. Batched
    damped-least-squares IK and joint target writes stay on the GPU.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, config: GrootNewtonEnvConfig | None = None, **config_overrides: Any):
        self.config = replace(config or GrootNewtonEnvConfig(), **config_overrides)
        self.num_envs = self.config.num_envs
        self.device = wp.get_device(self.config.device)
        self.frames_per_action = self.config.simulation_hz // self.config.control_hz
        self.control_dt = 1.0 / float(self.config.control_hz)
        self.control_mode = self.config.control_mode
        self.action_size = ACTION_SIZE if self.control_mode == "pd_eef_pose_abs" else JOINT_ACTION_SIZE
        self.obs_mode = self.config.obs_mode
        self.reward_mode = self.config.reward_mode
        self.render_mode = None
        self._control_mode_id = _CONTROL_MODE_IDS[self.control_mode]
        self._reward_mode_id = _REWARD_MODE_IDS[self.reward_mode]
        self._expose_images = self.obs_mode in {"rgb", "state_dict+rgb", "policy"}
        self._render_images = self.config.render_images and self._expose_images

        args = scene_runtime.Example.create_parser().parse_args([])
        args.device = self.config.device
        args.viewer = "null"
        args.headless = True
        args.fps = float(self.config.simulation_hz)
        args.substeps = self.config.substeps_per_frame
        args.capture_graph = self.config.capture_graph
        args.world_count = self.num_envs
        args.replicate_worlds = True
        args.request_qfrc_actuator = False
        args.gpu_env_mode = True
        args.quest_teleop = False
        args.d455_preview = False
        args.d405_preview = False
        if not self._render_images or not self.config.load_scene_visuals:
            args.scene_glb = scene_runtime.REPO_ROOT / "__headless_visuals_disabled__.glb"
        if not self._render_images:
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
        self.contacts = self._scene.contacts
        if self.model.world_count != self.num_envs:
            raise RuntimeError(f"Expected {self.num_envs} worlds, built {self.model.world_count}")
        if self.solver is None:
            raise RuntimeError("GrootNewtonEnv requires the MuJoCo solver")

        self.coords_per_world = self.model.joint_coord_count // self.num_envs
        self.dofs_per_world = self.model.joint_dof_count // self.num_envs
        self._setup_joint_indices()
        self._initialize_hand_pose()
        self._setup_gpu_ik()
        self._setup_task_indices()
        self._setup_observation_arrays()
        self._setup_episode_arrays()
        self._setup_task_arrays()
        self._setup_cameras(args)
        self._initialize_task_goal(None)
        self._refresh_observation()
        self._setup_spaces()

    @property
    def unwrapped(self) -> GrootNewtonEnv:
        """Return this environment, matching the Gymnasium convention."""
        return self

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
        self._arm_local_q_np = arm_q
        self._arm_local_qd_np = arm_qd
        self._hand_local_q_np = hand_q
        self._hand_local_qd_np = hand_qd
        self._arm_local_q = wp.array(arm_q, dtype=wp.int32, device=self.device)
        self._arm_local_qd = wp.array(arm_qd, dtype=wp.int32, device=self.device)
        self._hand_local_q = wp.array(hand_q, dtype=wp.int32, device=self.device)
        self._hand_local_qd = wp.array(hand_qd, dtype=wp.int32, device=self.device)
        self._action_local_q = wp.array(np.concatenate((arm_q, hand_q)), dtype=wp.int32, device=self.device)
        self._action_local_qd = wp.array(np.concatenate((arm_qd, hand_qd)), dtype=wp.int32, device=self.device)
        action_scale = np.concatenate(
            (
                np.full(len(ARM_JOINT_NAMES), self.config.arm_action_delta, dtype=np.float32),
                np.full(len(HAND_JOINT_NAMES), self.config.hand_action_delta, dtype=np.float32),
            )
        )
        self._action_scale = wp.array(action_scale, dtype=wp.float32, device=self.device)

    def _initialize_hand_pose(self) -> None:
        """Match every replicated hand to the first-frame GR00T posture."""
        expanded = load_linker_l10_right_hand_spec().expand_mimic_joint_values(
            NamedJointValues(
                joint_names=HAND_JOINT_NAMES,
                joint_positions=tuple(float(value) for value in self.config.initial_hand_q),
            )
        )
        joint_q = self.state_0.joint_q.numpy().copy()
        joint_target_q = self.control.joint_target_q.numpy().copy()
        joint_world_start = self.model.joint_world_start.numpy()
        joint_q_start = self.model.joint_q_start.numpy()
        for world in range(self.num_envs):
            first_joint = int(joint_world_start[world])
            last_joint = int(joint_world_start[world + 1])
            for name, value in zip(expanded.joint_names, expanded.joint_positions, strict=True):
                suffix = f"/right_l10_{name}"
                matches = [
                    joint
                    for joint in range(first_joint, last_joint)
                    if self.model.joint_label[joint] == suffix[1:] or self.model.joint_label[joint].endswith(suffix)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one replicated hand joint {name!r} in world {world}, got {len(matches)}"
                    )
                q_index = int(joint_q_start[matches[0]])
                joint_q[q_index] = float(value)
                joint_target_q[q_index] = float(value)

        joint_q_wp = wp.array(joint_q, dtype=wp.float32, device=self.device)
        joint_target_q_wp = wp.array(joint_target_q, dtype=wp.float32, device=self.device)
        wp.copy(self.model.joint_q, joint_q_wp)
        wp.copy(self.model.joint_target_q, joint_target_q_wp)
        wp.copy(self.state_0.joint_q, joint_q_wp)
        wp.copy(self.control.joint_target_q, joint_target_q_wp)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        wp.copy(self.model.body_q, self.state_0.body_q)
        self.state_1.assign(self.state_0)
        if hasattr(self._scene, "_initial_joint_q"):
            self._scene._initial_joint_q = joint_q.copy()
        if hasattr(self._scene, "_initial_joint_target_q"):
            self._scene._initial_joint_target_q = joint_target_q.copy()
        if hasattr(self._scene, "_initial_body_q"):
            self._scene._initial_body_q = self.state_0.body_q.numpy().copy()
        if hasattr(self._scene, "_initial_model_body_q"):
            self._scene._initial_model_body_q = self.model.body_q.numpy().copy()

    def _setup_gpu_ik(self) -> None:
        """Create fixed CUDA index and limit tensors used by batched EEF IK."""
        if self.control_mode != "pd_eef_pose_abs":
            return
        try:
            import torch
        except ImportError as exc:
            raise ImportError("pd_eef_pose_abs requires PyTorch for batched CUDA IK") from exc

        q_starts = self.model.joint_coord_world_start.numpy().astype(np.int64, copy=False)[: self.num_envs]
        qd_starts = self.model.joint_dof_world_start.numpy().astype(np.int64, copy=False)[: self.num_envs]
        arm_q = q_starts[:, None] + self._arm_local_q_np[None, :]
        arm_qd = qd_starts[:, None] + self._arm_local_qd_np[None, :]
        hand_q = q_starts[:, None] + self._hand_local_q_np[None, :]
        hand_qd = qd_starts[:, None] + self._hand_local_qd_np[None, :]
        torch_device = str(self.device)
        self._arm_q_indices_torch = torch.as_tensor(arm_q, dtype=torch.long, device=torch_device)
        self._arm_qd_indices_torch = torch.as_tensor(arm_qd, dtype=torch.long, device=torch_device)
        self._hand_q_indices_torch = torch.as_tensor(hand_q, dtype=torch.long, device=torch_device)
        self._hand_qd_indices_torch = torch.as_tensor(hand_qd, dtype=torch.long, device=torch_device)

        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        self._arm_lower_torch = torch.as_tensor(lower[arm_qd], dtype=torch.float32, device=torch_device)
        self._arm_upper_torch = torch.as_tensor(upper[arm_qd], dtype=torch.float32, device=torch_device)
        hand_lower = np.maximum(lower[hand_qd], np.asarray(_HAND_COMMAND_LIMITS, dtype=np.float32)[None, :, 0])
        hand_upper = np.minimum(upper[hand_qd], np.asarray(_HAND_COMMAND_LIMITS, dtype=np.float32)[None, :, 1])
        self._hand_lower_torch = torch.as_tensor(hand_lower, dtype=torch.float32, device=torch_device)
        self._hand_upper_torch = torch.as_tensor(hand_upper, dtype=torch.float32, device=torch_device)
        self._ik_identity_torch = torch.eye(7, dtype=torch.float32, device=torch_device).expand(self.num_envs, -1, -1)
        self._state_to_action_rotation_torch = torch.as_tensor(
            _STATE_TO_ACTION_ROTATION, dtype=torch.float32, device=torch_device
        ).expand(self.num_envs, -1, -1)
        self._state_to_action_translation_torch = torch.as_tensor(
            _STATE_TO_ACTION_TRANSLATION, dtype=torch.float32, device=torch_device
        ).expand(self.num_envs, -1)
        self._action_eef_offset_rotation_torch = torch.as_tensor(
            _ACTION_EEF_OFFSET_ROTATION, dtype=torch.float32, device=torch_device
        ).expand(self.num_envs, -1, -1)
        self._action_eef_offset_translation_torch = torch.as_tensor(
            _ACTION_EEF_OFFSET_TRANSLATION, dtype=torch.float32, device=torch_device
        ).expand(self.num_envs, -1)

    @staticmethod
    def _limit_vector_norm_torch(value: Any, maximum: float) -> Any:
        import torch

        norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        scale = torch.clamp(float(maximum) / torch.clamp(norm, min=1.0e-8), max=1.0)
        return value * scale

    def _eef_fk_jacobian_torch(self, q: Any) -> tuple[Any, Any, Any]:
        """Evaluate the Nero MDH pose and spatial Jacobian for every world."""
        import torch

        batch = q.shape[0]
        rotation = torch.eye(3, dtype=q.dtype, device=q.device).expand(batch, -1, -1).clone()
        position = torch.zeros((batch, 3), dtype=q.dtype, device=q.device)
        origins = []
        axes = []
        zeros = torch.zeros(batch, dtype=q.dtype, device=q.device)
        for joint, (d_i, a_i, alpha_i, theta_offset) in enumerate(_NERO_MDH):
            ca = math.cos(alpha_i)
            sa = math.sin(alpha_i)
            local_position = q.new_tensor((a_i, -sa * d_i, ca * d_i)).expand(batch, -1)
            origin = position + torch.bmm(rotation, local_position.unsqueeze(-1)).squeeze(-1)
            local_axis = q.new_tensor((0.0, -sa, ca)).expand(batch, -1)
            axis = torch.bmm(rotation, local_axis.unsqueeze(-1)).squeeze(-1)
            origins.append(origin)
            axes.append(axis)

            theta = q[:, joint] + theta_offset
            ct = torch.cos(theta)
            st = torch.sin(theta)
            row0 = torch.stack((ct, -st, zeros), dim=-1)
            row1 = torch.stack((ca * st, ca * ct, zeros - sa), dim=-1)
            row2 = torch.stack((sa * st, sa * ct, zeros + ca), dim=-1)
            link_rotation = torch.stack((row0, row1, row2), dim=1)
            position = origin
            rotation = torch.bmm(rotation, link_rotation)

        joint_origins = torch.stack(origins, dim=1)
        joint_axes = torch.stack(axes, dim=1)
        linear = torch.linalg.cross(joint_axes, position[:, None, :] - joint_origins, dim=-1)
        jacobian = torch.cat((linear.transpose(1, 2), joint_axes.transpose(1, 2)), dim=1)
        return position, rotation, jacobian

    @staticmethod
    def _rotation_6d_to_matrix_torch(rot6d: Any, fallback: Any) -> Any:
        """Convert the first-two-columns 6D convention used by the dataset."""
        import torch

        raw0 = rot6d[:, 0:3]
        raw1 = rot6d[:, 3:6]
        norm0 = torch.linalg.vector_norm(raw0, dim=-1, keepdim=True)
        column0 = raw0 / torch.clamp(norm0, min=1.0e-8)
        orthogonal1 = raw1 - torch.sum(column0 * raw1, dim=-1, keepdim=True) * column0
        norm1 = torch.linalg.vector_norm(orthogonal1, dim=-1, keepdim=True)
        column1 = orthogonal1 / torch.clamp(norm1, min=1.0e-8)
        column2 = torch.linalg.cross(column0, column1, dim=-1)
        rotation = torch.stack((column0, column1, column2), dim=2)
        valid = torch.isfinite(rot6d).all(dim=-1) & (norm0[:, 0] > 1.0e-8) & (norm1[:, 0] > 1.0e-8)
        return torch.where(valid[:, None, None], rotation, fallback)

    def _apply_eef_pose_action_torch(self) -> None:
        """Decode absolute dataset actions into batched arm and hand targets."""
        import torch

        with torch.autocast(device_type="cuda", enabled=False):
            self._apply_eef_pose_action_torch_fp32(torch)

    def _apply_eef_pose_action_torch_fp32(self, torch: Any) -> None:
        action = wp.to_torch(self._action)
        joint_q = wp.to_torch(self.state_0.joint_q)
        current_q = joint_q[self._arm_q_indices_torch]
        q_command = current_q
        current_position, current_rotation, _ = self._eef_fk_jacobian_torch(current_q)
        command_position = action[:, 0:3]
        command_rotation_fallback = torch.bmm(
            torch.bmm(self._state_to_action_rotation_torch, current_rotation),
            self._action_eef_offset_rotation_torch,
        )
        command_rotation = self._rotation_6d_to_matrix_torch(action[:, 3:9], command_rotation_fallback)
        target_rotation = torch.bmm(
            torch.bmm(self._state_to_action_rotation_torch.transpose(1, 2), command_rotation),
            self._action_eef_offset_rotation_torch.transpose(1, 2),
        )
        target_position = torch.bmm(
            self._state_to_action_rotation_torch.transpose(1, 2),
            (command_position - self._state_to_action_translation_torch).unsqueeze(-1),
        ).squeeze(-1)
        target_position = target_position - torch.bmm(
            target_rotation, self._action_eef_offset_translation_torch.unsqueeze(-1)
        ).squeeze(-1)
        target_position = torch.where(torch.isfinite(target_position), target_position, current_position)

        for _ in range(self.config.ik_iterations):
            position, rotation, jacobian = self._eef_fk_jacobian_torch(q_command)
            position_error = self._limit_vector_norm_torch(target_position - position, self.config.ik_max_task_step_m)
            orientation_error = 0.5 * (
                torch.linalg.cross(rotation[:, :, 0], target_rotation[:, :, 0], dim=-1)
                + torch.linalg.cross(rotation[:, :, 1], target_rotation[:, :, 1], dim=-1)
                + torch.linalg.cross(rotation[:, :, 2], target_rotation[:, :, 2], dim=-1)
            )
            orientation_error = self._limit_vector_norm_torch(orientation_error, self.config.ik_max_rotation_step_rad)
            error = torch.cat((position_error, orientation_error), dim=-1)
            weights = error.new_tensor((self.config.ik_position_weight,) * 3 + (self.config.ik_orientation_weight,) * 3)
            weighted_jacobian = jacobian * weights[None, :, None]
            weighted_error = error * weights[None, :]
            system = torch.bmm(weighted_jacobian.transpose(1, 2), weighted_jacobian)
            system = system + (self.config.ik_damping_lambda**2) * self._ik_identity_torch
            rhs = torch.bmm(weighted_jacobian.transpose(1, 2), weighted_error.unsqueeze(-1))
            step = torch.linalg.solve(system, rhs).squeeze(-1)
            step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
            q_command = torch.clamp(q_command + step, self._arm_lower_torch, self._arm_upper_torch)
            q_command = torch.clamp(
                q_command,
                current_q - self.config.ik_max_joint_step_rad,
                current_q + self.config.ik_max_joint_step_rad,
            )

        hand_current = joint_q[self._hand_q_indices_torch]
        hand_target = torch.where(torch.isfinite(action[:, 9:19]), action[:, 9:19], hand_current)
        hand_target = torch.clamp(
            hand_target,
            hand_current - self.config.hand_max_joint_step_rad,
            hand_current + self.config.hand_max_joint_step_rad,
        )
        hand_target = torch.clamp(hand_target, self._hand_lower_torch, self._hand_upper_torch)

        target_q = wp.to_torch(self.control.joint_target_q)
        target_qd = wp.to_torch(self.control.joint_target_qd)
        target_q[self._arm_q_indices_torch] = q_command
        target_q[self._hand_q_indices_torch] = hand_target
        target_qd[self._arm_qd_indices_torch] = (q_command - current_q) / self.control_dt
        target_qd[self._hand_qd_indices_torch] = 0.0

    def _setup_task_indices(self) -> None:
        self._bottle_body_local = self._find_local_body_index("dynamic_bottle")
        self._connector_body_local = self._find_local_body_index("/right_connector")
        fingertip_suffixes = tuple(f"/right_l10_{finger}_distal" for finger in _FINGER_NAMES)
        fingertip_locals = np.asarray(
            [self._find_local_body_index(suffix) for suffix in fingertip_suffixes], dtype=np.int32
        )
        self._fingertip_body_locals = wp.array(fingertip_locals, dtype=wp.int32, device=self.device)

        shape_world_start = self.model.shape_world_start.numpy()
        shape_body = self.model.shape_body.numpy()
        shape_world = np.full(self.model.shape_count, -1, dtype=np.int32)
        for world in range(self.num_envs):
            shape_world[int(shape_world_start[world]) : int(shape_world_start[world + 1])] = world

        shape_finger = np.full(self.model.shape_count, -1, dtype=np.int32)
        shape_is_bottle = np.zeros(self.model.shape_count, dtype=np.int32)
        for shape_index, body_index in enumerate(shape_body):
            if body_index < 0:
                continue
            body_label = self.model.body_label[int(body_index)].lower()
            if "dynamic_bottle" in body_label:
                shape_is_bottle[shape_index] = 1
            if "right_l10" not in body_label:
                continue
            for finger_index, finger in enumerate(_FINGER_NAMES):
                if finger in body_label:
                    shape_finger[shape_index] = finger_index
                    break
        self._shape_world = wp.array(shape_world, dtype=wp.int32, device=self.device)
        self._shape_finger = wp.array(shape_finger, dtype=wp.int32, device=self.device)
        self._shape_is_bottle = wp.array(shape_is_bottle, dtype=wp.int32, device=self.device)

    def _setup_observation_arrays(self) -> None:
        self._action = wp.zeros((self.num_envs, self.action_size), dtype=wp.float32, device=self.device)
        self._eef_9d = wp.zeros((self.num_envs, 9), dtype=wp.float32, device=self.device)
        self._arm_joint_pos = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=self.device)
        self._hand_joint_pos = wp.zeros((self.num_envs, 10), dtype=wp.float32, device=self.device)
        self._agent_qpos = wp.zeros((self.num_envs, JOINT_ACTION_SIZE), dtype=wp.float32, device=self.device)
        self._agent_qvel = wp.zeros((self.num_envs, JOINT_ACTION_SIZE), dtype=wp.float32, device=self.device)
        self._policy_state = wp.zeros((self.num_envs, POLICY_PROPRIO_SIZE), dtype=wp.float32, device=self.device)
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
        self.episode_return = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.success_once = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self.reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.terminated = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self.truncated = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._reset_mask = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)

    def _setup_task_arrays(self) -> None:
        self._goal_pos = wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device)
        self._obj_pose = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=self.device)
        self._tcp_pose = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=self.device)
        self._tcp_to_obj = wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device)
        self._obj_to_goal = wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device)
        self._finger_contacts = wp.zeros((self.num_envs, len(_FINGER_NAMES)), dtype=wp.int32, device=self.device)
        self._is_grasped = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._is_obj_placed = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._is_robot_static = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._success = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._fail = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._reaching_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._place_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._static_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._dense_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

    def _setup_spaces(self) -> None:
        """Create Gymnasium-compatible spaces without dense visual bound arrays."""
        if gym is None:
            self.single_action_space = None
            self.action_space = None
            self.single_observation_space = None
            self.observation_space = None
            return

        class TensorBox(gym.Space):
            def __init__(self, low: float | int, high: float | int, shape: tuple[int, ...], dtype: np.dtype):
                super().__init__(shape=shape, dtype=dtype)
                self.low = np.asarray(low, dtype=dtype)
                self.high = np.asarray(high, dtype=dtype)

            def sample(self, mask: Any | None = None, probability: Any | None = None) -> np.ndarray:
                if mask is not None or probability is not None:
                    raise ValueError("TensorBox does not support masked sampling")
                if np.issubdtype(self.dtype, np.bool_):
                    return self.np_random.integers(0, 2, size=self.shape, dtype=np.int8).astype(np.bool_)
                if np.issubdtype(self.dtype, np.integer):
                    return self.np_random.integers(self.low, self.high + 1, size=self.shape, dtype=self.dtype)
                low = -1.0 if not np.isfinite(self.low) else self.low
                high = 1.0 if not np.isfinite(self.high) else self.high
                return self.np_random.uniform(low, high, size=self.shape).astype(self.dtype)

            def contains(self, value: Any) -> bool:
                array = np.asarray(value)
                return array.shape == self.shape and np.can_cast(array.dtype, self.dtype)

        def array_space(value: wp.array, *, batched: bool) -> gym.Space:
            shape = tuple(value.shape if batched else value.shape[1:])
            if value.dtype == wp.uint8:
                return TensorBox(0, 255, shape, np.dtype(np.uint8))
            if value.dtype == wp.bool:
                return TensorBox(False, True, shape, np.dtype(np.bool_))
            if value.dtype == wp.int32:
                return TensorBox(np.iinfo(np.int32).min, np.iinfo(np.int32).max, shape, np.dtype(np.int32))
            return TensorBox(-np.inf, np.inf, shape, np.dtype(np.float32))

        def tree_space(value: Any, *, batched: bool) -> gym.Space:
            if isinstance(value, dict):
                return gym.spaces.Dict({key: tree_space(child, batched=batched) for key, child in value.items()})
            return array_space(value, batched=batched)

        action_low = -np.inf if self.control_mode == "pd_eef_pose_abs" else -1.0
        action_high = np.inf if self.control_mode == "pd_eef_pose_abs" else 1.0
        self.single_action_space = gym.spaces.Box(action_low, action_high, shape=(self.action_size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            action_low, action_high, shape=(self.num_envs, self.action_size), dtype=np.float32
        )
        observation = self.observation_warp()
        if isinstance(observation, wp.array):
            self.single_observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=tuple(observation.shape[1:]), dtype=np.float32
            )
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=tuple(observation.shape), dtype=np.float32)
        else:
            self.single_observation_space = tree_space(observation, batched=False)
            self.observation_space = tree_space(observation, batched=True)

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
        self._ego_rgb = None
        self._wrist_rgb = None
        if not self._expose_images:
            return
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
        if not self._render_images:
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
        elif isinstance(action, np.ndarray):
            action_wp = wp.array(action, dtype=wp.float32, device=self.device)
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
        if self.control_mode == "pd_eef_pose_abs":
            self._apply_eef_pose_action_torch()
            return
        wp.launch(
            _apply_joint_targets,
            dim=(self.num_envs, JOINT_ACTION_SIZE),
            inputs=[
                self._action,
                self.state_0.joint_q,
                self.model.joint_coord_world_start,
                self.model.joint_dof_world_start,
                self._action_local_q,
                self._action_local_qd,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self._action_scale,
                self._control_mode_id,
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

    def _initialize_task_goal(self, world_mask: wp.array | None) -> None:
        wp.launch(
            _initialize_task_goal,
            dim=self.num_envs,
            inputs=[
                self.state_0.body_q,
                self.model.body_world_start,
                self._bottle_body_local,
                self.config.bottle_lift_height,
                world_mask,
                self._goal_pos,
            ],
            device=self.device,
        )

    def _refresh_task_state(self, *, read_contacts: bool, reset_mask: wp.array | None = None) -> None:
        if read_contacts:
            self._finger_contacts.zero_()
            wp.launch(
                _accumulate_hand_bottle_contacts,
                dim=self.contacts.rigid_contact_max,
                inputs=[
                    self.contacts.rigid_contact_count,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self._shape_world,
                    self._shape_finger,
                    self._shape_is_bottle,
                    self._finger_contacts,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _clear_finger_contact_rows,
                dim=(self.num_envs, len(_FINGER_NAMES)),
                inputs=[reset_mask, self._finger_contacts],
                device=self.device,
            )
        wp.launch(
            _evaluate_pick_bottle,
            dim=self.num_envs,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.model.body_world_start,
                self._bottle_body_local,
                self._connector_body_local,
                self._fingertip_body_locals,
                self._finger_contacts,
                self._goal_pos,
                self.state_0.joint_qd,
                self.model.joint_dof_world_start,
                self._action_local_qd,
                self.config.goal_threshold,
                self.config.static_velocity_threshold,
                self.config.grasp_finger_count,
                self._obj_pose,
                self._tcp_pose,
                self._tcp_to_obj,
                self._obj_to_goal,
                self._is_grasped,
                self._is_obj_placed,
                self._is_robot_static,
                self._success,
                self._reaching_reward,
                self._place_reward,
                self._static_reward,
            ],
            device=self.device,
        )

    def _refresh_observation(self, *, read_contacts: bool = False, reset_mask: wp.array | None = None) -> None:
        wp.launch(
            _extract_state,
            dim=self.num_envs,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.model.joint_coord_world_start,
                self.model.joint_dof_world_start,
                self._arm_local_q,
                self._arm_local_qd,
                self._hand_local_q,
                self._hand_local_qd,
                self._hand_command_limits,
                self._hand_sdk_limits,
                self._hand_raw_reversed,
                self._hand_observation_lower,
                self._mdh,
                self._arm_joint_pos,
                self._hand_joint_pos,
                self._eef_9d,
                self._agent_qpos,
                self._agent_qvel,
                self._policy_state,
            ],
            device=self.device,
        )
        self._refresh_task_state(read_contacts=read_contacts, reset_mask=reset_mask)
        self._render_cameras()

    def observation_warp(self) -> Any:
        """Return the current observation as device-resident Warp views."""
        agent = {
            "qpos": self._agent_qpos,
            "qvel": self._agent_qvel,
            "arm_joint_pos": self._arm_joint_pos,
            "hand_joint_pos": self._hand_joint_pos,
        }
        extra = {
            "tcp_pose": self._tcp_pose,
            "obj_pose": self._obj_pose,
            "goal_pos": self._goal_pos,
            "tcp_to_obj_pos": self._tcp_to_obj,
            "obj_to_goal_pos": self._obj_to_goal,
            "is_grasped": self._is_grasped,
            "eef_9d": self._eef_9d,
        }
        sensor_data = {
            "ego_view": {"rgb": self._ego_rgb},
            "wrist_view": {"rgb": self._wrist_rgb},
        }
        if self.obs_mode == "state":
            return self._policy_state
        if self.obs_mode == "state_dict":
            return {"agent": agent, "extra": extra}
        if self.obs_mode == "rgb":
            return {
                "agent": agent,
                "extra": {
                    "tcp_pose": self._tcp_pose,
                    "goal_pos": self._goal_pos,
                    "is_grasped": self._is_grasped,
                    "eef_9d": self._eef_9d,
                },
                "sensor_data": sensor_data,
            }
        if self.obs_mode == "policy":
            return self.policy_observation_warp()
        return {"agent": agent, "extra": extra, "sensor_data": sensor_data}

    def policy_observation_warp(self) -> dict[str, Any]:
        """Return fields using the existing LeRobot dataset feature names."""
        observation = {"observation.state": self._policy_state}
        if self._expose_images:
            observation["observation.images.ego_view"] = self._ego_rgb
            observation["observation.images.wrist_view"] = self._wrist_rgb
        return observation

    def policy_observation(self) -> dict[str, Any]:
        """Return zero-copy CUDA Torch views for a Diffusion Policy encoder."""
        return self._to_torch_tree(self.policy_observation_warp())

    @staticmethod
    def _to_torch_tree(value: Any) -> Any:
        if isinstance(value, wp.array):
            return wp.to_torch(value)
        if isinstance(value, dict):
            return {key: GrootNewtonEnv._to_torch_tree(child) for key, child in value.items()}
        return value

    def observation(self) -> Any:
        """Return zero-copy CUDA Torch views using the configured observation mode."""
        return self._to_torch_tree(self.observation_warp())

    def observation_torch(self) -> Any:
        """Alias for :meth:`observation`, retained for explicit call sites."""
        return self.observation()

    def evaluate_warp(self) -> dict[str, wp.array]:
        """Return batched task evaluation arrays without copying from the GPU."""
        return {
            "success": self._success,
            "fail": self._fail,
            "is_grasped": self._is_grasped,
            "is_lifted": self._is_obj_placed,
            "is_obj_placed": self._is_obj_placed,
            "is_robot_static": self._is_robot_static,
        }

    def evaluate(self) -> dict[str, Any]:
        """Return ManiSkill-style batched task evaluation as CUDA Torch views."""
        return self._to_torch_tree(self.evaluate_warp())

    def compute_dense_reward(self, obs: Any = None, action: Any = None, info: Any = None) -> Any:
        """Return the latest PickBottle dense reward as a CUDA Torch view."""
        del obs, action, info
        return wp.to_torch(self._dense_reward)

    def compute_normalized_dense_reward(self, obs: Any = None, action: Any = None, info: Any = None) -> Any:
        """Return the latest dense reward normalized to the scale used by PPO."""
        del obs, action, info
        return wp.to_torch(self._dense_reward) / 5.0

    def _info_warp(self) -> dict[str, Any]:
        return {
            **self.evaluate_warp(),
            "elapsed_steps": self.episode_step,
            "episode": {
                "return": self.episode_return,
                "length": self.episode_step,
                "success_once": self.success_once,
                "success_at_end": self._success,
            },
            "reward_components": {
                "reaching": self._reaching_reward,
                "lift": self._place_reward,
                "place": self._place_reward,
                "static": self._static_reward,
                "dense": self._dense_reward,
            },
        }

    def hold_action(self) -> wp.array:
        """Return a GPU action that holds the current absolute targets."""
        if self.control_mode == "pd_eef_pose_abs":
            action = wp.to_torch(self._action)
            joint_q = wp.to_torch(self.state_0.joint_q)
            position, rotation, _ = self._eef_fk_jacobian_torch(joint_q[self._arm_q_indices_torch])
            command_rotation = wp.to_torch(self._action).new_empty((self.num_envs, 3, 3))
            command_rotation.copy_(
                self._state_to_action_rotation_torch @ rotation @ self._action_eef_offset_rotation_torch
            )
            command_position = self._state_to_action_translation_torch + (
                self._state_to_action_rotation_torch
                @ (
                    position + (rotation @ self._action_eef_offset_translation_torch.unsqueeze(-1)).squeeze(-1)
                ).unsqueeze(-1)
            ).squeeze(-1)
            action[:, :3].copy_(command_position)
            action[:, 3:6].copy_(command_rotation[:, :, 0])
            action[:, 6:9].copy_(command_rotation[:, :, 1])
            action[:, 9:19].copy_(wp.to_torch(self._hand_joint_pos))
            return self._action
        wp.launch(
            _gather_joint_position_action,
            dim=(self.num_envs, JOINT_ACTION_SIZE),
            inputs=[
                self.state_0.joint_q,
                self.model.joint_coord_world_start,
                self.model.joint_dof_world_start,
                self._action_local_q,
                self._action_local_qd,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self._control_mode_id,
                self._action,
            ],
            device=self.device,
        )
        return self._action

    def hold_action_torch(self) -> Any:
        """Return a zero-copy CUDA Torch view of :meth:`hold_action`."""
        return wp.to_torch(self.hold_action())

    def reset_warp(
        self,
        world_mask: Any | None = None,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset all or selected worlds and return Warp observations.

        ``options={"env_idx": indices}`` follows the ManiSkill partial-reset
        convention. The fixed bottle setup currently has no randomization, so
        ``seed`` is accepted for API compatibility but does not change state.
        """
        del seed
        if options is not None and options.get("reconfigure", False):
            raise ValueError("Runtime scene reconfiguration is not supported; construct a new GrootNewtonEnv")
        if world_mask is not None and options is not None and "env_idx" in options:
            raise ValueError("Specify either world_mask or options['env_idx'], not both")
        if options is not None and "env_idx" in options:
            world_mask = self._world_mask_from_indices(options["env_idx"])
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
            inputs=[
                mask_wp,
                self.episode_step,
                self.episode_return,
                self.success_once,
                self._dense_reward,
                self.reward,
                self.terminated,
                self.truncated,
            ],
            device=self.device,
        )
        self._initialize_task_goal(mask_wp)
        self.model.bvh_refit_shapes(self.state_0)
        self._refresh_observation(read_contacts=False, reset_mask=mask_wp)
        return self.observation_warp(), self._info_warp()

    def reset(
        self,
        world_mask: Any | None = None,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset worlds and return ManiSkill-style CUDA Torch observations and info."""
        observation, info = self.reset_warp(world_mask, seed=seed, options=options)
        return self._to_torch_tree(observation), self._to_torch_tree(info)

    def reset_torch(
        self,
        world_mask: Any | None = None,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Alias for :meth:`reset`."""
        return self.reset(world_mask, seed=seed, options=options)

    def _world_mask_from_indices(self, env_idx: Any) -> wp.array:
        if isinstance(env_idx, wp.array):
            indices_wp = env_idx
        else:
            try:
                import torch

                if isinstance(env_idx, torch.Tensor):
                    indices_wp = wp.from_torch(env_idx.to(dtype=torch.int32).contiguous(), requires_grad=False)
                else:
                    indices = np.asarray(env_idx, dtype=np.int32).reshape(-1)
                    if np.any(indices < 0) or np.any(indices >= self.num_envs):
                        raise IndexError(f"env_idx must be in [0, {self.num_envs})")
                    indices_wp = wp.array(indices, dtype=wp.int32, device=self.device)
            except ImportError:
                indices = np.asarray(env_idx, dtype=np.int32).reshape(-1)
                indices_wp = wp.array(indices, dtype=wp.int32, device=self.device)
        if indices_wp.device != self.device:
            raise ValueError(f"env_idx must be on {self.device}, got {indices_wp.device}")
        if indices_wp.dtype != wp.int32 or len(indices_wp.shape) != 1:
            raise ValueError(f"env_idx must be a 1-D int32 array, got {indices_wp.dtype} {indices_wp.shape}")
        self._reset_mask.zero_()
        wp.launch(
            _mark_world_indices,
            dim=indices_wp.shape[0],
            inputs=[indices_wp, self.num_envs, self._reset_mask],
            device=self.device,
        )
        return self._reset_mask

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

    def step_warp(self, action: Any) -> tuple[Any, wp.array, wp.array, wp.array, dict[str, Any]]:
        """Apply one control interval and return device-resident Warp values."""
        self._copy_action(action)
        self._apply_action()
        for _ in range(self.frames_per_action):
            if self._scene.graph is not None:
                wp.capture_launch(self._scene.graph)
            else:
                self._scene.simulate()
        self._refresh_observation(read_contacts=True)
        wp.launch(
            _advance_episode,
            dim=self.num_envs,
            inputs=[
                self.episode_step,
                self.episode_return,
                self.success_once,
                self._reaching_reward,
                self._place_reward,
                self._static_reward,
                self._is_grasped,
                self._is_obj_placed,
                self._success,
                self._dense_reward,
                self.reward,
                self.terminated,
                self.truncated,
                self.config.max_episode_steps,
                self._reward_mode_id,
                self.config.terminate_on_success,
            ],
            device=self.device,
        )
        return (
            self.observation_warp(),
            self.reward,
            self.terminated,
            self.truncated,
            self._info_warp(),
        )

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Apply one action and return the Gymnasium five-tuple on CUDA."""
        result = self.step_warp(action)
        return tuple(self._to_torch_tree(value) for value in result)

    def step_torch(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Alias for :meth:`step`."""
        return self.step(action)

    def close(self) -> None:
        """Release scene-owned auxiliary resources."""
        self._scene.close_l10_bottle_contact_log()
