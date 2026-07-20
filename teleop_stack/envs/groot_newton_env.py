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
from teleop_stack.retargeting.hand_config import (
    LINKER_L10_FINGERTIP_LINK_NAMES,
    LINKER_L10_FINGERTIP_LOCAL_OFFSETS_M,
    load_linker_l10_right_hand_spec,
)

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

_TASK_PHASE_APPROACH = 0
_TASK_PHASE_CARRYING = 1
_TASK_PHASE_RELEASED = 2
_TASK_PHASE_SUCCESS = 3
_TASK_PHASE_FAIL = 4
_STAGE_REWARD_MAX = 8.0
_TAKEOFF_REWARD_HEIGHT = 0.01
_NON_THUMB_CONTACT_REWARD_PER_FINGER = 0.125
_OPPOSED_GRASP_REWARD = 0.5
_PREGRASP_DISTANCE_SCALE_M = 0.08
_PREGRASP_Z_PAIR_SCALE_M = 0.04
_PREGRASP_REWARD_WEIGHT = 0.35
_PREGRASP_REWARD_MAX = 1.35
_PARTIAL_CONTACT_REWARD_MAX = 1.50
_PARTIAL_CONTACT_FRACTION_WEIGHT = 0.10
_PARTIAL_CONTACT_GEOMETRY_WEIGHT = 0.90
_MISSING_SIDE_PROXIMITY_BASE = 0.60
_MISSING_SIDE_OPPOSITION_WEIGHT = 0.25
_MISSING_SIDE_Z_WEIGHT = 0.15
_UNCONFIRMED_OPPOSED_REWARD_BASE = 1.51
_UNCONFIRMED_OPPOSED_STREAK_WEIGHT = 0.07
_UNCONFIRMED_OPPOSED_FINGER_WEIGHT = 0.02
_UNCONFIRMED_OPPOSED_STREAK_FRAMES = 5
_UNCONFIRMED_OPPOSED_REWARD_MAX = 1.60

_OBS_MODES = {"state", "state_dict", "rgb", "state_dict+rgb", "policy"}
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_FINGER_ROOT_HAND_INDICES = (0, 2, 3, 4, 5)
_FINGER_ROOT_JOINT_NAMES = tuple(HAND_JOINT_NAMES[index] for index in _FINGER_ROOT_HAND_INDICES)
_FINGER_ROOT_LOAD_SIZE = len(_FINGER_ROOT_HAND_INDICES)

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
    bottle_settle_frames: int = 60
    bottle_lift_height: float = 0.1
    bottle_min_xy_displacement: float = 0.1
    transport_start_distance: float = 0.01
    goal_threshold: float = 0.005
    final_z_threshold: float = 0.01
    final_orientation_threshold_rad: float = math.radians(15.0)
    contact_max_separation: float = 0.0002
    static_velocity_threshold: float = 0.2
    object_linear_velocity_threshold: float = 0.02
    object_angular_velocity_threshold: float = 0.5
    grasp_finger_count: int = 2
    grasp_confirm_frames: int = 6
    release_confirm_frames: int = 6
    settle_confirm_frames: int = 12
    terminate_on_success: bool = False
    terminate_on_fail: bool = True
    capture_graph: bool = True
    render_images: bool = True
    camera_textures: bool = True
    load_scene_visuals: bool = True
    hydroelastic_contacts: bool = True
    request_finger_root_load: bool = False
    finger_root_load_bias: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    finger_root_load_scale: tuple[float, ...] | None = None
    finger_root_closing_sign: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    ego_width: int = 320
    ego_height: int = 180
    wrist_width: int = 640
    wrist_height: int = 480
    rigid_contacts_per_env: int = 1024
    triangle_pairs_per_env: int = 65_536
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
        if self.bottle_settle_frames < 0:
            raise ValueError("bottle_settle_frames cannot be negative")
        if self.bottle_settle_frames > 0 and self.substeps_per_frame % 2 != 0:
            raise ValueError("bottle settling requires an even substeps_per_frame so state buffers do not alias")
        if (
            min(
                self.bottle_lift_height,
                self.bottle_min_xy_displacement,
                self.transport_start_distance,
                self.goal_threshold,
                self.final_z_threshold,
                self.final_orientation_threshold_rad,
            )
            <= 0.0
        ):
            raise ValueError("task distances, thresholds, and orientation tolerance must be positive")
        if self.goal_threshold >= self.bottle_lift_height:
            raise ValueError("goal_threshold must be smaller than bottle_lift_height")
        if self.contact_max_separation < 0.0:
            raise ValueError("contact_max_separation cannot be negative")
        if (
            min(
                self.static_velocity_threshold,
                self.object_linear_velocity_threshold,
                self.object_angular_velocity_threshold,
            )
            <= 0.0
        ):
            raise ValueError("velocity thresholds must be positive")
        if self.grasp_finger_count < 1 or self.grasp_finger_count > len(_FINGER_NAMES):
            raise ValueError(f"grasp_finger_count must be in [1, {len(_FINGER_NAMES)}]")
        if min(self.grasp_confirm_frames, self.release_confirm_frames, self.settle_confirm_frames) < 1:
            raise ValueError("task confirmation frame counts must be positive")
        if self.capture_graph and self.substeps_per_frame % 2 != 0:
            raise ValueError("capture_graph requires an even substeps_per_frame so state buffers do not alias")
        if min(self.ego_width, self.ego_height, self.wrist_width, self.wrist_height) < 1:
            raise ValueError("camera dimensions must be positive")
        if min(self.rigid_contacts_per_env, self.triangle_pairs_per_env) < 1:
            raise ValueError("contact buffer capacities must be positive")
        for name, values in (
            ("finger_root_load_bias", self.finger_root_load_bias),
            ("finger_root_closing_sign", self.finger_root_closing_sign),
        ):
            if len(values) != _FINGER_ROOT_LOAD_SIZE or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain {_FINGER_ROOT_LOAD_SIZE} finite values")
        if not all(abs(value) == 1.0 for value in self.finger_root_closing_sign):
            raise ValueError("finger_root_closing_sign values must be -1 or 1")
        if self.finger_root_load_scale is not None and (
            len(self.finger_root_load_scale) != _FINGER_ROOT_LOAD_SIZE
            or not all(math.isfinite(value) and value > 0.0 for value in self.finger_root_load_scale)
        ):
            raise ValueError(
                f"finger_root_load_scale must be None or contain {_FINGER_ROOT_LOAD_SIZE} positive finite values"
            )


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


@wp.func
def _positive_lift_height(current_z: wp.float32, initial_z: wp.float32) -> wp.float32:
    return wp.max(current_z - initial_z, 0.0)


@wp.func
def _normalized_lift_progress(
    current_z: wp.float32,
    initial_z: wp.float32,
    lift_height: wp.float32,
) -> wp.float32:
    return wp.clamp(_positive_lift_height(current_z, initial_z) / lift_height, 0.0, 1.0)


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
def _sync_hand_mimic_targets(
    world_mask: wp.array[wp.bool],
    leader_q_indices: wp.array2d[wp.int32],
    follower_q_indices: wp.array2d[wp.int32],
    follower_qd_indices: wp.array2d[wp.int32],
    multiplier: wp.array[wp.float32],
    offset: wp.array[wp.float32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
):
    world, mimic = wp.tid()
    if world_mask and not world_mask[world]:
        return
    leader_q = leader_q_indices[world, mimic]
    follower_q = follower_q_indices[world, mimic]
    follower_qd = follower_qd_indices[world, mimic]
    target = offset[mimic] + multiplier[mimic] * target_q[leader_q]
    target_q[follower_q] = wp.clamp(target, joint_limit_lower[follower_qd], joint_limit_upper[follower_qd])
    target_qd[follower_qd] = 0.0


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
    eef_out[world, 4] = rotation[0, 1]
    eef_out[world, 5] = rotation[0, 2]
    eef_out[world, 6] = rotation[1, 0]
    eef_out[world, 7] = rotation[1, 1]
    eef_out[world, 8] = rotation[1, 2]
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
def _extract_finger_root_load(
    qfrc_actuator: wp.array[wp.float32],
    joint_dof_world_start: wp.array[wp.int32],
    finger_root_local_qd: wp.array[wp.int32],
    closing_sign: wp.array[wp.float32],
    load_bias: wp.array[wp.float32],
    load_scale: wp.array[wp.float32],
    qfrc_out: wp.array2d[wp.float32],
    load_out: wp.array2d[wp.float32],
):
    world, finger = wp.tid()
    qd_index = joint_dof_world_start[world] + finger_root_local_qd[finger]
    qfrc = qfrc_actuator[qd_index]
    signed_load = closing_sign[finger] * qfrc
    qfrc_out[world, finger] = qfrc
    load_out[world, finger] = wp.clamp((signed_load - load_bias[finger]) / load_scale[finger], 0.0, 1.0)


@wp.kernel(enable_backward=False)
def _clear_finger_root_load_rows(
    world_mask: wp.array[wp.bool],
    qfrc: wp.array2d[wp.float32],
    load: wp.array2d[wp.float32],
):
    world, finger = wp.tid()
    if not world_mask or world_mask[world]:
        qfrc[world, finger] = 0.0
        load[world, finger] = 0.0


@wp.kernel(enable_backward=False)
def _initialize_task_goal(
    body_q: wp.array[wp.transform],
    body_world_start: wp.array[wp.int32],
    bottle_local_body: wp.int32,
    lift_height: wp.float32,
    world_mask: wp.array[wp.bool],
    goal_pos: wp.array2d[wp.float32],
    initial_obj_pose: wp.array2d[wp.float32],
    max_bottle_z: wp.array[wp.float32],
):
    world = wp.tid()
    if not world_mask or world_mask[world]:
        transform = body_q[body_world_start[world] + bottle_local_body]
        position = wp.transform_get_translation(transform)
        rotation = wp.transform_get_rotation(transform)
        goal_pos[world, 0] = position[0]
        goal_pos[world, 1] = position[1]
        goal_pos[world, 2] = position[2] + lift_height
        initial_obj_pose[world, 0] = position[0]
        initial_obj_pose[world, 1] = position[1]
        initial_obj_pose[world, 2] = position[2]
        initial_obj_pose[world, 3] = rotation[0]
        initial_obj_pose[world, 4] = rotation[1]
        initial_obj_pose[world, 5] = rotation[2]
        initial_obj_pose[world, 6] = rotation[3]
        max_bottle_z[world] = position[2]


@wp.kernel(enable_backward=False)
def _accumulate_hand_bottle_contacts(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_finger: wp.array[wp.int32],
    shape_is_bottle: wp.array[wp.int32],
    max_separation: wp.float32,
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
    finger_shape = int(-1)
    world = int(-1)
    if shape_is_bottle[shape0] != 0 and shape_finger[shape1] >= 0:
        finger = shape_finger[shape1]
        finger_shape = shape1
        world = shape_world[shape0]
    elif shape_is_bottle[shape1] != 0 and shape_finger[shape0] >= 0:
        finger = shape_finger[shape0]
        finger_shape = shape0
        world = shape_world[shape1]
    if world < 0 or finger < 0:
        return
    if shape_world[finger_shape] != world:
        return

    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    point0 = contact_point0[contact]
    point1 = contact_point1[contact]
    if body0 >= 0:
        point0 = wp.transform_point(body_q[body0], point0)
    if body1 >= 0:
        point1 = wp.transform_point(body_q[body1], point1)
    separation = wp.dot(contact_normal[contact], point1 - point0) - contact_margin0[contact] - contact_margin1[contact]
    if separation <= max_separation:
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
def _clear_control_step_contact(
    world_mask: wp.array[wp.bool],
    had_hand_contact: wp.array[wp.bool],
):
    world = wp.tid()
    if not world_mask or world_mask[world]:
        had_hand_contact[world] = False


@wp.kernel(enable_backward=False)
def _accumulate_control_step_contact(
    has_hand_contact: wp.array[wp.bool],
    had_hand_contact: wp.array[wp.bool],
):
    world = wp.tid()
    if has_hand_contact[world]:
        had_hand_contact[world] = True


@wp.kernel(enable_backward=False)
def _clear_control_step_contact_topology(
    world_mask: wp.array[wp.bool],
    finger_contact_any_frame: wp.array2d[wp.bool],
    opposed_grasp_any_frame: wp.array[wp.bool],
    opposed_grasp_consecutive_frames: wp.array[wp.int32],
    opposed_grasp_max_consecutive_frames: wp.array[wp.int32],
    non_thumb_anchor_contact_fraction: wp.array[wp.float32],
    non_thumb_missing_thumb_geometry_progress: wp.array[wp.float32],
    non_thumb_guidance_opposition_progress: wp.array[wp.float32],
    non_thumb_guidance_z_progress: wp.array[wp.float32],
    thumb_anchor_contact_fraction: wp.array[wp.float32],
    thumb_missing_non_thumb_geometry_progress: wp.array[wp.float32],
):
    world = wp.tid()
    if not world_mask or world_mask[world]:
        for finger in range(5):
            finger_contact_any_frame[world, finger] = False
        opposed_grasp_any_frame[world] = False
        opposed_grasp_consecutive_frames[world] = 0
        opposed_grasp_max_consecutive_frames[world] = 0
        non_thumb_anchor_contact_fraction[world] = 0.0
        non_thumb_missing_thumb_geometry_progress[world] = 0.0
        non_thumb_guidance_opposition_progress[world] = 0.0
        non_thumb_guidance_z_progress[world] = 0.0
        thumb_anchor_contact_fraction[world] = 0.0
        thumb_missing_non_thumb_geometry_progress[world] = 0.0


@wp.kernel(enable_backward=False)
def _accumulate_control_step_contact_topology(
    finger_contacts: wp.array2d[wp.int32],
    is_grasped: wp.array[wp.bool],
    finger_surface_gap: wp.array2d[wp.float32],
    thumb_partner_opposition: wp.array2d[wp.float32],
    thumb_partner_z_score: wp.array2d[wp.float32],
    inverse_frames_per_action: wp.float32,
    finger_contact_any_frame: wp.array2d[wp.bool],
    opposed_grasp_any_frame: wp.array[wp.bool],
    opposed_grasp_consecutive_frames: wp.array[wp.int32],
    opposed_grasp_max_consecutive_frames: wp.array[wp.int32],
    non_thumb_anchor_contact_fraction: wp.array[wp.float32],
    non_thumb_missing_thumb_geometry_progress: wp.array[wp.float32],
    non_thumb_guidance_opposition_progress: wp.array[wp.float32],
    non_thumb_guidance_z_progress: wp.array[wp.float32],
    thumb_anchor_contact_fraction: wp.array[wp.float32],
    thumb_missing_non_thumb_geometry_progress: wp.array[wp.float32],
):
    world = wp.tid()
    for finger in range(5):
        if finger_contacts[world, finger] > 0:
            finger_contact_any_frame[world, finger] = True

    if is_grasped[world]:
        opposed_grasp_any_frame[world] = True
        consecutive_frames = opposed_grasp_consecutive_frames[world] + 1
        opposed_grasp_consecutive_frames[world] = consecutive_frames
        opposed_grasp_max_consecutive_frames[world] = wp.max(
            opposed_grasp_max_consecutive_frames[world],
            consecutive_frames,
        )
    else:
        opposed_grasp_consecutive_frames[world] = 0

    thumb_anchor = finger_contacts[world, 0] > 0
    non_thumb_anchor = False
    thumb_proximity = 1.0 - wp.tanh(finger_surface_gap[world, 0] / wp.static(_PREGRASP_DISTANCE_SCALE_M))
    non_thumb_missing_thumb_geometry = float(0.0)
    selected_non_thumb_opposition = float(0.0)
    selected_non_thumb_z_score = float(0.0)
    thumb_missing_non_thumb_geometry = float(0.0)
    for partner in range(4):
        finger = partner + 1
        factor = wp.static(_MISSING_SIDE_PROXIMITY_BASE)
        opposition = wp.clamp(thumb_partner_opposition[world, partner], 0.0, 1.0)
        z_score = wp.clamp(thumb_partner_z_score[world, partner], 0.0, 1.0)
        factor = factor + wp.static(_MISSING_SIDE_OPPOSITION_WEIGHT) * opposition
        factor = factor + wp.static(_MISSING_SIDE_Z_WEIGHT) * z_score
        if finger_contacts[world, finger] > 0:
            non_thumb_anchor = True
            candidate_geometry = thumb_proximity * factor
            if candidate_geometry >= non_thumb_missing_thumb_geometry:
                non_thumb_missing_thumb_geometry = candidate_geometry
                selected_non_thumb_opposition = opposition
                selected_non_thumb_z_score = z_score
        if thumb_anchor:
            partner_proximity = 1.0 - wp.tanh(finger_surface_gap[world, finger] / wp.static(_PREGRASP_DISTANCE_SCALE_M))
            thumb_missing_non_thumb_geometry = wp.max(
                thumb_missing_non_thumb_geometry,
                partner_proximity * factor,
            )

    if non_thumb_anchor:
        updated_non_thumb_fraction = wp.min(
            non_thumb_anchor_contact_fraction[world] + inverse_frames_per_action,
            1.0,
        )
        non_thumb_anchor_contact_fraction[world] = updated_non_thumb_fraction
        non_thumb_missing_thumb_geometry_progress[world] = wp.min(
            non_thumb_missing_thumb_geometry_progress[world]
            + inverse_frames_per_action * non_thumb_missing_thumb_geometry,
            updated_non_thumb_fraction,
        )
        non_thumb_guidance_opposition_progress[world] = wp.min(
            non_thumb_guidance_opposition_progress[world] + inverse_frames_per_action * selected_non_thumb_opposition,
            updated_non_thumb_fraction,
        )
        non_thumb_guidance_z_progress[world] = wp.min(
            non_thumb_guidance_z_progress[world] + inverse_frames_per_action * selected_non_thumb_z_score,
            updated_non_thumb_fraction,
        )
    if thumb_anchor:
        updated_thumb_fraction = wp.min(
            thumb_anchor_contact_fraction[world] + inverse_frames_per_action,
            1.0,
        )
        thumb_anchor_contact_fraction[world] = updated_thumb_fraction
        thumb_missing_non_thumb_geometry_progress[world] = wp.min(
            thumb_missing_non_thumb_geometry_progress[world]
            + inverse_frames_per_action * thumb_missing_non_thumb_geometry,
            updated_thumb_fraction,
        )


@wp.kernel(enable_backward=False)
def _accumulate_control_step_collision_buffer(
    count: wp.array[wp.int32],
    capacity: wp.int32,
    frame_max: wp.array[wp.int32],
    overflow_frame_count: wp.array[wp.int32],
    overflow_excess_count: wp.array[wp.int32],
):
    observed = count[0]
    frame_max[0] = wp.max(frame_max[0], observed)
    if observed > capacity:
        overflow_frame_count[0] = overflow_frame_count[0] + 1
        overflow_excess_count[0] = overflow_excess_count[0] + observed - capacity


@wp.func
def _is_opposed_grasp(
    thumb_contact: wp.bool,
    non_thumb_contact: wp.bool,
    touching_fingers: wp.int32,
    grasp_finger_count: wp.int32,
) -> wp.bool:
    return thumb_contact and non_thumb_contact and touching_fingers >= grasp_finger_count


@wp.func
def _finite_cylinder_side_gap(
    point: wp.vec3,
    radius: wp.float32,
    half_height: wp.float32,
) -> wp.float32:
    radial = wp.sqrt(point[0] * point[0] + point[1] * point[1])
    radial_gap = wp.abs(radial - radius)
    axial_gap = wp.max(wp.abs(point[2]) - half_height, 0.0)
    return wp.sqrt(radial_gap * radial_gap + axial_gap * axial_gap)


@wp.func
def _radial_opposition_score(
    thumb: wp.vec3,
    partner: wp.vec3,
) -> wp.float32:
    thumb_radius_squared = thumb[0] * thumb[0] + thumb[1] * thumb[1]
    partner_radius_squared = partner[0] * partner[0] + partner[1] * partner[1]
    if thumb_radius_squared <= 1.0e-12 or partner_radius_squared <= 1.0e-12:
        return 0.0
    radial_dot = (thumb[0] * partner[0] + thumb[1] * partner[1]) / wp.sqrt(
        thumb_radius_squared * partner_radius_squared
    )
    return wp.clamp(0.5 * (1.0 - radial_dot), 0.0, 1.0)


@wp.func
def _fingertip_z_pair_score(
    thumb: wp.vec3,
    partner: wp.vec3,
) -> wp.float32:
    z_delta = (thumb[2] - partner[2]) / wp.static(_PREGRASP_Z_PAIR_SCALE_M)
    return wp.exp(-(z_delta * z_delta))


@wp.kernel(enable_backward=False)
def _evaluate_opposed_pregrasp_geometry(
    body_q: wp.array[wp.transform],
    body_world_start: wp.array[wp.int32],
    bottle_local_body: wp.int32,
    shape_world_start: wp.array[wp.int32],
    bottle_collision_local_shape: wp.int32,
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    fingertip_local_bodies: wp.array[wp.int32],
    fingertip_local_offsets: wp.array[wp.vec3],
    finger_surface_gap: wp.array2d[wp.float32],
    thumb_partner_opposition: wp.array2d[wp.float32],
    thumb_partner_z_score: wp.array2d[wp.float32],
    opposed_pregrasp_score: wp.array[wp.float32],
):
    world = wp.tid()
    body_start = body_world_start[world]
    shape = shape_world_start[world] + bottle_collision_local_shape
    cylinder_transform = wp.transform_multiply(
        body_q[body_start + bottle_local_body],
        shape_transform[shape],
    )
    cylinder_inverse = wp.transform_inverse(cylinder_transform)
    radius = shape_scale[shape][0]
    half_height = shape_scale[shape][1]

    thumb_transform = body_q[body_start + fingertip_local_bodies[0]]
    thumb_world = wp.transform_point(thumb_transform, fingertip_local_offsets[0])
    thumb = wp.transform_point(cylinder_inverse, thumb_world)
    thumb_gap = _finite_cylinder_side_gap(thumb, radius, half_height)
    thumb_proximity = 1.0 - wp.tanh(thumb_gap / wp.static(_PREGRASP_DISTANCE_SCALE_M))
    finger_surface_gap[world, 0] = thumb_gap

    best_score = float(0.0)
    for finger in range(1, 5):
        partner_transform = body_q[body_start + fingertip_local_bodies[finger]]
        partner_world = wp.transform_point(partner_transform, fingertip_local_offsets[finger])
        partner = wp.transform_point(cylinder_inverse, partner_world)
        partner_gap = _finite_cylinder_side_gap(partner, radius, half_height)
        partner_proximity = 1.0 - wp.tanh(partner_gap / wp.static(_PREGRASP_DISTANCE_SCALE_M))
        finger_surface_gap[world, finger] = partner_gap
        opposition = _radial_opposition_score(thumb, partner)
        z_score = _fingertip_z_pair_score(thumb, partner)
        thumb_partner_opposition[world, finger - 1] = opposition
        thumb_partner_z_score[world, finger - 1] = z_score
        best_score = wp.max(
            best_score,
            wp.min(thumb_proximity, partner_proximity) * opposition * z_score,
        )
    opposed_pregrasp_score[world] = wp.clamp(best_score, 0.0, 1.0)


@wp.kernel(enable_backward=False)
def _evaluate_transfer_bottle(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_world_start: wp.array[wp.int32],
    bottle_local_body: wp.int32,
    connector_local_body: wp.int32,
    fingertip_local_bodies: wp.array[wp.int32],
    fingertip_local_offsets: wp.array[wp.vec3],
    finger_contacts: wp.array2d[wp.int32],
    goal_pos: wp.array2d[wp.float32],
    initial_obj_pose: wp.array2d[wp.float32],
    task_phase: wp.array[wp.int32],
    reached_lift_height: wp.array[wp.bool],
    joint_qd: wp.array[wp.float32],
    joint_dof_world_start: wp.array[wp.int32],
    action_local_qd: wp.array[wp.int32],
    lift_height: wp.float32,
    min_xy_displacement: wp.float32,
    final_z_threshold: wp.float32,
    final_orientation_threshold: wp.float32,
    static_velocity_threshold: wp.float32,
    object_linear_velocity_threshold: wp.float32,
    object_angular_velocity_threshold: wp.float32,
    grasp_finger_count: wp.int32,
    obj_pose: wp.array2d[wp.float32],
    tcp_pose: wp.array2d[wp.float32],
    tcp_to_obj: wp.array2d[wp.float32],
    obj_to_goal: wp.array2d[wp.float32],
    touching_finger_count: wp.array[wp.int32],
    has_hand_contact: wp.array[wp.bool],
    is_grasped: wp.array[wp.bool],
    placement_pose_valid: wp.array[wp.bool],
    release_ready: wp.array[wp.bool],
    is_obj_placed: wp.array[wp.bool],
    is_obj_static: wp.array[wp.bool],
    is_robot_static: wp.array[wp.bool],
    xy_displacement: wp.array[wp.float32],
    final_z_error: wp.array[wp.float32],
    orientation_error: wp.array[wp.float32],
    current_lift_height: wp.array[wp.float32],
    physical_max_lift_height: wp.array[wp.float32],
    reaching_reward: wp.array[wp.float32],
    lift_reward: wp.array[wp.float32],
    transport_reward: wp.array[wp.float32],
    place_reward: wp.array[wp.float32],
    orientation_reward: wp.array[wp.float32],
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
        tcp_position = tcp_position + wp.transform_point(tip_transform, fingertip_local_offsets[finger])
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
    thumb_contact = finger_contacts[world, 0] > 0
    non_thumb_contact = False
    for finger in range(5):
        if finger_contacts[world, finger] > 0:
            touching_fingers = touching_fingers + 1
            if finger > 0:
                non_thumb_contact = True
    grasped = _is_opposed_grasp(thumb_contact, non_thumb_contact, touching_fingers, grasp_finger_count)
    hand_contact = touching_fingers > 0
    current_lift = _positive_lift_height(bottle_position[2], initial_obj_pose[world, 2])

    initial_dx = bottle_position[0] - initial_obj_pose[world, 0]
    initial_dy = bottle_position[1] - initial_obj_pose[world, 1]
    xy_distance = wp.sqrt(initial_dx * initial_dx + initial_dy * initial_dy)
    z_error = wp.abs(bottle_position[2] - initial_obj_pose[world, 2])
    quat_dot = wp.abs(
        bottle_rotation[0] * initial_obj_pose[world, 3]
        + bottle_rotation[1] * initial_obj_pose[world, 4]
        + bottle_rotation[2] * initial_obj_pose[world, 5]
        + bottle_rotation[3] * initial_obj_pose[world, 6]
    )
    angle_error = 2.0 * wp.acos(wp.clamp(quat_dot, 0.0, 1.0))
    pose_valid = (
        xy_distance >= min_xy_displacement
        and z_error <= final_z_threshold
        and angle_error <= final_orientation_threshold
    )

    robot_max_speed = float(0.0)
    qd_start = joint_dof_world_start[world]
    for joint in range(JOINT_ACTION_SIZE):
        robot_max_speed = wp.max(robot_max_speed, wp.abs(joint_qd[qd_start + action_local_qd[joint]]))
    bottle_linear_velocity = wp.spatial_top(body_qd[body_start + bottle_local_body])
    bottle_angular_velocity = wp.spatial_bottom(body_qd[body_start + bottle_local_body])
    linear_speed = wp.length(bottle_linear_velocity)
    angular_speed = wp.length(bottle_angular_velocity)
    object_static = (
        linear_speed <= object_linear_velocity_threshold and angular_speed <= object_angular_velocity_threshold
    )
    robot_static = robot_max_speed <= static_velocity_threshold

    xy_remaining = wp.max(min_xy_displacement - xy_distance, 0.0)
    xy_reward = 1.0 - wp.tanh(10.0 * xy_remaining)
    z_reward = 1.0 - wp.tanh(20.0 * z_error)
    rotation_reward = 1.0 - wp.tanh(2.0 * angle_error)
    placement_reward = (xy_reward + z_reward + rotation_reward) / 3.0
    phase = task_phase[world]
    released_or_success = phase == wp.static(_TASK_PHASE_RELEASED) or phase == wp.static(_TASK_PHASE_SUCCESS)

    touching_finger_count[world] = touching_fingers
    has_hand_contact[world] = hand_contact
    is_grasped[world] = grasped
    placement_pose_valid[world] = pose_valid
    release_ready[world] = (
        phase == wp.static(_TASK_PHASE_CARRYING) and reached_lift_height[world] and pose_valid and hand_contact
    )
    is_obj_placed[world] = released_or_success and pose_valid and not hand_contact
    is_obj_static[world] = object_static
    is_robot_static[world] = robot_static
    xy_displacement[world] = xy_distance
    final_z_error[world] = z_error
    orientation_error[world] = angle_error
    current_lift_height[world] = current_lift
    physical_max_lift_height[world] = wp.max(physical_max_lift_height[world], current_lift)
    reaching_reward[world] = 1.0 - wp.tanh(5.0 * wp.length(tcp_delta))
    lift_reward[world] = _normalized_lift_progress(
        bottle_position[2],
        initial_obj_pose[world, 2],
        lift_height,
    )
    transport_reward[world] = xy_reward
    place_reward[world] = placement_reward
    orientation_reward[world] = rotation_reward
    static_reward[world] = 1.0 - wp.tanh(10.0 * linear_speed + angular_speed)


@wp.kernel(enable_backward=False)
def _advance_transfer_phase(
    obj_pose: wp.array2d[wp.float32],
    initial_obj_pose: wp.array2d[wp.float32],
    is_grasped: wp.array[wp.bool],
    has_hand_contact: wp.array[wp.bool],
    placement_pose_valid: wp.array[wp.bool],
    is_obj_static: wp.array[wp.bool],
    current_lift_height: wp.array[wp.float32],
    lift_height: wp.float32,
    lift_threshold: wp.float32,
    transport_start_distance: wp.float32,
    grasp_confirm_frames: wp.int32,
    release_confirm_frames: wp.int32,
    settle_confirm_frames: wp.int32,
    task_phase: wp.array[wp.int32],
    grasp_contact_frames: wp.array[wp.int32],
    grasp_support_gap_frames: wp.array[wp.int32],
    contact_gap_frames: wp.array[wp.int32],
    settle_frames: wp.array[wp.int32],
    grasp_confirmed: wp.array[wp.bool],
    transport_started: wp.array[wp.bool],
    reached_lift_height: wp.array[wp.bool],
    release_armed: wp.array[wp.bool],
    released: wp.array[wp.bool],
    early_release: wp.array[wp.bool],
    max_bottle_z: wp.array[wp.float32],
    max_lift_height: wp.array[wp.float32],
    success: wp.array[wp.bool],
    fail: wp.array[wp.bool],
):
    world = wp.tid()
    if success[world] or fail[world]:
        return

    phase = task_phase[world]
    if phase == wp.static(_TASK_PHASE_APPROACH):
        grasp_support_gap_frames[world] = 0
        if is_grasped[world]:
            grasp_contact_frames[world] = grasp_contact_frames[world] + 1
            if grasp_contact_frames[world] >= grasp_confirm_frames:
                grasp_confirmed[world] = True
                contact_gap_frames[world] = 0
                task_phase[world] = wp.static(_TASK_PHASE_CARRYING)
                phase = wp.static(_TASK_PHASE_CARRYING)
        else:
            grasp_contact_frames[world] = 0
        if phase == wp.static(_TASK_PHASE_APPROACH):
            return

    if phase == wp.static(_TASK_PHASE_CARRYING):
        if has_hand_contact[world]:
            grasp_support_gap_frames[world] = 0
            contact_gap_frames[world] = 0
            max_lift_height[world] = wp.max(max_lift_height[world], current_lift_height[world])
            if current_lift_height[world] >= transport_start_distance:
                transport_started[world] = True
            if transport_started[world]:
                max_bottle_z[world] = wp.max(max_bottle_z[world], obj_pose[world, 2])
                if max_lift_height[world] >= lift_height - lift_threshold:
                    reached_lift_height[world] = True
            # Initial entry still requires confirmed opposition; once carrying,
            # any retained hand contact is valid support for transfer and release.
            release_armed[world] = reached_lift_height[world] and placement_pose_valid[world]
            return

        grasp_support_gap_frames[world] = 0
        contact_gap_frames[world] = contact_gap_frames[world] + 1
        if contact_gap_frames[world] < release_confirm_frames:
            return
        released[world] = True
        if not release_armed[world]:
            early_release[world] = True
            fail[world] = True
            task_phase[world] = wp.static(_TASK_PHASE_FAIL)
            return
        settle_frames[world] = 0
        task_phase[world] = wp.static(_TASK_PHASE_RELEASED)
        return

    if phase == wp.static(_TASK_PHASE_RELEASED):
        if has_hand_contact[world]:
            fail[world] = True
            task_phase[world] = wp.static(_TASK_PHASE_FAIL)
            return
        if is_obj_static[world]:
            settle_frames[world] = settle_frames[world] + 1
        else:
            settle_frames[world] = 0
        if settle_frames[world] >= settle_confirm_frames:
            if placement_pose_valid[world]:
                success[world] = True
                task_phase[world] = wp.static(_TASK_PHASE_SUCCESS)
            else:
                fail[world] = True
                task_phase[world] = wp.static(_TASK_PHASE_FAIL)


@wp.kernel(enable_backward=False)
def _reset_transfer_task(
    world_mask: wp.array[wp.bool],
    current_lift_height: wp.array[wp.float32],
    physical_max_lift_height: wp.array[wp.float32],
    max_lift_height: wp.array[wp.float32],
    task_phase: wp.array[wp.int32],
    grasp_contact_frames: wp.array[wp.int32],
    grasp_support_gap_frames: wp.array[wp.int32],
    contact_gap_frames: wp.array[wp.int32],
    settle_frames: wp.array[wp.int32],
    grasp_confirmed: wp.array[wp.bool],
    transport_started: wp.array[wp.bool],
    reached_lift_height: wp.array[wp.bool],
    release_armed: wp.array[wp.bool],
    released: wp.array[wp.bool],
    early_release: wp.array[wp.bool],
    success: wp.array[wp.bool],
    fail: wp.array[wp.bool],
):
    world = wp.tid()
    if not world_mask or world_mask[world]:
        current_lift_height[world] = 0.0
        physical_max_lift_height[world] = 0.0
        max_lift_height[world] = 0.0
        task_phase[world] = wp.static(_TASK_PHASE_APPROACH)
        grasp_contact_frames[world] = 0
        grasp_support_gap_frames[world] = 0
        contact_gap_frames[world] = 0
        settle_frames[world] = 0
        grasp_confirmed[world] = False
        transport_started[world] = False
        reached_lift_height[world] = False
        release_armed[world] = False
        released[world] = False
        early_release[world] = False
        success[world] = False
        fail[world] = False


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
    opposed_pregrasp_score: wp.array[wp.float32],
    max_lift_height: wp.array[wp.float32],
    bottle_lift_height: wp.float32,
    place_reward: wp.array[wp.float32],
    static_reward: wp.array[wp.float32],
    finger_contact_counts: wp.array2d[wp.int32],
    is_grasped: wp.array[wp.bool],
    finger_contact_any_frame: wp.array2d[wp.bool],
    opposed_grasp_any_frame: wp.array[wp.bool],
    opposed_grasp_max_consecutive_frames: wp.array[wp.int32],
    non_thumb_anchor_contact_fraction: wp.array[wp.float32],
    non_thumb_missing_thumb_geometry_progress: wp.array[wp.float32],
    thumb_anchor_contact_fraction: wp.array[wp.float32],
    thumb_missing_non_thumb_geometry_progress: wp.array[wp.float32],
    task_phase: wp.array[wp.int32],
    reached_lift_height: wp.array[wp.bool],
    release_ready: wp.array[wp.bool],
    is_obj_placed: wp.array[wp.bool],
    success: wp.array[wp.bool],
    fail: wp.array[wp.bool],
    approach_base_reward: wp.array[wp.float32],
    unilateral_guidance_gain: wp.array[wp.float32],
    unilateral_contact_reward: wp.array[wp.float32],
    dense_reward: wp.array[wp.float32],
    reward: wp.array[wp.float32],
    terminated: wp.array[wp.bool],
    truncated: wp.array[wp.bool],
    max_episode_steps: wp.int32,
    reward_mode: wp.int32,
    terminate_on_success: wp.bool,
    terminate_on_fail: wp.bool,
):
    world = wp.tid()
    episode_step[world] = episode_step[world] + 1
    dense = wp.clamp(reaching_reward[world], 0.0, 1.0)
    phase = task_phase[world]
    approach_base_reward[world] = 0.0
    unilateral_guidance_gain[world] = 0.0
    unilateral_contact_reward[world] = 0.0
    if phase == wp.static(_TASK_PHASE_APPROACH):
        pregrasp_score = wp.clamp(opposed_pregrasp_score[world], 0.0, 1.0)
        dense = wp.min(
            wp.static(_PREGRASP_REWARD_MAX),
            dense + wp.static(_PREGRASP_REWARD_WEIGHT) * pregrasp_score,
        )
        approach_base_reward[world] = dense
        current_non_thumb_fingers = int(0)
        for finger in range(1, 5):
            if finger_contact_counts[world, finger] > 0:
                current_non_thumb_fingers = current_non_thumb_fingers + 1
        if is_grasped[world]:
            dense = 1.0 + wp.static(_NON_THUMB_CONTACT_REWARD_PER_FINGER) * float(current_non_thumb_fingers)
            dense = dense + wp.static(_OPPOSED_GRASP_REWARD)
        else:
            any_frame_non_thumb_fingers = int(0)
            for finger in range(1, 5):
                if finger_contact_any_frame[world, finger]:
                    any_frame_non_thumb_fingers = any_frame_non_thumb_fingers + 1
            if opposed_grasp_any_frame[world]:
                streak_frames = wp.min(
                    opposed_grasp_max_consecutive_frames[world],
                    wp.static(_UNCONFIRMED_OPPOSED_STREAK_FRAMES),
                )
                streak_progress = float(streak_frames) / float(wp.static(_UNCONFIRMED_OPPOSED_STREAK_FRAMES))
                dense = wp.static(_UNCONFIRMED_OPPOSED_REWARD_BASE)
                dense = dense + wp.static(_UNCONFIRMED_OPPOSED_STREAK_WEIGHT) * streak_progress
                dense = dense + wp.static(_UNCONFIRMED_OPPOSED_FINGER_WEIGHT) * (
                    float(any_frame_non_thumb_fingers) / 4.0
                )
                dense = wp.min(dense, wp.static(_UNCONFIRMED_OPPOSED_REWARD_MAX))
            else:
                non_thumb_contact_fraction = wp.clamp(non_thumb_anchor_contact_fraction[world], 0.0, 1.0)
                non_thumb_geometry = wp.clamp(
                    non_thumb_missing_thumb_geometry_progress[world],
                    0.0,
                    non_thumb_contact_fraction,
                )
                non_thumb_progress = (
                    wp.static(_PARTIAL_CONTACT_FRACTION_WEIGHT) * non_thumb_contact_fraction
                    + wp.static(_PARTIAL_CONTACT_GEOMETRY_WEIGHT) * non_thumb_geometry
                )
                non_thumb_reward = dense + (wp.static(_PARTIAL_CONTACT_REWARD_MAX) - dense) * non_thumb_progress

                thumb_contact_fraction = wp.clamp(thumb_anchor_contact_fraction[world], 0.0, 1.0)
                thumb_geometry = wp.clamp(
                    thumb_missing_non_thumb_geometry_progress[world],
                    0.0,
                    thumb_contact_fraction,
                )
                thumb_progress = (
                    wp.static(_PARTIAL_CONTACT_FRACTION_WEIGHT) * thumb_contact_fraction
                    + wp.static(_PARTIAL_CONTACT_GEOMETRY_WEIGHT) * thumb_geometry
                )
                thumb_reward = dense + (wp.static(_PARTIAL_CONTACT_REWARD_MAX) - dense) * thumb_progress
                partial_contact_reward = wp.max(non_thumb_reward, thumb_reward)
                if wp.max(non_thumb_contact_fraction, thumb_contact_fraction) > 0.0:
                    unilateral_guidance_gain[world] = wp.max(partial_contact_reward - dense, 0.0)
                    unilateral_contact_reward[world] = partial_contact_reward
                dense = partial_contact_reward
    elif phase == wp.static(_TASK_PHASE_CARRYING):
        if release_ready[world]:
            dense = 6.0 + static_reward[world]
        elif reached_lift_height[world]:
            dense = 4.0 + place_reward[world]
        else:
            lift_height = max_lift_height[world]
            takeoff_progress = wp.clamp(lift_height / wp.static(_TAKEOFF_REWARD_HEIGHT), 0.0, 1.0)
            full_lift_progress = wp.clamp(lift_height / bottle_lift_height, 0.0, 1.0)
            dense = 2.0 + 0.5 * takeoff_progress + 1.5 * full_lift_progress
    elif phase == wp.static(_TASK_PHASE_RELEASED):
        dense = 0.0
        if is_obj_placed[world]:
            dense = 6.0 + static_reward[world]
    if success[world]:
        dense = wp.static(_STAGE_REWARD_MAX)
    elif fail[world]:
        dense = -wp.static(_STAGE_REWARD_MAX)
    dense_reward[world] = dense

    value = float(0.0)
    if reward_mode == wp.static(_REWARD_MODE_SPARSE):
        if success[world]:
            value = 1.0
        elif fail[world]:
            value = -1.0
    elif reward_mode == wp.static(_REWARD_MODE_DENSE):
        value = dense
    elif reward_mode == wp.static(_REWARD_MODE_NORMALIZED_DENSE):
        value = dense / wp.static(_STAGE_REWARD_MAX)
    reward[world] = value
    episode_return[world] = episode_return[world] + value
    success_once[world] = success_once[world] or success[world]
    terminated[world] = (terminate_on_success and success[world]) or (terminate_on_fail and fail[world])
    truncated[world] = max_episode_steps > 0 and episode_step[world] >= max_episode_steps


@wp.kernel(enable_backward=False)
def _reset_episode_arrays(
    world_mask: wp.array[wp.bool],
    episode_step: wp.array[wp.int32],
    episode_return: wp.array[wp.float32],
    success_once: wp.array[wp.bool],
    approach_base_reward: wp.array[wp.float32],
    unilateral_guidance_gain: wp.array[wp.float32],
    unilateral_contact_reward: wp.array[wp.float32],
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
        approach_base_reward[world] = 0.0
        unilateral_guidance_gain[world] = 0.0
        unilateral_contact_reward[world] = 0.0
        dense_reward[world] = 0.0
        reward[world] = 0.0
        terminated[world] = False
        truncated[world] = False


@wp.kernel(enable_backward=False)
def _capture_bottle_reset_defaults(
    settled_joint_q: wp.array[wp.float32],
    settled_joint_qd: wp.array[wp.float32],
    bottle_q_start: wp.array[wp.int32],
    bottle_qd_start: wp.array[wp.int32],
    default_joint_q: wp.array[wp.float32],
    default_joint_qd: wp.array[wp.float32],
):
    world, coordinate = wp.tid()
    if coordinate < 7:
        q_index = bottle_q_start[world] + coordinate
        default_joint_q[q_index] = settled_joint_q[q_index]
    if coordinate < 6:
        qd_index = bottle_qd_start[world] + coordinate
        settled_joint_qd[qd_index] = 0.0
        default_joint_qd[qd_index] = 0.0


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
        args.request_qfrc_actuator = self.config.request_finger_root_load
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
        args.max_triangle_pairs = max(1_000_000, self.config.triangle_pairs_per_env * self.num_envs)
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
        self._qfrc_actuator = getattr(getattr(self.state_0, "mujoco", None), "qfrc_actuator", None)
        if self.config.request_finger_root_load and self._qfrc_actuator is None:
            raise RuntimeError("Finger-root load requires the MuJoCo qfrc_actuator extended state")

        self.coords_per_world = self.model.joint_coord_count // self.num_envs
        self.dofs_per_world = self.model.joint_dof_count // self.num_envs
        self._setup_joint_indices()
        self._initialize_hand_pose()
        self._setup_gpu_ik()
        self._setup_task_indices()
        self._settle_bottle_reset_defaults()
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

    @property
    def bottle_settle_metadata(self) -> dict[str, Any]:
        """Describe the construction-time bottle reset-pose settling pass."""
        return {
            "enabled": self.config.bottle_settle_frames > 0,
            "frames": self.config.bottle_settle_frames,
            "duration_seconds": self.config.bottle_settle_frames / float(self.config.simulation_hz),
            "backend": "cuda_graph" if self._scene.graph is not None else "direct_gpu",
            "copied_free_joint_coordinates_per_env": 7,
            "zeroed_free_joint_velocities_per_env": 6,
        }

    @property
    def hand_target_metadata(self) -> dict[str, Any]:
        """Describe how active L10 targets drive their mimic followers."""
        return {
            "semantics": "active_and_clamped_mimic_follower_position_targets",
            "active_joint_names": HAND_JOINT_NAMES,
            "follower_joint_names": self._hand_mimic_joint_names,
            "source_joint_names": self._hand_mimic_source_joint_names,
            "multiplier": tuple(float(mimic.multiplier) for mimic in self._hand_spec.mimic_joints),
            "offset": tuple(float(mimic.offset) for mimic in self._hand_spec.mimic_joints),
            "target_velocity": 0.0,
            "limit_policy": "clamp_each_follower_to_its_joint_limits",
        }

    @property
    def finger_root_load_metadata(self) -> dict[str, Any]:
        """Describe the real-machine-aligned finger-root load observation."""
        return {
            "enabled": self.config.request_finger_root_load,
            "joint_names": _FINGER_ROOT_JOINT_NAMES,
            "source": "mujoco_qfrc_actuator_root_pitch_endpoint",
            "normalization": "clamp((closing_sign * qfrc - bias) / scale, 0, 1)",
            "bias": tuple(float(value) for value in self.config.finger_root_load_bias),
            "scale": tuple(float(value) for value in self._finger_root_load_scale_np),
            "closing_sign": tuple(float(value) for value in self.config.finger_root_closing_sign),
            "timing": "last_physics_substep_of_previous_control_interval",
            "reset": "zero_until_first_post_reset_control_step",
        }

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
        hand_spec = load_linker_l10_right_hand_spec()
        active_hand_names = set(HAND_JOINT_NAMES)
        invalid_mimics = [
            mimic.joint_name for mimic in hand_spec.mimic_joints if mimic.source_joint_name not in active_hand_names
        ]
        if invalid_mimics:
            raise ValueError(f"L10 mimic followers must reference active hand joints, got {invalid_mimics}")
        mimic_follower_labels = tuple(f"right_l10_{mimic.joint_name}" for mimic in hand_spec.mimic_joints)
        mimic_leader_labels = tuple(f"right_l10_{mimic.source_joint_name}" for mimic in hand_spec.mimic_joints)
        mimic_follower_q, mimic_follower_qd = find_local_indices(mimic_follower_labels)
        mimic_leader_q, _ = find_local_indices(mimic_leader_labels)
        self._arm_local_q_np = arm_q
        self._arm_local_qd_np = arm_qd
        self._hand_local_q_np = hand_q
        self._hand_local_qd_np = hand_qd
        self._hand_spec = hand_spec
        self._hand_mimic_count = len(hand_spec.mimic_joints)
        self._hand_mimic_joint_names = tuple(mimic.joint_name for mimic in hand_spec.mimic_joints)
        self._hand_mimic_source_joint_names = tuple(mimic.source_joint_name for mimic in hand_spec.mimic_joints)
        finger_root_hand_indices = np.asarray(_FINGER_ROOT_HAND_INDICES, dtype=np.int32)
        finger_root_local_qd = hand_qd[finger_root_hand_indices]
        effort_limit = self.model.joint_effort_limit.numpy()
        inferred_scale = np.abs(effort_limit[qd_origin + finger_root_local_qd]).astype(np.float32, copy=False)
        if self.config.finger_root_load_scale is None:
            if not np.all(np.isfinite(inferred_scale)) or np.any(inferred_scale <= 0.0):
                raise ValueError(
                    f"Finger-root effort limits must be positive and finite, got {inferred_scale.tolist()}"
                )
            load_scale = inferred_scale
        else:
            load_scale = np.asarray(self.config.finger_root_load_scale, dtype=np.float32)
        self._finger_root_load_scale_np = load_scale.copy()
        self._arm_local_q = wp.array(arm_q, dtype=wp.int32, device=self.device)
        self._arm_local_qd = wp.array(arm_qd, dtype=wp.int32, device=self.device)
        self._hand_local_q = wp.array(hand_q, dtype=wp.int32, device=self.device)
        self._hand_local_qd = wp.array(hand_qd, dtype=wp.int32, device=self.device)
        self._finger_root_local_qd = wp.array(finger_root_local_qd, dtype=wp.int32, device=self.device)
        self._finger_root_load_bias = wp.array(
            np.asarray(self.config.finger_root_load_bias, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self._finger_root_load_scale = wp.array(load_scale, dtype=wp.float32, device=self.device)
        self._finger_root_closing_sign = wp.array(
            np.asarray(self.config.finger_root_closing_sign, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self._action_local_q = wp.array(np.concatenate((arm_q, hand_q)), dtype=wp.int32, device=self.device)
        self._action_local_qd = wp.array(np.concatenate((arm_qd, hand_qd)), dtype=wp.int32, device=self.device)
        q_starts = self.model.joint_coord_world_start.numpy().astype(np.int32, copy=False)[: self.num_envs]
        qd_starts = self.model.joint_dof_world_start.numpy().astype(np.int32, copy=False)[: self.num_envs]
        self._hand_mimic_leader_q_indices = wp.array(
            q_starts[:, None] + mimic_leader_q[None, :], dtype=wp.int32, device=self.device
        )
        self._hand_mimic_follower_q_indices = wp.array(
            q_starts[:, None] + mimic_follower_q[None, :], dtype=wp.int32, device=self.device
        )
        self._hand_mimic_follower_qd_indices = wp.array(
            qd_starts[:, None] + mimic_follower_qd[None, :], dtype=wp.int32, device=self.device
        )
        self._hand_mimic_multiplier = wp.array(
            np.asarray([mimic.multiplier for mimic in hand_spec.mimic_joints], dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self._hand_mimic_offset = wp.array(
            np.asarray([mimic.offset for mimic in hand_spec.mimic_joints], dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        action_scale = np.concatenate(
            (
                np.full(len(ARM_JOINT_NAMES), self.config.arm_action_delta, dtype=np.float32),
                np.full(len(HAND_JOINT_NAMES), self.config.hand_action_delta, dtype=np.float32),
            )
        )
        self._action_scale = wp.array(action_scale, dtype=wp.float32, device=self.device)

    def _initialize_hand_pose(self) -> None:
        """Match every replicated hand to the first-frame GR00T posture."""
        expanded = self._hand_spec.expand_mimic_joint_values(
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
        """Convert GR00T's row-major first-two-rows rotation representation."""
        import torch

        raw0 = rot6d[:, 0:3]
        raw1 = rot6d[:, 3:6]
        norm0 = torch.linalg.vector_norm(raw0, dim=-1, keepdim=True)
        row0 = raw0 / torch.clamp(norm0, min=1.0e-8)
        orthogonal1 = raw1 - torch.sum(row0 * raw1, dim=-1, keepdim=True) * row0
        norm1 = torch.linalg.vector_norm(orthogonal1, dim=-1, keepdim=True)
        row1 = orthogonal1 / torch.clamp(norm1, min=1.0e-8)
        row2 = torch.linalg.cross(row0, row1, dim=-1)
        rotation = torch.stack((row0, row1, row2), dim=1)
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
        target_position = action[:, 0:3]
        target_rotation = self._rotation_6d_to_matrix_torch(action[:, 3:9], current_rotation)
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
        body_world_start = self.model.body_world_start.numpy()
        joint_world_start = self.model.joint_world_start.numpy()
        joint_child = self.model.joint_child.numpy()
        joint_type = self.model.joint_type.numpy()
        joint_q_start = self.model.joint_q_start.numpy()
        joint_qd_start = self.model.joint_qd_start.numpy()
        bottle_q_start = np.empty(self.num_envs, dtype=np.int32)
        bottle_qd_start = np.empty(self.num_envs, dtype=np.int32)
        for world in range(self.num_envs):
            bottle_body = int(body_world_start[world]) + self._bottle_body_local
            matches = [
                joint
                for joint in range(int(joint_world_start[world]), int(joint_world_start[world + 1]))
                if int(joint_child[joint]) == bottle_body and int(joint_type[joint]) == int(newton.JointType.FREE)
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one free joint for the bottle in world {world}, got {len(matches)}")
            bottle_joint = matches[0]
            bottle_q_start[world] = int(joint_q_start[bottle_joint])
            bottle_qd_start[world] = int(joint_qd_start[bottle_joint])
        self._bottle_q_start_np = bottle_q_start
        self._bottle_qd_start_np = bottle_qd_start
        self._bottle_q_start = wp.array(bottle_q_start, dtype=wp.int32, device=self.device)
        self._bottle_qd_start = wp.array(bottle_qd_start, dtype=wp.int32, device=self.device)
        self._connector_body_local = self._find_local_body_index("/right_connector")
        if tuple(self._hand_spec.fingertip_link_names) != LINKER_L10_FINGERTIP_LINK_NAMES:
            raise ValueError(
                "L10 fingertip link order differs from the calibrated fingertip offsets: "
                f"{self._hand_spec.fingertip_link_names}"
            )
        fingertip_suffixes = tuple(f"/right_l10_{link_name}" for link_name in LINKER_L10_FINGERTIP_LINK_NAMES)
        fingertip_locals = np.asarray(
            [self._find_local_body_index(suffix) for suffix in fingertip_suffixes], dtype=np.int32
        )
        self._fingertip_body_locals = wp.array(fingertip_locals, dtype=wp.int32, device=self.device)
        self._fingertip_local_offsets = wp.array(
            np.asarray(LINKER_L10_FINGERTIP_LOCAL_OFFSETS_M, dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )

        shape_world_start = self.model.shape_world_start.numpy()
        shape_body = self.model.shape_body.numpy()
        shape_type = self.model.shape_type.numpy()
        shape_scale = self.model.shape_scale.numpy()
        bottle_collision_local_shape = None
        for world in range(self.num_envs):
            first_shape = int(shape_world_start[world])
            last_shape = int(shape_world_start[world + 1])
            matches = [
                shape
                for shape in range(first_shape, last_shape)
                if self.model.shape_label[shape].endswith("dynamic_bottle_collision_cylinder")
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one bottle collision cylinder in world {world}, got {len(matches)}")
            shape = matches[0]
            local_shape = shape - first_shape
            if bottle_collision_local_shape is None:
                bottle_collision_local_shape = local_shape
            elif local_shape != bottle_collision_local_shape:
                raise ValueError("Bottle collision cylinder must have the same local shape index in every world")
            if int(shape_type[shape]) != int(newton.GeoType.CYLINDER):
                raise ValueError(f"Bottle collision shape in world {world} must be a cylinder")
            if float(shape_scale[shape, 0]) <= 0.0 or float(shape_scale[shape, 1]) <= 0.0:
                raise ValueError(
                    f"Bottle collision cylinder in world {world} must have positive radius and half-height"
                )
        if bottle_collision_local_shape is None:
            raise ValueError("Bottle collision cylinder was not found")
        self._bottle_collision_local_shape = bottle_collision_local_shape

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

    def _settle_bottle_reset_defaults(self) -> None:
        """Settle the bottle once and retain only its free-joint pose as the reset default."""
        if self.config.bottle_settle_frames == 0:
            return
        if self.device.is_cuda and self.config.capture_graph and self._scene.graph is None:
            raise RuntimeError("Bottle settling requires the scene CUDA graph when capture_graph is enabled")

        self.solver.reset(self.state_0)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.model.bvh_refit_shapes(self.state_0)
        for _ in range(self.config.bottle_settle_frames):
            if self._scene.graph is not None:
                wp.capture_launch(self._scene.graph)
            else:
                self._scene.simulate()
        if self._scene.state_0 is not self.state_0:
            raise RuntimeError("Bottle settling requires an even number of physics substeps per frame")

        wp.launch(
            _capture_bottle_reset_defaults,
            dim=(self.num_envs, 7),
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._bottle_q_start,
                self._bottle_qd_start,
                self.model.joint_q,
                self.model.joint_qd,
            ],
            device=self.device,
        )
        self.solver.reset(self.state_0)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.model.bvh_refit_shapes(self.state_0)

    def _setup_observation_arrays(self) -> None:
        self._action = wp.zeros((self.num_envs, self.action_size), dtype=wp.float32, device=self.device)
        self._eef_9d = wp.zeros((self.num_envs, 9), dtype=wp.float32, device=self.device)
        self._arm_joint_pos = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=self.device)
        self._hand_joint_pos = wp.zeros((self.num_envs, 10), dtype=wp.float32, device=self.device)
        self._agent_qpos = wp.zeros((self.num_envs, JOINT_ACTION_SIZE), dtype=wp.float32, device=self.device)
        self._agent_qvel = wp.zeros((self.num_envs, JOINT_ACTION_SIZE), dtype=wp.float32, device=self.device)
        self._policy_state = wp.zeros((self.num_envs, POLICY_PROPRIO_SIZE), dtype=wp.float32, device=self.device)
        self._finger_root_qfrc_actuator = wp.zeros(
            (self.num_envs, _FINGER_ROOT_LOAD_SIZE), dtype=wp.float32, device=self.device
        )
        self._finger_root_load = wp.zeros((self.num_envs, _FINGER_ROOT_LOAD_SIZE), dtype=wp.float32, device=self.device)
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
        self._initial_obj_pose = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=self.device)
        self._obj_pose = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=self.device)
        self._tcp_pose = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=self.device)
        self._tcp_to_obj = wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device)
        self._obj_to_goal = wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device)
        self._finger_contacts = wp.zeros((self.num_envs, len(_FINGER_NAMES)), dtype=wp.int32, device=self.device)
        self._finger_surface_gap = wp.zeros((self.num_envs, len(_FINGER_NAMES)), dtype=wp.float32, device=self.device)
        self._thumb_partner_opposition = wp.zeros((self.num_envs, 4), dtype=wp.float32, device=self.device)
        self._thumb_partner_z_score = wp.zeros((self.num_envs, 4), dtype=wp.float32, device=self.device)
        self._opposed_pregrasp_score = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._task_phase = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self._touching_finger_count = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self._grasp_contact_frames = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self._grasp_support_gap_frames = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self._contact_gap_frames = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self._settle_frames = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self._has_hand_contact = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._had_hand_contact_this_control_step = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._finger_contact_any_frame_this_control_step = wp.zeros(
            (self.num_envs, len(_FINGER_NAMES)), dtype=wp.bool, device=self.device
        )
        self._opposed_grasp_any_frame_this_control_step = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._opposed_grasp_consecutive_frames = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self._opposed_grasp_max_consecutive_frames_this_control_step = wp.zeros(
            self.num_envs, dtype=wp.int32, device=self.device
        )
        self._non_thumb_anchor_contact_fraction_this_control_step = wp.zeros(
            self.num_envs, dtype=wp.float32, device=self.device
        )
        self._non_thumb_missing_thumb_geometry_progress_this_control_step = wp.zeros(
            self.num_envs, dtype=wp.float32, device=self.device
        )
        self._non_thumb_guidance_opposition_progress_this_control_step = wp.zeros(
            self.num_envs, dtype=wp.float32, device=self.device
        )
        self._non_thumb_guidance_z_progress_this_control_step = wp.zeros(
            self.num_envs, dtype=wp.float32, device=self.device
        )
        self._thumb_anchor_contact_fraction_this_control_step = wp.zeros(
            self.num_envs, dtype=wp.float32, device=self.device
        )
        self._thumb_missing_non_thumb_geometry_progress_this_control_step = wp.zeros(
            self.num_envs, dtype=wp.float32, device=self.device
        )
        self._is_grasped = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._grasp_confirmed = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._transport_started = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._reached_lift_height = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._release_armed = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._released = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._early_release = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._placement_pose_valid = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._release_ready = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._is_obj_placed = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._is_obj_static = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._is_robot_static = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._success = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._fail = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._max_bottle_z = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._current_lift_height = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._physical_max_lift_height = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._max_lift_height = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._xy_displacement = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._final_z_error = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._orientation_error = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._reaching_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._lift_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._transport_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._place_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._orientation_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._static_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._approach_base_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._unilateral_guidance_gain = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._unilateral_contact_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._dense_reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

        self._rigid_contact_capacity = int(self.contacts.rigid_contact_max)
        self._rigid_contact_frame_max = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._rigid_contact_overflow_frame_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._rigid_contact_overflow_excess_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        narrow_phase = self._scene.collision_pipeline.narrow_phase
        self._triangle_pair_count = narrow_phase.triangle_pairs_count
        self._triangle_pair_buffer_available = (
            self._triangle_pair_count is not None and narrow_phase.triangle_pairs is not None
        )
        self._triangle_pair_capacity = (
            int(narrow_phase.triangle_pairs.shape[0]) if self._triangle_pair_buffer_available else 0
        )
        self._triangle_pair_frame_max = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._triangle_pair_overflow_frame_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._triangle_pair_overflow_excess_count = wp.zeros(1, dtype=wp.int32, device=self.device)

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
        else:
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
        self._sync_hand_mimic_targets()

    def _sync_hand_mimic_targets(self, world_mask: wp.array | None = None) -> None:
        wp.launch(
            _sync_hand_mimic_targets,
            dim=(self.num_envs, self._hand_mimic_count),
            inputs=[
                world_mask,
                self._hand_mimic_leader_q_indices,
                self._hand_mimic_follower_q_indices,
                self._hand_mimic_follower_qd_indices,
                self._hand_mimic_multiplier,
                self._hand_mimic_offset,
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
                self._initial_obj_pose,
                self._max_bottle_z,
            ],
            device=self.device,
        )

    def _clear_finger_contacts(self, world_mask: wp.array | None) -> None:
        wp.launch(
            _clear_finger_contact_rows,
            dim=(self.num_envs, len(_FINGER_NAMES)),
            inputs=[world_mask, self._finger_contacts],
            device=self.device,
        )

    def _clear_control_step_diagnostics(self, world_mask: wp.array | None) -> None:
        wp.launch(
            _clear_control_step_contact,
            dim=self.num_envs,
            inputs=[world_mask, self._had_hand_contact_this_control_step],
            device=self.device,
        )
        wp.launch(
            _clear_control_step_contact_topology,
            dim=self.num_envs,
            inputs=[
                world_mask,
                self._finger_contact_any_frame_this_control_step,
                self._opposed_grasp_any_frame_this_control_step,
                self._opposed_grasp_consecutive_frames,
                self._opposed_grasp_max_consecutive_frames_this_control_step,
                self._non_thumb_anchor_contact_fraction_this_control_step,
                self._non_thumb_missing_thumb_geometry_progress_this_control_step,
                self._non_thumb_guidance_opposition_progress_this_control_step,
                self._non_thumb_guidance_z_progress_this_control_step,
                self._thumb_anchor_contact_fraction_this_control_step,
                self._thumb_missing_non_thumb_geometry_progress_this_control_step,
            ],
            device=self.device,
        )
        self._rigid_contact_frame_max.zero_()
        self._rigid_contact_overflow_frame_count.zero_()
        self._rigid_contact_overflow_excess_count.zero_()
        self._triangle_pair_frame_max.zero_()
        self._triangle_pair_overflow_frame_count.zero_()
        self._triangle_pair_overflow_excess_count.zero_()

    def _accumulate_control_step_diagnostics(self) -> None:
        wp.launch(
            _accumulate_control_step_contact,
            dim=self.num_envs,
            inputs=[self._has_hand_contact, self._had_hand_contact_this_control_step],
            device=self.device,
        )
        wp.launch(
            _accumulate_control_step_contact_topology,
            dim=self.num_envs,
            inputs=[
                self._finger_contacts,
                self._is_grasped,
                self._finger_surface_gap,
                self._thumb_partner_opposition,
                self._thumb_partner_z_score,
                1.0 / float(self.frames_per_action),
                self._finger_contact_any_frame_this_control_step,
                self._opposed_grasp_any_frame_this_control_step,
                self._opposed_grasp_consecutive_frames,
                self._opposed_grasp_max_consecutive_frames_this_control_step,
                self._non_thumb_anchor_contact_fraction_this_control_step,
                self._non_thumb_missing_thumb_geometry_progress_this_control_step,
                self._non_thumb_guidance_opposition_progress_this_control_step,
                self._non_thumb_guidance_z_progress_this_control_step,
                self._thumb_anchor_contact_fraction_this_control_step,
                self._thumb_missing_non_thumb_geometry_progress_this_control_step,
            ],
            device=self.device,
        )
        wp.launch(
            _accumulate_control_step_collision_buffer,
            dim=1,
            inputs=[
                self.contacts.rigid_contact_count,
                self._rigid_contact_capacity,
                self._rigid_contact_frame_max,
                self._rigid_contact_overflow_frame_count,
                self._rigid_contact_overflow_excess_count,
            ],
            device=self.device,
        )
        if self._triangle_pair_buffer_available:
            wp.launch(
                _accumulate_control_step_collision_buffer,
                dim=1,
                inputs=[
                    self._triangle_pair_count,
                    self._triangle_pair_capacity,
                    self._triangle_pair_frame_max,
                    self._triangle_pair_overflow_frame_count,
                    self._triangle_pair_overflow_excess_count,
                ],
                device=self.device,
            )

    def _collect_finger_contacts(self) -> None:
        self._finger_contacts.zero_()
        wp.launch(
            _accumulate_hand_bottle_contacts,
            dim=self.contacts.rigid_contact_max,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.contacts.rigid_contact_point0,
                self.contacts.rigid_contact_point1,
                self.contacts.rigid_contact_normal,
                self.contacts.rigid_contact_margin0,
                self.contacts.rigid_contact_margin1,
                self.state_0.body_q,
                self.model.shape_body,
                self._shape_world,
                self._shape_finger,
                self._shape_is_bottle,
                self.config.contact_max_separation,
                self._finger_contacts,
            ],
            device=self.device,
        )

    def _evaluate_task_state(self) -> None:
        wp.launch(
            _evaluate_opposed_pregrasp_geometry,
            dim=self.num_envs,
            inputs=[
                self.state_0.body_q,
                self.model.body_world_start,
                self._bottle_body_local,
                self.model.shape_world_start,
                self._bottle_collision_local_shape,
                self.model.shape_transform,
                self.model.shape_scale,
                self._fingertip_body_locals,
                self._fingertip_local_offsets,
                self._finger_surface_gap,
                self._thumb_partner_opposition,
                self._thumb_partner_z_score,
                self._opposed_pregrasp_score,
            ],
            device=self.device,
        )
        wp.launch(
            _evaluate_transfer_bottle,
            dim=self.num_envs,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.model.body_world_start,
                self._bottle_body_local,
                self._connector_body_local,
                self._fingertip_body_locals,
                self._fingertip_local_offsets,
                self._finger_contacts,
                self._goal_pos,
                self._initial_obj_pose,
                self._task_phase,
                self._reached_lift_height,
                self.state_0.joint_qd,
                self.model.joint_dof_world_start,
                self._action_local_qd,
                self.config.bottle_lift_height,
                self.config.bottle_min_xy_displacement,
                self.config.final_z_threshold,
                self.config.final_orientation_threshold_rad,
                self.config.static_velocity_threshold,
                self.config.object_linear_velocity_threshold,
                self.config.object_angular_velocity_threshold,
                self.config.grasp_finger_count,
                self._obj_pose,
                self._tcp_pose,
                self._tcp_to_obj,
                self._obj_to_goal,
                self._touching_finger_count,
                self._has_hand_contact,
                self._is_grasped,
                self._placement_pose_valid,
                self._release_ready,
                self._is_obj_placed,
                self._is_obj_static,
                self._is_robot_static,
                self._xy_displacement,
                self._final_z_error,
                self._orientation_error,
                self._current_lift_height,
                self._physical_max_lift_height,
                self._reaching_reward,
                self._lift_reward,
                self._transport_reward,
                self._place_reward,
                self._orientation_reward,
                self._static_reward,
            ],
            device=self.device,
        )

    def _advance_task_phase(self) -> None:
        wp.launch(
            _advance_transfer_phase,
            dim=self.num_envs,
            inputs=[
                self._obj_pose,
                self._initial_obj_pose,
                self._is_grasped,
                self._has_hand_contact,
                self._placement_pose_valid,
                self._is_obj_static,
                self._current_lift_height,
                self.config.bottle_lift_height,
                self.config.goal_threshold,
                self.config.transport_start_distance,
                self.config.grasp_confirm_frames,
                self.config.release_confirm_frames,
                self.config.settle_confirm_frames,
                self._task_phase,
                self._grasp_contact_frames,
                self._grasp_support_gap_frames,
                self._contact_gap_frames,
                self._settle_frames,
                self._grasp_confirmed,
                self._transport_started,
                self._reached_lift_height,
                self._release_armed,
                self._released,
                self._early_release,
                self._max_bottle_z,
                self._max_lift_height,
                self._success,
                self._fail,
            ],
            device=self.device,
        )

    def _refresh_task_state(self, *, read_contacts: bool, reset_mask: wp.array | None = None) -> None:
        if read_contacts:
            self._collect_finger_contacts()
        elif reset_mask is not None:
            self._clear_finger_contacts(reset_mask)
        self._evaluate_task_state()

    def _refresh_finger_root_load(self) -> None:
        if not self.config.request_finger_root_load:
            return
        wp.launch(
            _extract_finger_root_load,
            dim=(self.num_envs, _FINGER_ROOT_LOAD_SIZE),
            inputs=[
                self._qfrc_actuator,
                self.model.joint_dof_world_start,
                self._finger_root_local_qd,
                self._finger_root_closing_sign,
                self._finger_root_load_bias,
                self._finger_root_load_scale,
                self._finger_root_qfrc_actuator,
                self._finger_root_load,
            ],
            device=self.device,
        )

    def _clear_finger_root_load(self, world_mask: wp.array | None) -> None:
        wp.launch(
            _clear_finger_root_load_rows,
            dim=(self.num_envs, _FINGER_ROOT_LOAD_SIZE),
            inputs=[world_mask, self._finger_root_qfrc_actuator, self._finger_root_load],
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
        self._refresh_finger_root_load()
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
        if self.config.request_finger_root_load:
            observation["observation.finger_root_load"] = self._finger_root_load
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
            "task_phase": self._task_phase,
            "has_hand_contact": self._has_hand_contact,
            "had_hand_contact_this_control_step": self._had_hand_contact_this_control_step,
            "touching_finger_count": self._touching_finger_count,
            "finger_contact_counts": self._finger_contacts,
            "finger_contact_any_frame_this_control_step": self._finger_contact_any_frame_this_control_step,
            "opposed_grasp_any_frame_this_control_step": self._opposed_grasp_any_frame_this_control_step,
            "opposed_grasp_max_consecutive_physics_frames_this_control_step": (
                self._opposed_grasp_max_consecutive_frames_this_control_step
            ),
            "finger_surface_gap": self._finger_surface_gap,
            "thumb_partner_opposition": self._thumb_partner_opposition,
            "thumb_partner_z_score": self._thumb_partner_z_score,
            "opposed_pregrasp_score": self._opposed_pregrasp_score,
            "non_thumb_anchor_contact_fraction_this_control_step": (
                self._non_thumb_anchor_contact_fraction_this_control_step
            ),
            "non_thumb_missing_thumb_geometry_progress_this_control_step": (
                self._non_thumb_missing_thumb_geometry_progress_this_control_step
            ),
            "non_thumb_guidance_opposition_progress_this_control_step": (
                self._non_thumb_guidance_opposition_progress_this_control_step
            ),
            "non_thumb_guidance_z_progress_this_control_step": (self._non_thumb_guidance_z_progress_this_control_step),
            "thumb_anchor_contact_fraction_this_control_step": (self._thumb_anchor_contact_fraction_this_control_step),
            "thumb_missing_non_thumb_geometry_progress_this_control_step": (
                self._thumb_missing_non_thumb_geometry_progress_this_control_step
            ),
            "finger_root_qfrc_actuator": self._finger_root_qfrc_actuator,
            "finger_root_load": self._finger_root_load,
            "is_grasped": self._is_grasped,
            "grasp_confirmed": self._grasp_confirmed,
            "grasp_support_gap_frames": self._grasp_support_gap_frames,
            "transport_started": self._transport_started,
            "is_lifted": self._reached_lift_height,
            "release_armed": self._release_armed,
            "released": self._released,
            "early_release": self._early_release,
            "release_ready": self._release_ready,
            "is_obj_placed": self._is_obj_placed,
            "is_obj_static": self._is_obj_static,
            "is_robot_static": self._is_robot_static,
            "current_lift_height": self._current_lift_height,
            "lift_height": self._current_lift_height,
            "physical_max_lift_height": self._physical_max_lift_height,
            "max_contacted_carry_lift_height": self._max_lift_height,
            "max_lift_height": self._max_lift_height,
            "xy_displacement": self._xy_displacement,
            "final_z_error": self._final_z_error,
            "orientation_error": self._orientation_error,
        }

    def evaluate(self) -> dict[str, Any]:
        """Return ManiSkill-style batched task evaluation as CUDA Torch views."""
        return self._to_torch_tree(self.evaluate_warp())

    def compute_dense_reward(self, obs: Any = None, action: Any = None, info: Any = None) -> Any:
        """Return the latest bottle-transfer dense reward as a CUDA Torch view."""
        del obs, action, info
        return wp.to_torch(self._dense_reward)

    def compute_normalized_dense_reward(self, obs: Any = None, action: Any = None, info: Any = None) -> Any:
        """Return the latest dense reward normalized to the scale used by PPO."""
        del obs, action, info
        return wp.to_torch(self._dense_reward) / _STAGE_REWARD_MAX

    def _info_warp(self) -> dict[str, Any]:
        return {
            **self.evaluate_warp(),
            "elapsed_steps": self.episode_step,
            "control_step_diagnostics": {
                "rigid_contact_capacity": self._rigid_contact_capacity,
                "rigid_contact_frame_max": self._rigid_contact_frame_max,
                "rigid_contact_overflow_frame_count": self._rigid_contact_overflow_frame_count,
                "rigid_contact_overflow_excess_count": self._rigid_contact_overflow_excess_count,
                "triangle_pair_buffer_available": self._triangle_pair_buffer_available,
                "triangle_pair_capacity": self._triangle_pair_capacity,
                "triangle_pair_frame_max": self._triangle_pair_frame_max,
                "triangle_pair_overflow_frame_count": self._triangle_pair_overflow_frame_count,
                "triangle_pair_overflow_excess_count": self._triangle_pair_overflow_excess_count,
            },
            "episode": {
                "return": self.episode_return,
                "length": self.episode_step,
                "success_once": self.success_once,
                "success_at_end": self._success,
                "fail_at_end": self._fail,
                "task_phase": self._task_phase,
                "grasp_confirmed": self._grasp_confirmed,
                "reached_lift_height": self._reached_lift_height,
                "release_armed": self._release_armed,
                "released": self._released,
                "early_release": self._early_release,
                "has_hand_contact_at_end": self._has_hand_contact,
                "max_bottle_z": self._max_bottle_z,
                "current_lift_height": self._current_lift_height,
                "physical_max_lift_height": self._physical_max_lift_height,
                "max_contacted_carry_lift_height": self._max_lift_height,
                "max_lift_height": self._max_lift_height,
                "xy_displacement": self._xy_displacement,
                "final_z_error": self._final_z_error,
                "orientation_error": self._orientation_error,
            },
            "reward_components": {
                "reaching": self._reaching_reward,
                "opposed_pregrasp": self._opposed_pregrasp_score,
                "approach_base": self._approach_base_reward,
                "unilateral_guidance_gain": self._unilateral_guidance_gain,
                "unilateral_contact_reward": self._unilateral_contact_reward,
                "lift": self._lift_reward,
                "transport": self._transport_reward,
                "place": self._place_reward,
                "orientation": self._orientation_reward,
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
            action[:, :3].copy_(position)
            action[:, 3:6].copy_(rotation[:, 0, :])
            action[:, 6:9].copy_(rotation[:, 1, :])
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
        self._sync_hand_mimic_targets(mask_wp)
        wp.launch(
            _reset_episode_arrays,
            dim=self.num_envs,
            inputs=[
                mask_wp,
                self.episode_step,
                self.episode_return,
                self.success_once,
                self._approach_base_reward,
                self._unilateral_guidance_gain,
                self._unilateral_contact_reward,
                self._dense_reward,
                self.reward,
                self.terminated,
                self.truncated,
            ],
            device=self.device,
        )
        wp.launch(
            _reset_transfer_task,
            dim=self.num_envs,
            inputs=[
                mask_wp,
                self._current_lift_height,
                self._physical_max_lift_height,
                self._max_lift_height,
                self._task_phase,
                self._grasp_contact_frames,
                self._grasp_support_gap_frames,
                self._contact_gap_frames,
                self._settle_frames,
                self._grasp_confirmed,
                self._transport_started,
                self._reached_lift_height,
                self._release_armed,
                self._released,
                self._early_release,
                self._success,
                self._fail,
            ],
            device=self.device,
        )
        self._initialize_task_goal(mask_wp)
        self._clear_finger_contacts(mask_wp)
        self._clear_control_step_diagnostics(mask_wp)
        self.model.bvh_refit_shapes(self.state_0)
        self._refresh_observation()
        self._clear_finger_root_load(mask_wp)
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
        self._clear_control_step_diagnostics(None)
        for _ in range(self.frames_per_action):
            if self._scene.graph is not None:
                wp.capture_launch(self._scene.graph)
            else:
                self._scene.simulate()
            self._collect_finger_contacts()
            self._evaluate_task_state()
            self._accumulate_control_step_diagnostics()
            self._advance_task_phase()
        self._refresh_observation()
        wp.launch(
            _advance_episode,
            dim=self.num_envs,
            inputs=[
                self.episode_step,
                self.episode_return,
                self.success_once,
                self._reaching_reward,
                self._opposed_pregrasp_score,
                self._max_lift_height,
                self.config.bottle_lift_height,
                self._place_reward,
                self._static_reward,
                self._finger_contacts,
                self._is_grasped,
                self._finger_contact_any_frame_this_control_step,
                self._opposed_grasp_any_frame_this_control_step,
                self._opposed_grasp_max_consecutive_frames_this_control_step,
                self._non_thumb_anchor_contact_fraction_this_control_step,
                self._non_thumb_missing_thumb_geometry_progress_this_control_step,
                self._thumb_anchor_contact_fraction_this_control_step,
                self._thumb_missing_non_thumb_geometry_progress_this_control_step,
                self._task_phase,
                self._reached_lift_height,
                self._release_ready,
                self._is_obj_placed,
                self._success,
                self._fail,
                self._approach_base_reward,
                self._unilateral_guidance_gain,
                self._unilateral_contact_reward,
                self._dense_reward,
                self.reward,
                self.terminated,
                self.truncated,
                self.config.max_episode_steps,
                self._reward_mode_id,
                self.config.terminate_on_success,
                self.config.terminate_on_fail,
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
