from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

import newton
from newton._src.sim.ik.ik_common import eval_fk_batched as _eval_fk_batched
from teleop_stack.ik.differential_ik import PositionJacobian, SpatialJacobian
from teleop_stack.ik.full_pose import solve_full_pose_damped_least_squares_step
from teleop_stack.ik.full_pose_controller import (
    FullPoseDifferentialIkController,
    FullPoseDifferentialIkControllerConfig,
)
from teleop_stack.ik.so3 import (
    QuaternionXYZW,
    quat_align_hemisphere_xyzw,
    quat_inverse_xyzw,
    quat_log_rotvec_xyzw,
    quat_multiply_xyzw,
    quat_normalize_xyzw,
)
from teleop_stack.ik.types import RobotStateSnapshot, TaskSpaceTarget
from teleop_stack.models import ArmSide, NamedJointValues, Pose7, SingleArmTeleopCommand
from teleop_stack.robots.base import RobotInterface
from teleop_stack.robots.nero_runtime import NeroTeleopMappingConfig
from teleop_stack.teleop.openxr_genesis_adapter import (
    adapt_openxr_hand_frame_to_genesis_parent,
    adapt_openxr_hand_frame_to_genesis_wrist_frame,
)
from teleop_stack.teleop.orientation_tracker import (
    OrientationTargetTracker,
    OrientationTrackerResult,
    QuaternionWXYZ,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from teleop_stack.teleop.spatial_frames import (
    BeavrHandFrameSmoother,
    FrameAxes,
    HandAnatomicalFrame,
    hand_anatomical_frame_from_debug,
    hand_beavr_anatomical_frame_from_debug,
    matrix_from_axes,
    matrix_to_quat_xyzw,
    quat_xyzw_to_matrix,
)

if TYPE_CHECKING:
    from debug.import_dual_nero_linker_l10 import Example


_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_FINGER_COUNT = len(_FINGER_NAMES)


@wp.kernel(enable_backward=False)
def _extract_eef_pose_batch_kernel(
    body_q: wp.array2d[wp.transform],
    eef_body_index: int,
    pose_out: wp.array2d[wp.float32],
):
    batch_index = wp.tid()
    eef_transform = body_q[batch_index, eef_body_index]
    position = wp.transform_get_translation(eef_transform)
    rotation = wp.transform_get_rotation(eef_transform)
    pose_out[batch_index, 0] = position[0]
    pose_out[batch_index, 1] = position[1]
    pose_out[batch_index, 2] = position[2]
    pose_out[batch_index, 3] = rotation[0]
    pose_out[batch_index, 4] = rotation[1]
    pose_out[batch_index, 5] = rotation[2]
    pose_out[batch_index, 6] = rotation[3]


@wp.kernel(enable_backward=False)
def _full_pose_dls_solve_kernel(
    spatial_jacobian: wp.array2d[wp.float32],
    position_error_xyz: wp.array[wp.float32],
    orientation_error_rotvec: wp.array[wp.float32],
    bias_step: wp.array[wp.float32],
    has_bias_step: int,
    damping_lambda: float,
    position_weight: float,
    orientation_weight: float,
    max_position_step_m: float,
    max_rotation_step_rad: float,
    bias_weight: float,
    system: wp.array2d[wp.float32],
    cholesky: wp.array2d[wp.float32],
    rhs: wp.array[wp.float32],
    y: wp.array[wp.float32],
    dq_out: wp.array[wp.float32],
    status: wp.array[wp.int32],
    current_q: wp.array[wp.float32],
    previous_velocity: wp.array[wp.float32],
    lower_limits: wp.array[wp.float32],
    upper_limits: wp.array[wp.float32],
    arm_joint_q_indices: wp.array[wp.int32],
    arm_joint_qd_indices: wp.array[wp.int32],
    target_joint_q: wp.array[wp.float32],
    target_joint_qd: wp.array[wp.float32],
    has_previous_velocity: int,
    max_joint_step_rad: float,
    max_joint_velocity_rad_s: float,
    max_joint_acceleration_rad_s2: float,
    dt_s: float,
    write_joint_targets: int,
    q_cmd_out: wp.array[wp.float32],
    dq_cmd_out: wp.array[wp.float32],
    unclipped_q_cmd_out: wp.array[wp.float32],
    flags_out: wp.array[wp.int32],
):
    flags_out[0] = int(0)
    flags_out[1] = int(0)

    pos_norm = wp.sqrt(
        position_error_xyz[0] * position_error_xyz[0]
        + position_error_xyz[1] * position_error_xyz[1]
        + position_error_xyz[2] * position_error_xyz[2]
    )
    pos_scale = float(1.0)
    if max_position_step_m > 0.0 and pos_norm > max_position_step_m and pos_norm > 1.0e-12:
        pos_scale = max_position_step_m / pos_norm

    rot_norm = wp.sqrt(
        orientation_error_rotvec[0] * orientation_error_rotvec[0]
        + orientation_error_rotvec[1] * orientation_error_rotvec[1]
        + orientation_error_rotvec[2] * orientation_error_rotvec[2]
    )
    rot_scale = float(1.0)
    if max_rotation_step_rad > 0.0 and rot_norm > max_rotation_step_rad and rot_norm > 1.0e-12:
        rot_scale = max_rotation_step_rad / rot_norm

    diagonal_regularization = wp.max(float(0.0), damping_lambda)
    diagonal_regularization = diagonal_regularization * diagonal_regularization + wp.max(float(0.0), bias_weight)

    i = int(0)
    while i < 7:
        rhs[i] = float(0.0)
        j = int(0)
        while j < 7:
            system[i, j] = float(0.0)
            cholesky[i, j] = float(0.0)
            j = j + 1
        i = i + 1

    row = int(0)
    while row < 6:
        weight = position_weight
        task_delta = position_weight * position_error_xyz[row] * pos_scale
        if row >= 3:
            weight = orientation_weight
            task_delta = orientation_weight * orientation_error_rotvec[row - 3] * rot_scale

        col = int(0)
        while col < 7:
            j_col = spatial_jacobian[row, col] * weight
            rhs[col] = rhs[col] + j_col * task_delta
            other_col = int(0)
            while other_col < 7:
                system[col, other_col] = system[col, other_col] + j_col * spatial_jacobian[row, other_col] * weight
                other_col = other_col + 1
            col = col + 1
        row = row + 1

    i = int(0)
    while i < 7:
        system[i, i] = system[i, i] + diagonal_regularization
        if has_bias_step != 0 and bias_weight > 0.0:
            rhs[i] = rhs[i] + bias_weight * bias_step[i]
        i = i + 1

    success = int(1)
    i = int(0)
    while i < 7:
        j = int(0)
        while j <= i:
            subtotal = float(0.0)
            k = int(0)
            while k < j:
                subtotal = subtotal + cholesky[i, k] * cholesky[j, k]
                k = k + 1
            if i == j:
                diagonal = system[i, i] - subtotal
                if diagonal <= 1.0e-12:
                    success = int(0)
                if success != 0:
                    cholesky[i, j] = wp.sqrt(diagonal)
            else:
                denominator = cholesky[j, j]
                if wp.abs(denominator) <= 1.0e-12:
                    success = int(0)
                if success != 0:
                    cholesky[i, j] = (system[i, j] - subtotal) / denominator
            j = j + 1
        i = i + 1

    if success == 0:
        i = int(0)
        while i < 7:
            dq_out[i] = float(0.0)
            q_cmd_out[i] = current_q[i]
            dq_cmd_out[i] = float(0.0)
            unclipped_q_cmd_out[i] = current_q[i]
            i = i + 1
        status[0] = int(0)
        return

    i = int(0)
    while i < 7:
        subtotal = float(0.0)
        k = int(0)
        while k < i:
            subtotal = subtotal + cholesky[i, k] * y[k]
            k = k + 1
        y[i] = (rhs[i] - subtotal) / cholesky[i, i]
        i = i + 1

    ii = int(0)
    while ii < 7:
        i = 6 - ii
        subtotal = float(0.0)
        k = i + 1
        while k < 7:
            subtotal = subtotal + cholesky[k, i] * dq_out[k]
            k = k + 1
        dq_out[i] = (y[i] - subtotal) / cholesky[i, i]
        ii = ii + 1

    if write_joint_targets != 0:
        safe_dt = wp.max(float(1.0e-6), dt_s)
        allowed_step = wp.max(float(0.0), max_joint_step_rad)
        safe_velocity = wp.max(float(0.0), max_joint_velocity_rad_s)
        if safe_velocity > 0.0 and safe_dt > 0.0:
            velocity_step = safe_velocity * safe_dt
            if allowed_step > 0.0:
                allowed_step = wp.min(allowed_step, velocity_step)
            else:
                allowed_step = velocity_step

        max_delta_velocity = wp.max(float(0.0), max_joint_acceleration_rad_s2) * safe_dt
        i = int(0)
        while i < 7:
            step = dq_out[i]
            if allowed_step <= 0.0:
                step = float(0.0)
            else:
                step = wp.clamp(step, -allowed_step, allowed_step)

            if has_previous_velocity != 0 and max_joint_acceleration_rad_s2 > 0.0:
                desired_velocity = step / safe_dt
                lower_velocity = previous_velocity[i] - max_delta_velocity
                upper_velocity = previous_velocity[i] + max_delta_velocity
                bounded_velocity = wp.clamp(desired_velocity, lower_velocity, upper_velocity)
                if wp.abs(bounded_velocity - desired_velocity) > 1.0e-12:
                    flags_out[0] = int(1)
                step = bounded_velocity * safe_dt

            unclipped_q = current_q[i] + step
            q_cmd = wp.clamp(unclipped_q, lower_limits[i], upper_limits[i])
            if wp.abs(q_cmd - unclipped_q) > 1.0e-12:
                flags_out[1] = int(1)

            applied_step = q_cmd - current_q[i]
            dq_cmd = applied_step / safe_dt
            q_cmd_out[i] = q_cmd
            dq_cmd_out[i] = dq_cmd
            unclipped_q_cmd_out[i] = unclipped_q

            q_index = arm_joint_q_indices[i]
            if q_index >= 0:
                target_joint_q[q_index] = q_cmd
            qd_index = arm_joint_qd_indices[i]
            if qd_index >= 0:
                target_joint_qd[qd_index] = dq_cmd
            previous_velocity[i] = dq_cmd
            i = i + 1

    status[0] = int(1)


@wp.kernel(enable_backward=False)
def _gather_arm_joint_state_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    arm_joint_q_indices: wp.array[wp.int32],
    arm_joint_qd_indices: wp.array[wp.int32],
    arm_joint_q_out: wp.array[wp.float32],
    arm_joint_qd_out: wp.array[wp.float32],
):
    joint_index = wp.tid()
    arm_joint_q_out[joint_index] = joint_q[arm_joint_q_indices[joint_index]]
    arm_joint_qd_out[joint_index] = joint_qd[arm_joint_qd_indices[joint_index]]


@wp.kernel(enable_backward=False)
def _write_hand_target_slices_kernel(
    current_joint_q: wp.array[wp.float32],
    hand_joint_q_indices: wp.array[wp.int32],
    hand_joint_qd_indices: wp.array[wp.int32],
    hand_joint_targets: wp.array[wp.float32],
    target_joint_q: wp.array[wp.float32],
    target_joint_qd: wp.array[wp.float32],
    count: int,
    max_step_rad: float,
    frame_dt: float,
    publish_kinematic_velocity: int,
):
    slot = wp.tid()
    if slot >= count:
        return

    q_index = hand_joint_q_indices[slot]
    if q_index < 0:
        return

    current_value = current_joint_q[q_index]
    target_value = hand_joint_targets[slot]
    if max_step_rad > 0.0:
        target_value = wp.clamp(target_value, current_value - max_step_rad, current_value + max_step_rad)

    target_joint_q[q_index] = target_value
    qd_index = hand_joint_qd_indices[slot]
    if qd_index >= 0:
        if publish_kinematic_velocity != 0:
            target_joint_qd[qd_index] = (target_value - current_value) / frame_dt
        else:
            target_joint_qd[qd_index] = 0.0


@wp.kernel(enable_backward=False)
def _l10_bottle_contact_stop_update_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[wp.float32],
    rigid_contact_margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_finger_id: wp.array[wp.int32],
    bottle_shape_index: int,
    rigid_contact_max: int,
    activation_m: float,
    threshold_m: float,
    release_m: float,
    stopped: wp.array[wp.int32],
    max_metric: wp.array[wp.float32],
):
    finger_id = wp.tid()
    raw_count = rigid_contact_count[0]
    active_count = int(raw_count)
    if active_count > rigid_contact_max:
        active_count = rigid_contact_max

    finger_metric = float(0.0)
    contact_index = int(0)
    while contact_index < active_count:
        shape0 = rigid_contact_shape0[contact_index]
        shape1 = rigid_contact_shape1[contact_index]
        shape0_finger_id = int(-1)
        shape1_finger_id = int(-1)
        if shape0 >= 0:
            shape0_finger_id = shape_finger_id[shape0]
        if shape1 >= 0:
            shape1_finger_id = shape_finger_id[shape1]
        shape0_matches = shape0_finger_id == finger_id and shape1 == bottle_shape_index
        shape1_matches = shape1_finger_id == finger_id and shape0 == bottle_shape_index

        if shape0_matches or shape1_matches:
            body0 = shape_body[shape0]
            body1 = shape_body[shape1]
            support0_world = wp.transform_point(body_q[body0], rigid_contact_point0[contact_index])
            support1_world = wp.transform_point(body_q[body1], rigid_contact_point1[contact_index])
            normal_shape0_to_shape1 = rigid_contact_normal[contact_index]
            normal_norm = wp.length(normal_shape0_to_shape1)
            if normal_norm > 0.0:
                normal_shape0_to_shape1 = normal_shape0_to_shape1 / normal_norm
            separation_m = wp.dot(normal_shape0_to_shape1, support1_world - support0_world) - (
                rigid_contact_margin0[contact_index] + rigid_contact_margin1[contact_index]
            )
            penetration_m = wp.max(float(0.0), -separation_m)
            stop_metric_m = penetration_m
            if separation_m <= activation_m:
                stop_metric_m = wp.max(stop_metric_m, threshold_m)
            finger_metric = wp.max(finger_metric, stop_metric_m)

        contact_index = contact_index + 1

    max_metric[finger_id] = finger_metric
    if finger_metric >= threshold_m:
        stopped[finger_id] = 1
    elif finger_metric <= release_m:
        stopped[finger_id] = 0


@wp.kernel(enable_backward=False)
def _l10_bottle_contact_stop_clamp_targets_kernel(
    current_joint_q: wp.array[wp.float32],
    target_joint_q: wp.array[wp.float32],
    target_joint_qd: wp.array[wp.float32],
    joint_finger_id: wp.array[wp.int32],
    joint_closing_direction: wp.array[wp.float32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    joint_coord_to_dof: wp.array[wp.int32],
    stopped: wp.array[wp.int32],
    stop_retreat_rad: float,
    release_retreat_rad: float,
    frame_dt: float,
    publish_kinematic_velocity: int,
):
    q_index = wp.tid()
    finger_id = joint_finger_id[q_index]
    if finger_id < 0:
        return
    if stopped[finger_id] == 0:
        return

    closing_direction = joint_closing_direction[q_index]
    if wp.abs(closing_direction) <= 0.0:
        return

    current_value = current_joint_q[q_index]
    target_value = target_joint_q[q_index]
    closing_delta = (target_value - current_value) * closing_direction

    if closing_delta < 0.0:
        retreat = release_retreat_rad
        if retreat <= 0.0:
            return
        signed_target = target_value * closing_direction
        signed_release = (current_value - closing_direction * retreat) * closing_direction
        target_value = wp.min(signed_target, signed_release) * closing_direction
    elif closing_delta > 0.0:
        target_value = current_value - closing_direction * stop_retreat_rad
    else:
        return

    target_value = wp.clamp(target_value, joint_limit_lower[q_index], joint_limit_upper[q_index])
    target_joint_q[q_index] = target_value

    qd_index = joint_coord_to_dof[q_index]
    if qd_index >= 0:
        if publish_kinematic_velocity != 0:
            target_joint_qd[qd_index] = (target_value - current_value) / frame_dt
        else:
            target_joint_qd[qd_index] = 0.0


@dataclass(frozen=True)
class NewtonRuntimeRobotConfig:
    arm_side: ArmSide = "right"
    publish_mode: str = "drive_target"
    drive_ik: bool = True
    relative_control: bool = True
    require_initial_anchor: bool = True
    eef_body_suffix_by_side: dict[ArmSide, str] = field(
        default_factory=lambda: {"left": "/left_revo2_flange", "right": "/right_revo2_flange"}
    )
    openxr_yaw_recenter: bool = True
    finite_difference_rad: float = 1.0e-4
    hand_max_joint_step_rad: float = 0.0
    hand_publish_kinematic_velocity: bool = True
    hand_contact_stop_enabled: bool = True
    hand_contact_stop_retreat_rad: float = 0.01
    hand_contact_release_retreat_rad: float = 0.0
    mapping: NeroTeleopMappingConfig = field(default_factory=NeroTeleopMappingConfig)
    ik_config_overrides: dict[str, object] = field(default_factory=dict)


class NewtonLinkKinematicsModel:
    def __init__(
        self,
        *,
        model: newton.Model,
        side: ArmSide,
        arm_joint_q_indices: tuple[int, ...],
        eef_body_suffix: str,
        finite_difference_rad: float,
    ) -> None:
        if len(arm_joint_q_indices) != 7:
            raise ValueError(f"Newton Nero kinematics expects seven arm joints, got {len(arm_joint_q_indices)}")
        self.model = model
        self.side = side
        self.arm_joint_q_indices = tuple(int(v) for v in arm_joint_q_indices)
        self._arm_joint_q_indices_np = np.asarray(self.arm_joint_q_indices, dtype=np.int64)
        self.eef_body_index = _find_body_index(model, eef_body_suffix)
        self.finite_difference_rad = max(abs(float(finite_difference_rad)), 1.0e-6)
        self._joint_q_host = model.joint_q.numpy().copy()
        self._fk_batch_size = 1 + 2 * len(self.arm_joint_q_indices)
        self._joint_q_batch_host = np.zeros(
            (self._fk_batch_size, int(model.joint_coord_count)),
            dtype=np.float32,
        )
        self._joint_q_batch_wp = wp.zeros(
            (self._fk_batch_size, int(model.joint_coord_count)),
            dtype=wp.float32,
            device=model.device,
        )
        self._joint_qd_batch_wp = wp.zeros(
            (self._fk_batch_size, int(model.joint_dof_count)),
            dtype=wp.float32,
            device=model.device,
        )
        self._body_q_batch_wp = wp.empty(
            (self._fk_batch_size, int(model.body_count)),
            dtype=wp.transform,
            device=model.device,
        )
        self._body_qd_batch_wp = wp.empty(
            (self._fk_batch_size, int(model.body_count)),
            dtype=wp.spatial_vector,
            device=model.device,
        )
        self._eef_pose_batch_wp = wp.empty((self._fk_batch_size, 7), dtype=wp.float32, device=model.device)
        self._last_jacobian_q: tuple[float, ...] | None = None
        self._last_spatial_jacobian: SpatialJacobian | None = None
        self._last_pose: Pose7 | None = None

    def sync_joint_q(self, joint_q: np.ndarray) -> None:
        self._joint_q_host = np.asarray(joint_q, dtype=np.float32).copy()
        self.clear_cache()

    def forward_pose(self, joint_positions_rad: tuple[float, ...]) -> Pose7:
        q = _require_seven_dof(joint_positions_rad)
        if self._last_jacobian_q is None or not _same_joint_tuple(q, self._last_jacobian_q):
            self._compute_pose_and_spatial_jacobian(q)
        assert self._last_pose is not None
        return self._last_pose

    def position_jacobian(self, joint_positions_rad: tuple[float, ...]) -> PositionJacobian:
        spatial = self.spatial_jacobian(joint_positions_rad)
        return (
            tuple(float(v) for v in spatial[0]),
            tuple(float(v) for v in spatial[1]),
            tuple(float(v) for v in spatial[2]),
        )

    def spatial_jacobian(self, joint_positions_rad: tuple[float, ...]) -> SpatialJacobian:
        q = _require_seven_dof(joint_positions_rad)
        if self._last_jacobian_q is not None and _same_joint_tuple(q, self._last_jacobian_q):
            assert self._last_spatial_jacobian is not None
            return self._last_spatial_jacobian

        self._compute_pose_and_spatial_jacobian(q)
        assert self._last_spatial_jacobian is not None
        return self._last_spatial_jacobian

    def _compute_pose_and_spatial_jacobian(self, q: tuple[float, ...]) -> None:
        eps = float(self.finite_difference_rad)
        q_np = np.asarray(q, dtype=np.float32)
        self._joint_q_batch_host[:, :] = self._joint_q_host
        self._joint_q_batch_host[:, self._arm_joint_q_indices_np] = q_np
        for joint_index in range(7):
            plus_row = 1 + joint_index
            minus_row = 1 + 7 + joint_index
            target_index = self.arm_joint_q_indices[joint_index]
            self._joint_q_batch_host[plus_row, target_index] = np.float32(q[joint_index] + eps)
            self._joint_q_batch_host[minus_row, target_index] = np.float32(q[joint_index] - eps)

        self._joint_q_batch_wp.assign(self._joint_q_batch_host)
        _eval_fk_batched(
            self.model,
            self._joint_q_batch_wp,
            self._joint_qd_batch_wp,
            self._body_q_batch_wp,
            self._body_qd_batch_wp,
        )
        wp.launch(
            kernel=_extract_eef_pose_batch_kernel,
            dim=self._fk_batch_size,
            inputs=[self._body_q_batch_wp, self.eef_body_index, self._eef_pose_batch_wp],
            device=self.model.device,
        )
        pose_batch = self._eef_pose_batch_wp.numpy()

        center = pose_batch[0]
        center_quaternion = quat_normalize_xyzw(tuple(float(v) for v in center[3:7]))  # type: ignore[arg-type]
        self._last_pose = Pose7(
            position_xyz=tuple(float(v) for v in center[:3]),
            quaternion_xyzw=center_quaternion,
        )

        columns: list[tuple[float, float, float, float, float, float]] = []
        for joint_index in range(7):
            plus = pose_batch[1 + joint_index]
            minus = pose_batch[1 + 7 + joint_index]
            pos_plus = np.asarray(plus[:3], dtype=np.float64)
            pos_minus = np.asarray(minus[:3], dtype=np.float64)
            quat_plus = quat_normalize_xyzw(tuple(float(v) for v in plus[3:7]))  # type: ignore[arg-type]
            quat_minus = quat_normalize_xyzw(tuple(float(v) for v in minus[3:7]))  # type: ignore[arg-type]
            linear = (pos_plus - pos_minus) / (2.0 * eps)
            quat_plus = quat_align_hemisphere_xyzw(quat_plus, quat_minus)
            delta_quat = quat_multiply_xyzw(quat_plus, quat_inverse_xyzw(quat_minus))
            angular = np.asarray(quat_log_rotvec_xyzw(delta_quat), dtype=np.float64) / (2.0 * eps)
            columns.append(
                (
                    float(linear[0]),
                    float(linear[1]),
                    float(linear[2]),
                    float(angular[0]),
                    float(angular[1]),
                    float(angular[2]),
                )
            )

        spatial: SpatialJacobian = tuple(tuple(float(columns[col][row]) for col in range(7)) for row in range(6))  # type: ignore[assignment]
        self._last_jacobian_q = q
        self._last_spatial_jacobian = spatial
        return spatial

    def clear_cache(self) -> None:
        self._last_jacobian_q = None
        self._last_spatial_jacobian = None
        self._last_pose = None


class NewtonGpuFullPoseDlsStepSolver:
    def __init__(self, *, device: Any, result_readback_stride: int = 1):
        self.device = device
        self.last_wrote_joint_targets = False
        self.last_result_was_synced = False
        self._result_readback_stride = max(1, int(result_readback_stride))
        self._solve_count = 0
        self._has_previous_velocity = False
        self._last_q_cmd_host = np.zeros(7, dtype=np.float32)
        self._last_dq_cmd_host = np.zeros(7, dtype=np.float32)
        self._last_unclipped_q_cmd_host = np.zeros(7, dtype=np.float32)
        self._last_acceleration_limited = False
        self._last_clipped = False
        self._spatial_host = np.zeros((6, 7), dtype=np.float32)
        self._position_error_host = np.zeros(3, dtype=np.float32)
        self._orientation_error_host = np.zeros(3, dtype=np.float32)
        self._bias_step_host = np.zeros(7, dtype=np.float32)
        self._current_q_host = np.zeros(7, dtype=np.float32)
        self._lower_limits_host = np.zeros(7, dtype=np.float32)
        self._upper_limits_host = np.zeros(7, dtype=np.float32)
        self._spatial_wp = wp.zeros((6, 7), dtype=wp.float32, device=device)
        self._position_error_wp = wp.zeros(3, dtype=wp.float32, device=device)
        self._orientation_error_wp = wp.zeros(3, dtype=wp.float32, device=device)
        self._bias_step_wp = wp.zeros(7, dtype=wp.float32, device=device)
        self._current_q_wp = wp.zeros(7, dtype=wp.float32, device=device)
        self._previous_velocity_wp = wp.zeros(7, dtype=wp.float32, device=device)
        self._lower_limits_wp = wp.zeros(7, dtype=wp.float32, device=device)
        self._upper_limits_wp = wp.zeros(7, dtype=wp.float32, device=device)
        self._system_wp = wp.zeros((7, 7), dtype=wp.float32, device=device)
        self._cholesky_wp = wp.zeros((7, 7), dtype=wp.float32, device=device)
        self._rhs_wp = wp.zeros(7, dtype=wp.float32, device=device)
        self._y_wp = wp.zeros(7, dtype=wp.float32, device=device)
        self._dq_out_wp = wp.zeros(7, dtype=wp.float32, device=device)
        self._status_wp = wp.zeros(1, dtype=wp.int32, device=device)
        self._q_cmd_out_wp = wp.zeros(7, dtype=wp.float32, device=device)
        self._dq_cmd_out_wp = wp.zeros(7, dtype=wp.float32, device=device)
        self._unclipped_q_cmd_out_wp = wp.zeros(7, dtype=wp.float32, device=device)
        self._flags_wp = wp.zeros(2, dtype=wp.int32, device=device)
        self._arm_joint_q_indices_wp = wp.zeros(7, dtype=wp.int32, device=device)
        self._arm_joint_qd_indices_wp = wp.zeros(7, dtype=wp.int32, device=device)
        self._dummy_target_joint_q_wp = wp.zeros(1, dtype=wp.float32, device=device)
        self._dummy_target_joint_qd_wp = wp.zeros(1, dtype=wp.float32, device=device)
        self._target_joint_q_wp: wp.array | None = None
        self._target_joint_qd_wp: wp.array | None = None

    def reset(self) -> None:
        self.last_wrote_joint_targets = False
        self.last_result_was_synced = False
        self._solve_count = 0
        self._has_previous_velocity = False
        self._last_acceleration_limited = False
        self._last_clipped = False
        self._last_q_cmd_host.fill(0.0)
        self._last_dq_cmd_host.fill(0.0)
        self._last_unclipped_q_cmd_host.fill(0.0)
        self._previous_velocity_wp.zero_()

    def configure_target_writer(
        self,
        *,
        arm_joint_q_indices: tuple[int, ...],
        arm_joint_qd_indices: tuple[int, ...],
        target_joint_q: wp.array,
        target_joint_qd: wp.array,
    ) -> None:
        if len(arm_joint_q_indices) != 7 or len(arm_joint_qd_indices) != 7:
            raise ValueError("Newton GPU DLS target writer expects seven arm q and qd indices.")
        self._arm_joint_q_indices_wp.assign(np.asarray(arm_joint_q_indices, dtype=np.int32))
        self._arm_joint_qd_indices_wp.assign(np.asarray(arm_joint_qd_indices, dtype=np.int32))
        self._target_joint_q_wp = target_joint_q
        self._target_joint_qd_wp = target_joint_qd

    def _upload_task_inputs(
        self,
        spatial_jacobian: SpatialJacobian,
        *,
        position_error_xyz: tuple[float, float, float],
        orientation_error_rotvec: tuple[float, float, float],
        bias_step: tuple[float, ...] | None,
    ) -> int:
        self._spatial_host[:, :] = np.asarray(spatial_jacobian, dtype=np.float32)
        self._position_error_host[:] = np.asarray(position_error_xyz, dtype=np.float32)
        self._orientation_error_host[:] = np.asarray(orientation_error_rotvec, dtype=np.float32)
        has_bias_step = bias_step is not None
        if has_bias_step:
            self._bias_step_host[:] = np.asarray(bias_step, dtype=np.float32)
        else:
            self._bias_step_host.fill(0.0)

        self._spatial_wp.assign(self._spatial_host)
        self._position_error_wp.assign(self._position_error_host)
        self._orientation_error_wp.assign(self._orientation_error_host)
        self._bias_step_wp.assign(self._bias_step_host)
        return 1 if has_bias_step else 0

    def _launch_dls_kernel(
        self,
        *,
        has_bias_step: int,
        damping_lambda: float,
        position_weight: float,
        orientation_weight: float,
        max_position_step_m: float | None,
        max_rotation_step_rad: float | None,
        bias_weight: float,
        has_previous_velocity: int = 0,
        max_joint_step_rad: float = 0.0,
        max_joint_velocity_rad_s: float = 0.0,
        max_joint_acceleration_rad_s2: float = 0.0,
        dt_s: float = 0.0,
        write_joint_targets: int = 0,
    ) -> None:
        target_joint_q = (
            self._target_joint_q_wp if self._target_joint_q_wp is not None else self._dummy_target_joint_q_wp
        )
        target_joint_qd = (
            self._target_joint_qd_wp if self._target_joint_qd_wp is not None else self._dummy_target_joint_qd_wp
        )
        wp.launch(
            kernel=_full_pose_dls_solve_kernel,
            dim=1,
            inputs=[
                self._spatial_wp,
                self._position_error_wp,
                self._orientation_error_wp,
                self._bias_step_wp,
                int(has_bias_step),
                float(damping_lambda),
                float(position_weight),
                float(orientation_weight),
                -1.0 if max_position_step_m is None else float(max_position_step_m),
                -1.0 if max_rotation_step_rad is None else float(max_rotation_step_rad),
                float(bias_weight),
                self._system_wp,
                self._cholesky_wp,
                self._rhs_wp,
                self._y_wp,
                self._dq_out_wp,
                self._status_wp,
                self._current_q_wp,
                self._previous_velocity_wp,
                self._lower_limits_wp,
                self._upper_limits_wp,
                self._arm_joint_q_indices_wp,
                self._arm_joint_qd_indices_wp,
                target_joint_q,
                target_joint_qd,
                int(has_previous_velocity),
                float(max_joint_step_rad),
                float(max_joint_velocity_rad_s),
                float(max_joint_acceleration_rad_s2),
                float(dt_s),
                int(write_joint_targets),
                self._q_cmd_out_wp,
                self._dq_cmd_out_wp,
                self._unclipped_q_cmd_out_wp,
                self._flags_wp,
            ],
            device=self.device,
        )

    def solve_full_pose_step(
        self,
        spatial_jacobian: SpatialJacobian,
        *,
        position_error_xyz: tuple[float, float, float],
        orientation_error_rotvec: tuple[float, float, float],
        damping_lambda: float,
        position_weight: float,
        orientation_weight: float,
        max_position_step_m: float | None,
        max_rotation_step_rad: float | None,
        bias_step: tuple[float, ...] | None,
        bias_weight: float,
    ) -> tuple[float, ...]:
        self.last_wrote_joint_targets = False
        has_bias_step = self._upload_task_inputs(
            spatial_jacobian,
            position_error_xyz=position_error_xyz,
            orientation_error_rotvec=orientation_error_rotvec,
            bias_step=bias_step,
        )
        self._launch_dls_kernel(
            has_bias_step=has_bias_step,
            damping_lambda=damping_lambda,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            max_position_step_m=max_position_step_m,
            max_rotation_step_rad=max_rotation_step_rad,
            bias_weight=bias_weight,
        )
        if int(self._status_wp.numpy()[0]) == 1:
            return tuple(float(v) for v in self._dq_out_wp.numpy())

        return solve_full_pose_damped_least_squares_step(
            spatial_jacobian,
            position_error_xyz=position_error_xyz,
            orientation_error_rotvec=orientation_error_rotvec,
            damping_lambda=damping_lambda,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            max_position_step_m=max_position_step_m,
            max_rotation_step_rad=max_rotation_step_rad,
            bias_step=bias_step,
            bias_weight=bias_weight,
        )

    def solve_full_pose_joint_step(
        self,
        spatial_jacobian: SpatialJacobian,
        *,
        position_error_xyz: tuple[float, float, float],
        orientation_error_rotvec: tuple[float, float, float],
        damping_lambda: float,
        position_weight: float,
        orientation_weight: float,
        max_position_step_m: float | None,
        max_rotation_step_rad: float | None,
        bias_step: tuple[float, ...] | None,
        bias_weight: float,
        current_q: tuple[float, ...],
        previous_velocity_rad_s: tuple[float, ...] | None,
        lower_limits_rad: tuple[float, ...],
        upper_limits_rad: tuple[float, ...],
        max_joint_step_rad: float,
        max_joint_velocity_rad_s: float,
        max_joint_acceleration_rad_s2: float,
        dt_s: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], bool, bool]:
        self.last_wrote_joint_targets = False
        if self._target_joint_q_wp is None or self._target_joint_qd_wp is None:
            dq_step = self.solve_full_pose_step(
                spatial_jacobian,
                position_error_xyz=position_error_xyz,
                orientation_error_rotvec=orientation_error_rotvec,
                damping_lambda=damping_lambda,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
                max_position_step_m=max_position_step_m,
                max_rotation_step_rad=max_rotation_step_rad,
                bias_step=bias_step,
                bias_weight=bias_weight,
            )
            return _postprocess_joint_step_host(
                dq_step,
                current_q=current_q,
                previous_velocity_rad_s=previous_velocity_rad_s,
                lower_limits_rad=lower_limits_rad,
                upper_limits_rad=upper_limits_rad,
                max_joint_step_rad=max_joint_step_rad,
                max_joint_velocity_rad_s=max_joint_velocity_rad_s,
                max_joint_acceleration_rad_s2=max_joint_acceleration_rad_s2,
                dt_s=dt_s,
            )

        has_bias_step = self._upload_task_inputs(
            spatial_jacobian,
            position_error_xyz=position_error_xyz,
            orientation_error_rotvec=orientation_error_rotvec,
            bias_step=bias_step,
        )
        self._current_q_host[:] = np.asarray(current_q, dtype=np.float32)
        if (
            not self._has_previous_velocity
            and previous_velocity_rad_s is not None
            and len(previous_velocity_rad_s) == 7
        ):
            self._last_dq_cmd_host[:] = np.asarray(previous_velocity_rad_s, dtype=np.float32)
            self._previous_velocity_wp.assign(self._last_dq_cmd_host)
            self._has_previous_velocity = True
        has_previous_velocity = 1 if self._has_previous_velocity else 0
        self._lower_limits_host[:] = np.asarray(lower_limits_rad, dtype=np.float32)
        self._upper_limits_host[:] = np.asarray(upper_limits_rad, dtype=np.float32)
        self._current_q_wp.assign(self._current_q_host)
        self._lower_limits_wp.assign(self._lower_limits_host)
        self._upper_limits_wp.assign(self._upper_limits_host)

        self._launch_dls_kernel(
            has_bias_step=has_bias_step,
            damping_lambda=damping_lambda,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            max_position_step_m=max_position_step_m,
            max_rotation_step_rad=max_rotation_step_rad,
            bias_weight=bias_weight,
            has_previous_velocity=has_previous_velocity,
            max_joint_step_rad=max_joint_step_rad,
            max_joint_velocity_rad_s=max_joint_velocity_rad_s,
            max_joint_acceleration_rad_s2=max_joint_acceleration_rad_s2,
            dt_s=dt_s,
            write_joint_targets=1,
        )
        self._solve_count += 1
        should_sync_result = self._solve_count == 1 or self._solve_count % self._result_readback_stride == 0
        if not should_sync_result:
            self.last_wrote_joint_targets = True
            self.last_result_was_synced = False
            self._has_previous_velocity = True
            dq_step = solve_full_pose_damped_least_squares_step(
                spatial_jacobian,
                position_error_xyz=position_error_xyz,
                orientation_error_rotvec=orientation_error_rotvec,
                damping_lambda=damping_lambda,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
                max_position_step_m=max_position_step_m,
                max_rotation_step_rad=max_rotation_step_rad,
                bias_step=bias_step,
                bias_weight=bias_weight,
            )
            q_cmd, dq_cmd, unclipped_q_cmd, acceleration_limited, clipped = _postprocess_joint_step_host(
                dq_step,
                current_q=current_q,
                previous_velocity_rad_s=previous_velocity_rad_s,
                lower_limits_rad=lower_limits_rad,
                upper_limits_rad=upper_limits_rad,
                max_joint_step_rad=max_joint_step_rad,
                max_joint_velocity_rad_s=max_joint_velocity_rad_s,
                max_joint_acceleration_rad_s2=max_joint_acceleration_rad_s2,
                dt_s=dt_s,
            )
            self._last_q_cmd_host[:] = np.asarray(q_cmd, dtype=np.float32)
            self._last_dq_cmd_host[:] = np.asarray(dq_cmd, dtype=np.float32)
            self._last_unclipped_q_cmd_host[:] = np.asarray(unclipped_q_cmd, dtype=np.float32)
            self._last_acceleration_limited = bool(acceleration_limited)
            self._last_clipped = bool(clipped)
            return (
                tuple(float(v) for v in self._last_q_cmd_host),
                tuple(float(v) for v in self._last_dq_cmd_host),
                tuple(float(v) for v in self._last_unclipped_q_cmd_host),
                self._last_acceleration_limited,
                self._last_clipped,
            )

        self.last_result_was_synced = True
        if int(self._status_wp.numpy()[0]) != 1:
            self._has_previous_velocity = False
            dq_step = solve_full_pose_damped_least_squares_step(
                spatial_jacobian,
                position_error_xyz=position_error_xyz,
                orientation_error_rotvec=orientation_error_rotvec,
                damping_lambda=damping_lambda,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
                max_position_step_m=max_position_step_m,
                max_rotation_step_rad=max_rotation_step_rad,
                bias_step=bias_step,
                bias_weight=bias_weight,
            )
            return _postprocess_joint_step_host(
                dq_step,
                current_q=current_q,
                previous_velocity_rad_s=previous_velocity_rad_s,
                lower_limits_rad=lower_limits_rad,
                upper_limits_rad=upper_limits_rad,
                max_joint_step_rad=max_joint_step_rad,
                max_joint_velocity_rad_s=max_joint_velocity_rad_s,
                max_joint_acceleration_rad_s2=max_joint_acceleration_rad_s2,
                dt_s=dt_s,
            )

        self.last_wrote_joint_targets = True
        self._has_previous_velocity = True
        self._last_q_cmd_host[:] = self._q_cmd_out_wp.numpy()
        self._last_dq_cmd_host[:] = self._dq_cmd_out_wp.numpy()
        self._last_unclipped_q_cmd_host[:] = self._unclipped_q_cmd_out_wp.numpy()
        flags = self._flags_wp.numpy()
        self._last_acceleration_limited = bool(int(flags[0]))
        self._last_clipped = bool(int(flags[1]))
        return (
            tuple(float(v) for v in self._last_q_cmd_host),
            tuple(float(v) for v in self._last_dq_cmd_host),
            tuple(float(v) for v in self._last_unclipped_q_cmd_host),
            self._last_acceleration_limited,
            self._last_clipped,
        )


def _postprocess_joint_step_host(
    joint_step_rad: tuple[float, ...],
    *,
    current_q: tuple[float, ...],
    previous_velocity_rad_s: tuple[float, ...] | None,
    lower_limits_rad: tuple[float, ...],
    upper_limits_rad: tuple[float, ...],
    max_joint_step_rad: float,
    max_joint_velocity_rad_s: float,
    max_joint_acceleration_rad_s2: float,
    dt_s: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], bool, bool]:
    safe_dt = max(1e-6, float(dt_s))
    allowed_step = max(0.0, float(max_joint_step_rad))
    safe_velocity = max(0.0, float(max_joint_velocity_rad_s))
    if safe_velocity > 0.0 and safe_dt > 0.0:
        velocity_step = safe_velocity * safe_dt
        allowed_step = min(allowed_step, velocity_step) if allowed_step > 0.0 else velocity_step

    step_values = [float(value) for value in joint_step_rad]
    if allowed_step <= 0.0:
        step_values = [0.0 for _ in step_values]
    else:
        step_values = [max(-allowed_step, min(allowed_step, value)) for value in step_values]

    acceleration_limited = False
    if (
        previous_velocity_rad_s is not None
        and len(previous_velocity_rad_s) == len(step_values)
        and float(max_joint_acceleration_rad_s2) > 0.0
    ):
        max_delta_velocity = max(0.0, float(max_joint_acceleration_rad_s2)) * safe_dt
        limited_steps: list[float] = []
        for step, previous_velocity in zip(step_values, previous_velocity_rad_s, strict=True):
            desired_velocity = float(step) / safe_dt
            lower_velocity = float(previous_velocity) - max_delta_velocity
            upper_velocity = float(previous_velocity) + max_delta_velocity
            bounded_velocity = max(lower_velocity, min(upper_velocity, desired_velocity))
            if abs(bounded_velocity - desired_velocity) > 1e-12:
                acceleration_limited = True
            limited_steps.append(bounded_velocity * safe_dt)
        step_values = limited_steps

    unclipped_q_cmd = tuple(float(current_q[index]) + step_values[index] for index in range(len(step_values)))
    clipped = False
    q_cmd_values: list[float] = []
    for index, value in enumerate(unclipped_q_cmd):
        bounded = max(float(lower_limits_rad[index]), min(float(upper_limits_rad[index]), float(value)))
        if abs(bounded - float(value)) > 1e-12:
            clipped = True
        q_cmd_values.append(bounded)

    q_cmd = tuple(q_cmd_values)
    dq_cmd = tuple((q_cmd[index] - float(current_q[index])) / safe_dt for index in range(len(q_cmd)))
    return q_cmd, dq_cmd, unclipped_q_cmd, acceleration_limited, clipped


class NewtonRuntimeRobotInterface(RobotInterface):
    def __init__(self, example: Example, config: NewtonRuntimeRobotConfig | None = None, *, print_every_n: int = 30):
        self.example = example
        self.config = config or NewtonRuntimeRobotConfig()
        self.print_every_n = max(1, int(print_every_n))
        self.command_count = 0
        self._connected = False
        self._human_anchor_xyz: tuple[float, float, float] | None = None
        self._target_anchor_xyz: tuple[float, float, float] | None = None
        self._joint_q_host: np.ndarray | None = None
        self._joint_qd_host: np.ndarray | None = None
        self._target_joint_q_host: np.ndarray | None = None
        self._target_joint_qd_host: np.ndarray | None = None
        self._target_joint_q_wp: wp.array | None = None
        self._target_joint_qd_wp: wp.array | None = None
        self._target_joint_host_dirty = False
        self._arm_joint_q_indices: tuple[int, ...] = ()
        self._arm_joint_qd_indices: tuple[int, ...] = ()
        self._arm_joint_q_indices_wp: wp.array | None = None
        self._arm_joint_qd_indices_wp: wp.array | None = None
        self._arm_joint_q_live_wp: wp.array | None = None
        self._arm_joint_qd_live_wp: wp.array | None = None
        self._arm_joint_q_live_host = np.zeros(7, dtype=np.float32)
        self._arm_joint_qd_live_host = np.zeros(7, dtype=np.float32)
        self._joint_q_index_by_label: dict[str, int] = {}
        self._joint_qd_index_by_label: dict[str, int] = {}
        self._hand_closing_direction_by_joint: dict[str, float] = {}
        self._hand_joint_limits_by_joint: dict[str, tuple[float, float]] = {}
        self._hand_target_capacity = 0
        self._hand_target_q_indices_host: np.ndarray | None = None
        self._hand_target_qd_indices_host: np.ndarray | None = None
        self._hand_target_values_host: np.ndarray | None = None
        self._hand_target_q_indices_wp: wp.array | None = None
        self._hand_target_qd_indices_wp: wp.array | None = None
        self._hand_target_values_wp: wp.array | None = None
        self._kinematics: NewtonLinkKinematicsModel | None = None
        self._ik: FullPoseDifferentialIkController | None = None
        self._last_hand_debug: dict[str, object] | None = None
        self._last_hand_debug_timestamp_s = 0.0
        self._target_anchor_quaternion_wxyz: QuaternionWXYZ | None = None
        self._orientation_tracker: OrientationTargetTracker | None = None
        self._orientation_debug: OrientationTrackerResult | None = None
        self._last_orientation_timestamp_s: float | None = None
        self._last_orientation_source_quaternion_xyzw: QuaternionXYZW | None = None
        self._last_orientation_source_debug: dict[str, object] | None = None
        self._orientation_anchor_source_actual: str | None = None
        self._beavr_hand_frame_smoother = BeavrHandFrameSmoother(moving_average_limit=5)
        self._openxr_yaw_correction_rad: float | None = None
        self._openxr_yaw_recenter_debug: dict[str, object] | None = None
        self._command_gate_enabled = True
        self._gate_mode = "engaged"
        self._gate_last_event = "direct_control"
        self._last_ik_result = None
        self._eef_body_suffix = ""
        self._gpu_dls_solver: NewtonGpuFullPoseDlsStepSolver | None = None
        self._contact_stop_shape_finger_id_wp: wp.array | None = None
        self._contact_stop_joint_finger_id_wp: wp.array | None = None
        self._contact_stop_joint_closing_direction_wp: wp.array | None = None
        self._contact_stop_joint_limit_lower_wp: wp.array | None = None
        self._contact_stop_joint_limit_upper_wp: wp.array | None = None
        self._contact_stop_joint_coord_to_dof_wp: wp.array | None = None
        self._contact_stop_stopped_wp: wp.array | None = None
        self._contact_stop_max_metric_wp: wp.array | None = None

    def connect(self) -> None:
        model = self.example.model
        self._joint_q_host = self.example.state_0.joint_q.numpy().copy()
        self._joint_qd_host = self.example.state_0.joint_qd.numpy().copy()
        self._target_joint_q_host = self._joint_q_host.copy()
        self._target_joint_qd_host = self._joint_qd_host.copy()
        self._target_joint_q_wp = wp.array(self._target_joint_q_host, dtype=wp.float32, device=model.device)
        self._target_joint_qd_wp = wp.array(self._target_joint_qd_host, dtype=wp.float32, device=model.device)
        self._target_joint_host_dirty = False
        self._joint_q_index_by_label, self._joint_qd_index_by_label = _joint_scalar_index_maps(model)
        self._hand_closing_direction_by_joint = _l10_closing_direction_by_joint()
        self._hand_joint_limits_by_joint = _l10_joint_limits_by_joint()
        self._arm_joint_q_indices = _arm_joint_indices(self._joint_q_index_by_label, self.config.arm_side)
        self._arm_joint_qd_indices = _arm_joint_indices(self._joint_qd_index_by_label, self.config.arm_side)
        self._arm_joint_q_indices_wp = wp.array(
            np.asarray(self._arm_joint_q_indices, dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        )
        self._arm_joint_qd_indices_wp = wp.array(
            np.asarray(self._arm_joint_qd_indices, dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        )
        self._arm_joint_q_live_wp = wp.zeros(7, dtype=wp.float32, device=model.device)
        self._arm_joint_qd_live_wp = wp.zeros(7, dtype=wp.float32, device=model.device)
        self._ensure_hand_target_buffers(int(model.joint_coord_count))
        self._kinematics = NewtonLinkKinematicsModel(
            model=model,
            side=self.config.arm_side,
            arm_joint_q_indices=self._arm_joint_q_indices,
            eef_body_suffix=self.config.eef_body_suffix_by_side[self.config.arm_side],
            finite_difference_rad=self.config.finite_difference_rad,
        )
        self._eef_body_suffix = self.config.eef_body_suffix_by_side[self.config.arm_side]
        self._init_contact_stop_gpu_state()
        current_q = self._current_arm_q()
        self._gpu_dls_solver = NewtonGpuFullPoseDlsStepSolver(
            device=model.device,
            result_readback_stride=self.print_every_n,
        )
        self._gpu_dls_solver.configure_target_writer(
            arm_joint_q_indices=self._arm_joint_q_indices,
            arm_joint_qd_indices=self._arm_joint_qd_indices,
            target_joint_q=self._target_joint_q_wp,
            target_joint_qd=self._target_joint_qd_wp,
        )
        ik_config_overrides = dict(self.config.ik_config_overrides)
        ik_config_overrides.setdefault("dls_step_solver", self._gpu_dls_solver)
        ik_config = FullPoseDifferentialIkControllerConfig(
            seed_joint_positions_rad=current_q,
            neutral_joint_positions_rad=current_q,
            kinematics_model=self._kinematics,
            **ik_config_overrides,
        )
        self._ik = FullPoseDifferentialIkController(ik_config)
        self._ik.reset(self._robot_state(timestamp_s=0.0))
        self._orientation_tracker = (
            OrientationTargetTracker(self.config.mapping.orientation_tracker_config())
            if bool(self.config.mapping.use_teleop_orientation)
            else None
        )
        self._connected = True
        print(
            "[newton-quest-teleop] connected"
            f" side={self.config.arm_side}"
            f" publish_mode={self.config.publish_mode}"
            f" drive_ik={'on' if self.config.drive_ik else 'hand-only'}"
            f" relative={'on' if self.config.relative_control else 'off'}"
            f" openxr_adapter={self.config.mapping.openxr_coordinate_adapter}"
            f" eef={self._eef_body_suffix}"
        )

    def _ensure_hand_target_buffers(self, capacity: int) -> None:
        capacity = max(1, int(capacity))
        if self._hand_target_capacity >= capacity:
            return
        device = self.example.model.device
        self._hand_target_capacity = capacity
        self._hand_target_q_indices_host = np.full(capacity, -1, dtype=np.int32)
        self._hand_target_qd_indices_host = np.full(capacity, -1, dtype=np.int32)
        self._hand_target_values_host = np.zeros(capacity, dtype=np.float32)
        self._hand_target_q_indices_wp = wp.zeros(capacity, dtype=wp.int32, device=device)
        self._hand_target_qd_indices_wp = wp.zeros(capacity, dtype=wp.int32, device=device)
        self._hand_target_values_wp = wp.zeros(capacity, dtype=wp.float32, device=device)

    def _init_contact_stop_gpu_state(self) -> None:
        model = self.example.model
        device = model.device

        shape_body_host = model.shape_body.numpy().copy()
        shape_finger_id = np.full(int(model.shape_count), -1, dtype=np.int32)
        for shape_index, body_index in enumerate(shape_body_host):
            body_id = int(body_index)
            if 0 <= body_id < len(model.body_label):
                finger_id = _l10_finger_id_from_body_label(model.body_label[body_id])
                if finger_id is not None:
                    shape_finger_id[int(shape_index)] = int(finger_id)

        joint_finger_id = np.full(int(model.joint_coord_count), -1, dtype=np.int32)
        joint_closing_direction = np.zeros(int(model.joint_coord_count), dtype=np.float32)
        joint_limit_lower = np.full(int(model.joint_coord_count), -np.inf, dtype=np.float32)
        joint_limit_upper = np.full(int(model.joint_coord_count), np.inf, dtype=np.float32)
        joint_coord_to_dof = np.full(int(model.joint_coord_count), -1, dtype=np.int32)

        for label_suffix, q_index in self._joint_q_index_by_label.items():
            base_name = _l10_base_joint_name(label_suffix)
            finger_id = _l10_finger_id_from_joint_name(base_name)
            closing_direction = float(self._hand_closing_direction_by_joint.get(base_name, 0.0))
            if finger_id is None or abs(closing_direction) <= 0.0:
                continue

            coord_index = int(q_index)
            joint_finger_id[coord_index] = int(finger_id)
            joint_closing_direction[coord_index] = closing_direction
            limits = self._hand_joint_limits_by_joint.get(base_name)
            if limits is not None:
                joint_limit_lower[coord_index] = float(limits[0])
                joint_limit_upper[coord_index] = float(limits[1])
            qd_index = self._joint_qd_index_by_label.get(label_suffix)
            if qd_index is not None:
                joint_coord_to_dof[coord_index] = int(qd_index)

        self._contact_stop_shape_finger_id_wp = wp.array(shape_finger_id, dtype=wp.int32, device=device)
        self._contact_stop_joint_finger_id_wp = wp.array(joint_finger_id, dtype=wp.int32, device=device)
        self._contact_stop_joint_closing_direction_wp = wp.array(
            joint_closing_direction,
            dtype=wp.float32,
            device=device,
        )
        self._contact_stop_joint_limit_lower_wp = wp.array(joint_limit_lower, dtype=wp.float32, device=device)
        self._contact_stop_joint_limit_upper_wp = wp.array(joint_limit_upper, dtype=wp.float32, device=device)
        self._contact_stop_joint_coord_to_dof_wp = wp.array(joint_coord_to_dof, dtype=wp.int32, device=device)
        self._contact_stop_stopped_wp = wp.zeros(_FINGER_COUNT, dtype=wp.int32, device=device)
        self._contact_stop_max_metric_wp = wp.zeros(_FINGER_COUNT, dtype=wp.float32, device=device)

    def send_command(self, command: SingleArmTeleopCommand) -> None:
        if not self._connected:
            raise RuntimeError("NewtonRuntimeRobotInterface is not connected")
        if command.arm_side != self.config.arm_side:
            return

        if not self._command_gate_enabled:
            self.command_count += 1
            if self.command_count == 1 or self.command_count % self.print_every_n == 0:
                print(
                    f"[newton-quest-teleop] frame={command.frame_id} mode={self._gate_mode} "
                    f"event={self._gate_last_event} holding_until_voice_start"
                )
            return

        if self.config.publish_mode == "drive_target":
            self._sync_live_joint_state()
        self._ensure_anchor(command)
        if self.config.drive_ik:
            self._apply_arm_command(command)
        if command.hand_target is not None:
            self._apply_hand_target(command.hand_target)
        self._publish_joint_state()

        self.command_count += 1
        if self.command_count == 1 or self.command_count % self.print_every_n == 0:
            target = self._target_position(command.ee_target)
            eef = self._current_eef_pose().position_xyz
            ik_status = getattr(self._last_ik_result, "status", "none")
            ik_pos_error = getattr(self._last_ik_result, "target_position_error_m", None)
            ik_ori_error = getattr(self._last_ik_result, "target_orientation_error_rad", None)
            ik_error_text = ""
            if ik_pos_error is not None:
                ik_error_text += f" ik_pos_err={float(ik_pos_error):.4f}"
            if ik_ori_error is not None:
                ik_error_text += f" ik_ori_err={float(ik_ori_error):.4f}"
            print(
                f"[newton-quest-teleop] frame={command.frame_id} side={self.config.arm_side} "
                f"target=({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f}) "
                f"eef=({eef[0]:+.3f},{eef[1]:+.3f},{eef[2]:+.3f}) "
                f"ik_status={ik_status}{ik_error_text} "
                f"gripper={command.gripper.normalized_position:.3f}"
            )

    def update_hand_debug(self, hand_debug: dict[str, object] | None, *, timestamp_s: float) -> None:
        self._last_hand_debug = hand_debug
        self._last_hand_debug_timestamp_s = float(timestamp_s)

    def stop(self) -> None:
        if self._joint_q_host is not None:
            self._publish_joint_state()

    def disconnect(self) -> None:
        self._connected = False

    def set_command_gate(self, enabled: bool, *, mode: str, last_event: str) -> None:
        self._command_gate_enabled = bool(enabled)
        self._gate_mode = str(mode)
        self._gate_last_event = str(last_event)

    def reset_relative_anchor(self) -> None:
        self._human_anchor_xyz = None
        self._target_anchor_xyz = None
        self._target_anchor_quaternion_wxyz = None
        self._last_orientation_timestamp_s = None
        self._last_orientation_source_quaternion_xyzw = None
        self._last_orientation_source_debug = None
        self._orientation_anchor_source_actual = None
        self._orientation_debug = None
        self._openxr_yaw_correction_rad = None
        self._openxr_yaw_recenter_debug = None
        self._beavr_hand_frame_smoother.reset()
        if self._ik is not None:
            self._ik.reset(self._robot_state(timestamp_s=0.0))

    def recenter_teleop(self) -> None:
        self.reset_relative_anchor()
        if self._joint_q_host is not None and self._kinematics is not None:
            self._kinematics.sync_joint_q(self._joint_q_host)
        print(f"[newton-quest-teleop] recentered side={self.config.arm_side}")

    def reset_to_scene_state(self) -> None:
        if not self._connected:
            return
        self._joint_q_host = self.example.state_0.joint_q.numpy().copy()
        self._joint_qd_host = self.example.state_0.joint_qd.numpy().copy()
        self._target_joint_q_host = self._joint_q_host.copy()
        self._target_joint_qd_host = self._joint_qd_host.copy()
        if self._target_joint_q_wp is not None:
            self._target_joint_q_wp.assign(self._target_joint_q_host)
        if self._target_joint_qd_wp is not None:
            self._target_joint_qd_wp.assign(self._target_joint_qd_host)
        self._target_joint_host_dirty = False
        self._last_ik_result = None
        self.command_count = 0
        if self._contact_stop_stopped_wp is not None:
            self._contact_stop_stopped_wp.zero_()
        if self._contact_stop_max_metric_wp is not None:
            self._contact_stop_max_metric_wp.zero_()
        if self._kinematics is not None:
            self._kinematics.sync_joint_q(self._joint_q_host)
        self.reset_relative_anchor()
        print(f"[newton-quest-teleop] reset_to_scene_state side={self.config.arm_side}")

    def xr_status_snapshot(self, *, mode: str, last_event: str) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "mode": mode,
            "last_event": last_event,
            "mapper_control_profile": "voice",
            "arm_side": self.config.arm_side,
            "command_count": self.command_count,
            "controller_available": True,
            "command_gate_enabled": self._command_gate_enabled,
        }
        if self._last_hand_debug is None:
            snapshot["input_tracking_state"] = "missing"
        else:
            snapshot["input_tracking_state"] = "tracking"
            snapshot["hand_debug_timestamp_s"] = self._last_hand_debug_timestamp_s
            snapshot.update(self._last_hand_debug)
        if self._last_orientation_source_debug is not None:
            snapshot["orientation_source_debug"] = self._last_orientation_source_debug
        if self._orientation_debug is not None:
            snapshot["orientation_debug"] = self._orientation_debug.as_dict()
        if self._last_ik_result is not None:
            snapshot["ik_status"] = getattr(self._last_ik_result, "status", None)
            snapshot["ik_target_position_error_m"] = getattr(self._last_ik_result, "target_position_error_m", None)
            snapshot["ik_target_orientation_error_rad"] = getattr(
                self._last_ik_result,
                "target_orientation_error_rad",
                None,
            )
            snapshot["ik_events"] = getattr(self._last_ik_result, "events", ())
        return snapshot

    def _ensure_anchor(self, command: SingleArmTeleopCommand) -> None:
        if self._human_anchor_xyz is not None and self._target_anchor_xyz is not None:
            return
        self._human_anchor_xyz = tuple(float(v) for v in command.ee_target.position_xyz)
        self._target_anchor_xyz = tuple(float(v) for v in self._current_eef_pose().position_xyz)
        self._target_anchor_quaternion_wxyz = xyzw_to_wxyz(self._current_eef_pose().quaternion_xyzw)
        if self.config.openxr_yaw_recenter and self._openxr_yaw_correction_rad is None:
            self._recenter_openxr_yaw_from_hand(command.ee_target, source="anchor")
        if self._orientation_tracker is not None:
            source_quat_xyzw, source_debug = self._orientation_source_quaternion_xyzw(command.ee_target)
            self._orientation_tracker.reset_anchor(source_quat_xyzw, self._target_anchor_quaternion_wxyz)
            self._orientation_debug = None
            self._last_orientation_timestamp_s = None
            self._orientation_anchor_source_actual = str(source_debug.get("actual", "unknown"))
        print(
            f"[newton-quest-teleop] anchor side={self.config.arm_side} "
            f"human=({self._human_anchor_xyz[0]:+.3f},{self._human_anchor_xyz[1]:+.3f},{self._human_anchor_xyz[2]:+.3f}) "
            f"target=({self._target_anchor_xyz[0]:+.3f},{self._target_anchor_xyz[1]:+.3f},{self._target_anchor_xyz[2]:+.3f}) "
            f"orientation={'on' if self._orientation_tracker is not None else 'off'} "
            f"orientation_source={self.config.mapping.orientation_source} "
            f"orientation_reference_mode={self.config.mapping.orientation_reference_mode}"
        )

    def _target_position(self, pose: Pose7) -> tuple[float, float, float]:
        if not self.config.relative_control:
            mapped = self.config.mapping.map_vector(pose.position_xyz)
            mapped = self._apply_openxr_yaw_correction_to_vector(mapped)
            return tuple(
                float(self.config.mapping.workspace_origin_xyz[i])
                + float(self.config.mapping.translation_scale_xyz[i]) * float(mapped[i])
                for i in range(3)
            )  # type: ignore[return-value]
        if self._human_anchor_xyz is None or self._target_anchor_xyz is None:
            raise RuntimeError("Relative teleop anchor is not initialized")
        delta = tuple(float(pose.position_xyz[i]) - float(self._human_anchor_xyz[i]) for i in range(3))
        mapped_delta = self.config.mapping.map_vector(delta)  # type: ignore[arg-type]
        mapped_delta = self._apply_openxr_yaw_correction_to_vector(mapped_delta)
        return tuple(
            float(self._target_anchor_xyz[i])
            + float(self.config.mapping.translation_scale_xyz[i]) * float(mapped_delta[i])
            for i in range(3)
        )  # type: ignore[return-value]

    def _target_quaternion_xyzw(self, pose: Pose7, *, timestamp_s: float) -> QuaternionXYZW:
        if self._orientation_tracker is not None:
            if self._target_anchor_quaternion_wxyz is None:
                return self._current_eef_pose().quaternion_xyzw
            source_quat, source_debug = self._orientation_source_quaternion_xyzw(pose)
            source_actual = str(source_debug.get("actual", "unknown"))
            if (
                self._orientation_anchor_source_actual is not None
                and source_actual
                in {"hand_anatomical_frame", "hand_beavr_anatomical_frame", "hand_genesis_wrist_frame"}
                and self._orientation_anchor_source_actual != source_actual
            ):
                self.reset_relative_anchor()
                return self._current_eef_pose().quaternion_xyzw
            if self._last_orientation_timestamp_s is None:
                dt_s = float(self.example.frame_dt)
            else:
                dt_s = max(0.0, float(timestamp_s) - float(self._last_orientation_timestamp_s))
                if dt_s <= 0.0:
                    dt_s = float(self.example.frame_dt)
            self._orientation_debug = self._orientation_tracker.update(source_quat, dt_s=dt_s)
            self._last_orientation_timestamp_s = float(timestamp_s)
            self._orientation_anchor_source_actual = source_actual
            return wxyz_to_xyzw(self._orientation_debug.cmd_target_quat_wxyz)
        if self.config.mapping.fixed_quaternion_wxyz is not None:
            return quat_normalize_xyzw(wxyz_to_xyzw(self.config.mapping.fixed_quaternion_wxyz))
        return self._current_eef_pose().quaternion_xyzw

    @staticmethod
    def _yaw_rotation_matrix(rad: float) -> np.ndarray:
        cos_v = math.cos(float(rad))
        sin_v = math.sin(float(rad))
        return np.asarray(
            (
                (cos_v, -sin_v, 0.0),
                (sin_v, cos_v, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], ...]:
        return tuple(tuple(float(value) for value in row) for row in np.asarray(matrix, dtype=np.float64))

    def _set_openxr_yaw_correction_from_genesis_forward(
        self,
        forward_xyz: tuple[float, float, float],
        *,
        source: str,
    ) -> bool:
        if self.config.mapping.openxr_coordinate_adapter != "openxr_genesis":
            self._openxr_yaw_correction_rad = None
            self._openxr_yaw_recenter_debug = {
                "enabled": False,
                "source": source,
                "reason": "openxr_coordinate_adapter_not_openxr_genesis",
                "openxr_coordinate_adapter": self.config.mapping.openxr_coordinate_adapter,
            }
            return False

        measured = np.asarray(forward_xyz, dtype=np.float64).reshape(3)
        measured_xy = np.asarray((measured[0], measured[1]), dtype=np.float64)
        norm_xy = float(np.linalg.norm(measured_xy))
        if norm_xy <= 1.0e-9:
            self._openxr_yaw_correction_rad = None
            self._openxr_yaw_recenter_debug = {
                "enabled": False,
                "source": source,
                "reason": "forward_axis_horizontal_norm_too_small",
                "measured_forward_xyz": [float(v) for v in measured],
            }
            return False

        measured_xy /= norm_xy
        target_xy = np.asarray((-1.0, 0.0), dtype=np.float64)
        cross_z = float(measured_xy[0] * target_xy[1] - measured_xy[1] * target_xy[0])
        dot = float(np.dot(measured_xy, target_xy))
        yaw_rad = math.atan2(cross_z, dot)
        self._openxr_yaw_correction_rad = float(yaw_rad)
        corrected = self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in measured))
        self._openxr_yaw_recenter_debug = {
            "enabled": True,
            "source": source,
            "yaw_correction_rad": float(yaw_rad),
            "yaw_correction_deg": float(math.degrees(yaw_rad)),
            "measured_forward_xyz": [float(v) for v in measured],
            "measured_forward_xy_normalized": [float(v) for v in measured_xy],
            "target_forward_xyz": [-1.0, 0.0, 0.0],
            "corrected_forward_xyz": [float(v) for v in corrected],
        }
        print(
            "[newton-quest-teleop] openxr_yaw_recenter "
            f"source={source} yaw_deg={math.degrees(yaw_rad):+.2f} "
            f"measured_forward=({measured[0]:+.3f},{measured[1]:+.3f},{measured[2]:+.3f})"
        )
        return True

    def _apply_openxr_yaw_correction_to_vector(
        self,
        vector_xyz: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if self._openxr_yaw_correction_rad is None:
            return tuple(float(v) for v in vector_xyz)  # type: ignore[return-value]
        corrected = self._yaw_rotation_matrix(self._openxr_yaw_correction_rad) @ np.asarray(
            vector_xyz,
            dtype=np.float64,
        )
        return (float(corrected[0]), float(corrected[1]), float(corrected[2]))

    def _apply_openxr_yaw_correction_to_quaternion(
        self,
        quaternion_xyzw: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if self._openxr_yaw_correction_rad is None:
            return tuple(float(v) for v in quaternion_xyzw)  # type: ignore[return-value]
        yaw_matrix = self._yaw_rotation_matrix(self._openxr_yaw_correction_rad)
        rotation = np.asarray(quat_xyzw_to_matrix(quaternion_xyzw), dtype=np.float64)
        return matrix_to_quat_xyzw(self._matrix_tuple(yaw_matrix @ rotation))  # type: ignore[return-value]

    def _apply_openxr_yaw_correction_to_frame(self, frame: HandAnatomicalFrame) -> HandAnatomicalFrame:
        if self._openxr_yaw_correction_rad is None:
            return frame
        axes = FrameAxes(
            x=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.axes.x)),
            y=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.axes.y)),
            z=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.axes.z)),
        )
        return HandAnatomicalFrame(
            origin_xyz=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.origin_xyz)),
            axes=axes,
            quaternion_xyzw=matrix_to_quat_xyzw(matrix_from_axes(axes)),
            handedness_det=float(frame.handedness_det),
            thumb_alignment=float(frame.thumb_alignment),
            legacy_palm_normal_alignment=frame.legacy_palm_normal_alignment,
            construction=f"{frame.construction}_openxr_yaw_recentered",
            raw_axes=frame.raw_axes,
            axis_adapter={
                **(frame.axis_adapter or {}),
                "session_yaw_recenter": "yaw-only Genesis +Z correction; operator front -> robot front",
            },
        )

    def _hand_orientation_frame(
        self,
        requested: str,
        *,
        apply_openxr_yaw_correction: bool = True,
    ) -> HandAnatomicalFrame | None:
        if not isinstance(self._last_hand_debug, dict):
            return None
        if requested == "hand_anatomical_frame":
            frame = hand_anatomical_frame_from_debug(self._last_hand_debug)
        elif requested in {"hand_beavr_anatomical_frame", "hand_genesis_wrist_frame"}:
            frame = self._beavr_hand_frame_smoother.update(self._last_hand_debug)
            if frame is None:
                frame = hand_beavr_anatomical_frame_from_debug(self._last_hand_debug)
        else:
            return None
        if frame is None:
            return None
        if requested == "hand_genesis_wrist_frame":
            adapted = adapt_openxr_hand_frame_to_genesis_wrist_frame(frame)
            return self._apply_openxr_yaw_correction_to_frame(adapted) if apply_openxr_yaw_correction else adapted
        if self.config.mapping.openxr_coordinate_adapter == "openxr_genesis":
            adapted = adapt_openxr_hand_frame_to_genesis_parent(frame)
            return self._apply_openxr_yaw_correction_to_frame(adapted) if apply_openxr_yaw_correction else adapted
        return frame

    def _recenter_openxr_yaw_from_hand(self, pose: Pose7, *, source: str) -> bool:
        frame = self._hand_orientation_frame("hand_genesis_wrist_frame", apply_openxr_yaw_correction=False)
        if frame is None:
            source_quat = self.config.mapping.adapt_openxr_quaternion(tuple(float(v) for v in pose.quaternion_xyzw))
            forward = quat_xyzw_to_matrix(source_quat)
            return self._set_openxr_yaw_correction_from_genesis_forward(
                (float(forward[0][2]), float(forward[1][2]), float(forward[2][2])),
                source=f"{source}:wrist_quat_fallback",
            )
        return self._set_openxr_yaw_correction_from_genesis_forward(
            tuple(float(v) for v in frame.axes.z),
            source=f"{source}:hand_genesis_wrist_frame_z",
        )

    def _orientation_source_quaternion_xyzw(self, pose: Pose7) -> tuple[QuaternionXYZW, dict[str, object]]:
        requested = str(self.config.mapping.orientation_source)
        wrist_quat = tuple(float(v) for v in pose.quaternion_xyzw)
        if requested == "wrist_quat":
            adapted = self.config.mapping.adapt_openxr_quaternion(wrist_quat)
            adapted = self._apply_openxr_yaw_correction_to_quaternion(adapted)
            adapted = quat_normalize_xyzw(adapted)
            debug: dict[str, object] = {
                "requested": requested,
                "actual": "wrist_quat",
                "fallback": False,
                "reason": None,
                "openxr_coordinate_adapter": self.config.mapping.openxr_coordinate_adapter,
                "openxr_yaw_recenter": self._openxr_yaw_recenter_debug,
            }
            self._last_orientation_source_quaternion_xyzw = adapted
            self._last_orientation_source_debug = debug
            return adapted, debug

        if requested not in {"hand_anatomical_frame", "hand_beavr_anatomical_frame", "hand_genesis_wrist_frame"}:
            raise ValueError(f"unsupported teleop orientation source: {requested!r}")

        frame = self._hand_orientation_frame(requested)
        if frame is not None:
            quat = quat_normalize_xyzw(tuple(float(v) for v in frame.quaternion_xyzw))
            debug = {
                "requested": requested,
                "actual": requested,
                "fallback": False,
                "reason": None,
                "openxr_coordinate_adapter": self.config.mapping.openxr_coordinate_adapter,
                "openxr_yaw_recenter": self._openxr_yaw_recenter_debug,
                requested: frame.as_dict(),
            }
            self._last_orientation_source_quaternion_xyzw = quat
            self._last_orientation_source_debug = debug
            return quat, debug

        if self._last_orientation_source_quaternion_xyzw is not None:
            debug = {
                "requested": requested,
                "actual": f"last_{requested}",
                "fallback": True,
                "reason": f"{requested}_unavailable",
                "openxr_coordinate_adapter": self.config.mapping.openxr_coordinate_adapter,
                "openxr_yaw_recenter": self._openxr_yaw_recenter_debug,
            }
            self._last_orientation_source_debug = debug
            return self._last_orientation_source_quaternion_xyzw, debug

        adapted = self.config.mapping.adapt_openxr_quaternion(wrist_quat)
        adapted = self._apply_openxr_yaw_correction_to_quaternion(adapted)
        adapted = quat_normalize_xyzw(adapted)
        debug = {
            "requested": requested,
            "actual": "wrist_quat",
            "fallback": True,
            "reason": f"{requested}_unavailable",
            "openxr_coordinate_adapter": self.config.mapping.openxr_coordinate_adapter,
            "openxr_yaw_recenter": self._openxr_yaw_recenter_debug,
        }
        self._last_orientation_source_quaternion_xyzw = adapted
        self._last_orientation_source_debug = debug
        return adapted, debug

    def _apply_arm_command(self, command: SingleArmTeleopCommand) -> None:
        assert self._ik is not None
        target_pose = Pose7(
            position_xyz=self._target_position(command.ee_target),
            quaternion_xyzw=self._target_quaternion_xyzw(command.ee_target, timestamp_s=command.timestamp_s),
        )
        target = TaskSpaceTarget(
            arm_side=self.config.arm_side,
            source_name=command.source_name,
            timestamp_s=command.timestamp_s,
            frame_id=command.frame_id,
            ee_target=target_pose,
            orientation_mode="teleop_full_pose",
            target_frame="world",
        )
        self._ik.set_target(target)
        result = self._ik.step(self._robot_state(timestamp_s=command.timestamp_s), self.example.frame_dt)
        self._last_ik_result = result
        arm_targets_written_on_gpu = bool(
            self._gpu_dls_solver is not None and self._gpu_dls_solver.last_wrote_joint_targets
        )
        arm_targets_synced_to_host = not arm_targets_written_on_gpu or bool(
            self._gpu_dls_solver is not None and self._gpu_dls_solver.last_result_was_synced
        )
        assert self._joint_q_host is not None and self._joint_qd_host is not None
        assert self._target_joint_q_host is not None and self._target_joint_qd_host is not None
        if arm_targets_synced_to_host:
            for target_index, value in zip(self._arm_joint_q_indices, result.q_cmd, strict=True):
                self._target_joint_q_host[int(target_index)] = float(value)
            if result.dq_cmd is not None:
                for target_index, value in zip(self._arm_joint_qd_indices, result.dq_cmd, strict=True):
                    self._target_joint_qd_host[int(target_index)] = float(value)
        if not arm_targets_written_on_gpu:
            self._target_joint_host_dirty = True
        if self.config.publish_mode == "state" and self._kinematics is not None:
            self._kinematics.sync_joint_q(self._target_joint_q_host)

    def _apply_hand_target(self, hand_target: NamedJointValues) -> None:
        joint_values = _expand_l10_mimic_joint_values(hand_target)
        assert self._joint_q_host is not None and self._joint_qd_host is not None
        assert self._target_joint_q_host is not None and self._target_joint_qd_host is not None
        assert self._target_joint_q_wp is not None and self._target_joint_qd_wp is not None
        self._ensure_hand_target_buffers(len(joint_values.joint_names))
        assert self._hand_target_q_indices_host is not None
        assert self._hand_target_qd_indices_host is not None
        assert self._hand_target_values_host is not None
        assert self._hand_target_q_indices_wp is not None
        assert self._hand_target_qd_indices_wp is not None
        assert self._hand_target_values_wp is not None
        max_step = max(0.0, float(self.config.hand_max_joint_step_rad))
        count = 0
        for joint_name, value in zip(joint_values.joint_names, joint_values.joint_positions, strict=True):
            label_suffix = _l10_joint_label_suffix(self.config.arm_side, str(joint_name))
            q_index = self._joint_q_index_by_label.get(label_suffix)
            if q_index is None:
                continue
            target_value = float(value)
            qd_index = self._joint_qd_index_by_label.get(label_suffix)
            self._hand_target_q_indices_host[count] = int(q_index)
            self._hand_target_qd_indices_host[count] = -1 if qd_index is None else int(qd_index)
            self._hand_target_values_host[count] = target_value
            self._target_joint_q_host[int(q_index)] = target_value
            if qd_index is not None:
                self._target_joint_qd_host[int(qd_index)] = 0.0
            count += 1
        if count <= 0:
            return
        self._hand_target_q_indices_wp.assign(self._hand_target_q_indices_host)
        self._hand_target_qd_indices_wp.assign(self._hand_target_qd_indices_host)
        self._hand_target_values_wp.assign(self._hand_target_values_host)
        wp.launch(
            kernel=_write_hand_target_slices_kernel,
            dim=count,
            inputs=[
                self.example.state_0.joint_q,
                self._hand_target_q_indices_wp,
                self._hand_target_qd_indices_wp,
                self._hand_target_values_wp,
                self._target_joint_q_wp,
                self._target_joint_qd_wp,
                int(count),
                max_step,
                max(1.0e-6, float(self.example.frame_dt)),
                1 if self.config.hand_publish_kinematic_velocity else 0,
            ],
            device=self.example.model.device,
        )

    def _publish_joint_state(self) -> None:
        assert self._target_joint_q_host is not None and self._target_joint_qd_host is not None
        model = self.example.model
        if self._target_joint_q_wp is None:
            self._target_joint_q_wp = wp.array(self._target_joint_q_host, dtype=wp.float32, device=model.device)
            self._target_joint_host_dirty = False
        if self._target_joint_qd_wp is None:
            self._target_joint_qd_wp = wp.array(self._target_joint_qd_host, dtype=wp.float32, device=model.device)
            self._target_joint_host_dirty = False
        if self._target_joint_host_dirty:
            self._target_joint_q_wp.assign(self._target_joint_q_host)
            self._target_joint_qd_wp.assign(self._target_joint_qd_host)
            self._target_joint_host_dirty = False
        joint_q = self._target_joint_q_wp
        joint_qd = self._target_joint_qd_wp
        self._apply_contact_stop_gpu(joint_q, joint_qd)

        if self.config.publish_mode == "drive_target":
            self.example.control.joint_target_q = joint_q
            self.example.control.joint_target_qd = joint_qd
            return
        if self.config.publish_mode != "state":
            raise ValueError(f"Unsupported Newton publish_mode: {self.config.publish_mode!r}")

        model.joint_q = joint_q
        model.joint_qd = joint_qd
        self.example.state_0.joint_q = joint_q
        self.example.state_0.joint_qd = joint_qd
        self.example.state_1.joint_q = wp.clone(joint_q)
        self.example.state_1.joint_qd = wp.clone(joint_qd)
        self.example.control.joint_target_q = wp.clone(joint_q)
        self.example.control.joint_target_qd = wp.clone(joint_qd)
        newton.eval_fk(
            model,
            joint_q,
            joint_qd,
            self.example.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )
        model.bvh_refit_shapes(self.example.state_0)
        self._joint_q_host = self._target_joint_q_host.copy()
        self._joint_qd_host = self._target_joint_qd_host.copy()

    def _apply_contact_stop_gpu(self, target_joint_q: wp.array, target_joint_qd: wp.array) -> None:
        if not self.config.hand_contact_stop_enabled:
            return
        if (
            self._contact_stop_shape_finger_id_wp is None
            or self._contact_stop_joint_finger_id_wp is None
            or self._contact_stop_joint_closing_direction_wp is None
            or self._contact_stop_joint_limit_lower_wp is None
            or self._contact_stop_joint_limit_upper_wp is None
            or self._contact_stop_joint_coord_to_dof_wp is None
            or self._contact_stop_stopped_wp is None
            or self._contact_stop_max_metric_wp is None
        ):
            return

        contacts = getattr(self.example, "contacts", None)
        state = getattr(self.example, "state_0", None)
        if contacts is None or state is None or getattr(state, "body_q", None) is None:
            return

        bottle_shape_index = int(getattr(self.example, "_dynamic_bottle_collision_shape", -1))
        if bottle_shape_index < 0:
            return

        release_m = min(
            float(getattr(self.example, "l10_bottle_contact_stop_release_m", 0.0)),
            float(getattr(self.example, "l10_bottle_contact_stop_threshold_m", 0.0)),
        )
        wp.launch(
            kernel=_l10_bottle_contact_stop_update_kernel,
            dim=_FINGER_COUNT,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                state.body_q,
                self.example.model.shape_body,
                self._contact_stop_shape_finger_id_wp,
                bottle_shape_index,
                int(contacts.rigid_contact_max),
                float(getattr(self.example, "l10_bottle_contact_stop_activation_m", 0.0)),
                float(getattr(self.example, "l10_bottle_contact_stop_threshold_m", 0.0)),
                release_m,
                self._contact_stop_stopped_wp,
                self._contact_stop_max_metric_wp,
            ],
            device=self.example.model.device,
        )
        wp.launch(
            kernel=_l10_bottle_contact_stop_clamp_targets_kernel,
            dim=int(self.example.model.joint_coord_count),
            inputs=[
                self.example.state_0.joint_q,
                target_joint_q,
                target_joint_qd,
                self._contact_stop_joint_finger_id_wp,
                self._contact_stop_joint_closing_direction_wp,
                self._contact_stop_joint_limit_lower_wp,
                self._contact_stop_joint_limit_upper_wp,
                self._contact_stop_joint_coord_to_dof_wp,
                self._contact_stop_stopped_wp,
                float(self.config.hand_contact_stop_retreat_rad),
                float(self.config.hand_contact_release_retreat_rad),
                float(self.example.frame_dt),
                1 if self.config.hand_publish_kinematic_velocity else 0,
            ],
            device=self.example.model.device,
        )

    def _sync_live_joint_state(self) -> None:
        assert self._joint_q_host is not None and self._joint_qd_host is not None
        if (
            self._arm_joint_q_indices_wp is None
            or self._arm_joint_qd_indices_wp is None
            or self._arm_joint_q_live_wp is None
            or self._arm_joint_qd_live_wp is None
        ):
            self._sync_full_joint_state_from_scene()
            return

        wp.launch(
            kernel=_gather_arm_joint_state_kernel,
            dim=7,
            inputs=[
                self.example.state_0.joint_q,
                self.example.state_0.joint_qd,
                self._arm_joint_q_indices_wp,
                self._arm_joint_qd_indices_wp,
                self._arm_joint_q_live_wp,
                self._arm_joint_qd_live_wp,
            ],
            device=self.example.model.device,
        )
        self._arm_joint_q_live_host[:] = self._arm_joint_q_live_wp.numpy()
        self._arm_joint_qd_live_host[:] = self._arm_joint_qd_live_wp.numpy()
        if not np.isfinite(self._arm_joint_q_live_host).all() or not np.isfinite(self._arm_joint_qd_live_host).all():
            print("[newton-quest-teleop] warning: non-finite Newton joint state; resetting scene", flush=True)
            reset_scene = getattr(self.example, "reset_scene_to_initial", None)
            if callable(reset_scene):
                reset_scene()
                self._sync_full_joint_state_from_scene()
                return

        if not np.isfinite(self._arm_joint_q_live_host).all() or not np.isfinite(self._arm_joint_qd_live_host).all():
            if self._target_joint_q_host is not None and np.isfinite(self._target_joint_q_host).all():
                self._joint_q_host = self._target_joint_q_host.copy()
                if self._target_joint_qd_host is not None:
                    self._joint_qd_host = np.zeros_like(self._target_joint_qd_host)
            else:
                raise ValueError("Newton live joint state is non-finite and no finite target fallback is available")
        else:
            for slot, q_index in enumerate(self._arm_joint_q_indices):
                self._joint_q_host[int(q_index)] = float(self._arm_joint_q_live_host[slot])
            for slot, qd_index in enumerate(self._arm_joint_qd_indices):
                self._joint_qd_host[int(qd_index)] = float(self._arm_joint_qd_live_host[slot])

        if self._kinematics is not None:
            self._kinematics.sync_joint_q(self._joint_q_host)

    def _sync_full_joint_state_from_scene(self) -> None:
        joint_q = self.example.state_0.joint_q.numpy().copy()
        joint_qd = self.example.state_0.joint_qd.numpy().copy()
        if not np.isfinite(joint_q).all() or not np.isfinite(joint_qd).all():
            if self._target_joint_q_host is not None and np.isfinite(self._target_joint_q_host).all():
                joint_q = self._target_joint_q_host.copy()
                joint_qd = (
                    np.zeros_like(self._target_joint_qd_host) if self._target_joint_qd_host is not None else joint_qd
                )
            else:
                raise ValueError("Newton live joint state is non-finite and no finite target fallback is available")
        self._joint_q_host = joint_q
        self._joint_qd_host = joint_qd
        if self._kinematics is not None:
            self._kinematics.sync_joint_q(self._joint_q_host)

    def _robot_state(self, *, timestamp_s: float) -> RobotStateSnapshot:
        return RobotStateSnapshot(
            timestamp_s=float(timestamp_s),
            joint_positions_rad=self._current_arm_q(),
            joint_velocities_rad_s=self._current_arm_qd(),
            ee_pose=self._current_eef_pose(),
        )

    def _current_arm_q(self) -> tuple[float, ...]:
        if self._joint_q_host is None:
            joint_q = self.example.state_0.joint_q.numpy()
        else:
            joint_q = self._joint_q_host
        return tuple(float(joint_q[index]) for index in self._arm_joint_q_indices)

    def _current_arm_qd(self) -> tuple[float, ...]:
        if self._joint_qd_host is None:
            joint_qd = self.example.state_0.joint_qd.numpy()
        else:
            joint_qd = self._joint_qd_host
        return tuple(float(joint_qd[index]) for index in self._arm_joint_qd_indices)

    def _current_eef_pose(self) -> Pose7:
        assert self._kinematics is not None
        return self._kinematics.forward_pose(self._current_arm_q())


def _find_body_index(model: newton.Model, label_suffix: str) -> int:
    body_index = next((i for i, label in enumerate(model.body_label) if label.endswith(label_suffix)), None)
    if body_index is None:
        raise ValueError(f"Body ending with {label_suffix!r} not found")
    return int(body_index)


def _joint_scalar_index_maps(model: newton.Model) -> tuple[dict[str, int], dict[str, int]]:
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    q_map: dict[str, int] = {}
    qd_map: dict[str, int] = {}
    for joint_index, label in enumerate(model.joint_label):
        q0 = int(q_start[joint_index])
        q1 = int(q_start[joint_index + 1]) if joint_index + 1 < len(q_start) else int(model.joint_coord_count)
        qd0 = int(qd_start[joint_index])
        qd1 = int(qd_start[joint_index + 1]) if joint_index + 1 < len(qd_start) else int(model.joint_dof_count)
        if q1 - q0 == 1:
            q_map[_short_joint_label(label)] = q0
        if qd1 - qd0 == 1:
            qd_map[_short_joint_label(label)] = qd0
    return q_map, qd_map


def _short_joint_label(label: str) -> str:
    return str(label).rsplit("/", maxsplit=1)[-1]


def _arm_joint_indices(index_by_label: dict[str, int], side: ArmSide) -> tuple[int, ...]:
    indices = []
    for joint_index in range(1, 8):
        label = f"{side}_joint{joint_index}"
        if label not in index_by_label:
            raise ValueError(f"Missing Newton arm joint: {label}")
        indices.append(int(index_by_label[label]))
    return tuple(indices)


def _l10_joint_label_suffix(side: ArmSide, joint_name: str) -> str:
    if joint_name.startswith(f"{side}_l10_"):
        return joint_name
    if joint_name.startswith("l10_"):
        return f"{side}_{joint_name}"
    return f"{side}_l10_{joint_name}"


def _l10_base_joint_name(joint_name: str) -> str:
    value = str(joint_name)
    for side in ("left", "right"):
        prefix = f"{side}_l10_"
        if value.startswith(prefix):
            return value[len(prefix) :]
    if value.startswith("l10_"):
        return value[len("l10_") :]
    return value


def _l10_finger_family_from_joint_name(joint_name: str) -> str | None:
    base_name = _l10_base_joint_name(joint_name)
    for family in ("thumb", "index", "middle", "ring", "pinky"):
        if base_name.startswith(f"{family}_"):
            return family
    return None


def _l10_finger_id_from_joint_name(joint_name: str) -> int | None:
    family = _l10_finger_family_from_joint_name(joint_name)
    if family is None:
        return None
    return _FINGER_NAMES.index(family)


def _l10_finger_id_from_body_label(body_label: str) -> int | None:
    link_name = str(body_label).rsplit("/", maxsplit=1)[-1].lower()
    for side in ("right", "left"):
        prefix = f"{side}_l10_"
        if link_name.startswith(prefix):
            link_name = link_name[len(prefix) :]
            break
    for index, family in enumerate(_FINGER_NAMES):
        if link_name.startswith(f"{family}_"):
            return index
    return None


def _l10_is_contact_stopped_curl_joint(joint_name: str) -> bool:
    base_name = _l10_base_joint_name(joint_name)
    if base_name in {"thumb_mcp", "thumb_ip", "thumb_cmc_pitch"}:
        return True
    return base_name.endswith(("_mcp_pitch", "_pip", "_dip"))


def _l10_closing_direction_by_joint() -> dict[str, float]:
    try:
        from teleop_stack.retargeting.hand_config import load_linker_l10_right_hand_spec

        spec = load_linker_l10_right_hand_spec()
    except Exception:
        return {}

    open_by_name = dict(zip(spec.default_open_pose.joint_names, spec.default_open_pose.joint_positions, strict=True))
    close_by_name = dict(zip(spec.default_close_pose.joint_names, spec.default_close_pose.joint_positions, strict=True))
    direction_by_name: dict[str, float] = {}

    for joint_name in spec.active_joint_names:
        if not _l10_is_contact_stopped_curl_joint(joint_name):
            continue
        delta = float(close_by_name[joint_name]) - float(open_by_name[joint_name])
        if abs(delta) > 1.0e-9:
            direction_by_name[joint_name] = 1.0 if delta > 0.0 else -1.0

    for mimic_joint in spec.mimic_joints:
        if not _l10_is_contact_stopped_curl_joint(mimic_joint.joint_name):
            continue
        source_delta = float(close_by_name[mimic_joint.source_joint_name]) - float(
            open_by_name[mimic_joint.source_joint_name]
        )
        delta = float(mimic_joint.multiplier) * source_delta
        if abs(delta) > 1.0e-9:
            direction_by_name[mimic_joint.joint_name] = 1.0 if delta > 0.0 else -1.0

    return direction_by_name


def _l10_joint_limits_by_joint() -> dict[str, tuple[float, float]]:
    try:
        import xml.etree.ElementTree as ET

        from teleop_stack.retargeting.hand_config import load_linker_l10_right_hand_spec

        spec = load_linker_l10_right_hand_spec()
    except Exception:
        return {}

    limits_by_name = {
        str(joint_name): (float(limits[0]), float(limits[1]))
        for joint_name, limits in zip(spec.active_joint_names, spec.active_joint_limits, strict=True)
        if _l10_is_contact_stopped_curl_joint(str(joint_name))
    }

    try:
        root = ET.parse(spec.urdf_path).getroot()
    except Exception:
        return limits_by_name

    for child in root:
        if child.tag != "joint":
            continue
        joint_name = str(child.attrib.get("name", ""))
        if not _l10_is_contact_stopped_curl_joint(joint_name):
            continue
        limit_tag = child.find("limit")
        if limit_tag is None:
            continue
        limits_by_name[joint_name] = (
            float(limit_tag.attrib.get("lower", "0.0")),
            float(limit_tag.attrib.get("upper", "0.0")),
        )
    return limits_by_name


def _expand_l10_mimic_joint_values(joint_values: NamedJointValues) -> NamedJointValues:
    try:
        from teleop_stack.retargeting.hand_config import load_linker_l10_right_hand_spec

        spec = load_linker_l10_right_hand_spec()
        return spec.expand_mimic_joint_values(joint_values)
    except Exception:
        return joint_values


def _require_seven_dof(joint_positions_rad: tuple[float, ...]) -> tuple[float, ...]:
    if len(joint_positions_rad) != 7:
        raise ValueError(f"Newton Nero kinematics expects 7 joints, got {len(joint_positions_rad)}")
    values = tuple(float(v) for v in joint_positions_rad)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Newton Nero joint positions must be finite")
    return values


def _same_joint_tuple(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> bool:
    if len(lhs) != len(rhs):
        return False
    return all(abs(float(a) - float(b)) <= 1.0e-12 for a, b in zip(lhs, rhs, strict=True))
