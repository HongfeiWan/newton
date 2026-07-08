from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from teleop_stack.ik.controller import _joint_limit_clamped_events, _joint_limit_soft_zone_events
from teleop_stack.ik.differential_ik import PositionJacobian, SpatialJacobian
from teleop_stack.ik.filters import clamp_joint_step, clip_joint_positions
from teleop_stack.ik.full_pose import build_full_pose_task, solve_full_pose_damped_least_squares_step
from teleop_stack.ik.posture_bias import compute_posture_bias_step
from teleop_stack.ik.singularity import damping_scale_from_metric, normalized_singularity_metric
from teleop_stack.ik.so3 import (
    QuaternionXYZW,
    quat_angle_between_xyzw,
    orientation_error_rotvec_xyzw,
    quat_align_hemisphere_xyzw,
    quat_normalize_xyzw,
    quat_slerp_xyzw,
)
from teleop_stack.ik.types import JointServoStepResult, RobotStateSnapshot, TaskSpaceTarget
from teleop_stack.models import Pose7


class FullPoseKinematicsModel(Protocol):
    def forward_pose(self, joint_positions_rad: tuple[float, ...]) -> Pose7:
        raise NotImplementedError

    def position_jacobian(self, joint_positions_rad: tuple[float, ...]) -> PositionJacobian:
        raise NotImplementedError

    def spatial_jacobian(self, joint_positions_rad: tuple[float, ...]) -> SpatialJacobian:
        raise NotImplementedError


class FullPoseDlsStepSolver(Protocol):
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
        raise NotImplementedError


class FullPoseJointStepSolver(FullPoseDlsStepSolver, Protocol):
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
        raise NotImplementedError


@dataclass(frozen=True)
class FullPoseDifferentialIkControllerConfig:
    seed_joint_positions_rad: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    joint_lower_limits_rad: tuple[float, ...] = (-2.9, -2.0, -2.9, -2.5, -2.9, -2.0, -3.0)
    joint_upper_limits_rad: tuple[float, ...] = (2.9, 2.0, 2.9, 2.5, 2.9, 2.0, 3.0)
    neutral_joint_positions_rad: tuple[float, ...] = (0.0, -0.5, 0.8, 0.2, 0.0, 0.0, 0.0)
    position_tolerance_m: float = 0.002
    orientation_tolerance_rad: float = math.radians(1.0)
    max_task_step_m: float = 0.03
    max_rotation_step_rad: float = math.radians(5.0)
    position_weight: float = 1.0
    orientation_weight: float = 0.25
    max_joint_step_rad: float = 0.05
    max_joint_velocity_rad_s: float = 1.2
    max_joint_acceleration_rad_s2: float = 0.0
    damping_lambda: float = 0.02
    max_damping_scale: float = 25.0
    singularity_soft_threshold: float = 0.05
    singularity_hard_threshold: float = 0.005
    singularity_hard_stop_residual_ratio: float = 0.50
    singularity_escape_min_joint_step_rad: float = 1e-4
    posture_bias_gain: float = 0.08
    joint_limit_bias_gain: float = 0.20
    bias_weight: float = 0.08
    joint_limit_soft_margin_rad: float = 0.20
    kinematics_model: FullPoseKinematicsModel | None = None
    orientation_ramp_enabled: bool = False
    orientation_ramp_rate_rad_s: float = math.radians(10.0)
    orientation_alignment_gate_enabled: bool = False
    orientation_alignment_tolerance_rad: float = math.radians(5.0)
    orientation_alignment_timeout_s: float = 0.0
    dls_step_solver: FullPoseDlsStepSolver | None = None


class FullPoseDifferentialIkController:
    """Host-side differential IK controller for TCP position + orientation."""

    def __init__(self, config: FullPoseDifferentialIkControllerConfig):
        if config.kinematics_model is None:
            raise ValueError("FullPoseDifferentialIkController requires a kinematics_model with spatial_jacobian().")
        self.config = config
        self._kinematics_model = config.kinematics_model
        self._target: TaskSpaceTarget | None = None
        self._last_target_quaternion_xyzw: QuaternionXYZW | None = None
        self._ramped_target_quaternion_xyzw: QuaternionXYZW | None = None
        self._alignment_gate_target_quaternion_xyzw: QuaternionXYZW | None = None
        self._alignment_gate_start_timestamp_s: float | None = None
        self._alignment_gate_open = False
        self._last_joint_velocity_rad_s: tuple[float, ...] | None = None

    def reset(self, robot_state: RobotStateSnapshot) -> None:
        self._target = None
        self.reset_orientation_alignment()
        self._last_joint_velocity_rad_s = None
        if len(robot_state.joint_positions_rad) == 0:
            raise ValueError("robot_state.joint_positions_rad must not be empty.")

    def reset_orientation_alignment(self) -> None:
        self._last_target_quaternion_xyzw = None
        self._ramped_target_quaternion_xyzw = None
        self._alignment_gate_target_quaternion_xyzw = None
        self._alignment_gate_start_timestamp_s = None
        self._alignment_gate_open = False

    def set_target(self, target: TaskSpaceTarget) -> None:
        self._target = target

    def step(self, robot_state: RobotStateSnapshot, dt_s: float) -> JointServoStepResult:
        if len(robot_state.joint_positions_rad) == 0:
            q_cmd = tuple(float(v) for v in self.config.seed_joint_positions_rad)
            return JointServoStepResult(
                q_cmd=q_cmd,
                dq_cmd=tuple(0.0 for _ in q_cmd),
                status="fault_missing_joint_state",
                singularity_metric=None,
                damping_scale=1.0,
                limit_margin_min_rad=None,
                control_mode="joint_position",
                events=("missing_joint_state",),
            )

        current_q = tuple(float(v) for v in robot_state.joint_positions_rad)
        zero_dq = tuple(0.0 for _ in current_q)
        if self._target is None:
            self._last_joint_velocity_rad_s = zero_dq
            return JointServoStepResult(
                q_cmd=current_q,
                dq_cmd=zero_dq,
                status="hold_no_target",
                singularity_metric=None,
                damping_scale=1.0,
                limit_margin_min_rad=None,
                control_mode="joint_position",
                events=(),
            )

        current_pose = robot_state.ee_pose or self._kinematics_model.forward_pose(current_q)
        target_position = tuple(float(v) for v in self._target.ee_target.position_xyz)
        current_position = tuple(float(v) for v in current_pose.position_xyz)
        position_error = tuple(target_position[index] - current_position[index] for index in range(3))
        position_error_norm = math.sqrt(sum(value * value for value in position_error))

        current_quaternion = quat_normalize_xyzw(current_pose.quaternion_xyzw)
        raw_target_quaternion = quat_normalize_xyzw(self._target.ee_target.quaternion_xyzw)
        target_reference = self._last_target_quaternion_xyzw or current_quaternion
        aligned_raw_target_quaternion = quat_align_hemisphere_xyzw(raw_target_quaternion, target_reference)
        events: list[str] = []
        orientation_alignment_error_rad: float | None = None
        orientation_alignment_elapsed_s: float | None = None
        if bool(self.config.orientation_alignment_gate_enabled):
            if self._alignment_gate_target_quaternion_xyzw is None:
                self._alignment_gate_target_quaternion_xyzw = aligned_raw_target_quaternion
                self._alignment_gate_start_timestamp_s = float(self._target.timestamp_s)
                self._alignment_gate_open = False
                events.append("orientation_alignment_gate_initialized")
            if not self._alignment_gate_open:
                aligned_raw_target_quaternion = quat_align_hemisphere_xyzw(
                    self._alignment_gate_target_quaternion_xyzw,
                    target_reference,
                )
                orientation_alignment_elapsed_s = max(
                    0.0,
                    float(self._target.timestamp_s) - float(self._alignment_gate_start_timestamp_s or self._target.timestamp_s),
                )
                events.append("orientation_alignment_gate_active")
        target_quaternion = aligned_raw_target_quaternion
        orientation_ramp_remaining_rad: float | None = None
        if bool(self.config.orientation_ramp_enabled):
            if self._ramped_target_quaternion_xyzw is None:
                self._ramped_target_quaternion_xyzw = current_quaternion
                events.append("orientation_ramp_initialized")
            ramp_reference = quat_align_hemisphere_xyzw(
                aligned_raw_target_quaternion,
                self._ramped_target_quaternion_xyzw,
            )
            remaining_rad = quat_angle_between_xyzw(self._ramped_target_quaternion_xyzw, ramp_reference)
            orientation_ramp_remaining_rad = remaining_rad
            max_ramp_step = max(0.0, float(self.config.orientation_ramp_rate_rad_s)) * max(0.0, float(dt_s))
            if remaining_rad <= 1e-9:
                target_quaternion = ramp_reference
                orientation_ramp_remaining_rad = 0.0
            elif max_ramp_step <= 0.0:
                target_quaternion = self._ramped_target_quaternion_xyzw
                events.append("orientation_ramp_active")
            elif remaining_rad > max_ramp_step:
                alpha = max_ramp_step / remaining_rad
                target_quaternion = quat_slerp_xyzw(self._ramped_target_quaternion_xyzw, ramp_reference, alpha)
                orientation_ramp_remaining_rad = quat_angle_between_xyzw(target_quaternion, ramp_reference)
                events.append("orientation_ramp_active")
            else:
                target_quaternion = ramp_reference
                events.append("orientation_ramp_completed")
                orientation_ramp_remaining_rad = 0.0
            self._ramped_target_quaternion_xyzw = target_quaternion
        self._last_target_quaternion_xyzw = target_quaternion
        orientation_error = orientation_error_rotvec_xyzw(target_quaternion, current_quaternion)
        orientation_error_norm = math.sqrt(sum(value * value for value in orientation_error))

        ramp_finished = (
            orientation_ramp_remaining_rad is None
            or orientation_ramp_remaining_rad <= max(0.0, float(self.config.orientation_tolerance_rad))
        )
        if (
            bool(self.config.orientation_alignment_gate_enabled)
            and self._alignment_gate_target_quaternion_xyzw is not None
            and not self._alignment_gate_open
        ):
            alignment_target = quat_align_hemisphere_xyzw(
                self._alignment_gate_target_quaternion_xyzw,
                current_quaternion,
            )
            orientation_alignment_error_rad = quat_angle_between_xyzw(current_quaternion, alignment_target)
            alignment_tolerance = max(0.0, float(self.config.orientation_alignment_tolerance_rad))
            timeout_s = max(0.0, float(self.config.orientation_alignment_timeout_s))
            timed_out = (
                timeout_s > 0.0
                and orientation_alignment_elapsed_s is not None
                and orientation_alignment_elapsed_s >= timeout_s
            )
            if ramp_finished and orientation_alignment_error_rad <= alignment_tolerance:
                self._alignment_gate_open = True
                events.append("orientation_alignment_gate_aligned")
                events.append("orientation_alignment_gate_opened")
            elif timed_out:
                self._alignment_gate_open = True
                events.append("orientation_alignment_gate_timeout")
                events.append("orientation_alignment_gate_opened")
        if (
            position_error_norm <= max(0.0, float(self.config.position_tolerance_m))
            and orientation_error_norm <= max(0.0, float(self.config.orientation_tolerance_rad))
            and ramp_finished
            and self._alignment_gate_open_or_disabled()
        ):
            self._last_joint_velocity_rad_s = zero_dq
            return JointServoStepResult(
                q_cmd=current_q,
                dq_cmd=zero_dq,
                status="hold_target_converged",
                singularity_metric=None,
                damping_scale=1.0,
                limit_margin_min_rad=None,
                control_mode="joint_position",
                events=tuple(events + ["target_within_tolerance"]),
                target_position_error_m=position_error_norm,
                target_orientation_error_rad=orientation_error_norm,
                residual_position_error_m=position_error_norm,
                residual_orientation_error_rad=orientation_error_norm,
                orientation_ramp_remaining_rad=orientation_ramp_remaining_rad,
                orientation_alignment_error_rad=orientation_alignment_error_rad,
                orientation_alignment_elapsed_s=orientation_alignment_elapsed_s,
            )

        position_jacobian = self._kinematics_model.position_jacobian(current_q)
        spatial_jacobian = self._kinematics_model.spatial_jacobian(current_q)
        singularity_metric = normalized_singularity_metric(position_jacobian)
        damping_scale = damping_scale_from_metric(
            singularity_metric,
            soft_threshold=self.config.singularity_soft_threshold,
            hard_threshold=self.config.singularity_hard_threshold,
            max_scale=self.config.max_damping_scale,
        )

        if singularity_metric < float(self.config.singularity_soft_threshold):
            events.append("singularity_soft_zone")

        bias_step, limit_margin_min_rad = compute_posture_bias_step(
            current_q,
            lower_limits_rad=self.config.joint_lower_limits_rad,
            upper_limits_rad=self.config.joint_upper_limits_rad,
            neutral_positions_rad=self.config.neutral_joint_positions_rad,
            posture_gain=self.config.posture_bias_gain,
            joint_limit_gain=self.config.joint_limit_bias_gain,
            soft_margin_rad=self.config.joint_limit_soft_margin_rad,
        )
        if limit_margin_min_rad is not None and limit_margin_min_rad < float(self.config.joint_limit_soft_margin_rad):
            events.append("joint_limit_soft_zone")
            events.extend(
                _joint_limit_soft_zone_events(
                    current_q,
                    lower_limits_rad=self.config.joint_lower_limits_rad,
                    upper_limits_rad=self.config.joint_upper_limits_rad,
                    soft_margin_rad=float(self.config.joint_limit_soft_margin_rad),
                )
            )

        full_pose_task = build_full_pose_task(
            spatial_jacobian,
            position_error_xyz=position_error,
            orientation_error_rotvec=orientation_error,
            position_weight=self.config.position_weight,
            orientation_weight=self.config.orientation_weight,
            max_position_step_m=self.config.max_task_step_m,
            max_rotation_step_rad=self.config.max_rotation_step_rad,
        )
        dls_solver = self.config.dls_step_solver
        joint_step_solver = (
            dls_solver
            if dls_solver is not None and hasattr(dls_solver, "solve_full_pose_joint_step")
            else None
        )
        if joint_step_solver is not None:
            q_cmd, dq_cmd, unclipped_q_cmd, acceleration_limited, clipped = joint_step_solver.solve_full_pose_joint_step(
                spatial_jacobian,
                position_error_xyz=position_error,
                orientation_error_rotvec=orientation_error,
                damping_lambda=float(self.config.damping_lambda) * damping_scale,
                position_weight=self.config.position_weight,
                orientation_weight=self.config.orientation_weight,
                max_position_step_m=self.config.max_task_step_m,
                max_rotation_step_rad=self.config.max_rotation_step_rad,
                bias_step=bias_step,
                bias_weight=self.config.bias_weight,
                current_q=current_q,
                previous_velocity_rad_s=self._last_joint_velocity_rad_s,
                lower_limits_rad=self.config.joint_lower_limits_rad,
                upper_limits_rad=self.config.joint_upper_limits_rad,
                max_joint_step_rad=self.config.max_joint_step_rad,
                max_joint_velocity_rad_s=self.config.max_joint_velocity_rad_s,
                max_joint_acceleration_rad_s2=self.config.max_joint_acceleration_rad_s2,
                dt_s=dt_s,
            )
            if acceleration_limited:
                events.append("joint_acceleration_limited")
        else:
            dq_step = solve_full_pose_damped_least_squares_step(
                spatial_jacobian,
                position_error_xyz=position_error,
                orientation_error_rotvec=orientation_error,
                damping_lambda=float(self.config.damping_lambda) * damping_scale,
                position_weight=self.config.position_weight,
                orientation_weight=self.config.orientation_weight,
                max_position_step_m=self.config.max_task_step_m,
                max_rotation_step_rad=self.config.max_rotation_step_rad,
                bias_step=bias_step,
                bias_weight=self.config.bias_weight,
            )
            if dls_solver is not None:
                dq_step = dls_solver.solve_full_pose_step(
                    spatial_jacobian,
                    position_error_xyz=position_error,
                    orientation_error_rotvec=orientation_error,
                    damping_lambda=float(self.config.damping_lambda) * damping_scale,
                    position_weight=self.config.position_weight,
                    orientation_weight=self.config.orientation_weight,
                    max_position_step_m=self.config.max_task_step_m,
                    max_rotation_step_rad=self.config.max_rotation_step_rad,
                    bias_step=bias_step,
                    bias_weight=self.config.bias_weight,
                )
            dq_step = clamp_joint_step(
                dq_step,
                max_step_rad=self.config.max_joint_step_rad,
                max_velocity_rad_s=self.config.max_joint_velocity_rad_s,
                dt_s=dt_s,
            )
            dq_step, acceleration_limited = _clamp_joint_acceleration(
                dq_step,
                previous_velocity_rad_s=self._last_joint_velocity_rad_s,
                max_acceleration_rad_s2=float(self.config.max_joint_acceleration_rad_s2),
                dt_s=dt_s,
            )
            if acceleration_limited:
                events.append("joint_acceleration_limited")

            unclipped_q_cmd = tuple(current_q[index] + dq_step[index] for index in range(len(current_q)))
            q_cmd, clipped = clip_joint_positions(
                unclipped_q_cmd,
                lower_limits_rad=self.config.joint_lower_limits_rad,
                upper_limits_rad=self.config.joint_upper_limits_rad,
            )
            applied_dq = tuple(q_cmd[index] - current_q[index] for index in range(len(current_q)))
            safe_dt = max(1e-6, float(dt_s))
            dq_cmd = tuple(value / safe_dt for value in applied_dq)
        if clipped:
            events.append("joint_limit_clamped")
            events.extend(
                _joint_limit_clamped_events(
                    unclipped_q_cmd,
                    q_cmd,
                    lower_limits_rad=self.config.joint_lower_limits_rad,
                    upper_limits_rad=self.config.joint_upper_limits_rad,
                )
            )

        applied_dq = tuple(q_cmd[index] - current_q[index] for index in range(len(current_q)))
        safe_dt = max(1e-6, float(dt_s))
        self._last_joint_velocity_rad_s = dq_cmd
        applied_dq_norm = math.sqrt(sum(value * value for value in applied_dq))
        achieved_task_delta = tuple(
            sum(
                float(full_pose_task.weighted_jacobian[row_idx][col_idx]) * float(applied_dq[col_idx])
                for col_idx in range(len(applied_dq))
            )
            for row_idx in range(6)
        )
        task_residual = tuple(
            float(full_pose_task.task_delta[row_idx]) - float(achieved_task_delta[row_idx])
            for row_idx in range(6)
        )
        residual_position_error_m = _unweighted_norm(
            task_residual[:3],
            weight=float(self.config.position_weight),
        )
        residual_orientation_error_rad = _unweighted_norm(
            task_residual[3:],
            weight=float(self.config.orientation_weight),
        )
        desired_task_delta_norm = math.sqrt(sum(value * value for value in full_pose_task.task_delta))
        task_residual_norm = math.sqrt(sum(value * value for value in task_residual))

        if singularity_metric <= float(self.config.singularity_hard_threshold):
            events.append("singularity_hard_zone")
            residual_ratio = (
                task_residual_norm / desired_task_delta_norm
                if desired_task_delta_norm > 1e-9
                else 0.0
            )
            if residual_ratio > float(self.config.singularity_hard_stop_residual_ratio):
                if applied_dq_norm >= float(self.config.singularity_escape_min_joint_step_rad):
                    events.append("singularity_escape_step")
                else:
                    events.append("singularity_hard_stop")
                    self._last_joint_velocity_rad_s = zero_dq
                    return JointServoStepResult(
                        q_cmd=current_q,
                        dq_cmd=zero_dq,
                        status="hold_singularity_hard",
                        singularity_metric=singularity_metric,
                        damping_scale=damping_scale,
                        limit_margin_min_rad=limit_margin_min_rad,
                        control_mode="joint_position",
                        events=tuple(events),
                        target_position_error_m=position_error_norm,
                        target_orientation_error_rad=orientation_error_norm,
                        residual_position_error_m=residual_position_error_m,
                        residual_orientation_error_rad=residual_orientation_error_rad,
                        orientation_ramp_remaining_rad=orientation_ramp_remaining_rad,
                        orientation_alignment_error_rad=orientation_alignment_error_rad,
                        orientation_alignment_elapsed_s=orientation_alignment_elapsed_s,
                    )
            else:
                events.append("singularity_hard_zone_tracking")

        if applied_dq_norm <= 1e-9:
            status = "hold_zero_joint_step"
            events.append("zero_joint_step")
        else:
            status = "tracking_full_pose"
            events.append("full_pose_servo_active")

        return JointServoStepResult(
            q_cmd=q_cmd,
            dq_cmd=dq_cmd,
            status=status,
            singularity_metric=singularity_metric,
            damping_scale=damping_scale,
            limit_margin_min_rad=limit_margin_min_rad,
            control_mode="joint_position",
            events=tuple(events),
            target_position_error_m=position_error_norm,
            target_orientation_error_rad=orientation_error_norm,
            residual_position_error_m=residual_position_error_m,
            residual_orientation_error_rad=residual_orientation_error_rad,
            orientation_ramp_remaining_rad=orientation_ramp_remaining_rad,
            orientation_alignment_error_rad=orientation_alignment_error_rad,
            orientation_alignment_elapsed_s=orientation_alignment_elapsed_s,
        )

    def _alignment_gate_open_or_disabled(self) -> bool:
        return not bool(self.config.orientation_alignment_gate_enabled) or bool(self._alignment_gate_open)


def _unweighted_norm(values: tuple[float, ...], *, weight: float) -> float | None:
    safe_weight = abs(float(weight))
    if safe_weight <= 1e-12:
        return None
    return math.sqrt(sum((float(value) / safe_weight) ** 2 for value in values))


def _clamp_joint_acceleration(
    joint_step_rad: tuple[float, ...],
    *,
    previous_velocity_rad_s: tuple[float, ...] | None,
    max_acceleration_rad_s2: float,
    dt_s: float,
) -> tuple[tuple[float, ...], bool]:
    if previous_velocity_rad_s is None:
        return joint_step_rad, False
    if len(previous_velocity_rad_s) != len(joint_step_rad):
        return joint_step_rad, False
    safe_dt = max(1e-6, float(dt_s))
    safe_accel = max(0.0, float(max_acceleration_rad_s2))
    if safe_accel <= 0.0:
        return joint_step_rad, False

    max_delta_velocity = safe_accel * safe_dt
    limited = False
    output: list[float] = []
    for step, previous_velocity in zip(joint_step_rad, previous_velocity_rad_s, strict=True):
        desired_velocity = float(step) / safe_dt
        lower = float(previous_velocity) - max_delta_velocity
        upper = float(previous_velocity) + max_delta_velocity
        bounded_velocity = max(lower, min(upper, desired_velocity))
        if abs(bounded_velocity - desired_velocity) > 1e-12:
            limited = True
        output.append(bounded_velocity * safe_dt)
    return tuple(output), limited
