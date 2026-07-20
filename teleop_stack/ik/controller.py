from __future__ import annotations

import math
from dataclasses import dataclass

from teleop_stack.ik.differential_ik import (
    PositionKinematicsModel,
    SyntheticSevenDofPositionKinematics,
    solve_damped_least_squares_step,
)
from teleop_stack.ik.filters import clamp_joint_step, clamp_vector_norm, clip_joint_positions
from teleop_stack.ik.posture_bias import compute_posture_bias_step
from teleop_stack.ik.singularity import damping_scale_from_metric, normalized_singularity_metric
from teleop_stack.ik.types import JointServoStepResult, RobotStateSnapshot, TaskSpaceTarget


@dataclass(frozen=True)
class DifferentialIkControllerConfig:
    seed_joint_positions_rad: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    joint_lower_limits_rad: tuple[float, ...] = (-2.9, -2.0, -2.9, -2.5, -2.9, -2.0, -3.0)
    joint_upper_limits_rad: tuple[float, ...] = (2.9, 2.0, 2.9, 2.5, 2.9, 2.0, 3.0)
    neutral_joint_positions_rad: tuple[float, ...] = (0.0, -0.5, 0.8, 0.2, 0.0, 0.0, 0.0)
    task_gain: float = 1.0
    position_tolerance_m: float = 0.002
    max_task_step_m: float = 0.03
    max_joint_step_rad: float = 0.05
    max_joint_velocity_rad_s: float = 1.2
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
    kinematics_model: PositionKinematicsModel | None = None


class DifferentialIkController:
    """Host-side position-only differential IK controller.

    Phase 1 intentionally tracks only translation. It solves a regularized
    damped least-squares step each cycle, which is QP-equivalent for the small
    local servo objective used here.
    """

    def __init__(self, config: DifferentialIkControllerConfig | None = None):
        self.config = config or DifferentialIkControllerConfig()
        self._target: TaskSpaceTarget | None = None
        self._kinematics_model = self.config.kinematics_model or SyntheticSevenDofPositionKinematics()

    def reset(self, robot_state: RobotStateSnapshot) -> None:
        self._target = None
        if len(robot_state.joint_positions_rad) == 0:
            raise ValueError("robot_state.joint_positions_rad must not be empty.")

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
        if position_error_norm <= max(0.0, float(self.config.position_tolerance_m)):
            return JointServoStepResult(
                q_cmd=current_q,
                dq_cmd=zero_dq,
                status="hold_target_converged",
                singularity_metric=None,
                damping_scale=1.0,
                limit_margin_min_rad=None,
                control_mode="joint_position",
                events=("target_within_tolerance",),
            )

        desired_task_delta = clamp_vector_norm(
            tuple(float(self.config.task_gain) * value for value in position_error),
            max_norm=self.config.max_task_step_m,
        )
        jacobian = self._kinematics_model.position_jacobian(current_q)
        # Use a scale-invariant singularity metric. The raw translational
        # Jacobian singular values are in meters/radian, so a fixed hard
        # threshold on sigma_min alone can falsely classify well-conditioned
        # but short-reach postures as singular on the live robot.
        singularity_metric = normalized_singularity_metric(jacobian)
        damping_scale = damping_scale_from_metric(
            singularity_metric,
            soft_threshold=self.config.singularity_soft_threshold,
            hard_threshold=self.config.singularity_hard_threshold,
            max_scale=self.config.max_damping_scale,
        )

        events: list[str] = []
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

        dq_step = solve_damped_least_squares_step(
            jacobian,
            desired_task_delta,
            damping_lambda=float(self.config.damping_lambda) * damping_scale,
            bias_step=bias_step,
            bias_weight=self.config.bias_weight,
        )
        dq_step = clamp_joint_step(
            dq_step,
            max_step_rad=self.config.max_joint_step_rad,
            max_velocity_rad_s=self.config.max_joint_velocity_rad_s,
            dt_s=dt_s,
        )

        unclipped_q_cmd = tuple(current_q[index] + dq_step[index] for index in range(len(current_q)))
        q_cmd, clipped = clip_joint_positions(
            unclipped_q_cmd,
            lower_limits_rad=self.config.joint_lower_limits_rad,
            upper_limits_rad=self.config.joint_upper_limits_rad,
        )
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
        dq_cmd = tuple(value / safe_dt for value in applied_dq)
        applied_dq_norm = math.sqrt(sum(value * value for value in applied_dq))
        achieved_task_delta = tuple(
            sum(float(jacobian[row_idx][col_idx]) * float(applied_dq[col_idx]) for col_idx in range(len(applied_dq)))
            for row_idx in range(3)
        )
        task_residual = tuple(
            float(desired_task_delta[row_idx]) - float(achieved_task_delta[row_idx]) for row_idx in range(3)
        )
        desired_task_delta_norm = math.sqrt(sum(value * value for value in desired_task_delta))
        task_residual_norm = math.sqrt(sum(value * value for value in task_residual))

        if singularity_metric <= float(self.config.singularity_hard_threshold):
            events.append("singularity_hard_zone")
            residual_ratio = task_residual_norm / desired_task_delta_norm if desired_task_delta_norm > 1e-9 else 0.0
            if residual_ratio > float(self.config.singularity_hard_stop_residual_ratio):
                if applied_dq_norm >= float(self.config.singularity_escape_min_joint_step_rad):
                    events.append("singularity_escape_step")
                else:
                    events.append("singularity_hard_stop")
                    return JointServoStepResult(
                        q_cmd=current_q,
                        dq_cmd=zero_dq,
                        status="hold_singularity_hard",
                        singularity_metric=singularity_metric,
                        damping_scale=damping_scale,
                        limit_margin_min_rad=limit_margin_min_rad,
                        control_mode="joint_position",
                        events=tuple(events),
                    )
            else:
                events.append("singularity_hard_zone_tracking")

        if applied_dq_norm <= 1e-9:
            status = "hold_zero_joint_step"
            events.append("zero_joint_step")
        else:
            status = "tracking_position"
            events.append("position_servo_active")

        return JointServoStepResult(
            q_cmd=q_cmd,
            dq_cmd=dq_cmd,
            status=status,
            singularity_metric=singularity_metric,
            damping_scale=damping_scale,
            limit_margin_min_rad=limit_margin_min_rad,
            control_mode="joint_position",
            events=tuple(events),
        )


def _joint_limit_soft_zone_events(
    joint_positions_rad: tuple[float, ...],
    *,
    lower_limits_rad: tuple[float, ...] | None,
    upper_limits_rad: tuple[float, ...] | None,
    soft_margin_rad: float,
) -> tuple[str, ...]:
    if lower_limits_rad is None or upper_limits_rad is None:
        return ()
    if len(lower_limits_rad) != len(joint_positions_rad) or len(upper_limits_rad) != len(joint_positions_rad):
        return ()

    safe_margin = max(0.0, float(soft_margin_rad))
    events: list[str] = []
    for index, q in enumerate(joint_positions_rad):
        if abs(float(upper_limits_rad[index]) - float(lower_limits_rad[index])) <= 1e-12:
            continue
        lower_margin = float(q) - float(lower_limits_rad[index])
        upper_margin = float(upper_limits_rad[index]) - float(q)
        if lower_margin < safe_margin:
            events.append(f"joint_limit_soft_zone:j{index + 1}_lower")
        if upper_margin < safe_margin:
            events.append(f"joint_limit_soft_zone:j{index + 1}_upper")
    return tuple(events)


def _joint_limit_clamped_events(
    unclipped_joint_positions_rad: tuple[float, ...],
    clipped_joint_positions_rad: tuple[float, ...],
    *,
    lower_limits_rad: tuple[float, ...] | None,
    upper_limits_rad: tuple[float, ...] | None,
) -> tuple[str, ...]:
    if lower_limits_rad is None or upper_limits_rad is None:
        return ()
    if len(lower_limits_rad) != len(unclipped_joint_positions_rad) or len(upper_limits_rad) != len(
        unclipped_joint_positions_rad
    ):
        return ()

    events: list[str] = []
    for index, (unclipped, clipped) in enumerate(
        zip(unclipped_joint_positions_rad, clipped_joint_positions_rad, strict=True)
    ):
        if abs(float(unclipped) - float(clipped)) <= 1e-12:
            continue
        if float(unclipped) < float(lower_limits_rad[index]):
            events.append(f"joint_limit_clamped:j{index + 1}_lower")
        elif float(unclipped) > float(upper_limits_rad[index]):
            events.append(f"joint_limit_clamped:j{index + 1}_upper")
        else:
            events.append(f"joint_limit_clamped:j{index + 1}")
    return tuple(events)
