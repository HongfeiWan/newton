from __future__ import annotations

from dataclasses import dataclass

from teleop_stack.models import ArmSide, Pose7


@dataclass(frozen=True)
class TaskSpaceTarget:
    arm_side: ArmSide
    source_name: str
    timestamp_s: float
    frame_id: int
    ee_target: Pose7
    orientation_mode: str = "hold_fixed"
    target_frame: str = "base"


@dataclass(frozen=True)
class RobotStateSnapshot:
    timestamp_s: float
    joint_positions_rad: tuple[float, ...]
    ee_pose: Pose7 | None = None
    joint_velocities_rad_s: tuple[float, ...] | None = None
    psi: float | None = None
    motor_state: int | None = None
    safety_stop: str | None = None
    controller_mode: str | None = None
    motion_mode: str | None = None


@dataclass(frozen=True)
class JointServoStepResult:
    q_cmd: tuple[float, ...]
    dq_cmd: tuple[float, ...] | None
    status: str
    singularity_metric: float | None
    damping_scale: float
    limit_margin_min_rad: float | None
    control_mode: str
    events: tuple[str, ...] = ()
    target_position_error_m: float | None = None
    target_orientation_error_rad: float | None = None
    residual_position_error_m: float | None = None
    residual_orientation_error_rad: float | None = None
    orientation_ramp_remaining_rad: float | None = None
    orientation_alignment_error_rad: float | None = None
    orientation_alignment_elapsed_s: float | None = None
