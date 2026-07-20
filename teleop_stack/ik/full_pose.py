from __future__ import annotations

from dataclasses import dataclass

from teleop_stack.ik.differential_ik import TaskJacobian, solve_damped_least_squares_step
from teleop_stack.ik.filters import clamp_vector_norm

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class FullPoseTask:
    task_delta: tuple[float, float, float, float, float, float]
    weighted_jacobian: TaskJacobian


def build_full_pose_task(
    spatial_jacobian: TaskJacobian,
    *,
    position_error_xyz: Vector3,
    orientation_error_rotvec: Vector3,
    position_weight: float = 1.0,
    orientation_weight: float = 1.0,
    max_position_step_m: float | None = None,
    max_rotation_step_rad: float | None = None,
) -> FullPoseTask:
    _require_six_row_jacobian(spatial_jacobian)
    position_step = tuple(float(value) for value in position_error_xyz)
    rotation_step = tuple(float(value) for value in orientation_error_rotvec)
    if max_position_step_m is not None:
        position_step = clamp_vector_norm(position_step, max_norm=float(max_position_step_m))
    if max_rotation_step_rad is not None:
        rotation_step = clamp_vector_norm(rotation_step, max_norm=float(max_rotation_step_rad))

    w_pos = float(position_weight)
    w_rot = float(orientation_weight)
    task_delta = (
        w_pos * position_step[0],
        w_pos * position_step[1],
        w_pos * position_step[2],
        w_rot * rotation_step[0],
        w_rot * rotation_step[1],
        w_rot * rotation_step[2],
    )
    weighted_jacobian: TaskJacobian = tuple(
        _scale_row(row, w_pos if row_index < 3 else w_rot) for row_index, row in enumerate(spatial_jacobian)
    )
    return FullPoseTask(task_delta=task_delta, weighted_jacobian=weighted_jacobian)


def solve_full_pose_damped_least_squares_step(
    spatial_jacobian: TaskJacobian,
    *,
    position_error_xyz: Vector3,
    orientation_error_rotvec: Vector3,
    damping_lambda: float,
    position_weight: float = 1.0,
    orientation_weight: float = 1.0,
    max_position_step_m: float | None = None,
    max_rotation_step_rad: float | None = None,
    bias_step: tuple[float, ...] | None = None,
    bias_weight: float = 0.0,
) -> tuple[float, ...]:
    task = build_full_pose_task(
        spatial_jacobian,
        position_error_xyz=position_error_xyz,
        orientation_error_rotvec=orientation_error_rotvec,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
        max_position_step_m=max_position_step_m,
        max_rotation_step_rad=max_rotation_step_rad,
    )
    return solve_damped_least_squares_step(
        task.weighted_jacobian,
        task.task_delta,
        damping_lambda=damping_lambda,
        bias_step=bias_step,
        bias_weight=bias_weight,
    )


def _scale_row(row: tuple[float, ...], scale: float) -> tuple[float, ...]:
    return tuple(float(value) * scale for value in row)


def _require_six_row_jacobian(spatial_jacobian: TaskJacobian) -> None:
    if len(spatial_jacobian) != 6:
        raise ValueError(f"Full-pose IK expects a 6-row spatial Jacobian, got {len(spatial_jacobian)} rows.")
    if len(spatial_jacobian[0]) == 0:
        raise ValueError("Full-pose IK spatial Jacobian must have at least one column.")
    dof = len(spatial_jacobian[0])
    for row in spatial_jacobian:
        if len(row) != dof:
            raise ValueError("Full-pose IK spatial Jacobian rows must have the same number of columns.")
