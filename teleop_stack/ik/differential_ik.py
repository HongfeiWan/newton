from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from teleop_stack.models import Pose7

TaskJacobian = tuple[tuple[float, ...], ...]
PositionJacobian = tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
SpatialJacobian = tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]


class PositionKinematicsModel(Protocol):
    def forward_pose(self, joint_positions_rad: tuple[float, ...]) -> Pose7:
        raise NotImplementedError

    def position_jacobian(self, joint_positions_rad: tuple[float, ...]) -> PositionJacobian:
        raise NotImplementedError


@dataclass(frozen=True)
class SyntheticSevenDofPositionKinematics:
    """Small host-side position kinematics used by Phase-1 dry-run paths.

    This is intentionally a lightweight analytical model rather than the final
    Rokae `model.h` integration. It gives the host IK stack a real nonlinear FK
    + Jacobian pair so the controller can run a genuine damped least-squares
    solved-rate step without pulling in optional heavy dependencies.
    """

    base_position_xyz: tuple[float, float, float] = (0.20, 0.0, 0.24)
    link_lengths_m: tuple[float, float, float] = (0.14, 0.12, 0.08)
    tool_quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def forward_pose(self, joint_positions_rad: tuple[float, ...]) -> Pose7:
        q0, q1, q2, q3, _, _, _ = _require_seven_dof(joint_positions_rad)
        l1, l2, l3 = self.link_lengths_m
        theta1 = q1
        theta2 = q1 + q2
        theta3 = q1 + q2 + q3

        radial = l1 * math.cos(theta1) + l2 * math.cos(theta2) + l3 * math.cos(theta3)
        vertical = l1 * math.sin(theta1) + l2 * math.sin(theta2) + l3 * math.sin(theta3)
        cos_yaw = math.cos(q0)
        sin_yaw = math.sin(q0)
        x = self.base_position_xyz[0] + cos_yaw * radial
        y = self.base_position_xyz[1] + sin_yaw * radial
        z = self.base_position_xyz[2] + vertical
        return Pose7(
            position_xyz=(x, y, z),
            quaternion_xyzw=self.tool_quaternion_xyzw,
        )

    def position_jacobian(self, joint_positions_rad: tuple[float, ...]) -> PositionJacobian:
        q0, q1, q2, q3, _, _, _ = _require_seven_dof(joint_positions_rad)
        l1, l2, l3 = self.link_lengths_m
        theta1 = q1
        theta2 = q1 + q2
        theta3 = q1 + q2 + q3
        cos_yaw = math.cos(q0)
        sin_yaw = math.sin(q0)

        radial = l1 * math.cos(theta1) + l2 * math.cos(theta2) + l3 * math.cos(theta3)
        radial_dq1 = -l1 * math.sin(theta1) - l2 * math.sin(theta2) - l3 * math.sin(theta3)
        radial_dq2 = -l2 * math.sin(theta2) - l3 * math.sin(theta3)
        radial_dq3 = -l3 * math.sin(theta3)
        vertical_dq1 = l1 * math.cos(theta1) + l2 * math.cos(theta2) + l3 * math.cos(theta3)
        vertical_dq2 = l2 * math.cos(theta2) + l3 * math.cos(theta3)
        vertical_dq3 = l3 * math.cos(theta3)

        return (
            (
                -sin_yaw * radial,
                cos_yaw * radial_dq1,
                cos_yaw * radial_dq2,
                cos_yaw * radial_dq3,
                0.0,
                0.0,
                0.0,
            ),
            (
                cos_yaw * radial,
                sin_yaw * radial_dq1,
                sin_yaw * radial_dq2,
                sin_yaw * radial_dq3,
                0.0,
                0.0,
                0.0,
            ),
            (
                0.0,
                vertical_dq1,
                vertical_dq2,
                vertical_dq3,
                0.0,
                0.0,
                0.0,
            ),
        )


def solve_damped_least_squares_step(
    jacobian: TaskJacobian,
    task_delta: tuple[float, ...],
    *,
    damping_lambda: float,
    bias_step: tuple[float, ...] | None = None,
    bias_weight: float = 0.0,
) -> tuple[float, ...]:
    """Solve a regularized least-squares step.

    This is equivalent to the small quadratic program:

    minimize ||J dq - dx||^2 + lambda^2 ||dq||^2 + w ||dq - dq_bias||^2
    """

    dof = len(jacobian[0])
    j_t = _transpose(jacobian)
    j_t_j = _matmul(j_t, jacobian)
    rhs = _matvec(j_t, task_delta)
    diagonal_regularization = max(0.0, damping_lambda) ** 2 + max(0.0, bias_weight)

    system = [list(row) for row in j_t_j]
    for i in range(dof):
        system[i][i] += diagonal_regularization

    if bias_step is not None and bias_weight > 0.0:
        for i in range(dof):
            rhs[i] += bias_weight * float(bias_step[i])

    return _solve_positive_definite(tuple(tuple(v for v in row) for row in system), rhs)


def _require_seven_dof(
    joint_positions_rad: tuple[float, ...],
) -> tuple[float, float, float, float, float, float, float]:
    if len(joint_positions_rad) != 7:
        raise ValueError(f"SyntheticSevenDofPositionKinematics expects 7 joints, got {len(joint_positions_rad)}.")
    return tuple(float(v) for v in joint_positions_rad)  # type: ignore[return-value]


def _transpose(matrix: TaskJacobian) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(row[i]) for row in matrix) for i in range(len(matrix[0])))


def _matmul(left: TaskJacobian, right: TaskJacobian) -> tuple[tuple[float, ...], ...]:
    shared = len(left[0])
    cols = len(right[0])
    result: list[tuple[float, ...]] = []
    for left_row in left:
        row: list[float] = []
        for col_idx in range(cols):
            value = 0.0
            for inner_idx in range(shared):
                value += float(left_row[inner_idx]) * float(right[inner_idx][col_idx])
            row.append(value)
        result.append(tuple(row))
    return tuple(result)


def _matvec(matrix: tuple[tuple[float, ...], ...], vector: tuple[float, ...]) -> list[float]:
    result: list[float] = []
    for row in matrix:
        value = 0.0
        for col_idx, coefficient in enumerate(row):
            value += float(coefficient) * float(vector[col_idx])
        result.append(value)
    return result


def _solve_positive_definite(matrix: tuple[tuple[float, ...], ...], rhs: list[float]) -> tuple[float, ...]:
    n = len(matrix)
    adjusted = [[float(value) for value in row] for row in matrix]
    jitter = 0.0
    for _ in range(3):
        cholesky: list[list[float]] = [[0.0] * n for _ in range(n)]
        success = True
        for i in range(n):
            for j in range(i + 1):
                subtotal = sum(cholesky[i][k] * cholesky[j][k] for k in range(j))
                if i == j:
                    diagonal = adjusted[i][i] - subtotal
                    if diagonal <= 1e-12:
                        success = False
                        break
                    cholesky[i][j] = math.sqrt(diagonal)
                else:
                    denominator = cholesky[j][j]
                    if abs(denominator) <= 1e-12:
                        success = False
                        break
                    cholesky[i][j] = (adjusted[i][j] - subtotal) / denominator
            if not success:
                break
        if success:
            return _solve_cholesky(cholesky, rhs)
        jitter = 1e-9 if jitter == 0.0 else jitter * 10.0
        for index in range(n):
            adjusted[index][index] = float(matrix[index][index]) + jitter

    raise ValueError("Failed to solve damped least-squares system: matrix is not positive definite.")


def _solve_cholesky(cholesky: list[list[float]], rhs: list[float]) -> tuple[float, ...]:
    n = len(cholesky)
    y = [0.0] * n
    for i in range(n):
        subtotal = sum(cholesky[i][k] * y[k] for k in range(i))
        y[i] = (float(rhs[i]) - subtotal) / cholesky[i][i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        subtotal = sum(cholesky[k][i] * x[k] for k in range(i + 1, n))
        x[i] = (y[i] - subtotal) / cholesky[i][i]
    return tuple(x)
