from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

import numpy as np

from teleop_stack.ik.differential_ik import PositionJacobian, SpatialJacobian
from teleop_stack.ik.so3 import (
    QuaternionXYZW,
    quat_align_hemisphere_xyzw,
    quat_inverse_xyzw,
    quat_log_rotvec_xyzw,
    quat_multiply_xyzw,
    quat_normalize_xyzw,
)
from teleop_stack.models import Pose7


NeroSide = Literal["left", "right"]


@dataclass
class GenesisLinkKinematicsModel:
    """FK + finite-difference Jacobian adapter for a Genesis Nero end-effector link."""

    runtime: Any
    side: NeroSide
    finite_difference_rad: float = 1e-4
    _last_jacobian_q: tuple[float, ...] | None = field(default=None, init=False, repr=False)
    _last_spatial_jacobian: SpatialJacobian | None = field(default=None, init=False, repr=False)

    def forward_pose(self, joint_positions_rad: tuple[float, ...]) -> Pose7:
        position, quaternion_xyzw = self._pose_after_set(joint_positions_rad)
        return Pose7(
            position_xyz=tuple(float(v) for v in position),
            quaternion_xyzw=tuple(float(v) for v in quaternion_xyzw),  # type: ignore[arg-type]
        )

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

        eps = max(abs(float(self.finite_difference_rad)), 1e-6)
        columns: list[tuple[float, float, float, float, float, float]] = []
        for joint_index in range(7):
            q_plus = list(q)
            q_minus = list(q)
            q_plus[joint_index] += eps
            q_minus[joint_index] -= eps
            pos_plus, quat_plus = self._pose_after_set(tuple(q_plus))
            pos_minus, quat_minus = self._pose_after_set(tuple(q_minus))
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

        self._set_joint_positions(q)
        spatial: SpatialJacobian = tuple(
            tuple(float(columns[col][row]) for col in range(7))
            for row in range(6)
        )  # type: ignore[assignment]
        self._last_jacobian_q = q
        self._last_spatial_jacobian = spatial
        return spatial

    def clear_cache(self) -> None:
        self._last_jacobian_q = None
        self._last_spatial_jacobian = None

    def _pose_after_set(self, joint_positions_rad: tuple[float, ...]) -> tuple[np.ndarray, QuaternionXYZW]:
        self._set_joint_positions(joint_positions_rad)
        link = self.runtime.eef_links[self.side]
        position = _tensor_to_np(link.get_pos()).reshape(3).astype(np.float64)
        quat_wxyz = _tensor_to_np(link.get_quat()).reshape(4).astype(np.float64)
        return position, _wxyz_to_xyzw(quat_wxyz)

    def _set_joint_positions(self, joint_positions_rad: tuple[float, ...]) -> None:
        q = np.asarray(_require_seven_dof(joint_positions_rad), dtype=np.float32)
        robot = self.runtime.robots[self.side]
        dofs = self.runtime.arm_dofs[self.side]
        robot.set_dofs_position(q, dofs, zero_velocity=True)


def _tensor_to_np(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _wxyz_to_xyzw(quaternion_wxyz: np.ndarray) -> QuaternionXYZW:
    if quaternion_wxyz.shape[0] < 4:
        raise ValueError("Expected a 4D wxyz quaternion.")
    w, x, y, z = (float(v) for v in quaternion_wxyz[:4])
    return quat_normalize_xyzw((x, y, z, w))


def _require_seven_dof(joint_positions_rad: tuple[float, ...]) -> tuple[float, ...]:
    if len(joint_positions_rad) != 7:
        raise ValueError(f"Nero Genesis kinematics expects 7 joints, got {len(joint_positions_rad)}.")
    values = tuple(float(v) for v in joint_positions_rad)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nero Genesis joint positions must be finite.")
    return values


def _same_joint_tuple(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> bool:
    if len(lhs) != len(rhs):
        return False
    return all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(lhs, rhs, strict=True))
