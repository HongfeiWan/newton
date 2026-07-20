from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

NERO_MDH: tuple[tuple[float, float, float, float], ...] = (
    (0.138, 0.0, 0.0, 0.0),
    (0.0, 0.0, math.pi / 2.0, math.pi),
    (0.31, 0.0, math.pi / 2.0, math.pi),
    (0.0, 0.0, math.pi / 2.0, math.pi),
    (0.27001, 0.0, math.pi / 2.0, math.pi / 2.0),
    (0.0, 0.0, math.pi / 2.0, math.pi / 2.0),
    (0.0235, 0.0, math.pi / 2.0, 0.0),
)


def nero_can_flange_pose_from_joints(joint_positions_rad: Iterable[float]) -> np.ndarray:
    """Return Nero SDK/CAN get_flange_pose-equivalent pose from 7 joint angles.

    The returned 4x4 pose is in the Nero SDK/CAN robot base frame, matching
    pyAgxArm's offline MDH FK and get_flange_pose() semantics.
    """
    q = np.asarray(tuple(float(v) for v in joint_positions_rad), dtype=np.float64).reshape(-1)
    if q.size != 7:
        raise ValueError(f"Nero CAN FK expects 7 joints, got {q.size}")
    if not np.all(np.isfinite(q)):
        raise ValueError("Nero CAN FK joints must be finite")

    pose = np.eye(4, dtype=np.float64)
    for joint, (d_i, a_i, alpha_i, theta_offset_i) in zip(q, NERO_MDH, strict=True):
        pose = pose @ _link_mdh(float(d_i), float(a_i), float(alpha_i), float(joint + theta_offset_i))
    return pose


def nero_can_flange_xyz_rpy_from_joints(joint_positions_rad: Iterable[float]) -> tuple[float, ...]:
    pose = nero_can_flange_pose_from_joints(joint_positions_rad)
    roll, pitch, yaw = _rot_to_rpy_zyx(pose[:3, :3])
    return (
        float(pose[0, 3]),
        float(pose[1, 3]),
        float(pose[2, 3]),
        float(roll),
        float(pitch),
        float(yaw),
    )


def _link_mdh(d_i: float, a_i: float, alpha_i: float, theta_i: float) -> np.ndarray:
    ca, sa = math.cos(alpha_i), math.sin(alpha_i)
    ct, st = math.cos(theta_i), math.sin(theta_i)
    return np.asarray(
        (
            (ct, -st, 0.0, a_i),
            (ca * st, ca * ct, -sa, -sa * d_i),
            (sa * st, sa * ct, ca, ca * d_i),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _rot_to_rpy_zyx(rotation: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    cp = math.cos(pitch)
    if abs(cp) < 1e-9:
        roll = 0.0
        yaw = math.atan2(-float(r[0, 1]), float(r[1, 1]))
    else:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return roll, pitch, yaw
