from __future__ import annotations

from math import acos, atan2, sin, sqrt

QuaternionXYZW = tuple[float, float, float, float]
Vector3 = tuple[float, float, float]


def quat_normalize_xyzw(quaternion_xyzw: QuaternionXYZW) -> QuaternionXYZW:
    norm = sqrt(sum(float(value) * float(value) for value in quaternion_xyzw))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(value) / norm for value in quaternion_xyzw)  # type: ignore[return-value]


def quat_dot_xyzw(lhs: QuaternionXYZW, rhs: QuaternionXYZW) -> float:
    return (
        float(lhs[0]) * float(rhs[0])
        + float(lhs[1]) * float(rhs[1])
        + float(lhs[2]) * float(rhs[2])
        + float(lhs[3]) * float(rhs[3])
    )


def quat_align_hemisphere_xyzw(target_xyzw: QuaternionXYZW, reference_xyzw: QuaternionXYZW) -> QuaternionXYZW:
    target = quat_normalize_xyzw(target_xyzw)
    reference = quat_normalize_xyzw(reference_xyzw)
    if quat_dot_xyzw(target, reference) < 0.0:
        return (-target[0], -target[1], -target[2], -target[3])
    return target


def quat_slerp_xyzw(lhs_xyzw: QuaternionXYZW, rhs_xyzw: QuaternionXYZW, alpha: float) -> QuaternionXYZW:
    lhs = quat_normalize_xyzw(lhs_xyzw)
    rhs = quat_align_hemisphere_xyzw(rhs_xyzw, lhs)
    t = max(0.0, min(1.0, float(alpha)))
    dot = max(-1.0, min(1.0, quat_dot_xyzw(lhs, rhs)))
    if dot > 0.9995:
        return quat_normalize_xyzw(
            tuple(lhs[index] + t * (rhs[index] - lhs[index]) for index in range(4))  # type: ignore[arg-type]
        )

    theta_0 = acos(dot)
    theta = theta_0 * t
    sin_theta_0 = sin(theta_0)
    if abs(sin_theta_0) <= 1e-12:
        return rhs
    scale_lhs = sin(theta_0 - theta) / sin_theta_0
    scale_rhs = sin(theta) / sin_theta_0
    return quat_normalize_xyzw(
        tuple(scale_lhs * lhs[index] + scale_rhs * rhs[index] for index in range(4))  # type: ignore[arg-type]
    )


def quat_angle_between_xyzw(lhs_xyzw: QuaternionXYZW, rhs_xyzw: QuaternionXYZW) -> float:
    lhs = quat_normalize_xyzw(lhs_xyzw)
    rhs = quat_align_hemisphere_xyzw(rhs_xyzw, lhs)
    dot = max(-1.0, min(1.0, quat_dot_xyzw(lhs, rhs)))
    return 2.0 * acos(dot)


def quat_conjugate_xyzw(quaternion_xyzw: QuaternionXYZW) -> QuaternionXYZW:
    x, y, z, w = quaternion_xyzw
    return (-float(x), -float(y), -float(z), float(w))


def quat_inverse_xyzw(quaternion_xyzw: QuaternionXYZW) -> QuaternionXYZW:
    return quat_conjugate_xyzw(quat_normalize_xyzw(quaternion_xyzw))


def quat_multiply_xyzw(lhs: QuaternionXYZW, rhs: QuaternionXYZW) -> QuaternionXYZW:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return quat_normalize_xyzw(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def quat_log_rotvec_xyzw(quaternion_xyzw: QuaternionXYZW) -> Vector3:
    q = quat_normalize_xyzw(quaternion_xyzw)
    if q[3] < 0.0:
        q = (-q[0], -q[1], -q[2], -q[3])

    vector_norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2])
    if vector_norm <= 1e-12:
        return (2.0 * q[0], 2.0 * q[1], 2.0 * q[2])

    angle = 2.0 * atan2(vector_norm, q[3])
    scale = angle / vector_norm
    return (q[0] * scale, q[1] * scale, q[2] * scale)


def orientation_error_rotvec_xyzw(target_xyzw: QuaternionXYZW, current_xyzw: QuaternionXYZW) -> Vector3:
    q_err = quat_multiply_xyzw(quat_normalize_xyzw(target_xyzw), quat_inverse_xyzw(current_xyzw))
    return quat_log_rotvec_xyzw(q_err)
