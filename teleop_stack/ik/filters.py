from __future__ import annotations

import math


def clamp_vector_norm(vector: tuple[float, ...], *, max_norm: float) -> tuple[float, ...]:
    safe_max_norm = max(0.0, float(max_norm))
    current_norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if safe_max_norm <= 0.0 or current_norm <= safe_max_norm or current_norm <= 1e-12:
        return tuple(float(value) for value in vector)
    scale = safe_max_norm / current_norm
    return tuple(float(value) * scale for value in vector)


def clamp_joint_step(
    joint_step_rad: tuple[float, ...],
    *,
    max_step_rad: float,
    max_velocity_rad_s: float,
    dt_s: float,
) -> tuple[float, ...]:
    safe_dt = max(0.0, float(dt_s))
    safe_max_step = max(0.0, float(max_step_rad))
    safe_velocity = max(0.0, float(max_velocity_rad_s))

    allowed_step = safe_max_step
    if safe_velocity > 0.0 and safe_dt > 0.0:
        allowed_step = min(allowed_step, safe_velocity * safe_dt) if allowed_step > 0.0 else safe_velocity * safe_dt
    if allowed_step <= 0.0:
        return tuple(0.0 for _ in joint_step_rad)

    return tuple(max(-allowed_step, min(allowed_step, float(value))) for value in joint_step_rad)


def clip_joint_positions(
    joint_positions_rad: tuple[float, ...],
    *,
    lower_limits_rad: tuple[float, ...] | None,
    upper_limits_rad: tuple[float, ...] | None,
) -> tuple[tuple[float, ...], bool]:
    dof = len(joint_positions_rad)
    if dof == 0:
        return (), False
    if (
        lower_limits_rad is None
        or upper_limits_rad is None
        or len(lower_limits_rad) != dof
        or len(upper_limits_rad) != dof
    ):
        return tuple(float(v) for v in joint_positions_rad), False

    clipped = False
    q_cmd: list[float] = []
    for index, q in enumerate(joint_positions_rad):
        bounded = max(float(lower_limits_rad[index]), min(float(upper_limits_rad[index]), float(q)))
        if abs(bounded - float(q)) > 1e-12:
            clipped = True
        q_cmd.append(bounded)
    return tuple(q_cmd), clipped
