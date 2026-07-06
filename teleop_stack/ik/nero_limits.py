from __future__ import annotations


NERO_JOINT_LOWER_LIMITS_RAD: tuple[float, ...] = (
    -2.705261,
    -1.745330,
    -2.757621,
    -1.012291,
    -2.757621,
    -0.733039,
    -1.570797,
)

NERO_JOINT_UPPER_LIMITS_RAD: tuple[float, ...] = (
    2.705261,
    1.745330,
    2.757621,
    2.146755,
    2.757621,
    0.959932,
    1.570797,
)


def nero_effective_joint_lower_limits(
    *,
    joint4_limit_enabled: bool,
    joint4_lower_rad: float,
) -> tuple[float, ...]:
    values = list(NERO_JOINT_LOWER_LIMITS_RAD)
    if joint4_limit_enabled:
        values[3] = max(float(values[3]), float(joint4_lower_rad))
    return tuple(values)


def nero_effective_joint_upper_limits(
    *,
    joint4_limit_enabled: bool,
    joint4_upper_rad: float,
) -> tuple[float, ...]:
    values = list(NERO_JOINT_UPPER_LIMITS_RAD)
    if joint4_limit_enabled:
        values[3] = min(float(values[3]), float(joint4_upper_rad))
    return tuple(values)
