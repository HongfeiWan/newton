from __future__ import annotations


def compute_posture_bias_step(
    joint_positions_rad: tuple[float, ...],
    *,
    lower_limits_rad: tuple[float, ...] | None,
    upper_limits_rad: tuple[float, ...] | None,
    neutral_positions_rad: tuple[float, ...] | None,
    posture_gain: float,
    joint_limit_gain: float,
    soft_margin_rad: float,
) -> tuple[tuple[float, ...], float | None]:
    dof = len(joint_positions_rad)
    if dof == 0:
        return (), None

    lower = _fill_limit_tuple(lower_limits_rad, dof, fallback=-3.0)
    upper = _fill_limit_tuple(upper_limits_rad, dof, fallback=3.0)
    neutral = _fill_neutral_tuple(neutral_positions_rad, lower, upper)

    bias_step: list[float] = []
    limit_margin_min: float | None = None
    safe_soft_margin = max(1e-6, float(soft_margin_rad))
    for index, q in enumerate(joint_positions_rad):
        q_value = float(q)
        lower_margin = q_value - lower[index]
        upper_margin = upper[index] - q_value
        min_margin = min(lower_margin, upper_margin)
        limit_margin_min = min_margin if limit_margin_min is None else min(limit_margin_min, min_margin)

        bias = float(posture_gain) * (neutral[index] - q_value)
        if lower_margin < safe_soft_margin:
            bias += float(joint_limit_gain) * (safe_soft_margin - lower_margin) / safe_soft_margin
        if upper_margin < safe_soft_margin:
            bias -= float(joint_limit_gain) * (safe_soft_margin - upper_margin) / safe_soft_margin
        bias_step.append(bias)

    return tuple(bias_step), limit_margin_min


def _fill_limit_tuple(values: tuple[float, ...] | None, dof: int, *, fallback: float) -> tuple[float, ...]:
    if values is None or len(values) != dof:
        return tuple(float(fallback) for _ in range(dof))
    return tuple(float(v) for v in values)


def _fill_neutral_tuple(
    neutral_positions_rad: tuple[float, ...] | None,
    lower_limits_rad: tuple[float, ...],
    upper_limits_rad: tuple[float, ...],
) -> tuple[float, ...]:
    dof = len(lower_limits_rad)
    if neutral_positions_rad is not None and len(neutral_positions_rad) == dof:
        return tuple(float(v) for v in neutral_positions_rad)
    return tuple(
        0.5 * (float(lower_limits_rad[index]) + float(upper_limits_rad[index]))
        for index in range(dof)
    )
