from __future__ import annotations

import math

from teleop_stack.ik.differential_ik import PositionJacobian


def singular_values(jacobian: PositionJacobian) -> tuple[float, float, float]:
    jjt = _task_gram_matrix(jacobian)
    eigenvalues = _jacobi_eigenvalues_symmetric_3x3(jjt)
    return tuple(sorted((math.sqrt(max(0.0, value)) for value in eigenvalues), reverse=True))


def min_singular_value(jacobian: PositionJacobian) -> float:
    values = singular_values(jacobian)
    return float(values[-1]) if values else 0.0


def normalized_singularity_metric(jacobian: PositionJacobian) -> float:
    values = singular_values(jacobian)
    if not values:
        return 0.0
    sigma_max = float(values[0])
    sigma_min = float(values[-1])
    if sigma_max <= 1e-12:
        return 0.0
    return sigma_min / sigma_max


def damping_scale_from_metric(
    metric: float,
    *,
    soft_threshold: float,
    hard_threshold: float,
    max_scale: float,
) -> float:
    safe_soft = max(float(soft_threshold), 0.0)
    safe_hard = max(0.0, min(float(hard_threshold), safe_soft))
    safe_max = max(1.0, float(max_scale))
    if metric >= safe_soft:
        return 1.0
    if metric <= safe_hard:
        return safe_max
    span = max(1e-9, safe_soft - safe_hard)
    alpha = (safe_soft - float(metric)) / span
    return 1.0 + alpha * (safe_max - 1.0)


def _task_gram_matrix(jacobian: PositionJacobian) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            sum(
                float(jacobian[row_idx][col_idx]) * float(jacobian[other_row_idx][col_idx])
                for col_idx in range(len(jacobian[0]))
            )
            for other_row_idx in range(3)
        )
        for row_idx in range(3)
    )


def _jacobi_eigenvalues_symmetric_3x3(matrix: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float]:
    data = [[float(matrix[row][col]) for col in range(3)] for row in range(3)]
    for _ in range(16):
        p, q, off_diag = _largest_off_diagonal(data)
        if off_diag <= 1e-12:
            break

        app = data[p][p]
        aqq = data[q][q]
        apq = data[p][q]
        tau = (aqq - app) / (2.0 * apq)
        t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        for k in range(3):
            if k in {p, q}:
                continue
            aik = data[p][k]
            aqk = data[q][k]
            data[p][k] = c * aik - s * aqk
            data[k][p] = data[p][k]
            data[q][k] = s * aik + c * aqk
            data[k][q] = data[q][k]

        data[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        data[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        data[p][q] = 0.0
        data[q][p] = 0.0

    return (data[0][0], data[1][1], data[2][2])


def _largest_off_diagonal(matrix: list[list[float]]) -> tuple[int, int, float]:
    candidates = (
        (0, 1, abs(matrix[0][1])),
        (0, 2, abs(matrix[0][2])),
        (1, 2, abs(matrix[1][2])),
    )
    return max(candidates, key=lambda item: item[2])
