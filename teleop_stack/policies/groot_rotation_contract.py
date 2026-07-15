# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rotation-convention adapters for legacy Nero GR00T checkpoints."""

from __future__ import annotations

from typing import Final

import numpy as np

GROOT_ROW_MAJOR_FIRST_TWO_ROWS: Final = "groot_row_major_first_two_rows"
LEGACY_FIRST_TWO_COLUMNS_COLUMN_MAJOR: Final = "rotation_matrix_first_two_columns_column_major"
ROT6D_CONVERSION_ALGORITHM: Final = "gram_schmidt_cross_v1"

_MIN_AXIS_NORM = 1.0e-8


def _rotation_matrix_from_two_axes(rot6d: np.ndarray, *, axes_are_rows: bool) -> np.ndarray:
    values = np.asarray(rot6d)
    if values.ndim < 1 or values.shape[-1] != 6:
        raise ValueError(f"rot6d must have shape [..., 6], got {values.shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"rot6d must be numeric, got dtype={values.dtype}")

    work = values.astype(np.float64, copy=False)
    if not np.isfinite(work).all():
        raise ValueError("rot6d contains non-finite values")

    axis0 = work[..., :3]
    raw_axis1 = work[..., 3:6]
    norm0 = np.linalg.norm(axis0, axis=-1, keepdims=True)
    if np.any(norm0 <= _MIN_AXIS_NORM):
        raise ValueError("rot6d first axis is degenerate")
    axis0 = axis0 / norm0

    axis1 = raw_axis1 - np.sum(axis0 * raw_axis1, axis=-1, keepdims=True) * axis0
    norm1 = np.linalg.norm(axis1, axis=-1, keepdims=True)
    if np.any(norm1 <= _MIN_AXIS_NORM):
        raise ValueError("rot6d axes are degenerate or nearly parallel")
    axis1 = axis1 / norm1
    axis2 = np.cross(axis0, axis1)

    axis = -2 if axes_are_rows else -1
    return np.stack((axis0, axis1, axis2), axis=axis)


def row_first_rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert row-first rot6d values to proper rotation matrices.

    Args:
        rot6d: Values ordered as ``[r00, r01, r02, r10, r11, r12]``.

    Returns:
        Rotation matrices with shape ``[..., 3, 3]``.
    """

    return _rotation_matrix_from_two_axes(rot6d, axes_are_rows=True)


def legacy_rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert checkpoint-native rot6d values to proper rotation matrices.

    Args:
        rot6d: Values ordered as ``[r00, r10, r20, r01, r11, r21]``.

    Returns:
        Rotation matrices with shape ``[..., 3, 3]``.
    """

    return _rotation_matrix_from_two_axes(rot6d, axes_are_rows=False)


def row_first_to_legacy_rot6d(rot6d: np.ndarray) -> np.ndarray:
    """Convert canonical row-first rot6d into the legacy checkpoint contract."""

    matrix = row_first_rot6d_to_matrix(rot6d)
    result = np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)
    return result.astype(np.float32, copy=False)


def legacy_to_row_first_rot6d(rot6d: np.ndarray) -> np.ndarray:
    """Convert legacy checkpoint rot6d into the canonical row-first contract."""

    matrix = legacy_rot6d_to_matrix(rot6d)
    result = matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)
    return result.astype(np.float32, copy=False)


def convert_eef_9d_rotation(
    eef_9d: np.ndarray,
    *,
    source: str,
    target: str,
) -> np.ndarray:
    """Convert only the rot6d portion of an EEF pose array.

    Args:
        eef_9d: EEF values ordered as translation ``[3]`` plus rot6d ``[6]``.
        source: Source rotation convention.
        target: Target rotation convention.

    Returns:
        A converted float32 copy with the same shape.
    """

    values = np.asarray(eef_9d, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != 9:
        raise ValueError(f"eef_9d must have shape [..., 9], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("eef_9d contains non-finite values")
    output = values.copy()
    if source == target:
        return output
    if source == GROOT_ROW_MAJOR_FIRST_TWO_ROWS and target == LEGACY_FIRST_TWO_COLUMNS_COLUMN_MAJOR:
        output[..., 3:9] = row_first_to_legacy_rot6d(values[..., 3:9])
        return output
    if source == LEGACY_FIRST_TWO_COLUMNS_COLUMN_MAJOR and target == GROOT_ROW_MAJOR_FIRST_TWO_ROWS:
        output[..., 3:9] = legacy_to_row_first_rot6d(values[..., 3:9])
        return output
    raise ValueError(f"unsupported rot6d conversion: {source!r} -> {target!r}")
