# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared camera preprocessing for GR00T data and Newton policy inputs."""

from __future__ import annotations

import numpy as np

EGO_SOURCE_SIZE = (1280, 800)
EGO_INPUT_SIZE = (320, 180)
EGO_ROI_ZOOM = 2.0
EGO_ROI_CENTER = (0.5, 0.65)


def roi_crop_rect(
    width: int,
    height: int,
    *,
    zoom: float,
    center_x: float,
    center_y: float,
) -> tuple[int, int, int, int]:
    """Return the node0-compatible integer ROI rectangle."""
    if zoom <= 1.0:
        return 0, 0, int(width), int(height)
    center_x = min(max(float(center_x), 0.0), 1.0)
    center_y = min(max(float(center_y), 0.0), 1.0)
    crop_width = max(1, min(int(width), int(round(int(width) / float(zoom)))))
    crop_height = max(1, min(int(height), int(round(int(height) / float(zoom)))))
    crop_x = int(round(center_x * int(width) - crop_width / 2.0))
    crop_y = int(round(center_y * int(height) - crop_height / 2.0))
    crop_x = min(max(0, crop_x), max(0, int(width) - crop_width))
    crop_y = min(max(0, crop_y), max(0, int(height) - crop_height))
    return crop_x, crop_y, crop_width, crop_height


def preprocess_ego_rgb(
    image: np.ndarray,
    *,
    zoom: float = EGO_ROI_ZOOM,
    center_x: float = EGO_ROI_CENTER[0],
    center_y: float = EGO_ROI_CENTER[1],
    output_size: tuple[int, int] = EGO_INPUT_SIZE,
) -> np.ndarray:
    """Apply the real node0 ROI and frame-tap resize to an RGB HWC frame."""
    import cv2  # noqa: PLC0415

    source = np.asarray(image, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"ego_view source must be RGB HWC, got {source.shape}")
    source_height, source_width, _ = source.shape
    crop_x, crop_y, crop_width, crop_height = roi_crop_rect(
        source_width,
        source_height,
        zoom=zoom,
        center_x=center_x,
        center_y=center_y,
    )
    cropped = source[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    zoomed = cv2.resize(cropped, (source_width, source_height), interpolation=cv2.INTER_LINEAR)
    output = cv2.resize(zoomed, tuple(int(value) for value in output_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(output, dtype=np.uint8)
