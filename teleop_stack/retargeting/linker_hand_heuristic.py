from __future__ import annotations

from dataclasses import dataclass
from math import acos, asin, atan2, degrees

import numpy as np

from teleop_stack.models import NamedJointValues
from teleop_stack.retargeting.hand_config import DexHandModelSpec, load_linker_l10_right_hand_spec


try:
    from isaacteleop.retargeting_engine.tensor_types.indices import HandJointIndex
except ModuleNotFoundError:

    class HandJointIndex:
        PALM = 0
        WRIST = 1
        THUMB_METACARPAL = 2
        THUMB_PROXIMAL = 3
        THUMB_DISTAL = 4
        THUMB_TIP = 5
        INDEX_METACARPAL = 6
        INDEX_PROXIMAL = 7
        INDEX_INTERMEDIATE = 8
        INDEX_DISTAL = 9
        INDEX_TIP = 10
        MIDDLE_METACARPAL = 11
        MIDDLE_PROXIMAL = 12
        MIDDLE_INTERMEDIATE = 13
        MIDDLE_DISTAL = 14
        MIDDLE_TIP = 15
        RING_METACARPAL = 16
        RING_PROXIMAL = 17
        RING_INTERMEDIATE = 18
        RING_DISTAL = 19
        RING_TIP = 20
        LITTLE_METACARPAL = 21
        LITTLE_PROXIMAL = 22
        LITTLE_INTERMEDIATE = 23
        LITTLE_DISTAL = 24
        LITTLE_TIP = 25


@dataclass(frozen=True)
class LinkerHandHeuristicConfig:
    side: str = "right"


def _safe_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _angle_between(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_unit = _safe_normalize(lhs)
    rhs_unit = _safe_normalize(rhs)
    if not np.any(lhs_unit) or not np.any(rhs_unit):
        return 0.0
    cosine = float(np.clip(np.dot(lhs_unit, rhs_unit), -1.0, 1.0))
    return float(np.arccos(cosine))


def _project_onto_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vector - float(np.dot(vector, normal)) * normal


def _safe_quat_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    return quat / norm


def _quat_conjugate_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quat = _safe_quat_xyzw(quaternion_xyzw)
    return np.asarray((-quat[0], -quat[1], -quat[2], quat[3]), dtype=np.float64)


def _quat_multiply_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = _safe_quat_xyzw(lhs)
    rhs = _safe_quat_xyzw(rhs)
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return _safe_quat_xyzw(
        np.asarray(
            (
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ),
            dtype=np.float64,
        )
    )


def _quat_angle_rad_xyzw(quaternion_xyzw: np.ndarray) -> float:
    quat = _safe_quat_xyzw(quaternion_xyzw)
    cosine = float(np.clip(abs(quat[3]), -1.0, 1.0))
    return float(2.0 * acos(cosine))


def _quat_to_matrix_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _safe_quat_xyzw(quaternion_xyzw)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _euler_xyz_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    y = asin(float(np.clip(matrix[0, 2], -1.0, 1.0)))
    cos_y = float(np.cos(y))
    if abs(cos_y) > 1e-8:
        x = atan2(float(-matrix[1, 2]), float(matrix[2, 2]))
        z = atan2(float(-matrix[0, 1]), float(matrix[0, 0]))
    else:
        x = atan2(float(matrix[2, 1]), float(matrix[1, 1]))
        z = 0.0
    return (float(x), float(y), float(z))


def _segment_angle(points: np.ndarray, a: int, b: int, c: int) -> float:
    return _angle_between(points[b] - points[a], points[c] - points[b])


def _finger_curl_ratio(points: np.ndarray, indices: tuple[int, int, int, int, int]) -> float:
    angles = (
        _segment_angle(points, indices[0], indices[1], indices[2]),
        _segment_angle(points, indices[1], indices[2], indices[3]),
        _segment_angle(points, indices[2], indices[3], indices[4]),
    )
    weighted_angle = sum(weight * angle for weight, angle in zip(_NON_THUMB_CURL_ANGLE_WEIGHTS, angles, strict=True))
    return _clamp01(weighted_angle / 1.35)


def _thumb_curl_ratio(points: np.ndarray, indices: tuple[int, int, int, int]) -> float:
    angles = (
        _segment_angle(points, indices[0], indices[1], indices[2]),
        _segment_angle(points, indices[1], indices[2], indices[3]),
    )
    weighted_angle = 0.65 * angles[0] + 0.35 * angles[1]
    return _clamp01(weighted_angle / 1.10)


def _spread_ratio(direction_a: np.ndarray, direction_b: np.ndarray, *, max_angle_rad: float = 0.50) -> float:
    return _clamp01(_angle_between(direction_a, direction_b) / max_angle_rad)


_PINCH_PREPARE_DISTANCE_M = 0.015
_PINCH_COMPLETE_DISTANCE_M = 0.007
_NON_THUMB_CURL_OPEN_DEADZONE = 0.0
_NON_THUMB_CURL_ANGLE_WEIGHTS = (0.450625, 0.432410, 0.116965)
_NON_THUMB_CURL_OPEN_BASELINE = {
    "index_mcp_pitch": 0.081773,
    "middle_mcp_pitch": 0.0,
    "ring_mcp_pitch": 0.041509,
    "pinky_mcp_pitch": 0.070462,
}
_NON_THUMB_MCP_PITCH_GAIN = {
    "index_mcp_pitch": 1.001076,
    "middle_mcp_pitch": 0.990150,
    "ring_mcp_pitch": 0.992641,
    "pinky_mcp_pitch": 1.002966,
}


def _apply_deadzone_ratio(value: float, deadzone: float) -> float:
    deadzone = _clamp01(deadzone)
    if value <= deadzone:
        return 0.0
    return _clamp01((value - deadzone) / max(1.0 - deadzone, 1e-6))


def _remove_open_baseline_ratio(value: float, baseline: float) -> float:
    baseline = _clamp01(baseline)
    if value <= baseline:
        return 0.0
    return _clamp01((value - baseline) / max(1.0 - baseline, 1e-6))


def _pose_from_joint_ratios(spec: DexHandModelSpec, joint_ratio_map: dict[str, float]) -> NamedJointValues:
    joint_positions = []
    for joint_name, (lower, upper) in zip(spec.active_joint_names, spec.active_joint_limits, strict=True):
        ratio = _clamp01(joint_ratio_map[joint_name])
        joint_positions.append(lower + ratio * (upper - lower))
    return NamedJointValues(
        joint_names=spec.active_joint_names,
        joint_positions=tuple(joint_positions),
    )


def _orientation_thumb_ratios(
    joint_orientations_xyzw: np.ndarray | None,
    joint_valid: np.ndarray | None,
    *,
    palm_across: np.ndarray,
    palm_forward: np.ndarray,
    palm_normal: np.ndarray,
) -> dict[str, float]:
    if joint_orientations_xyzw is None:
        return {}

    orientations = np.asarray(joint_orientations_xyzw, dtype=np.float64)
    if orientations.shape != (26, 4):
        return {}

    valid = np.ones((26,), dtype=np.uint8) if joint_valid is None else np.asarray(joint_valid, dtype=np.uint8)
    if valid.shape != (26,):
        return {}

    required_indices = (
        HandJointIndex.THUMB_METACARPAL,
        HandJointIndex.THUMB_PROXIMAL,
        HandJointIndex.THUMB_DISTAL,
    )
    if not all(bool(valid[int(index)]) for index in required_indices):
        return {}

    thumb_metacarpal_quat = _safe_quat_xyzw(orientations[HandJointIndex.THUMB_METACARPAL])
    thumb_proximal_quat = _safe_quat_xyzw(orientations[HandJointIndex.THUMB_PROXIMAL])
    thumb_distal_quat = _safe_quat_xyzw(orientations[HandJointIndex.THUMB_DISTAL])

    palm_matrix = np.column_stack((palm_across, palm_forward, palm_normal))
    if abs(float(np.linalg.det(palm_matrix))) < 1e-6:
        return {}

    thumb_metacarpal_in_palm = palm_matrix.T @ _quat_to_matrix_xyzw(thumb_metacarpal_quat)
    thumb_roll_rad, _, _ = _euler_xyz_from_matrix(thumb_metacarpal_in_palm)

    metacarpal_to_proximal = _quat_multiply_xyzw(
        _quat_conjugate_xyzw(thumb_metacarpal_quat),
        thumb_proximal_quat,
    )
    proximal_to_distal = _quat_multiply_xyzw(
        _quat_conjugate_xyzw(thumb_proximal_quat),
        thumb_distal_quat,
    )
    pitch_angle_rad = 0.65 * _quat_angle_rad_xyzw(metacarpal_to_proximal) + 0.35 * _quat_angle_rad_xyzw(
        proximal_to_distal
    )

    # OpenXR joint frames are not L10 joint frames. These ranges are deliberately broad and
    # monotonic, using the latest real capture as a sanity check rather than as hard calibration.
    return {
        "thumb_cmc_roll": _clamp01((degrees(thumb_roll_rad) - 105.0) / 80.0),
        "thumb_cmc_pitch": _clamp01((degrees(pitch_angle_rad) - 5.0) / 45.0),
    }


def retarget_openxr_joint_positions_to_linker_l10_right(
    joint_positions_xyz: np.ndarray,
    *,
    joint_orientations_xyzw: np.ndarray | None = None,
    joint_valid: np.ndarray | None = None,
    spec: DexHandModelSpec | None = None,
) -> NamedJointValues:
    hand_spec = spec or load_linker_l10_right_hand_spec()
    points = np.asarray(joint_positions_xyz, dtype=np.float64)
    if points.shape != (26, 3):
        raise ValueError(f"Expected OpenXR hand positions with shape (26, 3), got {points.shape}")

    wrist = points[HandJointIndex.WRIST]
    index_base = points[HandJointIndex.INDEX_PROXIMAL]
    middle_base = points[HandJointIndex.MIDDLE_PROXIMAL]
    ring_base = points[HandJointIndex.RING_PROXIMAL]
    pinky_base = points[HandJointIndex.LITTLE_PROXIMAL]
    thumb_base = points[HandJointIndex.THUMB_PROXIMAL]

    palm_forward = _safe_normalize(middle_base - wrist)
    palm_across = _safe_normalize(index_base - pinky_base)
    palm_normal = _safe_normalize(np.cross(palm_across, palm_forward))
    palm_across = _safe_normalize(np.cross(palm_forward, palm_normal))

    if not np.any(palm_forward) or not np.any(palm_across) or not np.any(palm_normal):
        return hand_spec.default_open_pose

    index_dir = _safe_normalize(_project_onto_plane(points[HandJointIndex.INDEX_INTERMEDIATE] - index_base, palm_normal))
    middle_dir = _safe_normalize(
        _project_onto_plane(points[HandJointIndex.MIDDLE_INTERMEDIATE] - middle_base, palm_normal)
    )
    ring_dir = _safe_normalize(_project_onto_plane(points[HandJointIndex.RING_INTERMEDIATE] - ring_base, palm_normal))
    pinky_dir = _safe_normalize(_project_onto_plane(points[HandJointIndex.LITTLE_INTERMEDIATE] - pinky_base, palm_normal))
    thumb_dir = _safe_normalize(_project_onto_plane(thumb_base - wrist, palm_normal))

    index_curl = _finger_curl_ratio(
        points,
        (
            HandJointIndex.INDEX_METACARPAL,
            HandJointIndex.INDEX_PROXIMAL,
            HandJointIndex.INDEX_INTERMEDIATE,
            HandJointIndex.INDEX_DISTAL,
            HandJointIndex.INDEX_TIP,
        ),
    )
    index_curl = _apply_deadzone_ratio(index_curl, _NON_THUMB_CURL_OPEN_DEADZONE)
    index_curl = _remove_open_baseline_ratio(index_curl, _NON_THUMB_CURL_OPEN_BASELINE["index_mcp_pitch"])
    middle_curl = _finger_curl_ratio(
        points,
        (
            HandJointIndex.MIDDLE_METACARPAL,
            HandJointIndex.MIDDLE_PROXIMAL,
            HandJointIndex.MIDDLE_INTERMEDIATE,
            HandJointIndex.MIDDLE_DISTAL,
            HandJointIndex.MIDDLE_TIP,
        ),
    )
    middle_curl = _apply_deadzone_ratio(middle_curl, _NON_THUMB_CURL_OPEN_DEADZONE)
    middle_curl = _remove_open_baseline_ratio(middle_curl, _NON_THUMB_CURL_OPEN_BASELINE["middle_mcp_pitch"])
    ring_curl = _finger_curl_ratio(
        points,
        (
            HandJointIndex.RING_METACARPAL,
            HandJointIndex.RING_PROXIMAL,
            HandJointIndex.RING_INTERMEDIATE,
            HandJointIndex.RING_DISTAL,
            HandJointIndex.RING_TIP,
        ),
    )
    ring_curl = _apply_deadzone_ratio(ring_curl, _NON_THUMB_CURL_OPEN_DEADZONE)
    ring_curl = _remove_open_baseline_ratio(ring_curl, _NON_THUMB_CURL_OPEN_BASELINE["ring_mcp_pitch"])
    pinky_curl = _finger_curl_ratio(
        points,
        (
            HandJointIndex.LITTLE_METACARPAL,
            HandJointIndex.LITTLE_PROXIMAL,
            HandJointIndex.LITTLE_INTERMEDIATE,
            HandJointIndex.LITTLE_DISTAL,
            HandJointIndex.LITTLE_TIP,
        ),
    )
    pinky_curl = _apply_deadzone_ratio(pinky_curl, _NON_THUMB_CURL_OPEN_DEADZONE)
    pinky_curl = _remove_open_baseline_ratio(pinky_curl, _NON_THUMB_CURL_OPEN_BASELINE["pinky_mcp_pitch"])
    thumb_curl = _thumb_curl_ratio(
        points,
        (
            HandJointIndex.THUMB_METACARPAL,
            HandJointIndex.THUMB_PROXIMAL,
            HandJointIndex.THUMB_DISTAL,
            HandJointIndex.THUMB_TIP,
        ),
    )

    index_spread = _spread_ratio(index_dir, middle_dir)
    ring_spread = _spread_ratio(ring_dir, middle_dir)
    pinky_spread = _spread_ratio(pinky_dir, ring_dir, max_angle_rad=0.55)

    thumb_tip = points[HandJointIndex.THUMB_TIP]
    index_tip = points[HandJointIndex.INDEX_TIP]
    thumb_index_distance = float(np.linalg.norm(thumb_tip - index_tip))
    pinch_ratio = _clamp01(
        (_PINCH_PREPARE_DISTANCE_M - thumb_index_distance)
        / max(_PINCH_PREPARE_DISTANCE_M - _PINCH_COMPLETE_DISTANCE_M, 1e-6)
    )

    thumb_forward_ratio = _clamp01((float(np.dot(thumb_dir, palm_forward)) + 0.35) / 0.90)
    thumb_lateral_ratio = _clamp01(abs(float(np.dot(thumb_dir, palm_across))))
    orientation_thumb_ratios = _orientation_thumb_ratios(
        joint_orientations_xyzw,
        joint_valid,
        palm_across=palm_across,
        palm_forward=palm_forward,
        palm_normal=palm_normal,
    )

    thumb_roll_ratio = orientation_thumb_ratios.get(
        "thumb_cmc_roll",
        _clamp01(0.10 + 0.18 * max(thumb_lateral_ratio, pinch_ratio)),
    )
    orientation_thumb_pitch = orientation_thumb_ratios.get("thumb_cmc_pitch")
    thumb_pitch_ratio = (
        _clamp01(0.70 * orientation_thumb_pitch + 0.30 * thumb_curl)
        if orientation_thumb_pitch is not None
        else _clamp01(0.12 + 0.70 * thumb_curl)
    )
    thumb_yaw_ratio = _clamp01(0.22 + 0.25 * thumb_forward_ratio + 0.30 * pinch_ratio)
    if orientation_thumb_ratios:
        thumb_yaw_ratio = min(thumb_yaw_ratio, 0.72)

    joint_ratio_map = {
        "thumb_cmc_roll": thumb_roll_ratio,
        "thumb_cmc_yaw": thumb_yaw_ratio,
        "thumb_cmc_pitch": thumb_pitch_ratio,
        "index_mcp_roll": _clamp01(0.95 * index_spread),
        "index_mcp_pitch": _clamp01(_NON_THUMB_MCP_PITCH_GAIN["index_mcp_pitch"] * index_curl),
        "middle_mcp_pitch": _clamp01(_NON_THUMB_MCP_PITCH_GAIN["middle_mcp_pitch"] * middle_curl),
        "ring_mcp_roll": _clamp01(0.95 * ring_spread),
        "ring_mcp_pitch": _clamp01(_NON_THUMB_MCP_PITCH_GAIN["ring_mcp_pitch"] * ring_curl),
        "pinky_mcp_roll": _clamp01(1.00 * pinky_spread),
        "pinky_mcp_pitch": _clamp01(_NON_THUMB_MCP_PITCH_GAIN["pinky_mcp_pitch"] * pinky_curl),
    }

    return _pose_from_joint_ratios(hand_spec, joint_ratio_map)
