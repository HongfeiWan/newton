from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import acos, sin, sqrt

QuaternionXYZW = tuple[float, float, float, float]
QuaternionWXYZ = tuple[float, float, float, float]
Vector3 = tuple[float, float, float]

WRIST_INDEX = 1
INDEX_PROXIMAL_INDEX = 7
MIDDLE_PROXIMAL_INDEX = 12
LITTLE_PROXIMAL_INDEX = 22


@dataclass(frozen=True)
class PalmPlaneOrientation:
    quaternion_xyzw: QuaternionXYZW
    palm_origin_xyz: Vector3
    palm_across_xyz: Vector3
    palm_forward_xyz: Vector3
    palm_normal_xyz: Vector3

    @property
    def quaternion_wxyz(self) -> QuaternionWXYZ:
        x, y, z, w = self.quaternion_xyzw
        return (w, x, y, z)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "quaternion_xyzw": [float(value) for value in self.quaternion_xyzw],
            "quaternion_wxyz": [float(value) for value in self.quaternion_wxyz],
            "palm_origin_xyz": [float(value) for value in self.palm_origin_xyz],
            "palm_across_xyz": [float(value) for value in self.palm_across_xyz],
            "palm_forward_xyz": [float(value) for value in self.palm_forward_xyz],
            "palm_normal_xyz": [float(value) for value in self.palm_normal_xyz],
        }


@dataclass(frozen=True)
class PalmPlaneCorrectionResult:
    corrected_quaternion_xyzw: QuaternionXYZW
    corrected_quaternion_wxyz: QuaternionWXYZ
    raw_to_palm_error_rad: float
    palm_plane: PalmPlaneOrientation
    blend_alpha: float

    def as_dict(self) -> dict[str, object]:
        return {
            "corrected_quaternion_xyzw": [float(value) for value in self.corrected_quaternion_xyzw],
            "corrected_quaternion_wxyz": [float(value) for value in self.corrected_quaternion_wxyz],
            "raw_to_palm_error_rad": float(self.raw_to_palm_error_rad),
            "blend_alpha": float(self.blend_alpha),
            "palm_plane": self.palm_plane.as_dict(),
        }


def xyzw_to_wxyz(quaternion_xyzw: QuaternionXYZW) -> QuaternionWXYZ:
    x, y, z, w = quaternion_xyzw
    return (w, x, y, z)


def wxyz_to_xyzw(quaternion_wxyz: QuaternionWXYZ) -> QuaternionXYZW:
    w, x, y, z = quaternion_wxyz
    return (x, y, z, w)


def quat_normalize_xyzw(quaternion_xyzw: QuaternionXYZW) -> QuaternionXYZW:
    norm = sqrt(sum(float(value) * float(value) for value in quaternion_xyzw))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(value) / norm for value in quaternion_xyzw)  # type: ignore[return-value]


def quat_dot_xyzw(lhs: QuaternionXYZW, rhs: QuaternionXYZW) -> float:
    return sum(float(lhs[index]) * float(rhs[index]) for index in range(4))


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


def _normalize_vec3(values_xyz: Sequence[float]) -> Vector3 | None:
    norm = sqrt(sum(float(value) * float(value) for value in values_xyz))
    if norm <= 1e-9:
        return None
    return tuple(float(value) / norm for value in values_xyz)  # type: ignore[return-value]


def _sub_vec3(lhs_xyz: Sequence[float], rhs_xyz: Sequence[float]) -> Vector3:
    return (
        float(lhs_xyz[0]) - float(rhs_xyz[0]),
        float(lhs_xyz[1]) - float(rhs_xyz[1]),
        float(lhs_xyz[2]) - float(rhs_xyz[2]),
    )


def _cross_vec3(lhs_xyz: Sequence[float], rhs_xyz: Sequence[float]) -> Vector3:
    return (
        float(lhs_xyz[1]) * float(rhs_xyz[2]) - float(lhs_xyz[2]) * float(rhs_xyz[1]),
        float(lhs_xyz[2]) * float(rhs_xyz[0]) - float(lhs_xyz[0]) * float(rhs_xyz[2]),
        float(lhs_xyz[0]) * float(rhs_xyz[1]) - float(lhs_xyz[1]) * float(rhs_xyz[0]),
    )


def _average_vec3(vectors_xyz: Sequence[Sequence[float]]) -> Vector3:
    count = float(len(vectors_xyz))
    return (
        sum(float(vector[0]) for vector in vectors_xyz) / count,
        sum(float(vector[1]) for vector in vectors_xyz) / count,
        sum(float(vector[2]) for vector in vectors_xyz) / count,
    )


def _matrix_to_quat_xyzw(matrix: tuple[Vector3, Vector3, Vector3]) -> QuaternionXYZW:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = sqrt(trace + 1.0) * 2.0
        return quat_normalize_xyzw(((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale))
    if m00 > m11 and m00 > m22:
        scale = sqrt(max(0.0, 1.0 + m00 - m11 - m22)) * 2.0
        return quat_normalize_xyzw((0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale))
    if m11 > m22:
        scale = sqrt(max(0.0, 1.0 + m11 - m00 - m22)) * 2.0
        return quat_normalize_xyzw(((m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale))
    scale = sqrt(max(0.0, 1.0 + m22 - m00 - m11)) * 2.0
    return quat_normalize_xyzw(((m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale))


def palm_plane_orientation_from_hand_debug(hand_debug: dict[str, object]) -> PalmPlaneOrientation | None:
    """Rebuild palm orientation from OpenXR/Quest hand debug joint data.

    ``hand_debug`` follows the AGIMani Teleop debug payload shape:
    ``joint_positions_xyz`` is a 26x3 list and ``joint_valid`` is a 26-value
    boolean list. The palm frame columns are across, forward, normal.
    """

    positions = hand_debug.get("joint_positions_xyz")
    valid = hand_debug.get("joint_valid")
    if not isinstance(positions, list) or not isinstance(valid, list):
        return None
    required_indices = (WRIST_INDEX, INDEX_PROXIMAL_INDEX, MIDDLE_PROXIMAL_INDEX, LITTLE_PROXIMAL_INDEX)
    if len(positions) <= max(required_indices) or len(valid) <= max(required_indices):
        return None
    if not all(bool(valid[index]) for index in required_indices):
        return None

    try:
        wrist = tuple(float(value) for value in positions[WRIST_INDEX])
        index_proximal = tuple(float(value) for value in positions[INDEX_PROXIMAL_INDEX])
        middle_proximal = tuple(float(value) for value in positions[MIDDLE_PROXIMAL_INDEX])
        little_proximal = tuple(float(value) for value in positions[LITTLE_PROXIMAL_INDEX])
    except (TypeError, ValueError):
        return None
    if not all(len(vector) == 3 for vector in (wrist, index_proximal, middle_proximal, little_proximal)):
        return None

    palm_forward = _normalize_vec3(_sub_vec3(middle_proximal, wrist))
    palm_across = _normalize_vec3(_sub_vec3(index_proximal, little_proximal))
    if palm_forward is None or palm_across is None:
        return None
    palm_normal = _normalize_vec3(_cross_vec3(palm_across, palm_forward))
    if palm_normal is None:
        return None
    palm_across = _normalize_vec3(_cross_vec3(palm_forward, palm_normal))
    if palm_across is None:
        return None

    matrix = (
        (palm_across[0], palm_forward[0], palm_normal[0]),
        (palm_across[1], palm_forward[1], palm_normal[1]),
        (palm_across[2], palm_forward[2], palm_normal[2]),
    )
    return PalmPlaneOrientation(
        quaternion_xyzw=_matrix_to_quat_xyzw(matrix),
        palm_origin_xyz=_average_vec3((wrist, index_proximal, middle_proximal, little_proximal)),
        palm_across_xyz=palm_across,
        palm_forward_xyz=palm_forward,
        palm_normal_xyz=palm_normal,
    )


def apply_palm_plane_wrist_orientation_correction(
    raw_wrist_quaternion_xyzw: QuaternionXYZW,
    hand_debug: dict[str, object] | None,
    *,
    blend_alpha: float,
) -> PalmPlaneCorrectionResult | None:
    """Blend a raw OpenXR wrist quaternion toward the reconstructed palm plane."""

    alpha = max(0.0, min(1.0, float(blend_alpha)))
    if alpha <= 0.0 or hand_debug is None:
        return None
    palm_plane = palm_plane_orientation_from_hand_debug(hand_debug)
    if palm_plane is None:
        return None

    raw_wrist_quaternion = quat_normalize_xyzw(raw_wrist_quaternion_xyzw)
    corrected = quat_slerp_xyzw(raw_wrist_quaternion, palm_plane.quaternion_xyzw, alpha)
    return PalmPlaneCorrectionResult(
        corrected_quaternion_xyzw=corrected,
        corrected_quaternion_wxyz=xyzw_to_wxyz(corrected),
        raw_to_palm_error_rad=quat_angle_between_xyzw(raw_wrist_quaternion, palm_plane.quaternion_xyzw),
        palm_plane=palm_plane,
        blend_alpha=alpha,
    )
