from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import Any, Literal

from teleop_stack.ik.so3 import (
    QuaternionXYZW,
    quat_angle_between_xyzw,
    quat_inverse_xyzw,
    quat_log_rotvec_xyzw,
    quat_multiply_xyzw,
    quat_normalize_xyzw,
)

AxisName = Literal["x", "y", "z"]
Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
AxisMap = tuple[str, str, str]


IDENTITY_QUAT_XYZW: QuaternionXYZW = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class FrameAxes:
    """Basis vectors of a frame expressed in the parent frame."""

    x: Vector3
    y: Vector3
    z: Vector3

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "x": [float(value) for value in self.x],
            "y": [float(value) for value in self.y],
            "z": [float(value) for value in self.z],
        }


@dataclass(frozen=True)
class HandAnatomicalFrame:
    """Hand-derived frame axes expressed in the OpenXR/stage parent frame."""

    origin_xyz: Vector3
    axes: FrameAxes
    quaternion_xyzw: QuaternionXYZW
    handedness_det: float
    thumb_alignment: float
    legacy_palm_normal_alignment: float | None
    construction: str = "wrist_middle_thumb"
    raw_axes: FrameAxes | None = None
    axis_adapter: dict[str, str] | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "origin_xyz": [float(value) for value in self.origin_xyz],
            "axes": self.axes.as_dict(),
            "quaternion_xyzw": [float(value) for value in self.quaternion_xyzw],
            "handedness_det": float(self.handedness_det),
            "thumb_alignment": float(self.thumb_alignment),
            "legacy_palm_normal_alignment": (
                float(self.legacy_palm_normal_alignment) if self.legacy_palm_normal_alignment is not None else None
            ),
            "construction": self.construction,
        }
        if self.raw_axes is not None:
            payload["raw_axes"] = self.raw_axes.as_dict()
        if self.axis_adapter is not None:
            payload["axis_adapter"] = dict(self.axis_adapter)
        return payload


@dataclass(frozen=True)
class CanonicalAxisCase:
    name: str
    source_axis: AxisName
    quaternion_xyzw: QuaternionXYZW
    angle_rad: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_axis": self.source_axis,
            "quaternion_xyzw": [float(value) for value in self.quaternion_xyzw],
            "angle_rad": float(self.angle_rad),
        }


def quat_xyzw_to_matrix(quaternion_xyzw: QuaternionXYZW) -> Matrix3:
    x, y, z, w = quat_normalize_xyzw(quaternion_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def quat_xyzw_to_axes(quaternion_xyzw: QuaternionXYZW) -> FrameAxes:
    matrix = quat_xyzw_to_matrix(quaternion_xyzw)
    return FrameAxes(
        x=(matrix[0][0], matrix[1][0], matrix[2][0]),
        y=(matrix[0][1], matrix[1][1], matrix[2][1]),
        z=(matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def matrix_to_quat_xyzw(matrix: Matrix3) -> QuaternionXYZW:
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


def axis_rotation_xyzw(axis: AxisName, angle_rad: float) -> QuaternionXYZW:
    half = 0.5 * float(angle_rad)
    scale = sin(half)
    if axis == "x":
        return (scale, 0.0, 0.0, cos(half))
    if axis == "y":
        return (0.0, scale, 0.0, cos(half))
    if axis == "z":
        return (0.0, 0.0, scale, cos(half))
    raise ValueError(f"unsupported axis: {axis!r}")


def canonical_axis_cases(angle_rad: float) -> tuple[CanonicalAxisCase, ...]:
    return (
        CanonicalAxisCase("identity", "x", IDENTITY_QUAT_XYZW, 0.0),
        CanonicalAxisCase("+roll_x", "x", axis_rotation_xyzw("x", angle_rad), angle_rad),
        CanonicalAxisCase("+pitch_y", "y", axis_rotation_xyzw("y", angle_rad), angle_rad),
        CanonicalAxisCase("+yaw_z", "z", axis_rotation_xyzw("z", angle_rad), angle_rad),
    )


def mapped_source_axis(axis_map: AxisMap, source_axis: AxisName) -> Vector3:
    source_index = {"x": 0, "y": 1, "z": 2}[source_axis]
    values: list[float] = []
    for token in axis_map:
        spec = token.strip().lower()
        sign = -1.0 if spec.startswith("-") else 1.0
        axis = spec[1:] if spec.startswith(("-", "+")) else spec
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"unsupported axis map token: {token!r}")
        values.append(sign if {"x": 0, "y": 1, "z": 2}[axis] == source_index else 0.0)
    return (values[0], values[1], values[2])


def vector_sub(lhs: Vector3, rhs: Vector3) -> Vector3:
    return tuple(float(lhs[index]) - float(rhs[index]) for index in range(3))  # type: ignore[return-value]


def vector_norm(vector: Vector3) -> float:
    return sqrt(sum(float(value) * float(value) for value in vector))


def vector_dot(lhs: Vector3, rhs: Vector3) -> float:
    return sum(float(lhs[index]) * float(rhs[index]) for index in range(3))


def vector_cross(lhs: Vector3, rhs: Vector3) -> Vector3:
    return (
        float(lhs[1]) * float(rhs[2]) - float(lhs[2]) * float(rhs[1]),
        float(lhs[2]) * float(rhs[0]) - float(lhs[0]) * float(rhs[2]),
        float(lhs[0]) * float(rhs[1]) - float(lhs[1]) * float(rhs[0]),
    )


def vector_scale(vector: Vector3, scale: float) -> Vector3:
    return tuple(float(scale) * float(value) for value in vector)  # type: ignore[return-value]


def vector_normalize(vector: Vector3) -> Vector3:
    norm = vector_norm(vector)
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0)
    return tuple(float(value) / norm for value in vector)  # type: ignore[return-value]


def vector_project_orthogonal(vector: Vector3, unit_axis: Vector3) -> Vector3:
    return vector_sub(vector, vector_scale(unit_axis, vector_dot(vector, unit_axis)))


def matrix_from_axes(axes: FrameAxes) -> Matrix3:
    return (
        (axes.x[0], axes.y[0], axes.z[0]),
        (axes.x[1], axes.y[1], axes.z[1]),
        (axes.x[2], axes.y[2], axes.z[2]),
    )


def matrix_det(matrix: Matrix3) -> float:
    a, b, c = matrix
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def _joint_index_by_name(hand_debug: dict[str, Any]) -> dict[str, int]:
    names = hand_debug.get("joint_names")
    if not isinstance(names, list):
        return {}
    return {str(name): index for index, name in enumerate(names)}


def _hand_debug_point(hand_debug: dict[str, Any], name: str) -> Vector3 | None:
    index_by_name = _joint_index_by_name(hand_debug)
    index = index_by_name.get(name)
    positions = hand_debug.get("joint_positions_xyz")
    valid = hand_debug.get("joint_valid")
    if index is None or not isinstance(positions, list) or index >= len(positions):
        return None
    if isinstance(valid, list) and index < len(valid) and not bool(valid[index]):
        return None
    point = positions[index]
    if not isinstance(point, list) or len(point) < 3:
        return None
    try:
        return (float(point[0]), float(point[1]), float(point[2]))
    except (TypeError, ValueError):
        return None


def hand_anatomical_frame_from_debug(hand_debug: dict[str, Any]) -> HandAnatomicalFrame | None:
    """Build the right-handed hand frame requested for Nero wrist auditing.

    Contract:
    - +X follows wrist -> middle finger.
    - +Y follows wrist -> thumb side, orthogonalized against +X.
    - +Z is +X cross +Y so the frame is a proper robot/Genesis rotation.

    The legacy palm normal is reported separately because existing L10 thumb
    code uses roughly index/little across cross middle-forward, which has the
    opposite sign from +X cross +Y on the right-hand traces seen so far.
    """

    wrist = _hand_debug_point(hand_debug, "wrist")
    middle = _hand_debug_point(hand_debug, "middle_proximal") or _hand_debug_point(hand_debug, "middle_metacarpal")
    thumb = _hand_debug_point(hand_debug, "thumb_proximal") or _hand_debug_point(hand_debug, "thumb_metacarpal")
    if wrist is None or middle is None or thumb is None:
        return None

    x_axis = vector_normalize(vector_sub(middle, wrist))
    thumb_vector = vector_project_orthogonal(vector_sub(thumb, wrist), x_axis)
    y_axis = vector_normalize(thumb_vector)
    if vector_norm(x_axis) <= 1e-9 or vector_norm(y_axis) <= 1e-9:
        return None

    z_axis = vector_normalize(vector_cross(x_axis, y_axis))
    y_axis = vector_normalize(vector_cross(z_axis, x_axis))
    axes = FrameAxes(x=x_axis, y=y_axis, z=z_axis)
    matrix = matrix_from_axes(axes)
    det = matrix_det(matrix)
    if abs(det - 1.0) > 1e-6:
        return None

    legacy_alignment: float | None = None
    index_base = _hand_debug_point(hand_debug, "index_proximal")
    little_base = _hand_debug_point(hand_debug, "little_proximal")
    if index_base is not None and little_base is not None:
        across = vector_normalize(vector_project_orthogonal(vector_sub(index_base, little_base), x_axis))
        legacy_normal = vector_normalize(vector_cross(across, x_axis))
        if vector_norm(legacy_normal) > 1e-9:
            legacy_alignment = vector_dot(z_axis, legacy_normal)

    return HandAnatomicalFrame(
        origin_xyz=wrist,
        axes=axes,
        quaternion_xyzw=matrix_to_quat_xyzw(matrix),
        handedness_det=det,
        thumb_alignment=vector_dot(y_axis, vector_normalize(thumb_vector)),
        legacy_palm_normal_alignment=legacy_alignment,
    )


def hand_beavr_anatomical_frame_from_debug(hand_debug: dict[str, Any]) -> HandAnatomicalFrame | None:
    """Build a BEAVR-style stable hand frame with Nero-compatible public axes.

    Raw construction follows BEAVR's wrist/index/middle/pinky knuckle frame:
    palm_normal = cross(index - wrist, middle - wrist)
    palm_direction = average(index, middle, little) - wrist
    cross_product = cross(palm_direction, palm_normal)

    The returned public axes are then fixed-remapped so downstream Nero semantics
    stay unchanged:
    +X = wrist -> middle/finger-forward roll axis, +Y = thumb-side/lateral,
    +Z = +X cross +Y.
    """

    raw = _beavr_raw_frame_from_debug(hand_debug)
    if raw is None:
        return None
    origin, raw_axes = raw
    return _beavr_nero_frame_from_raw(origin, raw_axes)


class BeavrHandFrameSmoother:
    """BEAVR-style moving average over raw hand-frame origin and axes."""

    def __init__(self, moving_average_limit: int = 5) -> None:
        self.moving_average_limit = max(1, int(moving_average_limit))
        self._queue: list[tuple[Vector3, FrameAxes]] = []

    def reset(self) -> None:
        self._queue.clear()

    def update(self, hand_debug: dict[str, Any]) -> HandAnatomicalFrame | None:
        raw = _beavr_raw_frame_from_debug(hand_debug)
        if raw is None:
            return None
        self._queue.append(raw)
        if len(self._queue) > self.moving_average_limit:
            self._queue.pop(0)
        origin = _mean_vector(tuple(item[0] for item in self._queue))
        avg_x = _mean_vector(tuple(item[1].x for item in self._queue))
        avg_y = _mean_vector(tuple(item[1].y for item in self._queue))
        axes = _orthogonalize_axes(avg_x, avg_y)
        if axes is None:
            return None
        return _beavr_nero_frame_from_raw(
            origin,
            axes,
            construction=f"beavr_stable_knuckle_frame_ma{self.moving_average_limit}",
        )


def _beavr_raw_frame_from_debug(hand_debug: dict[str, Any]) -> tuple[Vector3, FrameAxes] | None:
    wrist = _hand_debug_point(hand_debug, "wrist")
    index = _hand_debug_point(hand_debug, "index_proximal") or _hand_debug_point(hand_debug, "index_metacarpal")
    middle = _hand_debug_point(hand_debug, "middle_proximal") or _hand_debug_point(hand_debug, "middle_metacarpal")
    little = _hand_debug_point(hand_debug, "little_proximal") or _hand_debug_point(hand_debug, "little_metacarpal")
    if wrist is None or index is None or middle is None or little is None:
        return None

    v_index = vector_sub(index, wrist)
    v_middle = vector_sub(middle, wrist)
    v_little = vector_sub(little, wrist)
    palm_normal = vector_normalize(vector_cross(v_index, v_middle))
    palm_direction = vector_normalize(
        (
            (v_index[0] + v_middle[0] + v_little[0]) / 3.0,
            (v_index[1] + v_middle[1] + v_little[1]) / 3.0,
            (v_index[2] + v_middle[2] + v_little[2]) / 3.0,
        )
    )
    cross_product = vector_normalize(vector_cross(palm_direction, palm_normal))
    axes = _orthogonalize_axes(cross_product, palm_normal)
    if axes is None:
        return None
    return wrist, axes


def _beavr_nero_frame_from_raw(
    origin: Vector3,
    raw_axes: FrameAxes,
    *,
    construction: str = "beavr_stable_knuckle_frame",
) -> HandAnatomicalFrame | None:
    axes = FrameAxes(
        x=_vector_scale(raw_axes.z, -1.0),
        y=raw_axes.x,
        z=_vector_scale(raw_axes.y, -1.0),
    )
    matrix = matrix_from_axes(axes)
    det = matrix_det(matrix)
    if abs(det - 1.0) > 1e-6:
        return None
    return HandAnatomicalFrame(
        origin_xyz=origin,
        axes=axes,
        quaternion_xyzw=matrix_to_quat_xyzw(matrix),
        handedness_det=det,
        thumb_alignment=0.0,
        legacy_palm_normal_alignment=vector_dot(axes.z, _vector_scale(raw_axes.y, -1.0)),
        construction=construction,
        raw_axes=raw_axes,
        axis_adapter={
            "x": "-beavr_z",
            "y": "beavr_x",
            "z": "-beavr_y",
        },
    )


def _orthogonalize_axes(x_axis: Vector3, y_axis: Vector3) -> FrameAxes | None:
    x = vector_normalize(x_axis)
    y_projected = vector_project_orthogonal(y_axis, x)
    y = vector_normalize(y_projected)
    if vector_norm(x) <= 1e-9 or vector_norm(y) <= 1e-9:
        return None
    z = vector_normalize(vector_cross(x, y))
    y = vector_normalize(vector_cross(z, x))
    return FrameAxes(x=x, y=y, z=z)


def _mean_vector(vectors: tuple[Vector3, ...]) -> Vector3:
    count = max(1, len(vectors))
    return (
        sum(vector[0] for vector in vectors) / count,
        sum(vector[1] for vector in vectors) / count,
        sum(vector[2] for vector in vectors) / count,
    )


def _vector_scale(vector: Vector3, scale: float) -> Vector3:
    return (float(vector[0]) * scale, float(vector[1]) * scale, float(vector[2]) * scale)


def local_delta_rotvec_xyzw(anchor_xyzw: QuaternionXYZW, target_xyzw: QuaternionXYZW) -> Vector3:
    """Rotation from anchor to target expressed in the anchor/body-local frame."""

    local_delta = quat_multiply_xyzw(quat_inverse_xyzw(anchor_xyzw), target_xyzw)
    return quat_log_rotvec_xyzw(local_delta)


def angle_between_quat_deg(lhs_xyzw: QuaternionXYZW, rhs_xyzw: QuaternionXYZW) -> float:
    return float(quat_angle_between_xyzw(lhs_xyzw, rhs_xyzw) * 180.0 / pi)
