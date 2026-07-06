from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Literal

from teleop_stack.ik.so3 import (
    QuaternionXYZW,
    quat_align_hemisphere_xyzw,
    quat_angle_between_xyzw,
    quat_inverse_xyzw,
    quat_multiply_xyzw,
    quat_normalize_xyzw,
    quat_slerp_xyzw,
)


QuaternionWXYZ = tuple[float, float, float, float]
AxisMap = tuple[str, str, str]
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
OrientationReferenceMode = Literal["world_delta", "tool_local_delta", "calibrated_tool_local"]


def xyzw_to_wxyz(quaternion_xyzw: QuaternionXYZW) -> QuaternionWXYZ:
    x, y, z, w = quat_normalize_xyzw(quaternion_xyzw)
    return (w, x, y, z)


def wxyz_to_xyzw(quaternion_wxyz: QuaternionWXYZ) -> QuaternionXYZW:
    w, x, y, z = quaternion_wxyz
    return quat_normalize_xyzw((x, y, z, w))


def axis_map_matrix(axis_map: AxisMap) -> Matrix3:
    rows: list[tuple[float, float, float]] = []
    for token in axis_map:
        spec = token.strip().lower()
        sign = -1.0 if spec.startswith("-") else 1.0
        axis = spec[1:] if spec.startswith(("-", "+")) else spec
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported orientation axis token: {token!r}")
        row = [0.0, 0.0, 0.0]
        row[{"x": 0, "y": 1, "z": 2}[axis]] = sign
        rows.append((row[0], row[1], row[2]))
    if len(rows) != 3:
        raise ValueError(f"orientation axis map must contain 3 tokens, got {len(rows)}")
    return (rows[0], rows[1], rows[2])


def matrix_det3(matrix: Matrix3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def validate_rotation_axis_map(axis_map: AxisMap) -> Matrix3:
    matrix = axis_map_matrix(axis_map)
    det = matrix_det3(matrix)
    if abs(det - 1.0) > 1e-9:
        raise ValueError(
            "orientation axis map must be a proper right-handed rotation "
            f"(det=+1), got det={det:.1f} for {axis_map!r}"
        )
    return matrix


def _matmul(lhs: Matrix3, rhs: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(lhs[row][k] * rhs[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _transpose(matrix: Matrix3) -> Matrix3:
    return tuple(tuple(matrix[row][col] for row in range(3)) for col in range(3))  # type: ignore[return-value]


def _matrix_to_tuple(matrix: Matrix3 | tuple[tuple[float, ...], ...]) -> Matrix3:
    return (
        (float(matrix[0][0]), float(matrix[0][1]), float(matrix[0][2])),
        (float(matrix[1][0]), float(matrix[1][1]), float(matrix[1][2])),
        (float(matrix[2][0]), float(matrix[2][1]), float(matrix[2][2])),
    )


def _quat_xyzw_to_matrix(quaternion_xyzw: QuaternionXYZW) -> Matrix3:
    x, y, z, w = quat_normalize_xyzw(quaternion_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _matrix_to_quat_xyzw(matrix: Matrix3) -> QuaternionXYZW:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = sqrt(trace + 1.0) * 2.0
        return quat_normalize_xyzw(((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s))
    if m00 > m11 and m00 > m22:
        s = sqrt(max(0.0, 1.0 + m00 - m11 - m22)) * 2.0
        return quat_normalize_xyzw((0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s))
    if m11 > m22:
        s = sqrt(max(0.0, 1.0 + m11 - m00 - m22)) * 2.0
        return quat_normalize_xyzw(((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s))
    s = sqrt(max(0.0, 1.0 + m22 - m00 - m11)) * 2.0
    return quat_normalize_xyzw(((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s))


def map_quaternion_by_axis_map_xyzw(quaternion_xyzw: QuaternionXYZW, axis_map: AxisMap) -> QuaternionXYZW:
    axis_matrix = validate_rotation_axis_map(axis_map)
    rotation = _quat_xyzw_to_matrix(quaternion_xyzw)
    mapped = _matmul(_matmul(axis_matrix, rotation), _transpose(axis_matrix))
    return _matrix_to_quat_xyzw(mapped)


def quaternion_angle_wxyz(lhs_wxyz: QuaternionWXYZ, rhs_wxyz: QuaternionWXYZ) -> float:
    return quat_angle_between_xyzw(wxyz_to_xyzw(lhs_wxyz), wxyz_to_xyzw(rhs_wxyz))


@dataclass(frozen=True)
class OrientationTrackerConfig:
    axis_map: AxisMap = ("x", "y", "z")
    max_speed_rad_s: float = 0.8
    tool_offset_wxyz: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)
    reference_mode: OrientationReferenceMode = "world_delta"

    def __post_init__(self) -> None:
        validate_rotation_axis_map(self.axis_map)
        quat_normalize_xyzw(wxyz_to_xyzw(self.tool_offset_wxyz))
        if self.reference_mode not in {"world_delta", "tool_local_delta", "calibrated_tool_local"}:
            raise ValueError(f"Unsupported orientation reference mode: {self.reference_mode!r}")
        if float(self.max_speed_rad_s) < 0.0:
            raise ValueError("max_speed_rad_s must be non-negative")


@dataclass(frozen=True)
class OrientationTrackerResult:
    wrist_quat_xyzw: QuaternionXYZW
    anchor_wrist_quat_xyzw: QuaternionXYZW
    anchor_ee_quat_wxyz: QuaternionWXYZ
    orientation_reference_mode: OrientationReferenceMode
    orientation_axis_map: AxisMap
    orientation_mapping_matrix: Matrix3
    source_delta_quat_wxyz: QuaternionWXYZ
    source_to_tool_matrix: Matrix3 | None
    mapped_delta_quat_wxyz: QuaternionWXYZ
    raw_target_quat_wxyz: QuaternionWXYZ
    cmd_target_quat_wxyz: QuaternionWXYZ
    raw_to_cmd_error_rad: float
    cmd_step_rad: float
    orientation_max_step_rad: float
    orientation_limited: bool
    events: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "wrist_quat_xyzw": [float(v) for v in self.wrist_quat_xyzw],
            "anchor_wrist_quat_xyzw": [float(v) for v in self.anchor_wrist_quat_xyzw],
            "anchor_ee_quat_wxyz": [float(v) for v in self.anchor_ee_quat_wxyz],
            "orientation_reference_mode": self.orientation_reference_mode,
            "orientation_axis_map": [str(v) for v in self.orientation_axis_map],
            "orientation_mapping_matrix": [[float(v) for v in row] for row in self.orientation_mapping_matrix],
            "source_delta_quat_wxyz": [float(v) for v in self.source_delta_quat_wxyz],
            "source_to_tool_matrix": (
                [[float(v) for v in row] for row in self.source_to_tool_matrix]
                if self.source_to_tool_matrix is not None
                else None
            ),
            "mapped_delta_quat_wxyz": [float(v) for v in self.mapped_delta_quat_wxyz],
            "raw_target_quat_wxyz": [float(v) for v in self.raw_target_quat_wxyz],
            "cmd_target_quat_wxyz": [float(v) for v in self.cmd_target_quat_wxyz],
            "raw_to_cmd_error_rad": float(self.raw_to_cmd_error_rad),
            "cmd_step_rad": float(self.cmd_step_rad),
            "orientation_max_step_rad": float(self.orientation_max_step_rad),
            "orientation_limited": bool(self.orientation_limited),
            "orientation_events": [str(v) for v in self.events],
        }


@dataclass
class OrientationTargetTracker:
    config: OrientationTrackerConfig = field(default_factory=OrientationTrackerConfig)
    _anchor_wrist_quat_xyzw: QuaternionXYZW | None = field(default=None, init=False, repr=False)
    _anchor_ee_quat_xyzw: QuaternionXYZW | None = field(default=None, init=False, repr=False)
    _cmd_quat_xyzw: QuaternionXYZW | None = field(default=None, init=False, repr=False)
    _source_to_tool_matrix: Matrix3 | None = field(default=None, init=False, repr=False)

    def reset_anchor(self, wrist_quat_xyzw: QuaternionXYZW, ee_quat_wxyz: QuaternionWXYZ) -> None:
        self._anchor_wrist_quat_xyzw = quat_normalize_xyzw(wrist_quat_xyzw)
        self._anchor_ee_quat_xyzw = wxyz_to_xyzw(ee_quat_wxyz)
        self._cmd_quat_xyzw = self._anchor_ee_quat_xyzw
        self._source_to_tool_matrix = None

    def _tool_mapped_delta(self, delta_xyzw: QuaternionXYZW) -> QuaternionXYZW:
        tool_offset = wxyz_to_xyzw(self.config.tool_offset_wxyz)
        return quat_multiply_xyzw(
            quat_multiply_xyzw(tool_offset, delta_xyzw),
            quat_inverse_xyzw(tool_offset),
        )

    def _calibrated_source_to_tool_matrix(self, axis_matrix: Matrix3) -> Matrix3:
        if self._anchor_wrist_quat_xyzw is None or self._anchor_ee_quat_xyzw is None:
            raise RuntimeError("OrientationTargetTracker source/tool calibration requires reset_anchor().")
        if self._source_to_tool_matrix is None:
            anchor_wrist_matrix = _quat_xyzw_to_matrix(self._anchor_wrist_quat_xyzw)
            anchor_ee_matrix = _quat_xyzw_to_matrix(self._anchor_ee_quat_xyzw)
            source_to_tool = _matmul(
                _matmul(_transpose(anchor_ee_matrix), axis_matrix),
                anchor_wrist_matrix,
            )
            self._source_to_tool_matrix = _matrix_to_tuple(source_to_tool)
        return self._source_to_tool_matrix

    def _target_delta(self, wrist_quat: QuaternionXYZW, axis_matrix: Matrix3) -> tuple[QuaternionXYZW, QuaternionXYZW, Matrix3 | None]:
        assert self._anchor_wrist_quat_xyzw is not None
        assert self._anchor_ee_quat_xyzw is not None
        mode = self.config.reference_mode
        if mode == "world_delta":
            source_delta = quat_multiply_xyzw(wrist_quat, quat_inverse_xyzw(self._anchor_wrist_quat_xyzw))
            mapped_delta = map_quaternion_by_axis_map_xyzw(source_delta, self.config.axis_map)
            return source_delta, self._tool_mapped_delta(mapped_delta), None

        source_delta = quat_multiply_xyzw(quat_inverse_xyzw(self._anchor_wrist_quat_xyzw), wrist_quat)
        if mode == "tool_local_delta":
            mapped_delta = map_quaternion_by_axis_map_xyzw(source_delta, self.config.axis_map)
            return source_delta, self._tool_mapped_delta(mapped_delta), None

        source_to_tool = self._calibrated_source_to_tool_matrix(axis_matrix)
        source_delta_matrix = _quat_xyzw_to_matrix(source_delta)
        mapped_delta_matrix = _matmul(
            _matmul(source_to_tool, source_delta_matrix),
            _transpose(source_to_tool),
        )
        return source_delta, self._tool_mapped_delta(_matrix_to_quat_xyzw(mapped_delta_matrix)), source_to_tool

    def update(self, wrist_quat_xyzw: QuaternionXYZW, dt_s: float) -> OrientationTrackerResult:
        if self._anchor_wrist_quat_xyzw is None or self._anchor_ee_quat_xyzw is None or self._cmd_quat_xyzw is None:
            raise RuntimeError("OrientationTargetTracker.update called before reset_anchor().")

        wrist_quat = quat_normalize_xyzw(wrist_quat_xyzw)
        matrix = validate_rotation_axis_map(self.config.axis_map)
        source_delta, tool_mapped_delta, source_to_tool = self._target_delta(wrist_quat, matrix)
        if self.config.reference_mode == "world_delta":
            raw_target = quat_multiply_xyzw(tool_mapped_delta, self._anchor_ee_quat_xyzw)
        else:
            raw_target = quat_multiply_xyzw(self._anchor_ee_quat_xyzw, tool_mapped_delta)
        raw_target = quat_align_hemisphere_xyzw(raw_target, self._cmd_quat_xyzw)

        error_rad = quat_angle_between_xyzw(self._cmd_quat_xyzw, raw_target)
        max_step = max(0.0, float(self.config.max_speed_rad_s)) * max(0.0, float(dt_s))
        events: list[str] = []
        if error_rad <= 1e-12:
            next_cmd = raw_target
            cmd_step = 0.0
            limited = False
        elif max_step <= 0.0:
            next_cmd = self._cmd_quat_xyzw
            cmd_step = 0.0
            limited = True
            events.append("orientation_rate_limited")
        elif error_rad > max_step:
            next_cmd = quat_slerp_xyzw(self._cmd_quat_xyzw, raw_target, max_step / error_rad)
            cmd_step = max_step
            limited = True
            events.append("orientation_rate_limited")
        else:
            next_cmd = raw_target
            cmd_step = error_rad
            limited = False
            events.append("orientation_target_reached")

        self._cmd_quat_xyzw = quat_normalize_xyzw(next_cmd)
        remaining_error = quat_angle_between_xyzw(self._cmd_quat_xyzw, raw_target)
        return OrientationTrackerResult(
            wrist_quat_xyzw=wrist_quat,
            anchor_wrist_quat_xyzw=self._anchor_wrist_quat_xyzw,
            anchor_ee_quat_wxyz=xyzw_to_wxyz(self._anchor_ee_quat_xyzw),
            orientation_reference_mode=self.config.reference_mode,
            orientation_axis_map=self.config.axis_map,
            orientation_mapping_matrix=matrix,
            source_delta_quat_wxyz=xyzw_to_wxyz(source_delta),
            source_to_tool_matrix=source_to_tool,
            mapped_delta_quat_wxyz=xyzw_to_wxyz(tool_mapped_delta),
            raw_target_quat_wxyz=xyzw_to_wxyz(raw_target),
            cmd_target_quat_wxyz=xyzw_to_wxyz(self._cmd_quat_xyzw),
            raw_to_cmd_error_rad=remaining_error,
            cmd_step_rad=cmd_step,
            orientation_max_step_rad=max_step,
            orientation_limited=limited,
            events=tuple(events),
        )
