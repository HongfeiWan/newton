from __future__ import annotations

from dataclasses import replace
from typing import Literal

from teleop_stack.ik.so3 import QuaternionXYZW
from teleop_stack.teleop.spatial_frames import (
    FrameAxes,
    HandAnatomicalFrame,
    Matrix3,
    Vector3,
    matrix_det,
    matrix_from_axes,
    matrix_to_quat_xyzw,
    quat_xyzw_to_matrix,
    vector_cross,
    vector_dot,
)


OpenXrCoordinateAdapterName = Literal["none", "openxr_genesis"]

# OpenXR raw world: +X=wearer right, +Y=up, -Z=wearer front.
# Canonical Genesis world for Nero teleop: +X=back, +Y=right, +Z=up.
OPENXR_TO_GENESIS_AXIS_MAP: tuple[str, str, str] = ("z", "x", "y")
OPENXR_TO_GENESIS_MATRIX: Matrix3 = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)

# Initial right-hand wrist/tool frame in canonical Genesis world when fingers
# point forward and the palm faces left.
GENESIS_INITIAL_RIGHT_HAND_WRIST_AXES = FrameAxes(
    x=(0.0, 0.0, -1.0),  # down
    y=(0.0, -1.0, 0.0),  # left
    z=(-1.0, 0.0, 0.0),  # front
)

# Convert the existing finger-forward hand frame
#   +X=fingers/front, +Y=thumb/up, +Z=right/back-of-hand
# into the requested wrist/tool frame
#   +X=down, +Y=left, +Z=front.
FINGER_FRAME_TO_WRIST_LOCAL_MATRIX: Matrix3 = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)


def validate_adapter_name(name: str) -> OpenXrCoordinateAdapterName:
    if name not in {"none", "openxr_genesis"}:
        raise ValueError(f"unsupported OpenXR coordinate adapter: {name!r}")
    return name  # type: ignore[return-value]


def map_openxr_vector_to_genesis(values_xyz: Vector3) -> Vector3:
    return _mat_vec_mul(OPENXR_TO_GENESIS_MATRIX, values_xyz)


def map_openxr_quaternion_to_genesis_parent(quaternion_xyzw: QuaternionXYZW) -> QuaternionXYZW:
    rotation = quat_xyzw_to_matrix(quaternion_xyzw)
    return matrix_to_quat_xyzw(_matmul(OPENXR_TO_GENESIS_MATRIX, rotation))


def map_openxr_axes_to_genesis_parent(axes: FrameAxes) -> FrameAxes:
    return FrameAxes(
        x=map_openxr_vector_to_genesis(axes.x),
        y=map_openxr_vector_to_genesis(axes.y),
        z=map_openxr_vector_to_genesis(axes.z),
    )


def adapt_openxr_hand_frame_to_genesis_parent(frame: HandAnatomicalFrame) -> HandAnatomicalFrame:
    axes = map_openxr_axes_to_genesis_parent(frame.axes)
    return _replace_frame_axes(
        frame,
        axes,
        construction=f"{frame.construction}_openxr_to_genesis",
        axis_adapter={
            **(frame.axis_adapter or {}),
            "parent": "openxr_to_genesis(+X=back,+Y=right,+Z=up)",
        },
    )


def adapt_finger_frame_to_genesis_wrist_frame(frame: HandAnatomicalFrame) -> HandAnatomicalFrame:
    axes = FrameAxes(
        x=_vector_scale(frame.axes.y, -1.0),
        y=_vector_scale(frame.axes.z, -1.0),
        z=frame.axes.x,
    )
    return _replace_frame_axes(
        frame,
        axes,
        construction=f"{frame.construction}_wrist_tool_axes",
        axis_adapter={
            **(frame.axis_adapter or {}),
            "local_x": "-finger_frame_y",
            "local_y": "-finger_frame_z",
            "local_z": "finger_frame_x",
        },
    )


def adapt_openxr_hand_frame_to_genesis_wrist_frame(frame: HandAnatomicalFrame) -> HandAnatomicalFrame:
    return adapt_finger_frame_to_genesis_wrist_frame(adapt_openxr_hand_frame_to_genesis_parent(frame))


def adapter_debug_payload() -> dict[str, object]:
    axes = GENESIS_INITIAL_RIGHT_HAND_WRIST_AXES
    return {
        "openxr_to_genesis_axis_map": list(OPENXR_TO_GENESIS_AXIS_MAP),
        "openxr_to_genesis_matrix": [[float(value) for value in row] for row in OPENXR_TO_GENESIS_MATRIX],
        "genesis_world": {
            "+x": "back",
            "+y": "right",
            "+z": "up",
        },
        "initial_right_hand_wrist_axes": axes.as_dict(),
        "initial_right_hand_wrist_semantics": {
            "+x": "down",
            "+y": "left",
            "+z": "front",
        },
        "hand_local_axis_adapter": {
            "local_matrix": [[float(value) for value in row] for row in FINGER_FRAME_TO_WRIST_LOCAL_MATRIX],
            "+x": "-finger_frame_y",
            "+y": "-finger_frame_z",
            "+z": "finger_frame_x",
        },
    }


def _replace_frame_axes(
    frame: HandAnatomicalFrame,
    axes: FrameAxes,
    *,
    construction: str,
    axis_adapter: dict[str, str],
) -> HandAnatomicalFrame:
    matrix = matrix_from_axes(axes)
    det = matrix_det(matrix)
    if abs(det - 1.0) > 1e-6:
        raise ValueError(f"adapted hand frame must be right-handed, got det={det:.6f}")
    return replace(
        frame,
        axes=axes,
        quaternion_xyzw=matrix_to_quat_xyzw(matrix),
        handedness_det=det,
        construction=construction,
        axis_adapter=axis_adapter,
    )


def _matmul(lhs: Matrix3, rhs: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(float(lhs[row][k]) * float(rhs[k][col]) for k in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _mat_vec_mul(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(
        sum(float(matrix[row][col]) * float(vector[col]) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _vector_scale(vector: Vector3, scale: float) -> Vector3:
    return tuple(float(scale) * float(value) for value in vector)  # type: ignore[return-value]


def assert_right_handed_axes(axes: FrameAxes) -> None:
    det = matrix_det(matrix_from_axes(axes))
    if abs(det - 1.0) > 1e-6:
        raise AssertionError(f"axes are not right-handed: det={det:.6f}")
    cross = vector_cross(axes.x, axes.y)
    if vector_dot(cross, axes.z) < 1.0 - 1e-6:
        raise AssertionError("axes do not satisfy +X cross +Y = +Z")
