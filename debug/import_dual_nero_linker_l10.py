# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visual debug runner for the generated dual Nero + Linker L10 URDF."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.sensors import SensorTiledCamera

try:
    from debug.edit_dynamic_bottle_body import build_dynamic_bottle, load_dynamic_bottle_spec
except ModuleNotFoundError:
    from edit_dynamic_bottle_body import build_dynamic_bottle, load_dynamic_bottle_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "assets" / "generated" / "dual_nero_linker_l10_combined.urdf"
DEFAULT_SCENE_GLB = REPO_ROOT / "scene" / "scene.glb"
DEFAULT_SCENE_COLLISION_SPEC = REPO_ROOT / "debug" / "scene_collision_boxes.json"
DEFAULT_DYNAMIC_BOTTLE_SPEC = REPO_ROOT / "debug" / "dynamic_bottle_body.json"
DEFAULT_HARNESS_ROOT = Path("/home/whf/Project/harness")
DEFAULT_D455_JSON = DEFAULT_HARNESS_ROOT / "assets" / "d455json.json"
DEFAULT_D405_JSON = REPO_ROOT / "assets" / "d405json.json"
DEFAULT_D405_MOUNT_JSON = REPO_ROOT / "assets" / "d405_mount_default.json"
DEFAULT_OVERLAY_HAND_TRACE_PATH = REPO_ROOT / "logs" / "xr_debug" / "camera_overlay_hand.jsonl"
URDF_UP_AXIS = "Z"
D455_BODY_LABEL_SUFFIX = "/d455_body"
D455_BODY_SIZE_FALLBACK = (0.026, 0.124, 0.029)
D455_RGB_FRONT_CLEARANCE_M = 0.002
D455_MODEL_IMAGE_SIZE = (224, 224)
D455_EGO_ROI_ZOOM = 2.0
D455_EGO_ROI_CENTER_X = 0.50
D455_EGO_ROI_CENTER_Y = 0.65
D455_PREVIEW_SCALE = 2
D405_BODY_SIZE_FALLBACK = (0.042, 0.042, 0.023)
RIGHT_D405_CONNECTOR_REL_POS_M = (0.022759, -0.004138, 0.013103)
RIGHT_D405_CONNECTOR_REL_EULER_DEG = (79.969, 0.0, 0.0)
D405_CAMERA_LOCAL_POS_RATIO = (0.0, 0.0, 0.5)
D405_CAMERA_NEAR_M = 1.0e-4
D405_CAMERA_FAR_M = 1.0e6
D405_PREVIEW_SCALE = 1
INITIAL_LEFT_ARM_Q = (
    -0.3010692959690218,
    1.4731277018532938,
    -1.1596840214876325,
    1.3072865163287928,
    0.005689773361501515,
    -0.06812020070533868,
    0.21753783796857323,
)
INITIAL_RIGHT_ARM_Q = (
    0.2530727415391778,
    1.5579507035002182,
    1.2218002895661106,
    1.3225232406987033,
    -0.0004886921905584122,
    -0.11129964639967839,
    0.11606439525762292,
)
DEFAULT_RIGID_GAP_M = 1.0e-4
L10_CONTACT_FRICTION = 0.45
L10_CONTACT_TORSIONAL_FRICTION = 0.0
L10_CONTACT_ROLLING_FRICTION = 0.0
L10_CONTACT_KE = 8.0e3
L10_CONTACT_KD = 1.5e3
L10_CONTACT_KF = 2.5e2
L10_CONTACT_MARGIN_M = 0.0
L10_CONTACT_GAP_M = 1.0e-4
BOTTLE_SCENE_COLLISION_CLEARANCE_M = 0.002


@dataclass
class SceneGlbMesh:
    mesh: newton.Mesh
    texture: np.ndarray | None
    color: tuple[float, float, float]


@dataclass
class SceneCollisionBox:
    name: str
    pos: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    size: tuple[float, float, float]
    friction: float
    visible: bool


@dataclass
class CameraPreview:
    name: str
    enabled: bool
    width: int
    height: int
    fov_deg: float
    camera_rays: wp.array[wp.vec3]
    color_image: wp.array[wp.uint32]
    camera_transform: wp.array[wp.transformf] | None = None


def _resolve_urdf(path: Path) -> Path:
    urdf_path = path if path.is_absolute() else (Path.cwd() / path)
    urdf_path = urdf_path.resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file does not exist: {urdf_path}")
    return urdf_path


def _print_model_summary(model: newton.Model) -> None:
    print("Imported model:")
    print(f"  bodies:        {model.body_count}")
    print(f"  joints:        {model.joint_count}")
    print(f"  shapes:        {model.shape_count}")
    print(f"  articulations: {model.articulation_count}")
    print(f"  joint coords:  {model.joint_coord_count}")
    print(f"  joint dofs:    {model.joint_dof_count}")
    print(f"  up axis:       {model.up_axis.name}")

    if model.body_label:
        print(f"  first bodies:  {', '.join(model.body_label[: min(8, len(model.body_label))])}")
    if model.joint_label:
        print(f"  first joints:  {', '.join(model.joint_label[: min(8, len(model.joint_label))])}")


def _is_l10_hand_body_label(body_label: str) -> bool:
    link_name = body_label.rsplit("/", maxsplit=1)[-1].lower()
    return link_name.startswith(("right_l10_", "left_l10_"))


def _filter_urdf_collisions_to_l10_hand(
    builder: newton.ModelBuilder,
    first_shape: int,
    last_shape: int,
    *,
    l10_friction: float,
    l10_ke: float,
    l10_kd: float,
    l10_kf: float,
    l10_mu_torsional: float,
    l10_mu_rolling: float,
) -> None:
    collision_mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
    kept_l10 = 0
    disabled_non_l10 = 0

    for shape_index in range(first_shape, last_shape):
        flags = int(builder.shape_flags[shape_index])
        if not flags & collision_mask:
            continue

        body_index = builder.shape_body[shape_index]
        body_label = builder.body_label[body_index] if 0 <= body_index < len(builder.body_label) else ""
        if _is_l10_hand_body_label(body_label):
            builder.shape_material_mu[shape_index] = float(l10_friction)
            builder.shape_material_restitution[shape_index] = 0.0
            builder.shape_material_ke[shape_index] = float(l10_ke)
            builder.shape_material_kd[shape_index] = float(l10_kd)
            builder.shape_material_kf[shape_index] = float(l10_kf)
            builder.shape_material_mu_torsional[shape_index] = float(l10_mu_torsional)
            builder.shape_material_mu_rolling[shape_index] = float(l10_mu_rolling)
            builder.shape_margin[shape_index] = L10_CONTACT_MARGIN_M
            builder.shape_gap[shape_index] = L10_CONTACT_GAP_M
            kept_l10 += 1
            continue

        builder.shape_flags[shape_index] = flags & ~collision_mask
        disabled_non_l10 += 1

    print(
        "URDF collision filter:"
        f" kept_l10_hand_shapes={kept_l10}"
        f" disabled_non_l10_shapes={disabled_non_l10}"
        f" l10_mu={l10_friction:g}"
        f" l10_ke={l10_ke:g}"
        f" margin={L10_CONTACT_MARGIN_M:g}"
        f" gap={L10_CONTACT_GAP_M:g}"
    )


def _make_urdf_bodies_kinematic(builder: newton.ModelBuilder, first_body: int, last_body: int) -> None:
    for body_index in range(first_body, last_body):
        builder.body_flags[body_index] = int(newton.BodyFlags.KINEMATIC)
        builder.body_mass[body_index] = 0.0
        builder.body_inertia[body_index] = wp.mat33(0.0)
        builder.body_inv_mass[body_index] = 0.0
        builder.body_inv_inertia[body_index] = wp.mat33(0.0)
        builder.body_qd[body_index] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    print(f"URDF body mode: kinematic bodies={max(0, last_body - first_body)}")


def _set_initial_arm_pose(
    builder: newton.ModelBuilder,
    *,
    side: str,
    values: tuple[float, float, float, float, float, float, float],
) -> None:
    applied = []
    for joint_index in range(1, 8):
        label_suffix = f"/{side}_joint{joint_index}"
        joint_id = next((i for i, label in enumerate(builder.joint_label) if label.endswith(label_suffix)), None)
        if joint_id is None:
            print(f"Warning: initial arm pose skipped missing joint {label_suffix}")
            continue

        q_start = builder.joint_q_start[joint_id]
        q_end = builder.joint_q_start[joint_id + 1] if joint_id + 1 < len(builder.joint_q_start) else len(builder.joint_q)
        qd_start = builder.joint_qd_start[joint_id]
        qd_end = (
            builder.joint_qd_start[joint_id + 1] if joint_id + 1 < len(builder.joint_qd_start) else len(builder.joint_qd)
        )
        if q_end - q_start != 1:
            print(f"Warning: initial arm pose expected 1 q for {label_suffix}, got {q_end - q_start}")
            continue

        value = float(values[joint_index - 1])
        builder.joint_q[q_start] = value
        builder.joint_target_q[q_start] = value
        for qd_index in range(qd_start, qd_end):
            builder.joint_qd[qd_index] = 0.0
            builder.joint_target_qd[qd_index] = 0.0
        applied.append(value)

    print(f"Initial {side} arm q from harness: {np.round(applied, 6).tolist()}")


def _set_initial_arm_poses(
    builder: newton.ModelBuilder,
    *,
    left_q: tuple[float, float, float, float, float, float, float],
    right_q: tuple[float, float, float, float, float, float, float],
) -> None:
    _set_initial_arm_pose(builder, side="left", values=left_q)
    _set_initial_arm_pose(builder, side="right", values=right_q)


def _assert_finite_state(state: newton.State, label: str) -> None:
    if state.body_q is None:
        return

    body_q = state.body_q.numpy()
    if not np.isfinite(body_q).all():
        raise RuntimeError(f"{label} produced non-finite body transforms")


def _resolve_optional_file(path: Path) -> Path | None:
    file_path = path if path.is_absolute() else (Path.cwd() / path)
    file_path = file_path.resolve()
    return file_path if file_path.exists() else None


def _image_size(value: str) -> tuple[int, int]:
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected height,width, e.g. 224,224")
    try:
        height, width = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer height,width") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("height and width must be positive")
    return height, width


def _vec3(value: str) -> tuple[float, float, float]:
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric x,y,z") from exc


def _axis_map(value: str) -> tuple[str, str, str]:
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated axis tokens")
    for token in parts:
        raw = token[1:] if token.startswith(("-", "+")) else token
        if raw not in {"x", "y", "z"}:
            raise argparse.ArgumentTypeError(f"unsupported axis token: {token!r}")
    return parts  # type: ignore[return-value]


def _normalize_axis_map(value: str | tuple[str, str, str]) -> tuple[str, str, str]:
    if isinstance(value, tuple):
        return value
    return _axis_map(value)


def _vec4(value: str) -> tuple[float, float, float, float]:
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected four comma-separated values")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric values") from exc


def _default_voice_control_port() -> int:
    for name in ("TELEOP_QUEST_VOICE_UDP_PORT", "TELEOP_VOICE_UDP_PORT"):
        raw_value = os.environ.get(name)
        if raw_value:
            return int(raw_value)
    return 9910


def _vec7(value: str) -> tuple[float, float, float, float, float, float, float]:
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("expected seven comma-separated joint values")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric joint values") from exc


def _resize_hwc_linear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    src_height, src_width = image.shape[:2]
    if (src_height, src_width) == (height, width):
        return image.astype(np.uint8, copy=False)

    y = (np.arange(height, dtype=np.float32) + 0.5) * (src_height / height) - 0.5
    x = (np.arange(width, dtype=np.float32) + 0.5) * (src_width / width) - 0.5
    y = np.clip(y, 0.0, src_height - 1.0)
    x = np.clip(x, 0.0, src_width - 1.0)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_height - 1)
    x1 = np.minimum(x0 + 1, src_width - 1)
    wy = (y - y0).reshape(height, 1, 1)
    wx = (x - x0).reshape(1, width, 1)

    src = image.astype(np.float32, copy=False)
    top = src[y0[:, None], x0[None, :]] * (1.0 - wx) + src[y0[:, None], x1[None, :]] * wx
    bottom = src[y1[:, None], x0[None, :]] * (1.0 - wx) + src[y1[:, None], x1[None, :]] * wx
    resized = top * (1.0 - wy) + bottom * wy
    return np.clip(np.rint(resized), 0, 255).astype(np.uint8)


def _resize_one_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / width, cur_height / height)
    resized_width = max(1, int(cur_width / ratio))
    resized_height = max(1, int(cur_height / ratio))
    resized_image = _resize_hwc_linear(image, resized_height, resized_width)
    output = np.zeros((height, width, image.shape[-1]), dtype=np.uint8)
    y0 = (height - resized_height) // 2
    x0 = (width - resized_width) // 2
    output[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized_image
    return output


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[-3:-1] == (height, width):
        return image.astype(np.uint8, copy=False)
    original_shape = image.shape
    image = image.reshape(-1, *original_shape[-3:])
    resized = [_resize_one_with_pad(frame, height, width) for frame in image]
    return np.stack(resized).reshape(*original_shape[:-3], height, width, original_shape[-1])


def _roi_crop_zoom_hwc(image: np.ndarray, *, zoom: float, center_x: float, center_y: float) -> np.ndarray:
    zoom = float(zoom)
    if zoom <= 1.0:
        return image
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image
    center_x = min(max(float(center_x), 0.0), 1.0)
    center_y = min(max(float(center_y), 0.0), 1.0)
    crop_width = max(1, min(width, int(round(width / zoom))))
    crop_height = max(1, min(height, int(round(height / zoom))))
    crop_x = int(round(center_x * width - crop_width / 2.0))
    crop_y = int(round(center_y * height - crop_height / 2.0))
    crop_x = min(max(0, crop_x), max(0, width - crop_width))
    crop_y = min(max(0, crop_y), max(0, height - crop_height))
    return image[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]


def _roi_crop_rect(
    width: int,
    height: int,
    *,
    zoom: float,
    center_x: float,
    center_y: float,
) -> tuple[int, int, int, int]:
    zoom = float(zoom)
    if zoom <= 1.0:
        return 0, 0, int(width), int(height)
    center_x = min(max(float(center_x), 0.0), 1.0)
    center_y = min(max(float(center_y), 0.0), 1.0)
    crop_width = max(1, min(int(width), int(round(int(width) / zoom))))
    crop_height = max(1, min(int(height), int(round(int(height) / zoom))))
    crop_x = int(round(center_x * int(width) - crop_width / 2.0))
    crop_y = int(round(center_y * int(height) - crop_height / 2.0))
    crop_x = min(max(0, crop_x), max(0, int(width) - crop_width))
    crop_y = min(max(0, crop_y), max(0, int(height) - crop_height))
    return crop_x, crop_y, crop_width, crop_height


def _packed_color_image_to_rgb_hwc(image: wp.array[wp.uint32]) -> np.ndarray:
    packed = image.numpy()
    while packed.ndim > 2:
        packed = packed[0]
    packed = np.ascontiguousarray(packed)
    rgba = packed.view(np.uint8).reshape(*packed.shape, 4)
    return rgba[..., :3].copy()


def _show_rgb_preview(owner: object, window_name: str, image: np.ndarray, *, scale: int = 2) -> None:
    warning_prefix = f"[{window_name}]"
    warning_attr_prefix = window_name.lower().replace(" ", "_").replace("-", "_")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        warning_attr = f"_{warning_attr_prefix}_display_warned"
        if not getattr(owner, warning_attr, False):
            print(f"{warning_prefix} OpenCV window disabled: DISPLAY/WAYLAND_DISPLAY is not set", flush=True)
            setattr(owner, warning_attr, True)
        return
    try:
        import cv2  # noqa: PLC0415
    except Exception as exc:
        warning_attr = f"_{warning_attr_prefix}_cv2_warned"
        if not getattr(owner, warning_attr, False):
            print(f"{warning_prefix} OpenCV window disabled: failed to import cv2: {exc}", flush=True)
            setattr(owner, warning_attr, True)
        return

    frame = np.asarray(image, dtype=np.uint8)
    if int(scale) > 1:
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width * int(scale), height * int(scale)), interpolation=cv2.INTER_NEAREST)
    try:
        cv2.imshow(window_name, frame[..., ::-1])
        cv2.waitKey(1)
    except Exception as exc:
        warning_attr = f"_{warning_attr_prefix}_imshow_warned"
        if not getattr(owner, warning_attr, False):
            print(f"{warning_prefix} OpenCV window disabled: failed to show window: {exc}", flush=True)
            setattr(owner, warning_attr, True)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> wp.quat:
    qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), roll)
    qy = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pitch)
    qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    return qz * qy * qx


def _rotation_from_euler_deg(euler_deg: tuple[float, float, float]) -> np.ndarray:
    x, y, z = (np.deg2rad(v) for v in euler_deg)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)), dtype=np.float64)
    ry = np.asarray(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), dtype=np.float64)
    rz = np.asarray(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    return rz @ ry @ rx


def _euler_deg_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = -float(rotation[2, 0])
    sy = min(max(sy, -1.0), 1.0)
    pitch = np.arcsin(sy)
    cp = np.cos(pitch)
    if abs(cp) > 1.0e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    return tuple(float(np.rad2deg(v)) for v in (roll, pitch, yaw))


def _rotation_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        ),
        dtype=np.float64,
    )


def _quat_xyzw_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(rotation)))
        if i == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray((x, y, z, w), dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)


def _camera_transform_from_forward_up(position: np.ndarray, forward: np.ndarray, up: np.ndarray) -> wp.transformf:
    forward = np.asarray(forward, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = np.asarray(up, dtype=np.float64)
    up -= forward * float(np.dot(forward, up))
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, up, -forward))
    qx, qy, qz, qw = _quat_xyzw_from_rotation(rotation)
    return wp.transformf(wp.vec3f(*position), wp.quatf(qx, qy, qz, qw))


def _mesh_color(mesh, texture: np.ndarray | None) -> tuple[float, float, float]:
    if texture is not None:
        material = getattr(getattr(mesh, "visual", None), "material", None)
        base_color = getattr(material, "baseColorFactor", None)
        if base_color is None:
            return (1.0, 1.0, 1.0)

    default = (0.65, 0.65, 0.65)
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    candidates = [
        getattr(material, "baseColorFactor", None),
        getattr(material, "main_color", None),
        getattr(visual, "main_color", None),
        getattr(visual, "vertex_colors", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        color = np.asarray(candidate, dtype=np.float32)
        if color.ndim == 2:
            color = color[:, :3].mean(axis=0)
        color = color.reshape(-1)
        if color.size >= 3:
            if np.max(color[:3]) > 1.0:
                color = color / 255.0
            return tuple(np.clip(color[:3], 0.0, 1.0).tolist())
    return default


def _mesh_texture(mesh) -> np.ndarray | None:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        texture = getattr(material, "base_color_texture", None)
    if texture is None:
        texture = getattr(material, "image", None)
    if texture is None:
        return None
    if hasattr(texture, "convert"):
        return np.asarray(texture.convert("RGBA"))
    return np.asarray(texture)


def _load_glb_meshes(glb_path: Path, *, label: str) -> list[SceneGlbMesh]:
    import trimesh  # noqa: PLC0415

    scene = trimesh.load(glb_path, force="scene")
    meshes: list[SceneGlbMesh] = []

    for index, node_name in enumerate(scene.graph.nodes_geometry):
        node_transform, geometry_name = scene.graph.get(node_name)
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(node_transform)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
        if vertices.size == 0 or faces.size == 0:
            continue

        normals_np = None
        if getattr(mesh, "vertex_normals", None) is not None and len(mesh.vertex_normals) == len(mesh.vertices):
            normals_np = np.asarray(mesh.vertex_normals, dtype=np.float32)
            if not normals_np.size:
                normals_np = None

        uvs_np = None
        visual_uvs = getattr(getattr(mesh, "visual", None), "uv", None)
        if visual_uvs is not None:
            uvs_np = np.asarray(visual_uvs, dtype=np.float32)
            if uvs_np.shape != (len(mesh.vertices), 2):
                uvs_np = None

        texture = _mesh_texture(mesh)
        color = _mesh_color(mesh, texture)
        meshes.append(
            SceneGlbMesh(
                mesh=newton.Mesh(
                    vertices,
                    faces,
                    normals=normals_np,
                    uvs=uvs_np,
                    compute_inertia=False,
                    is_solid=False,
                    color=color,
                    texture=texture,
                ),
                texture=texture,
                color=color,
            )
        )

    if not meshes:
        raise ValueError(f"No renderable meshes found in GLB: {glb_path}")

    bounds = scene.bounds
    print(f"Loaded {label} GLB: {glb_path}")
    print(f"  mesh parts:    {len(meshes)}")
    print(f"  textured:      {sum(part.texture is not None and part.mesh.uvs is not None for part in meshes)}")
    print(f"  colored:       {sum(part.texture is None for part in meshes)}")
    print(f"  bounds min:    {np.round(bounds[0], 6)}")
    print(f"  bounds max:    {np.round(bounds[1], 6)}")
    return meshes


def _load_scene_glb_meshes(scene_glb: Path) -> list[SceneGlbMesh]:
    return _load_glb_meshes(scene_glb, label="scene")


def _scene_visual_cfg() -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    cfg.is_visible = True
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    cfg.collision_group = 0
    return cfg


def _scene_collision_cfg(friction: float) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    cfg.mu = float(friction)
    cfg.restitution = 0.0
    cfg.ke = 5.0e4
    cfg.kd = 5.0e2
    cfg.kf = 1.0e3
    cfg.is_visible = False
    cfg.has_shape_collision = True
    cfg.has_particle_collision = True
    return cfg


def _load_scene_collision_boxes(path: Path) -> list[SceneCollisionBox]:
    spec_path = path if path.is_absolute() else (Path.cwd() / path)
    spec_path = spec_path.resolve()
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    if data.get("format") != "newton_scene_collision_boxes_v1":
        raise ValueError(f"Unsupported scene collision spec format: {data.get('format')!r}")

    boxes = []
    for index, item in enumerate(data.get("collision_boxes", [])):
        size = tuple(float(v) for v in item["size"])
        if len(size) != 3 or min(size) <= 0.0:
            raise ValueError(f"Scene collision box {index} must have positive size [x, y, z]")
        boxes.append(
            SceneCollisionBox(
                name=str(item.get("name", f"scene_collision_box_{index:02d}")),
                pos=tuple(float(v) for v in item.get("position", (0.0, 0.0, 0.0))),
                rpy_deg=tuple(float(v) for v in item.get("rpy_deg", (0.0, 0.0, 0.0))),
                size=size,
                friction=float(item.get("friction", 1.0)),
                visible=bool(item.get("visible", False)),
            )
        )
    return boxes


def _scene_collision_box_world_pose(
    box: SceneCollisionBox,
    *,
    scene_pos: tuple[float, float, float],
    scene_rpy_deg: tuple[float, float, float],
    scene_scale: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene_rotation = _rotation_from_euler_deg(scene_rpy_deg)
    scene_pos_array = np.asarray(scene_pos, dtype=np.float64)
    scene_scale_array = np.asarray(scene_scale, dtype=np.float64)
    box_pos = np.asarray(box.pos, dtype=np.float64)
    box_world_pos = scene_pos_array + scene_rotation @ (box_pos * scene_scale_array)
    box_world_rotation = scene_rotation @ _rotation_from_euler_deg(box.rpy_deg)
    box_size = np.asarray(box.size, dtype=np.float64) * scene_scale_array
    return box_world_pos, box_world_rotation, box_size


def _add_scene_collision_boxes(
    builder: newton.ModelBuilder,
    boxes: list[SceneCollisionBox],
    *,
    scene_pos: tuple[float, float, float],
    scene_rpy_deg: tuple[float, float, float],
    scene_scale: tuple[float, float, float],
) -> None:
    for index, box in enumerate(boxes):
        box_world_pos, box_world_rotation, box_size = _scene_collision_box_world_pose(
            box,
            scene_pos=scene_pos,
            scene_rpy_deg=scene_rpy_deg,
            scene_scale=scene_scale,
        )
        box_world_rpy = _euler_deg_from_rotation(box_world_rotation)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                wp.vec3(*box_world_pos.tolist()),
                _quat_from_rpy(*(np.deg2rad(box_world_rpy).tolist())),
            ),
            hx=0.5 * float(box_size[0]),
            hy=0.5 * float(box_size[1]),
            hz=0.5 * float(box_size[2]),
            cfg=_scene_collision_cfg(box.friction),
            color=(1.0, 0.72, 0.15),
            label=f"{box.name}_{index:02d}",
        )


def _oriented_cylinder_aabb_half_extents(rotation: np.ndarray, radius: float, height: float) -> np.ndarray:
    axis = np.asarray(rotation, dtype=np.float64).reshape(3, 3)[:, 2]
    half_height = 0.5 * float(height)
    radius = float(radius)
    radial = np.sqrt(np.maximum(0.0, 1.0 - axis * axis)) * radius
    return np.abs(axis) * half_height + radial


def _lift_bottle_above_scene_collision(
    bottle_spec,
    boxes: list[SceneCollisionBox],
    *,
    scene_pos: tuple[float, float, float],
    scene_rpy_deg: tuple[float, float, float],
    scene_scale: tuple[float, float, float],
    clearance: float,
) -> None:
    if not boxes:
        return

    bottle_pos = np.asarray(bottle_spec.pos, dtype=np.float64)
    bottle_rotation = _rotation_from_euler_deg(tuple(bottle_spec.rpy_deg))
    bottle_half_extents = _oriented_cylinder_aabb_half_extents(
        bottle_rotation,
        float(bottle_spec.radius),
        float(bottle_spec.height),
    )
    target_z = float(bottle_pos[2])

    for box in boxes:
        box_pos, box_rotation, box_size = _scene_collision_box_world_pose(
            box,
            scene_pos=scene_pos,
            scene_rpy_deg=scene_rpy_deg,
            scene_scale=scene_scale,
        )
        box_half_extents = np.abs(box_rotation) @ (0.5 * box_size)
        xy_overlap = np.all(
            np.abs(bottle_pos[:2] - box_pos[:2]) <= box_half_extents[:2] + bottle_half_extents[:2] + 0.02
        )
        if not xy_overlap:
            continue

        box_top_z = float(box_pos[2] + box_half_extents[2])
        target_z = max(target_z, box_top_z + float(bottle_half_extents[2]) + float(clearance))

    if target_z > bottle_pos[2] + 1.0e-6:
        old_z = float(bottle_spec.pos[2])
        bottle_spec.pos[2] = target_z
        print(f"Raised dynamic bottle above scene collision: z {old_z:.6f} -> {target_z:.6f}")


def _load_d455_config(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    body_size = tuple(float(v) for v in data.get("body", {}).get("body_size_m_xyz", D455_BODY_SIZE_FALLBACK))
    if len(body_size) != 3:
        raise ValueError(f"{path} body.body_size_m_xyz must contain three numbers")
    preset = data.get("genesis_presets", {}).get("rgb_native_1280x800_30fps", {})
    return {
        "body_size": body_size,
        "rgb_res": tuple(int(v) for v in preset.get("res", (1280, 800))),
        "rgb_fov": float(preset.get("fov", 65.0)),
        "rgb_near": float(preset.get("near", 0.05)),
        "rgb_far": float(preset.get("far", 100.0)),
    }


def _load_d405_config(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    body_size = tuple(float(v) for v in data.get("body", {}).get("body_size_m_xyz", D405_BODY_SIZE_FALLBACK))
    if len(body_size) != 3:
        raise ValueError(f"{path} body.body_size_m_xyz must contain three numbers")
    resolution = data.get("resolution", {})
    fov_degrees = data.get("fov_degrees", {})
    return {
        "body_size": body_size,
        "res": (int(resolution.get("width", 640)), int(resolution.get("height", 480))),
        "fov": float(fov_degrees.get("vertical", 58.0)),
        "near": D405_CAMERA_NEAR_M,
        "far": D405_CAMERA_FAR_M,
    }


def _load_d405_mount_json(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mount_path = path if path.is_absolute() else (Path.cwd() / path)
    mount_path = mount_path.resolve()
    data = json.loads(mount_path.read_text(encoding="utf-8"))
    constants = data.get("python_constants") if isinstance(data.get("python_constants"), dict) else {}

    pos = data.get("offset_xyz_in_connector_frame_m")
    if pos is None:
        pos = constants.get("RIGHT_D405_CONNECTOR_REL_POS_M")
    euler = data.get("euler_xyz_in_connector_frame_deg")
    if euler is None:
        euler = constants.get("RIGHT_D405_CONNECTOR_REL_EULER_DEG")
    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
        raise ValueError(f"{mount_path} missing D405 connector-frame position")
    if not isinstance(euler, (list, tuple)) or len(euler) != 3:
        raise ValueError(f"{mount_path} missing D405 connector-frame Euler angles")
    return tuple(float(v) for v in pos), tuple(float(v) for v in euler)


def _resolve_d405_mount_args(args) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
    source = "defaults"
    json_pos: tuple[float, float, float] | None = None
    json_euler: tuple[float, float, float] | None = None
    mount_path = _resolve_optional_file(args.d405_mount_json) if args.d405_mount_json is not None else None
    if mount_path is not None:
        json_pos, json_euler = _load_d405_mount_json(mount_path)
        source = f"json:{mount_path}"
    elif args.d405_mount_json is not None:
        print(f"Warning: D405 mount JSON not found, using defaults: {args.d405_mount_json}")

    final_pos = tuple(float(v) for v in (args.d405_connector_rel_pos or json_pos or RIGHT_D405_CONNECTOR_REL_POS_M))
    final_euler = tuple(
        float(v) for v in (args.d405_connector_rel_euler or json_euler or RIGHT_D405_CONNECTOR_REL_EULER_DEG)
    )
    if args.d405_connector_rel_pos is not None or args.d405_connector_rel_euler is not None:
        source = f"{source}+cli_override" if mount_path is not None else "cli"
    return final_pos, final_euler, source


def _find_builder_body_index(builder: newton.ModelBuilder, label_suffix: str) -> int | None:
    return next((i for i, label in enumerate(builder.body_label) if label.endswith(label_suffix)), None)


def _add_d405_body_visual(
    builder: newton.ModelBuilder,
    *,
    body_size: tuple[float, float, float],
    rel_pos: tuple[float, float, float],
    rel_euler_deg: tuple[float, float, float],
) -> None:
    connector_body = _find_builder_body_index(builder, "/right_connector")
    if connector_body is None:
        print("Warning: right_connector body not found, skipping D405 body visual")
        return

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    cfg.is_visible = True
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    cfg.collision_group = 0
    builder.add_shape_box(
        body=connector_body,
        xform=wp.transform(
            wp.vec3(*rel_pos),
            _quat_from_rpy(*(np.deg2rad(rel_euler_deg).tolist())),
        ),
        hx=0.5 * float(body_size[0]),
        hy=0.5 * float(body_size[1]),
        hz=0.5 * float(body_size[2]),
        cfg=cfg,
        color=(0.78, 0.78, 0.76),
        label="right_d405_body",
    )


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True

        self.viewer = viewer
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.update_step_interval = 1
        self.graph = None
        self.simulate_enabled = args.simulate
        self.scene_pos = [args.scene_pos_x, args.scene_pos_y, args.scene_pos_z]
        self.scene_rpy_deg = [args.scene_roll, args.scene_pitch, args.scene_yaw]
        self.scene_scale = [args.scene_scale, args.scene_scale, args.scene_scale]
        self.bottle_pos = [args.bottle_pos_x, args.bottle_pos_y, args.bottle_pos_z]
        self.bottle_rpy_deg = [args.bottle_roll, args.bottle_pitch, args.bottle_yaw]
        self.dynamic_bottle_handles: dict[str, object] | None = None
        self.d455_preview_enabled = args.d455_preview
        self.d405_preview_enabled = args.d405_preview
        self.d405_connector_rel_pos, self.d405_connector_rel_euler, self.d405_mount_source = _resolve_d405_mount_args(args)
        self.d405_body_size = tuple(float(v) for v in _load_d405_config(args.d405_json)["body_size"])
        self.d455_image_size = tuple(int(v) for v in args.d455_image_size)
        if args.d455_width is not None or args.d455_height is not None:
            self.d455_image_size = (
                int(args.d455_height or self.d455_image_size[0]),
                int(args.d455_width or self.d455_image_size[1]),
            )
        self.d455_roi_zoom = float(args.d455_roi_zoom)
        self.d455_roi_center_x = float(args.d455_roi_center_x)
        self.d455_roi_center_y = float(args.d455_roi_center_y)
        self.d455_preview_scale = int(args.d455_preview_scale)
        self.d455_opencv_window = bool(args.d455_opencv_window)
        self.d405_preview_scale = int(args.d405_preview_scale)
        self.d405_opencv_window = bool(args.d405_opencv_window)
        self.d455_front_clearance = float(args.d455_front_clearance)
        self._d455_preview_started = False
        self._d405_preview_started = False
        self._d455_pose_logged = False
        self._d405_pose_logged = False
        self.camera_sensor = None
        self.d455_preview: CameraPreview | None = None
        self.d405_preview: CameraPreview | None = None
        self.teleop_session = None
        self.teleop_robot = None
        self.teleop_voice_policy = None
        self.teleop_xr_status_publisher = None
        self.teleop_mode = "ready"
        self.teleop_last_event = "session_created"
        self.teleop_exit_requested = False
        self._teleop_session_entered = False

        urdf_path = _resolve_urdf(args.urdf)
        builder = newton.ModelBuilder(up_axis=URDF_UP_AXIS, gravity=args.gravity)
        builder.rigid_gap = float(args.rigid_gap)
        builder.default_joint_cfg.armature = args.armature
        builder.default_joint_cfg.target_ke = args.target_ke
        builder.default_joint_cfg.target_kd = args.target_kd
        builder.default_shape_cfg.mu = args.friction

        print(f"Importing URDF: {urdf_path}")
        urdf_first_body = len(builder.body_q)
        urdf_first_shape = len(builder.shape_body)
        builder.add_urdf(
            urdf_path,
            floating=args.floating,
            enable_self_collisions=args.self_collisions,
            ignore_inertial_definitions=args.ignore_inertial_definitions,
            collapse_massless_fixed_root=args.collapse_massless_fixed_root,
            up_axis=URDF_UP_AXIS,
        )
        _filter_urdf_collisions_to_l10_hand(
            builder,
            urdf_first_shape,
            len(builder.shape_body),
            l10_friction=args.l10_friction,
            l10_ke=args.l10_contact_ke,
            l10_kd=args.l10_contact_kd,
            l10_kf=args.l10_contact_kf,
            l10_mu_torsional=args.l10_torsional_friction,
            l10_mu_rolling=args.l10_rolling_friction,
        )
        _set_initial_arm_poses(
            builder,
            left_q=tuple(args.initial_left_arm_q),
            right_q=tuple(args.initial_right_arm_q),
        )
        if args.robot_kinematic:
            _make_urdf_bodies_kinematic(builder, urdf_first_body, len(builder.body_q))
        if args.d405_body_visual:
            _add_d405_body_visual(
                builder,
                body_size=self.d405_body_size,
                rel_pos=self.d405_connector_rel_pos,
                rel_euler_deg=self.d405_connector_rel_euler,
            )
            print(
                "Mounted D405 body visual:"
                f" source={self.d405_mount_source}"
                f" rel_pos={np.round(self.d405_connector_rel_pos, 6).tolist()}"
                f" rel_euler_deg={np.round(self.d405_connector_rel_euler, 3).tolist()}"
                f" body_size={np.round(self.d405_body_size, 6).tolist()}"
            )

        self.scene_glb = _resolve_optional_file(args.scene_glb)
        if self.scene_glb is not None:
            scene_xform = wp.transform(
                wp.vec3(self.scene_pos[0], self.scene_pos[1], self.scene_pos[2]),
                _quat_from_rpy(*(np.deg2rad(self.scene_rpy_deg).tolist())),
            )
            scene_visual_cfg = _scene_visual_cfg()
            for index, part in enumerate(_load_scene_glb_meshes(self.scene_glb)):
                builder.add_shape_mesh(
                    body=-1,
                    mesh=part.mesh,
                    xform=scene_xform,
                    scale=tuple(self.scene_scale),
                    cfg=scene_visual_cfg,
                    color=part.color,
                    label=f"scene_glb_part_{index:02d}",
                )
        else:
            print(f"Warning: scene GLB not found, skipping: {args.scene_glb}")

        self.scene_collision_spec_path = _resolve_optional_file(args.scene_collision_spec)
        scene_collision_boxes: list[SceneCollisionBox] = []
        if self.scene_collision_spec_path is not None:
            scene_collision_boxes = _load_scene_collision_boxes(self.scene_collision_spec_path)
            _add_scene_collision_boxes(
                builder,
                scene_collision_boxes,
                scene_pos=tuple(self.scene_pos),
                scene_rpy_deg=tuple(self.scene_rpy_deg),
                scene_scale=tuple(self.scene_scale),
            )
            print(
                "Loaded scene collision boxes:"
                f" spec={self.scene_collision_spec_path}"
                f" boxes={len(scene_collision_boxes)}"
            )
        elif args.scene_collision_spec != DEFAULT_SCENE_COLLISION_SPEC:
            print(f"Warning: scene collision spec not found, skipping: {args.scene_collision_spec}")

        self.dynamic_bottle_spec_path = _resolve_optional_file(args.dynamic_bottle_spec)
        if self.dynamic_bottle_spec_path is not None:
            bottle_position, bottle_rotation = self._bottle_world_pose()
            dynamic_bottle_spec = load_dynamic_bottle_spec(self.dynamic_bottle_spec_path)
            dynamic_bottle_spec.pos = [float(v) for v in bottle_position]
            dynamic_bottle_spec.rpy_deg = list(_euler_deg_from_rotation(bottle_rotation))
            dynamic_bottle_spec.friction = max(float(dynamic_bottle_spec.friction), float(args.dynamic_bottle_friction))
            if args.lift_bottle_above_scene_collision:
                _lift_bottle_above_scene_collision(
                    dynamic_bottle_spec,
                    scene_collision_boxes,
                    scene_pos=tuple(self.scene_pos),
                    scene_rpy_deg=tuple(self.scene_rpy_deg),
                    scene_scale=tuple(self.scene_scale),
                    clearance=args.bottle_scene_collision_clearance,
                )
            self.dynamic_bottle_handles = build_dynamic_bottle(builder, dynamic_bottle_spec)
            print(
                "Loaded dynamic bottle:"
                f" spec={self.dynamic_bottle_spec_path}"
                f" visual={dynamic_bottle_spec.visual_glb}"
                f" scene_pos={np.round(self.bottle_pos, 6).tolist()}"
                f" body_pos={np.round(dynamic_bottle_spec.pos, 6).tolist()}"
                f" body_rpy={np.round(dynamic_bottle_spec.rpy_deg, 6).tolist()}"
                f" radius={dynamic_bottle_spec.radius:g}"
                f" height={dynamic_bottle_spec.height:g}"
                f" mass={dynamic_bottle_spec.mass:g}"
                f" friction={dynamic_bottle_spec.friction:g}"
                " material=plastic"
            )
        else:
            print(f"Warning: dynamic bottle spec not found, skipping: {args.dynamic_bottle_spec}")

        if args.add_ground:
            builder.add_ground_plane()

        self.model = builder.finalize(device=args.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = (
            newton.solvers.SolverXPBD(
                self.model,
                iterations=args.solver_iterations,
                rigid_contact_relaxation=float(args.rigid_contact_relaxation),
                angular_damping=float(args.angular_damping),
            )
            if self.simulate_enabled
            else None
        )

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        _assert_finite_state(self.state_0, "Initial FK")
        self._capture_initial_state_snapshot()
        _print_model_summary(self.model)

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(1.8, -2.4, 1.2), pitch=-20.0, yaw=135.0)

        self.setup_camera_previews(args)

        if args.quest_teleop:
            self.setup_quest_teleop(args)

        if self.simulate_enabled and args.capture_graph and not args.quest_teleop:
            self.capture()
        elif args.quest_teleop and args.capture_graph:
            print("Quest teleop enabled: CUDA graph capture disabled for live joint-state edits.")

    def _bottle_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        scene_rotation = _rotation_from_euler_deg(tuple(self.scene_rpy_deg))
        bottle_rotation = _rotation_from_euler_deg(tuple(self.bottle_rpy_deg))
        scene_pos = np.asarray(self.scene_pos, dtype=np.float64)
        scene_scale = np.asarray(self.scene_scale, dtype=np.float64)
        bottle_rel_pos = np.asarray(self.bottle_pos, dtype=np.float64)
        bottle_world_pos = scene_pos + scene_rotation @ (bottle_rel_pos * scene_scale)
        bottle_world_rotation = scene_rotation @ bottle_rotation
        return bottle_world_pos, bottle_world_rotation

    def _capture_initial_state_snapshot(self) -> None:
        self._initial_joint_q = self.model.joint_q.numpy().copy()
        self._initial_joint_qd = self.model.joint_qd.numpy().copy()
        self._initial_body_q = self.state_0.body_q.numpy().copy()
        self._initial_body_qd = self.state_0.body_qd.numpy().copy()

    def reset_scene_to_initial(self) -> None:
        if not all(
            hasattr(self, name)
            for name in ("_initial_joint_q", "_initial_joint_qd", "_initial_body_q", "_initial_body_qd")
        ):
            return

        joint_q = wp.array(self._initial_joint_q.copy(), dtype=wp.float32, device=self.model.device)
        joint_qd = wp.array(self._initial_joint_qd.copy(), dtype=wp.float32, device=self.model.device)
        body_q = wp.array(self._initial_body_q.copy(), dtype=wp.transform, device=self.model.device)
        body_qd = wp.array(self._initial_body_qd.copy(), dtype=wp.spatial_vector, device=self.model.device)

        self.model.joint_q = joint_q
        self.model.joint_qd = joint_qd
        self.state_0.joint_q = joint_q
        self.state_0.joint_qd = joint_qd
        self.state_0.body_q = body_q
        self.state_0.body_qd = body_qd
        self.state_1.joint_q = wp.clone(joint_q)
        self.state_1.joint_qd = wp.clone(joint_qd)
        self.state_1.body_q = wp.clone(body_q)
        self.state_1.body_qd = wp.clone(body_qd)
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        self.control.clear()
        self.control.joint_target_q = wp.clone(joint_q)
        self.control.joint_target_qd = wp.clone(joint_qd)
        self.contacts = self.model.contacts()
        self.model.bvh_refit_shapes(self.state_0)
        if self.model.particle_count:
            self.model.bvh_refit_particles(self.state_0)
        self.sim_time = 0.0

        if self.teleop_robot is not None and hasattr(self.teleop_robot, "reset_to_scene_state"):
            self.teleop_robot.reset_to_scene_state()
        elif self.teleop_robot is not None and hasattr(self.teleop_robot, "reset_relative_anchor"):
            self.teleop_robot.reset_relative_anchor()
        print("[newton-quest-teleop] scene reset to initial state", flush=True)

    def capture(self) -> None:
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            if hasattr(self.viewer, "apply_forces"):
                self.viewer.apply_forces(self.state_0)

            refresh_contacts = (substep % self.update_step_interval) == 0
            if refresh_contacts:
                self.model.collide(self.state_0, self.contacts)

            if self.solver is None:
                continue
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        if self.teleop_voice_policy is not None:
            if self._apply_teleop_voice_events(self.teleop_voice_policy.update()):
                print("[newton-quest-teleop] voice exit requested", flush=True)
                self.teleop_exit_requested = True

        self._publish_teleop_xr_status()

        if self.teleop_exit_requested and hasattr(self.viewer, "close"):
            self.viewer.close()
            return

        if self.teleop_session is not None:
            self.teleop_session.step()

        if not self.simulate_enabled:
            self.sim_time += self.frame_dt
            return

        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def setup_quest_teleop(self, args) -> None:
        from teleop_stack.robots.newton_runtime import NewtonRuntimeRobotConfig, NewtonRuntimeRobotInterface
        from teleop_stack.robots.nero_runtime import NeroTeleopMappingConfig
        from teleop_stack.session.overlay_hand_log_session import OverlayHandLogSession, OverlayHandLogSessionConfig
        from teleop_stack.session.quest_session import QuestRobotSession, QuestRobotSessionConfig
        from teleop_stack.session.voice_controls import VoiceTeleopControlConfig, VoiceTeleopControlPolicy
        from teleop_stack.session.xr_status import XrTeleopStatusPublisher

        axis_map = _normalize_axis_map(args.teleop_input_axis_map)
        mapping = NeroTeleopMappingConfig(
            translation_scale_xyz=tuple(float(v) for v in args.teleop_translation_scale),
            workspace_origin_xyz=tuple(float(v) for v in args.teleop_workspace_origin),
            input_axis_map=axis_map,  # type: ignore[arg-type]
            openxr_coordinate_adapter=args.teleop_openxr_coordinate_adapter,
            use_teleop_orientation=bool(args.teleop_orientation),
            fixed_quaternion_wxyz=args.teleop_fixed_quaternion_wxyz,
            orientation_axis_map=args.teleop_orientation_axis_map,
            orientation_max_speed_rad_s=float(args.teleop_orientation_max_speed_rad_s),
            orientation_tool_offset_wxyz=args.teleop_orientation_tool_offset_wxyz,
            orientation_reference_mode=args.teleop_orientation_reference_mode,
            orientation_source=args.teleop_orientation_source,
        )
        robot = NewtonRuntimeRobotInterface(
            self,
            NewtonRuntimeRobotConfig(
                arm_side=args.teleop_arm_side,
                drive_ik=bool(args.teleop_drive_ik),
                relative_control=bool(args.teleop_relative_control),
                eef_body_suffix_by_side={
                    "left": str(args.teleop_left_eef_body_suffix),
                    "right": str(args.teleop_right_eef_body_suffix),
                },
                openxr_yaw_recenter=bool(args.teleop_openxr_yaw_recenter),
                finite_difference_rad=float(args.teleop_finite_difference_rad),
                hand_max_joint_step_rad=float(args.teleop_hand_max_joint_step_rad),
                hand_publish_kinematic_velocity=bool(args.teleop_hand_publish_kinematic_velocity),
                mapping=mapping,
                ik_config_overrides={
                    "max_task_step_m": float(args.teleop_ik_max_task_step_m),
                    "max_rotation_step_rad": float(args.teleop_ik_max_rotation_step_rad),
                    "orientation_weight": float(args.teleop_ik_orientation_weight),
                    "max_joint_step_rad": float(args.teleop_ik_max_joint_step_rad),
                    "max_joint_velocity_rad_s": float(args.teleop_ik_max_joint_velocity_rad_s),
                    "damping_lambda": float(args.teleop_ik_damping_lambda),
                },
            ),
            print_every_n=args.teleop_print_every_n_frames,
        )
        self.teleop_robot = robot
        self.teleop_mode = "ready" if args.teleop_require_engage else "engaged"
        self.teleop_last_event = "session_started"
        robot.set_command_gate(
            self.teleop_mode == "engaged",
            mode=self.teleop_mode,
            last_event=self.teleop_last_event,
        )
        if args.teleop_input_source == "overlay-log":
            self.teleop_session = OverlayHandLogSession(
                OverlayHandLogSessionConfig(
                    trace_path=str(args.teleop_overlay_hand_log_path),
                    arm_side=args.teleop_arm_side,
                    hand_side=args.teleop_arm_side,
                    use_teleop_orientation=args.teleop_arm_pose_command_mode == "raw_wrist_position_full_orientation",
                    loop_hz=float(args.teleop_loop_hz),
                    print_every_n_frames=int(args.teleop_print_every_n_frames),
                    stale_after_s=float(args.teleop_overlay_stale_after_s),
                    teleop_trace_path=args.teleop_trace_path,
                ),
                robot,
            )
        else:
            self.teleop_session = QuestRobotSession(
                QuestRobotSessionConfig(
                    app_name=args.teleop_app_name,
                    arm_side=args.teleop_arm_side,
                    pose_input_mode=args.teleop_pose_input_mode,
                    arm_pose_command_mode=args.teleop_arm_pose_command_mode,
                    fixed_arm_orientation_xyzw=args.teleop_fixed_arm_orientation_xyzw,
                    use_wrist_position_for_hand=bool(args.teleop_use_wrist_position_for_hand),
                    use_wrist_rotation_for_hand=bool(args.teleop_use_wrist_rotation_for_hand),
                    palm_plane_wrist_orientation_blend_alpha=float(args.teleop_palm_plane_blend_alpha),
                    loop_hz=float(args.teleop_loop_hz),
                    print_every_n_frames=int(args.teleop_print_every_n_frames),
                    enable_head_tracker=bool(args.teleop_enable_head_tracker),
                    enable_synthetic_hands_plugin=bool(args.teleop_synthetic_hands_plugin),
                    isaac_teleop_root=args.teleop_isaac_teleop_root,
                    startup_timeout_s=float(args.teleop_startup_timeout_s),
                    startup_retry_interval_s=float(args.teleop_startup_retry_interval_s),
                    teleop_trace_path=args.teleop_trace_path,
                ),
                robot,
            )
        self.teleop_session.__enter__()
        self._teleop_session_entered = True
        if args.teleop_enable_voice_controls:
            self.teleop_voice_policy = VoiceTeleopControlPolicy(
                VoiceTeleopControlConfig(
                    host=args.teleop_voice_control_host,
                    port=int(args.teleop_voice_control_port),
                )
            )
            self.teleop_voice_policy.connect()
        self.teleop_xr_status_publisher = XrTeleopStatusPublisher(args.teleop_xr_status_path)
        self._publish_teleop_xr_status(lifecycle_event="session_started", force=True)
        atexit.register(self.close_quest_teleop)
        print(
            f"[newton-quest-teleop] teleop ready source={args.teleop_input_source}. "
            "Say 开始 to engage, 暂停 to clutch, 继续 to resume, 重置 to recenter, 停止 to hold.",
            flush=True,
        )

    def _set_teleop_mode(self, mode: str, event: str) -> None:
        self.teleop_mode = str(mode)
        self.teleop_last_event = str(event)
        if self.teleop_robot is not None and hasattr(self.teleop_robot, "set_command_gate"):
            self.teleop_robot.set_command_gate(
                self.teleop_mode == "engaged",
                mode=self.teleop_mode,
                last_event=self.teleop_last_event,
            )
        print(f"[newton-quest-teleop] mode={self.teleop_mode} event={self.teleop_last_event}", flush=True)

    def _apply_teleop_voice_events(self, events) -> bool:
        if not events.commands_seen:
            return False
        if events.estop_requested:
            self._set_teleop_mode("fault", "estop")
        if events.recenter_requested:
            self.reset_scene_to_initial()
            self._set_teleop_mode("ready", "scene_reset")
            self._publish_teleop_xr_status(force=True)
            return bool(events.exit_requested)
        if events.clutch_requested:
            self._set_teleop_mode("clutched", "entered_clutch")
        if events.resume_requested:
            if self.teleop_robot is not None and hasattr(self.teleop_robot, "reset_relative_anchor"):
                self.teleop_robot.reset_relative_anchor()
            self._set_teleop_mode("engaged", "resumed_from_clutch")
        if events.engage_requested:
            if self.teleop_robot is not None and hasattr(self.teleop_robot, "reset_relative_anchor"):
                self.teleop_robot.reset_relative_anchor()
            self._set_teleop_mode("engaged", "engaged")
        if events.stop_requested:
            if self.teleop_robot is not None and hasattr(self.teleop_robot, "reset_relative_anchor"):
                self.teleop_robot.reset_relative_anchor()
            self._set_teleop_mode("ready", "disengaged")
        self._publish_teleop_xr_status(force=True)
        return bool(events.exit_requested)

    def _publish_teleop_xr_status(self, *, lifecycle_event: str | None = None, force: bool = False) -> None:
        if self.teleop_xr_status_publisher is None:
            return
        if self.teleop_robot is not None and hasattr(self.teleop_robot, "xr_status_snapshot"):
            snapshot = self.teleop_robot.xr_status_snapshot(mode=self.teleop_mode, last_event=self.teleop_last_event)
        else:
            snapshot = {"mode": self.teleop_mode, "last_event": self.teleop_last_event}
        self.teleop_xr_status_publisher.publish(
            snapshot=snapshot,
            lifecycle_event=lifecycle_event,
            force=force,
        )

    def close_quest_teleop(self) -> None:
        if self.teleop_xr_status_publisher is not None:
            self._publish_teleop_xr_status(lifecycle_event="session_stopped", force=True)
            self.teleop_xr_status_publisher = None
        if self.teleop_voice_policy is not None:
            self.teleop_voice_policy.disconnect()
            self.teleop_voice_policy = None
        if self.teleop_session is None or not self._teleop_session_entered:
            return
        session = self.teleop_session
        self._teleop_session_entered = False
        self.teleop_session = None
        self.teleop_robot = None
        session.__exit__(None, None, None)

    def render(self) -> None:
        self.render_camera_previews()

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def setup_camera_previews(self, args) -> None:
        if not self.d455_preview_enabled and not self.d405_preview_enabled:
            return

        camera_render_config = SensorTiledCamera.RenderConfig(
            enable_textures=bool(args.camera_textures),
            enable_shadows=False,
        )
        self.camera_sensor = SensorTiledCamera(
            model=self.model,
            config=camera_render_config,
            load_textures=bool(args.camera_textures),
        )
        self.camera_sensor.utils.create_default_light(enable_shadows=False, direction=wp.vec3f(0.0, 0.0, -1.0))

        d455_config = _load_d455_config(args.d455_json)
        d455_native_width, d455_native_height = tuple(int(v) for v in d455_config["rgb_res"])
        d455_width = int(args.d455_render_width or d455_native_width)
        d455_height = int(args.d455_render_height or d455_native_height)
        self.d455_preview = CameraPreview(
            name="D455 RGB",
            enabled=self.d455_preview_enabled,
            width=d455_width,
            height=d455_height,
            fov_deg=float(args.d455_fov if args.d455_fov is not None else d455_config["rgb_fov"]),
            camera_rays=self.camera_sensor.utils.compute_pinhole_camera_rays(
                d455_width,
                d455_height,
                np.deg2rad(float(args.d455_fov if args.d455_fov is not None else d455_config["rgb_fov"])),
            ),
            color_image=self.camera_sensor.utils.create_color_image_output(d455_width, d455_height, 1),
        )
        self.d455_body_size = tuple(float(v) for v in d455_config["body_size"])

        d405_config = _load_d405_config(args.d405_json)
        d405_width = int(args.d405_width or d405_config["res"][0])
        d405_height = int(args.d405_height or d405_config["res"][1])
        self.d405_preview = CameraPreview(
            name="D405 RGB",
            enabled=self.d405_preview_enabled,
            width=d405_width,
            height=d405_height,
            fov_deg=float(args.d405_fov if args.d405_fov is not None else d405_config["fov"]),
            camera_rays=self.camera_sensor.utils.compute_pinhole_camera_rays(
                d405_width,
                d405_height,
                np.deg2rad(float(args.d405_fov if args.d405_fov is not None else d405_config["fov"])),
            ),
            color_image=self.camera_sensor.utils.create_color_image_output(d405_width, d405_height, 1),
        )
        self.d405_body_size = tuple(float(v) for v in d405_config["body_size"])

        print(
            "Camera previews:"
            f" textures={bool(args.camera_textures)}"
            f" D455 pose_source=urdf:{D455_BODY_LABEL_SUFFIX}"
            f" D455 enabled={self.d455_preview_enabled} render={d455_width}x{d455_height}"
            f" output={self.d455_image_size[1]}x{self.d455_image_size[0]}"
            f" fov={self.d455_preview.fov_deg:g}"
            f" local_x={self.d455_body_size[0] * 0.5 + self.d455_front_clearance:g}"
            f" roi_zoom={self.d455_roi_zoom:g}"
            f" roi_center=({self.d455_roi_center_x:g},{self.d455_roi_center_y:g});"
            f" D405 enabled={self.d405_preview_enabled} {d405_width}x{d405_height} fov={self.d405_preview.fov_deg:g}"
            f" mount_source={self.d405_mount_source}"
        )

    def render_camera_previews(self) -> None:
        if self.camera_sensor is None:
            return

        self.model.bvh_refit_shapes(self.state_0)
        self.model.bvh_refit_particles(self.state_0)

        if self.d455_preview is not None and self.d455_preview.enabled:
            self.d455_preview.camera_transform = wp.array(
                [[self.compute_d455_camera_transform()]], dtype=wp.transformf, device=self.model.device
            )
            self.camera_sensor.update(
                self.state_0,
                self.d455_preview.camera_transform,
                self.d455_preview.camera_rays,
                color_image=self.d455_preview.color_image,
                clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
            )
            d455_image = self.d455_model_input_image()
            self.viewer.log_image("D455 ego_view model input", d455_image)
            if self.d455_opencv_window:
                _show_rgb_preview(
                    self,
                    "D455 ego_view model input",
                    d455_image,
                    scale=max(1, self.d455_preview_scale),
                )

        if self.d405_preview is not None and self.d405_preview.enabled:
            self.d405_preview.camera_transform = wp.array(
                [[self.compute_d405_camera_transform()]], dtype=wp.transformf, device=self.model.device
            )
            self.camera_sensor.update(
                self.state_0,
                self.d405_preview.camera_transform,
                self.d405_preview.camera_rays,
                color_image=self.d405_preview.color_image,
                clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
            )
            self.viewer.log_image(
                self.d405_preview.name,
                self.camera_sensor.utils.to_rgba_from_color(self.d405_preview.color_image),
            )
            d405_image = _packed_color_image_to_rgb_hwc(self.d405_preview.color_image)
            if not self._d405_preview_started:
                print(
                    "[d405-preview] showing raw D405 RGB "
                    f"source=({self.d405_preview.width},{self.d405_preview.height}) "
                    f"min={int(d405_image.min())} max={int(d405_image.max())} "
                    f"mean={float(d405_image.mean()):.2f}",
                    flush=True,
                )
                self._d405_preview_started = True
            if self.d405_opencv_window:
                _show_rgb_preview(
                    self,
                    "D405 RGB",
                    d405_image,
                    scale=max(1, self.d405_preview_scale),
                )

    def d455_model_input_image(self) -> np.ndarray:
        if self.d455_preview is None:
            return np.zeros((*self.d455_image_size, 3), dtype=np.uint8)
        image = _packed_color_image_to_rgb_hwc(self.d455_preview.color_image)
        image = _roi_crop_zoom_hwc(
            image,
            zoom=self.d455_roi_zoom,
            center_x=self.d455_roi_center_x,
            center_y=self.d455_roi_center_y,
        )
        image = _resize_with_pad(image, self.d455_image_size[0], self.d455_image_size[1])
        image = image.astype(np.uint8, copy=False)
        if not self._d455_preview_started:
            source_height = int(self.d455_preview.height) if self.d455_preview is not None else 0
            source_width = int(self.d455_preview.width) if self.d455_preview is not None else 0
            crop_x, crop_y, crop_width, crop_height = _roi_crop_rect(
                source_width,
                source_height,
                zoom=self.d455_roi_zoom,
                center_x=self.d455_roi_center_x,
                center_y=self.d455_roi_center_y,
            )
            print(
                "[d455-preview] showing D455 ego_view model input "
                f"source=({source_width},{source_height}) "
                f"crop=(x={crop_x},y={crop_y},w={crop_width},h={crop_height}) "
                f"output={self.d455_image_size} roi_zoom={self.d455_roi_zoom:.2f} "
                f"center=({self.d455_roi_center_x:.2f},{self.d455_roi_center_y:.2f}) "
                f"min={int(image.min())} max={int(image.max())} mean={float(image.mean()):.2f}",
                flush=True,
            )
            self._d455_preview_started = True
        return image

    def _body_pose(self, label_suffix: str) -> tuple[np.ndarray, np.ndarray]:
        labels = self.model.body_label
        body_index = next((i for i, label in enumerate(labels) if label.endswith(label_suffix)), None)
        if body_index is None:
            raise ValueError(f"Body ending with {label_suffix!r} not found")
        body_q = self.state_0.body_q.numpy()[body_index]
        return np.asarray(body_q[:3], dtype=np.float64), _rotation_from_quat_xyzw(np.asarray(body_q[3:7]))

    def compute_d455_camera_transform(self) -> wp.transformf:
        d455_pos, d455_rotation = self._body_pose(D455_BODY_LABEL_SUFFIX)
        local_camera_pos = np.asarray(
            (float(self.d455_body_size[0]) * 0.5 + self.d455_front_clearance, 0.0, 0.0),
            dtype=np.float64,
        )
        camera_pos = d455_pos + d455_rotation @ local_camera_pos
        camera_forward = d455_rotation @ np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        camera_up = d455_rotation @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        if not self._d455_pose_logged:
            print(
                "[d455-preview] camera pose from URDF "
                f"body_pos={np.round(d455_pos, 6).tolist()} "
                f"local_pos={np.round(local_camera_pos, 6).tolist()} "
                f"camera_pos={np.round(camera_pos, 6).tolist()} "
                f"forward={np.round(camera_forward, 6).tolist()} "
                f"up={np.round(camera_up, 6).tolist()}",
                flush=True,
            )
            self._d455_pose_logged = True
        return _camera_transform_from_forward_up(camera_pos, camera_forward, camera_up)

    def compute_d405_camera_transform(self) -> wp.transformf:
        connector_pos, connector_rotation = self._body_pose("/right_connector")
        d405_rotation = connector_rotation @ _rotation_from_euler_deg(tuple(self.d405_connector_rel_euler))
        d405_pos = connector_pos + connector_rotation @ np.asarray(self.d405_connector_rel_pos, dtype=np.float64)
        local_camera_pos = np.asarray(self.d405_body_size, dtype=np.float64) * np.asarray(
            D405_CAMERA_LOCAL_POS_RATIO, dtype=np.float64
        )
        camera_pos = d405_pos + d405_rotation @ local_camera_pos
        camera_forward = d405_rotation @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        camera_up = d405_rotation @ np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        if not self._d405_pose_logged:
            print(
                "[d405-preview] camera pose from right_connector "
                f"mount_source={self.d405_mount_source} "
                f"connector_pos={np.round(connector_pos, 6).tolist()} "
                f"body_pos={np.round(d405_pos, 6).tolist()} "
                f"body_rel_pos={np.round(self.d405_connector_rel_pos, 6).tolist()} "
                f"body_rel_euler={np.round(self.d405_connector_rel_euler, 3).tolist()} "
                f"local_camera_pos={np.round(local_camera_pos, 6).tolist()} "
                f"camera_pos={np.round(camera_pos, 6).tolist()} "
                f"forward={np.round(camera_forward, 6).tolist()} "
                f"up={np.round(camera_up, 6).tolist()}",
                flush=True,
            )
            self._d405_pose_logged = True
        return _camera_transform_from_forward_up(camera_pos, camera_forward, camera_up)

    def test_final(self) -> None:
        _assert_finite_state(self.state_0, "Simulation")
        self.close_quest_teleop()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.description = __doc__
        parser.set_defaults(device="cuda:0", viewer="gl", paused=False)
        parser.add_argument(
            "urdf",
            nargs="?",
            type=Path,
            default=DEFAULT_URDF,
            help=f"URDF file to import. Defaults to {DEFAULT_URDF.relative_to(REPO_ROOT)}.",
        )
        parser.add_argument(
            "--scene-glb",
            type=Path,
            default=DEFAULT_SCENE_GLB,
            help=f"GLB scene to overlay. Defaults to {DEFAULT_SCENE_GLB.relative_to(REPO_ROOT)}.",
        )
        parser.add_argument(
            "--scene-collision-spec",
            type=Path,
            default=DEFAULT_SCENE_COLLISION_SPEC,
            help="Optional scene-local box collision proxy JSON. scene.glb itself stays visual-only.",
        )
        parser.add_argument(
            "--fps",
            type=float,
            default=60.0,
            help="Viewer simulation frame rate [Hz].",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=12,
            help="XPBD substeps per rendered frame.",
        )
        parser.add_argument(
            "--solver-iterations",
            type=int,
            default=6,
            help="XPBD iterations per substep.",
        )
        parser.add_argument(
            "--rigid-gap",
            type=float,
            default=DEFAULT_RIGID_GAP_M,
            help="Default rigid contact detection gap [m] for shapes without an explicit gap.",
        )
        parser.add_argument(
            "--rigid-contact-relaxation",
            type=float,
            default=0.01,
            help="XPBD rigid contact relaxation. Lower values make kinematic hand contacts less impulsive.",
        )
        parser.add_argument(
            "--angular-damping",
            type=float,
            default=0.12,
            help="XPBD angular damping for dynamic rigid bodies.",
        )
        parser.add_argument(
            "--simulate",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run XPBD simulation so the dynamic bottle can fall, contact, and be grasped.",
        )
        parser.add_argument(
            "--gravity",
            type=float,
            default=-9.81,
            help="Gravity acceleration [m/s^2] along the URDF Z-up axis when --simulate is enabled.",
        )
        parser.add_argument(
            "--floating",
            action="store_true",
            help="Import the URDF root with a floating base joint.",
        )
        parser.add_argument(
            "--self-collisions",
            action="store_true",
            help="Enable self-collisions during URDF import.",
        )
        parser.add_argument(
            "--robot-kinematic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Treat the imported URDF robot as kinematic/infinite-mass while keeping L10 collision shapes active.",
        )
        parser.add_argument(
            "--initial-left-arm-q",
            type=_vec7,
            default=INITIAL_LEFT_ARM_Q,
            help="Initial left Nero arm joint pose [rad], formatted q1,...,q7. Defaults to harness INITIAL_LEFT_ARM_Q.",
        )
        parser.add_argument(
            "--initial-right-arm-q",
            type=_vec7,
            default=INITIAL_RIGHT_ARM_Q,
            help="Initial right Nero arm joint pose [rad], formatted q1,...,q7. Defaults to harness INITIAL_RIGHT_ARM_Q.",
        )
        parser.add_argument(
            "--ignore-inertial-definitions",
            action="store_true",
            help="Compute inertial properties from geometry instead of URDF inertial tags.",
        )
        parser.add_argument(
            "--collapse-massless-fixed-root",
            action="store_true",
            help="Collapse massless fixed-root chains during import.",
        )
        parser.add_argument(
            "--add-ground",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Add a ground plane before finalizing the model.",
        )
        parser.add_argument(
            "--capture-graph",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture the simulation step as a CUDA graph when running on CUDA.",
        )
        parser.add_argument(
            "--armature",
            type=float,
            default=0.01,
            help="Default joint armature.",
        )
        parser.add_argument(
            "--target-ke",
            type=float,
            default=2000.0,
            help="Default joint target stiffness.",
        )
        parser.add_argument(
            "--target-kd",
            type=float,
            default=1.0,
            help="Default joint target damping.",
        )
        parser.add_argument(
            "--friction",
            type=float,
            default=1.0,
            help="Default shape friction coefficient.",
        )
        parser.add_argument(
            "--l10-friction",
            type=float,
            default=L10_CONTACT_FRICTION,
            help="High friction coefficient for L10 hand contact shapes.",
        )
        parser.add_argument(
            "--l10-torsional-friction",
            type=float,
            default=L10_CONTACT_TORSIONAL_FRICTION,
            help="Torsional friction coefficient for L10 hand contact shapes.",
        )
        parser.add_argument(
            "--l10-rolling-friction",
            type=float,
            default=L10_CONTACT_ROLLING_FRICTION,
            help="Rolling friction coefficient for L10 hand contact shapes.",
        )
        parser.add_argument(
            "--l10-contact-ke",
            type=float,
            default=L10_CONTACT_KE,
            help="Contact stiffness for L10 hand contact shapes.",
        )
        parser.add_argument(
            "--l10-contact-kd",
            type=float,
            default=L10_CONTACT_KD,
            help="Contact damping for L10 hand contact shapes.",
        )
        parser.add_argument(
            "--l10-contact-kf",
            type=float,
            default=L10_CONTACT_KF,
            help="Tangential friction response gain for L10 hand contact shapes.",
        )
        parser.add_argument("--scene-pos-x", type=float, default=0.0, help="Initial scene.glb X offset [m].")
        parser.add_argument("--scene-pos-y", type=float, default=-0.0184, help="Initial scene.glb Y offset [m].")
        parser.add_argument("--scene-pos-z", type=float, default=0.129, help="Initial scene.glb Z offset [m].")
        parser.add_argument("--scene-roll", type=float, default=0.0, help="Initial scene.glb roll [deg].")
        parser.add_argument("--scene-pitch", type=float, default=180.0, help="Initial scene.glb pitch [deg].")
        parser.add_argument("--scene-yaw", type=float, default=0.0, help="Initial scene.glb yaw [deg].")
        parser.add_argument("--scene-scale", type=float, default=1.0, help="Initial scene.glb uniform scale.")
        parser.add_argument(
            "--dynamic-bottle-spec",
            type=Path,
            default=DEFAULT_DYNAMIC_BOTTLE_SPEC,
            help="Dynamic bottle JSON spec with a GLB visual and transparent cylinder collision.",
        )
        parser.add_argument(
            "--dynamic-bottle-friction",
            type=float,
            default=3.0,
            help="Minimum plastic friction coefficient for the dynamic bottle cylinder.",
        )
        parser.add_argument(
            "--lift-bottle-above-scene-collision",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Raise the dynamic bottle out of any overlapping scene collision box at startup.",
        )
        parser.add_argument(
            "--bottle-scene-collision-clearance",
            type=float,
            default=BOTTLE_SCENE_COLLISION_CLEARANCE_M,
            help="Clearance [m] used when lifting the bottle above scene collision boxes.",
        )
        parser.add_argument(
            "--bottle-pos-x",
            type=float,
            default=0.5374,
            help="Dynamic bottle X offset in scene frame [m].",
        )
        parser.add_argument(
            "--bottle-pos-y",
            type=float,
            default=0.0359,
            help="Dynamic bottle Y offset in scene frame [m].",
        )
        parser.add_argument(
            "--bottle-pos-z",
            type=float,
            default=-0.6752,
            help="Dynamic bottle Z offset in scene frame [m].",
        )
        parser.add_argument(
            "--bottle-roll",
            type=float,
            default=-180.0,
            help="Dynamic bottle roll in scene frame [deg].",
        )
        parser.add_argument("--bottle-pitch", type=float, default=0.0, help="Dynamic bottle pitch in scene frame [deg].")
        parser.add_argument("--bottle-yaw", type=float, default=0.0, help="Dynamic bottle yaw in scene frame [deg].")
        parser.add_argument(
            "--quest-teleop",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Run the migrated Quest/OpenXR teleop session and drive the Newton robot.",
        )
        parser.add_argument(
            "--teleop-input-source",
            choices=("overlay-log", "quest"),
            default="overlay-log",
            help="Teleop input source. overlay-log matches the harness camera_streamer hand skeleton path.",
        )
        parser.add_argument(
            "--teleop-overlay-hand-log-path",
            type=Path,
            default=DEFAULT_OVERLAY_HAND_TRACE_PATH,
            help="camera_streamer XR hand joint JSONL path used when --teleop-input-source=overlay-log.",
        )
        parser.add_argument(
            "--teleop-overlay-stale-after-s",
            type=float,
            default=1.0,
            help="Ignore overlay hand samples older than this many seconds.",
        )
        parser.add_argument(
            "--teleop-app-name",
            type=str,
            default="NewtonNeroQuestTeleop",
            help="OpenXR application name used by the Quest teleop session.",
        )
        parser.add_argument(
            "--teleop-arm-side",
            choices=("left", "right"),
            default="right",
            help="Nero arm side driven by Quest teleop.",
        )
        parser.add_argument(
            "--teleop-pose-input-mode",
            choices=("controller_abs", "hand_abs"),
            default="hand_abs",
            help="Quest pose input mode. Defaults to the harness hand-tracking mode.",
        )
        parser.add_argument(
            "--teleop-arm-pose-command-mode",
            choices=(
                "legacy_retargeted_ee",
                "raw_wrist_position_fixed_orientation",
                "raw_wrist_position_full_orientation",
            ),
            default="raw_wrist_position_full_orientation",
            help="Arm pose command mode used by the harness command converter.",
        )
        parser.add_argument(
            "--teleop-fixed-arm-orientation-xyzw",
            type=_vec4,
            default=(0.0, 0.0, 0.0, 1.0),
            help="Fixed arm orientation for fixed-orientation command mode, formatted x,y,z,w.",
        )
        parser.add_argument(
            "--teleop-use-wrist-position-for-hand",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use hand wrist position as the pose source in hand_abs mode.",
        )
        parser.add_argument(
            "--teleop-use-wrist-rotation-for-hand",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use hand wrist rotation as the pose source in hand_abs mode.",
        )
        parser.add_argument(
            "--teleop-palm-plane-blend-alpha",
            type=float,
            default=1.0,
            help="Blend alpha for the migrated palm-plane wrist orientation correction.",
        )
        parser.add_argument(
            "--teleop-loop-hz",
            type=float,
            default=60.0,
            help="Nominal Quest teleop loop frequency [Hz].",
        )
        parser.add_argument(
            "--teleop-print-every-n-frames",
            type=int,
            default=30,
            help="Print one teleop status line every N frames.",
        )
        parser.add_argument(
            "--teleop-enable-voice-controls",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Listen for Quest voice commands on the local UDP control socket.",
        )
        parser.add_argument(
            "--teleop-require-engage",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Require voice/controller engage before Newton follows Quest targets.",
        )
        parser.add_argument(
            "--teleop-voice-control-host",
            default=os.environ.get("TELEOP_QUEST_VOICE_UDP_HOST", os.environ.get("TELEOP_VOICE_UDP_HOST", "127.0.0.1")),
            help="Host/IP for the local teleop voice UDP receiver.",
        )
        parser.add_argument(
            "--teleop-voice-control-port",
            type=int,
            default=_default_voice_control_port(),
            help="Port for the local teleop voice UDP receiver.",
        )
        parser.add_argument(
            "--teleop-xr-status-path",
            default=None,
            help="Optional teleop_xr_status.json path used by the VR overlay.",
        )
        parser.add_argument(
            "--teleop-enable-head-tracker",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable Quest head tracker input in the migrated teleop session.",
        )
        parser.add_argument(
            "--teleop-synthetic-hands-plugin",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable the IsaacTeleop synthetic hands plugin.",
        )
        parser.add_argument(
            "--teleop-isaac-teleop-root",
            type=str,
            default=None,
            help="Optional IsaacTeleop checkout root. Defaults to TELEOP/IsaacTeleop search paths.",
        )
        parser.add_argument(
            "--teleop-startup-timeout-s",
            type=float,
            default=30.0,
            help="How long to wait for an active Quest/OpenXR session [s].",
        )
        parser.add_argument(
            "--teleop-startup-retry-interval-s",
            type=float,
            default=1.0,
            help="Retry interval while waiting for Quest/OpenXR startup [s].",
        )
        parser.add_argument(
            "--teleop-trace-path",
            type=str,
            default=None,
            help="Optional JSONL path for migrated teleop command traces.",
        )
        parser.add_argument(
            "--teleop-drive-ik",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Drive Newton Nero arm joints with the migrated full-pose differential IK.",
        )
        parser.add_argument(
            "--teleop-left-eef-body-suffix",
            type=str,
            default="/left_revo2_flange",
            help="Newton body-label suffix used as the left IK EEF frame. Defaults to the harness revo2_flange.",
        )
        parser.add_argument(
            "--teleop-right-eef-body-suffix",
            type=str,
            default="/right_revo2_flange",
            help="Newton body-label suffix used as the right IK EEF frame. Defaults to the harness revo2_flange.",
        )
        parser.add_argument(
            "--teleop-ik-max-task-step-m",
            type=float,
            default=0.05,
            help="Maximum Newton teleop IK translational task step per frame [m].",
        )
        parser.add_argument(
            "--teleop-ik-max-rotation-step-rad",
            type=float,
            default=float(np.deg2rad(10.0)),
            help="Maximum Newton teleop IK rotational task step per frame [rad].",
        )
        parser.add_argument(
            "--teleop-ik-orientation-weight",
            type=float,
            default=0.35,
            help="Newton teleop IK orientation weight.",
        )
        parser.add_argument(
            "--teleop-ik-max-joint-step-rad",
            type=float,
            default=0.045,
            help="Maximum Newton teleop IK joint step per frame [rad], matching the harness clamp.",
        )
        parser.add_argument(
            "--teleop-ik-max-joint-velocity-rad-s",
            type=float,
            default=0.0,
            help="Maximum Newton teleop IK joint velocity [rad/s]. 0 disables the extra velocity clamp.",
        )
        parser.add_argument(
            "--teleop-ik-damping-lambda",
            type=float,
            default=0.02,
            help="Newton teleop IK damped-least-squares lambda.",
        )
        parser.add_argument(
            "--teleop-hand-max-joint-step-rad",
            type=float,
            default=0.0,
            help="Maximum L10 hand joint command step per teleop frame [rad]. Lower values make grasp closure softer.",
        )
        parser.add_argument(
            "--teleop-hand-publish-kinematic-velocity",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Publish L10 finger joint velocities into Newton kinematic contact friction.",
        )
        parser.add_argument(
            "--teleop-relative-control",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use the harness relative-control anchor for Quest wrist motion.",
        )
        parser.add_argument(
            "--teleop-translation-scale",
            type=_vec3,
            default=(1.0, 1.0, 1.0),
            help="Quest-to-target translation scale x,y,z.",
        )
        parser.add_argument(
            "--teleop-workspace-origin",
            type=_vec3,
            default=(0.0, 0.0, 0.0),
            help="Absolute-control workspace origin x,y,z [m].",
        )
        parser.add_argument(
            "--teleop-input-axis-map",
            type=_axis_map,
            default=("x", "y", "z"),
            help="Harness input axis map, formatted as three tokens such as z,x,y.",
        )
        parser.add_argument(
            "--teleop-openxr-coordinate-adapter",
            choices=("none", "openxr_genesis"),
            default="openxr_genesis",
            help="Coordinate adapter applied to Quest/OpenXR wrist vectors and orientations.",
        )
        parser.add_argument(
            "--teleop-openxr-yaw-recenter",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Yaw-recenter OpenXR forward so the operator front matches the robot front.",
        )
        parser.add_argument(
            "--teleop-orientation",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use Quest wrist orientation for Newton full-pose IK.",
        )
        parser.add_argument(
            "--teleop-orientation-source",
            choices=("wrist_quat", "hand_anatomical_frame", "hand_beavr_anatomical_frame", "hand_genesis_wrist_frame"),
            default="hand_genesis_wrist_frame",
            help="Remote Nero wrist orientation source.",
        )
        parser.add_argument(
            "--teleop-orientation-axis-map",
            type=_axis_map,
            default=("x", "y", "z"),
            help="Remote Nero orientation axis map.",
        )
        parser.add_argument(
            "--teleop-orientation-max-speed-rad-s",
            type=float,
            default=3.0,
            help="Maximum commanded wrist orientation speed [rad/s].",
        )
        parser.add_argument(
            "--teleop-orientation-tool-offset-wxyz",
            type=_vec4,
            default=(1.0, 0.0, 0.0, 0.0),
            help="Tool-local orientation offset formatted w,x,y,z.",
        )
        parser.add_argument(
            "--teleop-orientation-reference-mode",
            choices=("world_delta", "tool_local_delta", "calibrated_tool_local"),
            default="calibrated_tool_local",
            help="Remote Nero orientation reference mode.",
        )
        parser.add_argument(
            "--teleop-fixed-quaternion-wxyz",
            type=_vec4,
            default=None,
            help="Optional fixed Newton target quaternion formatted w,x,y,z when --no-teleop-orientation is used.",
        )
        parser.add_argument(
            "--teleop-finite-difference-rad",
            type=float,
            default=1.0e-4,
            help="Finite-difference step [rad] for Newton IK Jacobian.",
        )
        parser.add_argument(
            "--d455-json",
            type=Path,
            default=DEFAULT_D455_JSON,
            help="Harness D455 camera parameter JSON.",
        )
        parser.add_argument(
            "--d405-json",
            type=Path,
            default=DEFAULT_D405_JSON,
            help="Harness D405 camera parameter JSON.",
        )
        parser.add_argument(
            "--d405-mount-json",
            type=Path,
            default=DEFAULT_D405_MOUNT_JSON,
            help="Harness D405 right-connector mount JSON.",
        )
        parser.add_argument(
            "--d405-connector-rel-pos",
            type=_vec3,
            default=None,
            help="Override D405 body xyz in right_connector frame [m], formatted x,y,z.",
        )
        parser.add_argument(
            "--d405-connector-rel-euler",
            type=_vec3,
            default=None,
            help="Override D405 body Euler XYZ in right_connector frame [deg], formatted r,p,y.",
        )
        parser.add_argument(
            "--d405-body-visual",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Attach a visual-only D405 body box to right_connector using the harness mount.",
        )
        parser.add_argument(
            "--d455-preview",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Render a separate D455 RGB preview image window.",
        )
        parser.add_argument(
            "--d405-preview",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Render a separate D405 RGB preview image window.",
        )
        parser.add_argument(
            "--camera-textures",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable texture sampling in Newton camera previews.",
        )
        parser.add_argument(
            "--d455-image-size",
            type=_image_size,
            default=D455_MODEL_IMAGE_SIZE,
            help="D455 ego preview/model input size as height,width.",
        )
        parser.add_argument(
            "--d455-width",
            type=int,
            default=None,
            help="Deprecated alias for the D455 output preview width [px].",
        )
        parser.add_argument(
            "--d455-height",
            type=int,
            default=None,
            help="Deprecated alias for the D455 output preview height [px].",
        )
        parser.add_argument(
            "--d455-render-width",
            type=int,
            default=None,
            help="D455 sensor render width [px]. Defaults to the harness D455 RGB preset.",
        )
        parser.add_argument(
            "--d455-render-height",
            type=int,
            default=None,
            help="D455 sensor render height [px]. Defaults to the harness D455 RGB preset.",
        )
        parser.add_argument(
            "--d455-roi-zoom",
            type=float,
            default=D455_EGO_ROI_ZOOM,
            help="D455 ego_view center-crop digital zoom. Default matches harness.",
        )
        parser.add_argument(
            "--d455-roi-center-x",
            type=float,
            default=D455_EGO_ROI_CENTER_X,
            help="D455 ego_view ROI center X in normalized image coordinates.",
        )
        parser.add_argument(
            "--d455-roi-center-y",
            type=float,
            default=D455_EGO_ROI_CENTER_Y,
            help="D455 ego_view ROI center Y in normalized image coordinates.",
        )
        parser.add_argument(
            "--d455-preview-scale",
            type=int,
            default=D455_PREVIEW_SCALE,
            help="Integer scale for the optional OpenCV D455 preview window.",
        )
        parser.add_argument(
            "--d455-front-clearance",
            type=float,
            default=D455_RGB_FRONT_CLEARANCE_M,
            help="Extra local +X offset from the D455 front face to avoid self-hitting the black camera body.",
        )
        parser.add_argument(
            "--d455-opencv-window",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Mirror the D455 preview into a separate OpenCV OS window when cv2 is installed.",
        )
        parser.add_argument("--d405-width", type=int, default=None, help="D405 preview width [px].")
        parser.add_argument("--d405-height", type=int, default=None, help="D405 preview height [px].")
        parser.add_argument(
            "--d405-preview-scale",
            type=int,
            default=D405_PREVIEW_SCALE,
            help="Integer scale for the optional OpenCV D405 preview window.",
        )
        parser.add_argument(
            "--d405-opencv-window",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Mirror the raw D405 preview into a separate OpenCV OS window when cv2 is installed.",
        )
        parser.add_argument("--d455-fov", type=float, default=None, help="Override D455 vertical FOV [deg].")
        parser.add_argument("--d405-fov", type=float, default=None, help="Override D405 vertical FOV [deg].")
        return parser


def main() -> None:
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
