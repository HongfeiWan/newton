#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run checkpoint-200000 with RTC in the dual Nero + L10 Newton scene.

The policy can consume either live Newton camera renders or frames from a
selected smooth dataset episode. State can independently come from Newton or
the selected smooth episode. Simulator ego images mirror node0's D455 ROI and
frame-tap resize before reaching the checkpoint processor.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import newton.examples  # noqa: E402
from debug import import_dual_nero_linker_l10 as scene_runtime  # noqa: E402
from teleop_stack.camera_preprocessing import preprocess_ego_rgb  # noqa: E402
from teleop_stack.ik import (  # noqa: E402
    FullPoseDifferentialIkController,
    FullPoseDifferentialIkControllerConfig,
    RobotStateSnapshot,
    TaskSpaceTarget,
)
from teleop_stack.ik.nero_can_fk import nero_can_flange_pose_from_joints  # noqa: E402
from teleop_stack.models import NamedJointValues, Pose7  # noqa: E402
from teleop_stack.retargeting.hand_config import load_linker_l10_right_hand_spec  # noqa: E402
from teleop_stack.robots.newton_runtime import NewtonLinkKinematicsModel  # noqa: E402
from teleop_stack.teleop.spatial_frames import matrix_to_quat_xyzw, quat_xyzw_to_matrix  # noqa: E402

DEFAULT_ISAAC_GROOT_ROOT = Path(os.environ.get("ISAAC_GROOT_ROOT", REPO_ROOT.parent / "Isaac-GR00T"))
DEFAULT_POLICY_CHECKPOINT = REPO_ROOT / "checkpoints" / "groot" / "checkpoint-200000"
DEFAULT_VLM_MODEL = REPO_ROOT / "checkpoints" / "nvidia" / "Cosmos-Reason2-2B"
DEFAULT_SMOOTH_DIR = REPO_ROOT / "local_data" / "groot" / "smooth"
NODE0_EGO_SOURCE_WIDTH = 1280
NODE0_EGO_SOURCE_HEIGHT = 800
NODE0_EGO_INPUT_WIDTH = 320
NODE0_EGO_INPUT_HEIGHT = 180
DEFAULT_TRACE_JSONL = REPO_ROOT / "logs" / "groot_newton_rtc" / "trace.jsonl"
DEFAULT_INSTRUCTION = "pick up the bottle with green cap and place it in the white rectangle area"
GROOT_INITIAL_RIGHT_ARM_Q = (
    0.2724284429,
    1.6012174157,
    1.4535451076,
    1.2643514167,
    0.2993937799,
    -0.0534419817,
    0.1828232391,
)
GROOT_D405_FOV_DEG = 72.0
GROOT_D405_CONNECTOR_REL_EULER_DEG = (89.483, -1.020, -2.995)
NODE0_STATE_TO_GENESIS_TRANSLATION_XYZ = (0.0, 0.059, 0.918)
NODE0_STATE_TO_GENESIS_QUATERNION_XYZW = (0.5, 0.5, 0.5, -0.5)
NODE0_STATE_TO_GENESIS_EEF_OFFSET_TRANSLATION_XYZ = (0.032, 0.0, -0.0235)
NODE0_STATE_TO_GENESIS_EEF_OFFSET_QUATERNION_XYZW = (-0.5, 0.5, -0.5, 0.5)
GROOT_INITIAL_HAND_COMMAND_Q = (
    0.1848468184,
    0.3151794076,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0581703521,
    0.0262242742,
    0.1140341759,
    0.0,
)
GROOT_RUNTIME_PACKAGES = {
    "albumentations": "1.4.18",
    "albucore": "0.0.17",
}


@wp.kernel
def _resize_rgb_nearest(
    source: wp.array3d[wp.uint8],
    target: wp.array3d[wp.uint8],
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
):
    y, x, channel = wp.tid()
    source_y = min((y * source_height) // target_height, source_height - 1)
    source_x = min((x * source_width) // target_width, source_width - 1)
    target[y, x, channel] = source[source_y, source_x, channel]


POLICY_HAND_JOINT_NAMES = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
    "index_mcp_roll",
    "ring_mcp_roll",
    "pinky_mcp_roll",
    "thumb_cmc_roll",
)
POLICY_HAND_COMMAND_LIMITS = {
    "thumb_cmc_roll": (0.0, 1.1339),
    "thumb_cmc_yaw": (0.0, 1.9189),
    "thumb_cmc_pitch": (0.0, 0.5146),
    "index_mcp_roll": (0.0, 0.2181),
    "index_mcp_pitch": (0.0, 1.3607),
    "middle_mcp_pitch": (0.0, 1.3607),
    "ring_mcp_roll": (0.0, 0.2181),
    "ring_mcp_pitch": (0.0, 1.3607),
    "pinky_mcp_roll": (0.0, 0.3489),
    "pinky_mcp_pitch": (0.0, 1.3607),
}
POLICY_HAND_OBSERVATION_LOWER = {
    "index_mcp_pitch": 0.005336078431372549,
    "middle_mcp_pitch": 0.005336078431372549,
    "ring_mcp_pitch": 0.005336078431372549,
    "pinky_mcp_pitch": 0.005336078431372549,
}
L10_SDK_JOINT_CALIBRATION = (
    ("thumb_cmc_pitch", 0.0, 0.75, True),
    ("thumb_cmc_yaw", 0.0, 1.43, True),
    ("index_mcp_pitch", 0.0, 1.62, True),
    ("middle_mcp_pitch", 0.0, 1.62, True),
    ("ring_mcp_pitch", 0.0, 1.62, True),
    ("pinky_mcp_pitch", 0.0, 1.62, True),
    ("index_mcp_roll", -0.26, 0.21, False),
    ("ring_mcp_roll", 0.0, 0.21, False),
    ("pinky_mcp_roll", 0.0, 0.34, False),
    ("thumb_cmc_roll", -0.52, 1.01, True),
)


@dataclass(frozen=True)
class ModalitySpec:
    keys: tuple[str, ...]
    delta_indices: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointModalities:
    video: ModalitySpec
    state: ModalitySpec
    action: ModalitySpec
    language: ModalitySpec


@dataclass(frozen=True)
class PolicyReplanRequest:
    replan_index: int
    policy_step: int
    timeline_s: float
    observation: dict[str, Any]
    source_metadata: dict[str, Any]
    rtc_seed: dict[str, np.ndarray] | None
    seed_metadata: dict[str, Any]
    options: dict[str, Any] | None
    rtc_metadata: dict[str, Any]


@dataclass(frozen=True)
class PolicyReplanResult:
    request: PolicyReplanRequest
    policy_action: dict[str, np.ndarray]
    policy_metadata: dict[str, Any]
    inference_s: float


class ViewerFifoPreview:
    """Copy the GPU viewer and exact model RGB inputs to a host FIFO."""

    def __init__(
        self,
        viewer: Any,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        input_width: int = 0,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Viewer FIFO preview dimensions must be positive")
        if fps <= 0.0:
            raise ValueError("Viewer FIFO preview FPS must be positive")
        if input_width < 0 or input_width >= width:
            raise ValueError("Viewer FIFO model-input width must be smaller than the output width")
        if not hasattr(viewer, "get_frame"):
            raise ValueError("Viewer FIFO preview requires the GL viewer")

        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.input_width = int(input_width)
        self.scene_width = self.width - self.input_width
        self._stream: Any | None = None
        self._target_image: Any | None = None
        self._resized_image: Any | None = None
        self._next_capture_time = 0.0
        self._failed = False

        renderer = getattr(viewer, "renderer", None)
        window = getattr(renderer, "window", None)
        if window is not None and hasattr(window, "set_size"):
            window.set_size(self.scene_width, self.height)

    def capture(self, viewer: Any, model_images: dict[str, np.ndarray] | None = None) -> None:
        if self._failed:
            return
        now = time.monotonic()
        if now < self._next_capture_time:
            return
        self._next_capture_time = now + 1.0 / self.fps

        try:
            if self._stream is None:
                self._stream = self.path.open("wb", buffering=0)
                print(
                    f"[groot-viewer] direct-GPU preview={self.width}x{self.height}@{self.fps:g} "
                    f"layout=simulator+ego_view+wrist_view fifo={self.path}",
                    flush=True,
                )
            frame = viewer.get_frame(target_image=self._target_image, render_ui=False)
            self._target_image = frame
            source_height, source_width, channels = tuple(int(value) for value in frame.shape)
            if channels != 3:
                raise ValueError(f"Viewer frame must have three RGB channels, got {frame.shape}")
            output_frame = frame
            if (source_height, source_width) != (self.height, self.scene_width):
                expected_shape = (self.height, self.scene_width, 3)
                if self._resized_image is None or self._resized_image.shape != expected_shape:
                    self._resized_image = wp.empty(
                        shape=expected_shape,
                        dtype=wp.uint8,
                        device=frame.device,
                    )
                wp.launch(
                    _resize_rgb_nearest,
                    dim=expected_shape,
                    inputs=[
                        frame,
                        self._resized_image,
                        source_height,
                        source_width,
                        self.height,
                        self.scene_width,
                    ],
                    device=frame.device,
                )
                output_frame = self._resized_image
            host_scene = np.ascontiguousarray(output_frame.numpy())
            host_frame = self._compose_frame(host_scene, model_images or {})
            remaining = memoryview(host_frame).cast("B")
            while remaining:
                written = self._stream.write(remaining)
                if written is None:
                    continue
                remaining = remaining[written:]
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._failed = True
            self.close()
            print(f"[groot-viewer] preview disabled: {exc}", flush=True)

    def _compose_frame(
        self,
        scene: np.ndarray,
        model_images: dict[str, np.ndarray],
    ) -> np.ndarray:
        if self.input_width <= 0:
            return scene
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, : self.scene_width] = scene
        frame[:, self.scene_width : self.scene_width + 2] = 210

        slot_height = self.height // 2
        slots = (
            ("ego_view", 0, np.asarray((45, 210, 120), dtype=np.uint8)),
            ("wrist_view", slot_height, np.asarray((255, 105, 70), dtype=np.uint8)),
        )
        for key, slot_y, color in slots:
            header_height = min(4, max(0, slot_height - 1))
            frame[slot_y : slot_y + header_height, self.scene_width :] = color
            image = model_images.get(key)
            if image is None:
                continue
            resized = _resize_rgb_preview(
                image,
                max_width=self.input_width,
                max_height=max(1, slot_height - header_height),
            )
            image_height, image_width, _ = resized.shape
            x = self.scene_width + (self.input_width - image_width) // 2
            y = slot_y + header_height + max(0, (slot_height - header_height - image_height) // 2)
            frame[y : y + image_height, x : x + image_width] = resized
        return frame

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def _resize_rgb_preview(image: np.ndarray, *, max_width: int, max_height: int) -> np.ndarray:
    source = np.asarray(image, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"Model input preview must be RGB HWC, got {source.shape}")
    source_height, source_width, _ = source.shape
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"Model input preview is empty: {source.shape}")
    scale = min(float(max_width) / source_width, float(max_height) / source_height)
    target_width = max(1, min(max_width, int(round(source_width * scale))))
    target_height = max(1, min(max_height, int(round(source_height * scale))))
    if (target_height, target_width) == (source_height, source_width):
        return np.ascontiguousarray(source)
    y_indices = np.minimum(
        np.arange(target_height, dtype=np.int64) * source_height // target_height,
        source_height - 1,
    )
    x_indices = np.minimum(
        np.arange(target_width, dtype=np.int64) * source_width // target_width,
        source_width - 1,
    )
    return np.ascontiguousarray(source[y_indices[:, None], x_indices[None, :]])


def _node0_ego_view_preprocess(
    image: np.ndarray,
    *,
    zoom: float,
    center_x: float,
    center_y: float,
) -> np.ndarray:
    """Apply node0's camera ROI and frame-tap resize without changing RGB order."""
    return preprocess_ego_rgb(
        image,
        zoom=float(zoom),
        center_x=float(center_x),
        center_y=float(center_y),
        output_size=(NODE0_EGO_INPUT_WIDTH, NODE0_EGO_INPUT_HEIGHT),
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _append_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_jsonable(row), ensure_ascii=True, sort_keys=True) + "\n")


def _checkpoint_modalities(checkpoint: Path) -> CheckpointModalities:
    payload = _load_json(checkpoint / "processor_config.json")
    configs = payload["processor_kwargs"]["modality_configs"]["new_embodiment"]

    def spec(name: str) -> ModalitySpec:
        config = configs[name]
        return ModalitySpec(
            keys=tuple(str(value) for value in config["modality_keys"]),
            delta_indices=tuple(int(value) for value in config["delta_indices"]),
        )

    return CheckpointModalities(
        video=spec("video"),
        state=spec("state"),
        action=spec("action"),
        language=spec("language"),
    )


def _processor_path(model_path: Path) -> Path:
    if (model_path / "processor_config.json").exists():
        return model_path
    if (model_path / "processor" / "processor_config.json").exists():
        return model_path / "processor"
    parent_processor = model_path.parent / "processor"
    if (parent_processor / "processor_config.json").exists():
        return parent_processor
    return model_path


def _validate_groot_python_runtime() -> None:
    mismatches = []
    for package_name, expected_version in GROOT_RUNTIME_PACKAGES.items():
        try:
            actual_version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            actual_version = "not installed"
        if actual_version != expected_version:
            mismatches.append(f"{package_name}={actual_version} (expected {expected_version})")
    if mismatches:
        pins = " ".join(f"{name}=={version}" for name, version in GROOT_RUNTIME_PACKAGES.items())
        raise RuntimeError(
            "The GR00T image processor runtime does not match Isaac-GR00T: "
            f"{', '.join(mismatches)}. In the active conda environment run: "
            f"python -m pip install {pins}"
        )


def _validate_asset_layout(args: argparse.Namespace) -> CheckpointModalities:
    checkpoint = args.policy_checkpoint.expanduser().resolve()
    vlm_model = args.vlm_model.expanduser().resolve()
    smooth_dir = args.smooth_dir.expanduser().resolve()
    required_checkpoint_files = (
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    missing = [name for name in required_checkpoint_files if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is incomplete at {checkpoint}: missing={missing}")
    if not (vlm_model / "config.json").is_file():
        raise FileNotFoundError(f"Cosmos VLM config is missing: {vlm_model / 'config.json'}")
    for relative in ("meta/info.json", "meta/modality.json", "meta/episodes.jsonl", "meta/tasks.jsonl"):
        if not (smooth_dir / relative).is_file():
            raise FileNotFoundError(f"Smooth dataset file is missing: {smooth_dir / relative}")

    modalities = _checkpoint_modalities(checkpoint)
    expected = {
        "video": ("ego_view", "wrist_view"),
        "state": ("eef_9d", "hand_joint_pos", "arm_joint_pos"),
        "action": ("eef_9d", "hand_joint_target", "arm_joint_target"),
    }
    actual = {
        "video": modalities.video.keys,
        "state": modalities.state.keys,
        "action": modalities.action.keys,
    }
    if actual != expected:
        raise ValueError(f"Unexpected checkpoint modality schema: expected={expected}, actual={actual}")
    if modalities.video.delta_indices != (0,) or modalities.state.delta_indices != (0,):
        raise ValueError(
            "This runtime expects one current video/state frame; "
            f"video={modalities.video.delta_indices} state={modalities.state.delta_indices}"
        )
    if modalities.action.delta_indices != tuple(range(32)):
        raise ValueError(f"Expected action delta indices 0..31, got {modalities.action.delta_indices}")
    return modalities


def _scale_linear(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if abs(float(src_max) - float(src_min)) < 1.0e-9:
        return float(dst_min)
    return (float(value) - float(src_min)) * (float(dst_max) - float(dst_min)) / (
        float(src_max) - float(src_min)
    ) + float(dst_min)


def _l10_base_raw_from_command(value_by_name: dict[str, float]) -> tuple[int, ...]:
    raw_values = []
    for joint_name, sdk_lower, sdk_upper, raw_reversed in L10_SDK_JOINT_CALIBRATION:
        source_lower, source_upper = POLICY_HAND_COMMAND_LIMITS[joint_name]
        value = float(np.clip(value_by_name[joint_name], source_lower, source_upper))
        ratio = _scale_linear(value, source_lower, source_upper, 0.0, 1.0)
        sdk_value = float(np.clip(_scale_linear(ratio, 0.0, 1.0, sdk_lower, sdk_upper), sdk_lower, sdk_upper))
        if raw_reversed:
            raw = _scale_linear(sdk_value, sdk_lower, sdk_upper, 255.0, 0.0)
        else:
            raw = _scale_linear(sdk_value, sdk_lower, sdk_upper, 0.0, 255.0)
        raw_values.append(int(round(float(np.clip(raw, 0.0, 255.0)))))
    return tuple(raw_values)


def _reported_hand_q_from_command(hand_q: np.ndarray) -> np.ndarray:
    command = np.zeros(len(POLICY_HAND_JOINT_NAMES), dtype=np.float32)
    values = np.asarray(hand_q, dtype=np.float32).reshape(-1)
    command[: min(command.size, values.size)] = values[: command.size]
    value_by_name = dict(zip(POLICY_HAND_JOINT_NAMES, (float(v) for v in command), strict=True))
    raw_values = _l10_base_raw_from_command(value_by_name)
    reported_by_name: dict[str, float] = {}
    for raw, (joint_name, sdk_lower, sdk_upper, raw_reversed) in zip(
        raw_values, L10_SDK_JOINT_CALIBRATION, strict=True
    ):
        if raw_reversed:
            sdk_value = _scale_linear(float(raw), 255.0, 0.0, sdk_lower, sdk_upper)
        else:
            sdk_value = _scale_linear(float(raw), 0.0, 255.0, sdk_lower, sdk_upper)
        source_lower, source_upper = POLICY_HAND_COMMAND_LIMITS[joint_name]
        ratio = _scale_linear(float(np.clip(sdk_value, sdk_lower, sdk_upper)), sdk_lower, sdk_upper, 0.0, 1.0)
        reported_by_name[joint_name] = float(
            np.clip(_scale_linear(ratio, 0.0, 1.0, source_lower, source_upper), source_lower, source_upper)
        )
    reported = np.asarray([reported_by_name[name] for name in POLICY_HAND_JOINT_NAMES], dtype=np.float32)
    for index, name in enumerate(POLICY_HAND_JOINT_NAMES):
        lower = POLICY_HAND_OBSERVATION_LOWER.get(name)
        if lower is not None:
            reported[index] = max(float(lower), float(reported[index]))
    return reported


def _reported_hand_value_to_command(joint_index: int, target_value: float) -> float:
    name = POLICY_HAND_JOINT_NAMES[int(joint_index)]
    lower, upper = POLICY_HAND_COMMAND_LIMITS[name]

    def reported_at(command_value: float) -> float:
        command = np.zeros(len(POLICY_HAND_JOINT_NAMES), dtype=np.float32)
        command[int(joint_index)] = float(command_value)
        return float(_reported_hand_q_from_command(command)[int(joint_index)])

    low_reported = reported_at(lower)
    high_reported = reported_at(upper)
    if high_reported < low_reported:
        lower, upper = upper, lower
        low_reported, high_reported = high_reported, low_reported
    target = float(np.clip(target_value, low_reported, high_reported))
    for _ in range(18):
        midpoint = 0.5 * (lower + upper)
        if reported_at(midpoint) < target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _reported_hand_q_to_command(hand_q: np.ndarray) -> np.ndarray:
    values = np.asarray(hand_q, dtype=np.float32).reshape(-1)
    command = np.zeros(len(POLICY_HAND_JOINT_NAMES), dtype=np.float32)
    for index in range(command.size):
        target = float(values[index]) if index < values.size else 0.0
        command[index] = _reported_hand_value_to_command(index, target)
    return command


def _rotmat_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    # Match the live node0 GR00T bridge exactly.
    return rotation[:2, :].reshape(6).astype(np.float32)


def _rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    rows = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    row0 = rows[0]
    row1 = rows[1]
    row0_norm = float(np.linalg.norm(row0))
    if row0_norm <= 1.0e-8:
        return np.eye(3, dtype=np.float64)
    row0 = row0 / row0_norm
    row1 = row1 - float(np.dot(row0, row1)) * row0
    row1_norm = float(np.linalg.norm(row1))
    if row1_norm <= 1.0e-8:
        return np.eye(3, dtype=np.float64)
    row1 = row1 / row1_norm
    row2 = np.cross(row0, row1)
    return np.vstack((row0, row1, row2))


def _eef_9d_to_pose(eef_9d: np.ndarray) -> np.ndarray:
    values = np.asarray(eef_9d, dtype=np.float64).reshape(-1)
    if values.size < 9 or not np.isfinite(values[:9]).all():
        raise ValueError(f"EEF target must contain nine finite values, got {values}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = values[:3]
    pose[:3, :3] = _rot6d_to_rotmat(values[3:9])
    return pose


def _require_finite_values(
    values: np.ndarray,
    *,
    name: str,
    error_type: type[Exception] = RuntimeError,
) -> np.ndarray:
    array = np.asarray(values)
    finite = np.isfinite(array)
    if finite.all():
        return array
    flat = array.reshape(-1)
    bad_indices = np.flatnonzero(~finite.reshape(-1))[:8]
    bad_values = [float(flat[index]) for index in bad_indices]
    raise error_type(
        f"{name} contains non-finite values at flat indices {bad_indices.astype(int).tolist()}: {bad_values}"
    )


def _pose7_to_matrix(pose: Pose7) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(pose.position_xyz, dtype=np.float64)
    matrix[:3, :3] = np.asarray(quat_xyzw_to_matrix(pose.quaternion_xyzw), dtype=np.float64)
    return matrix


def _matrix_to_pose7(matrix: np.ndarray) -> Pose7:
    pose = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rotation = tuple(tuple(float(value) for value in row) for row in pose[:3, :3])
    return Pose7(
        position_xyz=tuple(float(value) for value in pose[:3, 3]),
        quaternion_xyzw=matrix_to_quat_xyzw(rotation),
    )


def _rigid_transform_matrix(
    translation_xyz: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    transform[:3, :3] = np.asarray(quat_xyzw_to_matrix(quaternion_xyzw), dtype=np.float64)
    return transform


def _invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
    source = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = source[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ source[:3, 3]
    return inverse


def _packed_color_to_rgb(image: wp.array[wp.uint32]) -> np.ndarray:
    packed = image.numpy()
    while packed.ndim > 2:
        packed = packed[0]
    packed = np.ascontiguousarray(packed)
    return packed.view(np.uint8).reshape(*packed.shape, 4)[..., :3].copy()


def _initialize_right_hand_pose(example: scene_runtime.Example, command_q: tuple[float, ...]) -> None:
    values = np.asarray(command_q, dtype=np.float32).reshape(-1)
    if values.size != len(POLICY_HAND_JOINT_NAMES):
        raise ValueError(f"Initial right hand pose must contain 10 values, got {values.size}")

    spec = load_linker_l10_right_hand_spec()
    expanded = spec.expand_mimic_joint_values(
        NamedJointValues(
            joint_names=POLICY_HAND_JOINT_NAMES,
            joint_positions=tuple(float(value) for value in values),
        )
    )
    q_start = example.model.joint_q_start.numpy()
    q_indices = {
        str(label).rsplit("/", maxsplit=1)[-1]: int(q_start[index])
        for index, label in enumerate(example.model.joint_label)
    }
    joint_q = example.state_0.joint_q.numpy().copy()
    joint_target_q = example.control.joint_target_q.numpy().copy()
    applied = 0
    for name, value in zip(expanded.joint_names, expanded.joint_positions, strict=True):
        q_index = q_indices.get(f"right_l10_{name}")
        if q_index is None:
            continue
        joint_q[q_index] = float(value)
        joint_target_q[q_index] = float(value)
        applied += 1

    device = example.model.device
    joint_q_wp = wp.array(joint_q, dtype=wp.float32, device=device)
    target_q_wp = wp.array(joint_target_q, dtype=wp.float32, device=device)
    wp.copy(example.model.joint_q, joint_q_wp)
    wp.copy(example.model.joint_target_q, target_q_wp)
    wp.copy(example.state_0.joint_q, joint_q_wp)
    wp.copy(example.control.joint_target_q, target_q_wp)
    newton.eval_fk(example.model, example.state_0.joint_q, example.state_0.joint_qd, example.state_0)
    wp.copy(example.model.body_q, example.state_0.body_q)
    example.state_1.assign(example.state_0)

    if hasattr(example, "_initial_joint_q"):
        example._initial_joint_q = joint_q.copy()  # noqa: SLF001
    if hasattr(example, "_initial_joint_target_q"):
        example._initial_joint_target_q = joint_target_q.copy()  # noqa: SLF001
    if hasattr(example, "_initial_body_q"):
        example._initial_body_q = example.state_0.body_q.numpy().copy()  # noqa: SLF001
    if hasattr(example, "_initial_model_body_q"):
        example._initial_model_body_q = example.model.body_q.numpy().copy()  # noqa: SLF001
    print(
        f"[groot-control] initialized right L10 hand from checkpoint pose joints={applied} "
        f"command={np.round(values, 6).tolist()}",
        flush=True,
    )


def _first_batch_chunk(action: dict[str, np.ndarray], key: str) -> np.ndarray:
    value = np.asarray(action[key], dtype=np.float32)
    if value.ndim == 3:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"Action {key!r} must have shape [T,D] or [1,T,D], got {value.shape}")
    return value.astype(np.float32, copy=True)


def _unbatch_action(action: dict[str, np.ndarray], action_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {key: _first_batch_chunk(action, key) for key in action_keys if key in action}


def _stored_rtc_action(
    *,
    policy_action: dict[str, np.ndarray],
    rtc_seed_action: dict[str, np.ndarray] | None,
    action_keys: tuple[str, ...],
    frozen_steps: int,
) -> dict[str, np.ndarray]:
    stored = {key: np.asarray(value, dtype=np.float32).copy() for key, value in policy_action.items()}
    if rtc_seed_action is None or frozen_steps <= 0:
        return stored
    for key in action_keys:
        if key not in stored or key not in rtc_seed_action:
            continue
        seed = _first_batch_chunk(rtc_seed_action, key)
        count = min(int(frozen_steps), stored[key].shape[0], seed.shape[0])
        stored[key][:count] = seed[:count]
    return stored


def _elapsed_action_steps(*, start_s: float, current_s: float, action_dt_s: float) -> int:
    elapsed_s = max(0.0, float(current_s) - float(start_s))
    return max(0, int(math.floor(elapsed_s / max(float(action_dt_s), 1.0e-9) + 0.5)))


class TeleopRtcSeedManager:
    """Rolling decoded-action seed window matching the validated RTC probe."""

    def __init__(self, *, action_keys: tuple[str, ...], action_dt_s: float, max_chunks: int = 4) -> None:
        self.action_keys = action_keys
        self.action_dt_s = float(action_dt_s)
        self.max_chunks = max(1, int(max_chunks))
        self._epoch_s: float | None = None
        self._chunks: list[dict[str, Any]] = []

    def clear(self) -> None:
        self._epoch_s = None
        self._chunks.clear()

    def _time_to_step(self, timestamp_s: float) -> int:
        if self._epoch_s is None:
            self._epoch_s = float(timestamp_s)
        return int(round((float(timestamp_s) - self._epoch_s) / self.action_dt_s))

    def push(self, action: dict[str, np.ndarray], *, start_s: float, frame_id: int) -> None:
        self._chunks.append(
            {
                "start_step": self._time_to_step(start_s),
                "frame_id": int(frame_id),
                "action": {key: _first_batch_chunk(action, key) for key in self.action_keys if key in action},
            }
        )
        if len(self._chunks) > self.max_chunks:
            del self._chunks[: len(self._chunks) - self.max_chunks]

    def seed_window(
        self, *, anchor_start_s: float, anchor_frame_id: int, horizon: int
    ) -> tuple[dict[str, np.ndarray] | None, float | None, dict[str, Any]]:
        if not self._chunks:
            return None, None, {}
        start_step = self._time_to_step(anchor_start_s)
        indexed: dict[int, dict[str, np.ndarray]] = {}
        for chunk in self._chunks:
            action = chunk["action"]
            if not action:
                continue
            chunk_horizon = min(value.shape[0] for value in action.values())
            for offset in range(chunk_horizon):
                indexed[int(chunk["start_step"]) + offset] = {
                    key: value[offset].copy() for key, value in action.items()
                }
        rows = []
        for step in range(start_step, start_step + int(horizon)):
            row = indexed.get(step)
            if row is None:
                break
            rows.append(row)
        if not rows:
            return None, None, {"reason": "no_seed_window", "start_action_step": start_step}
        valid_steps = len(rows)
        rows.extend(rows[-1] for _ in range(valid_steps, int(horizon)))
        seed = {
            key: np.stack([row[key] for row in rows], axis=0).astype(np.float32)
            for key in self.action_keys
            if key in rows[0]
        }
        return (
            seed,
            float(anchor_start_s),
            {
                "reason": "ok",
                "anchor_frame_id": int(anchor_frame_id),
                "start_action_step": int(start_step),
                "seed_steps": len(rows),
                "seed_valid_steps": valid_steps,
                "seed_padded_steps": len(rows) - valid_steps,
            },
        )


def _rtc_options(
    *,
    enabled: bool,
    mode: str,
    previous_action: dict[str, np.ndarray] | None,
    previous_start_s: float | None,
    current_s: float,
    action_dt_s: float,
    fallback_replan_horizon: int,
    max_overlap_steps: int,
    frozen_steps: int,
    ramp_rate: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    metadata: dict[str, Any] = {"enabled": bool(enabled), "mode": str(mode)}
    if not enabled or mode == "off" or previous_action is None:
        metadata["reason"] = "disabled_or_no_previous_action"
        return None, metadata
    previous_horizon = min(value.shape[0] for value in previous_action.values())
    if previous_start_s is None:
        elapsed_steps = int(fallback_replan_horizon)
    else:
        elapsed_s = max(0.0, float(current_s) - float(previous_start_s))
        elapsed_steps = int(math.floor(elapsed_s / float(action_dt_s) + 0.5))
    elapsed_steps = max(0, min(elapsed_steps, previous_horizon))
    raw_overlap = max(0, previous_horizon - elapsed_steps)
    overlap = min(raw_overlap, max(0, int(max_overlap_steps)))
    metadata.update(
        {
            "previous_horizon": previous_horizon,
            "elapsed_steps": elapsed_steps,
            "raw_overlap_steps": raw_overlap,
            "overlap_steps": overlap,
        }
    )
    if overlap <= 0:
        metadata["reason"] = "no_overlap"
        return None, metadata
    options = {
        "action_horizon": previous_horizon,
        "rtc_mode": str(mode),
        "rtc_overlap_steps": overlap,
        "rtc_frozen_steps": min(max(0, int(frozen_steps)), overlap),
        "rtc_ramp_rate": float(ramp_rate),
        "rtc_previous_start_step": elapsed_steps,
    }
    metadata["reason"] = "ok"
    metadata["options"] = options
    return options, metadata


def _to_device_dtype(value: Any, *, device: str, dtype: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        if torch.is_floating_point(value):
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _to_device_dtype(item, device=device, dtype=dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device_dtype(item, device=device, dtype=dtype) for item in value]
    return value


def _action_input_previous_action(action_input: Any) -> Any:
    try:
        if "action" in action_input:
            return action_input["action"]
    except TypeError:
        pass
    return getattr(action_input, "action", None)


def _initial_actions_with_rtc(
    action_head: Any,
    action_input: Any,
    options: dict[str, Any] | None,
    *,
    batch_size: int,
    dtype: Any,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    if hasattr(action_head, "init_actions"):
        actions = action_head.init_actions.expand((batch_size, -1, -1)).to(dtype=dtype, device=device).clone()
    else:
        actions = torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.action_dim),
            dtype=dtype,
            device=device,
        )
    velocity_strength = torch.ones_like(actions)
    previous_action = _action_input_previous_action(action_input)
    if previous_action is None:
        return actions, velocity_strength
    if options is None:
        raise ValueError("RTC previous action requires options")
    required = ("action_horizon", "rtc_overlap_steps", "rtc_frozen_steps", "rtc_ramp_rate")
    missing = [key for key in required if key not in options]
    if missing:
        raise ValueError(f"RTC options missing keys: {missing}")

    previous_action = previous_action.to(dtype=dtype, device=device)
    action_slice_end = max(0, min(int(options["action_horizon"]), previous_action.shape[1]))
    overlap_steps = max(0, min(int(options["rtc_overlap_steps"]), action_slice_end, actions.shape[1]))
    frozen_steps = max(0, min(int(options["rtc_frozen_steps"]), overlap_steps))
    if overlap_steps <= 0:
        return actions, velocity_strength
    start = max(0, min(int(options.get("rtc_previous_start_step", 0)), action_slice_end))
    end = min(action_slice_end, start + overlap_steps)
    overlap_steps = max(0, end - start)
    frozen_steps = min(frozen_steps, overlap_steps)
    if overlap_steps <= 0:
        return actions, velocity_strength

    actions[:, :overlap_steps, :] = previous_action[:, start:end, :]
    velocity_strength[:, :frozen_steps, :] = 0.0
    intermediate_steps = overlap_steps - frozen_steps
    if intermediate_steps > 0:
        ramp = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
        ramp = 1.0 - torch.exp(-float(options["rtc_ramp_rate"]) * ramp)
        ramp = ramp / ramp[-1].clamp_min(1.0e-8)
        velocity_strength[:, frozen_steps:overlap_steps, :] = ramp[1:-1][None, :, None].to(dtype=dtype, device=device)
    return actions, velocity_strength


def _action_head_get_action_with_rtc(
    action_head: Any,
    backbone_features: Any,
    state_features: Any,
    embodiment_id: Any,
    backbone_output: Any,
    action_input: Any,
    options: dict[str, Any] | None = None,
) -> Any:
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    with torch.no_grad():
        vision_language_embeddings = backbone_features
        batch_size = vision_language_embeddings.shape[0]
        actions, velocity_strength = _initial_actions_with_rtc(
            action_head,
            action_input,
            options,
            batch_size=batch_size,
            dtype=vision_language_embeddings.dtype,
            device=vision_language_embeddings.device,
        )
        timestep_delta = 1.0 / action_head.num_inference_timesteps
        for timestep_index in range(action_head.num_inference_timesteps):
            continuous_timestep = timestep_index / float(action_head.num_inference_timesteps)
            discrete_timestep = int(continuous_timestep * action_head.num_timestep_buckets)
            timesteps = torch.full(
                size=(batch_size,), fill_value=discrete_timestep, device=vision_language_embeddings.device
            )
            action_features = action_head.action_encoder(actions, timesteps, embodiment_id)
            if action_head.config.add_pos_embed:
                position_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=vision_language_embeddings.device
                )
                action_features = action_features + action_head.position_embedding(position_ids).unsqueeze(0)
            state_action_embeddings = torch.cat((state_features, action_features), dim=1)
            if action_head.config.use_alternate_vl_dit:
                model_output = action_head.model(
                    hidden_states=state_action_embeddings,
                    encoder_hidden_states=vision_language_embeddings,
                    timestep=timesteps,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = action_head.model(
                    hidden_states=state_action_embeddings,
                    encoder_hidden_states=vision_language_embeddings,
                    timestep=timesteps,
                )
            prediction = action_head.action_decoder(model_output, embodiment_id)
            velocity = prediction[:, -action_head.action_horizon :]
            actions = actions + timestep_delta * velocity * velocity_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vision_language_embeddings,
                "state_features": state_features,
            }
        )


def _patch_pytorch_action_head_rtc(model: Any) -> None:
    action_head = getattr(model, "action_head", None)
    if action_head is None or not hasattr(action_head, "get_action_with_features"):
        raise RuntimeError("Loaded GR00T model does not expose a patchable PyTorch action head")
    if getattr(action_head, "_newton_rtc_patch_status", None) == "enabled":
        return
    action_head._newton_original_get_action_with_features = action_head.get_action_with_features
    action_head.get_action_with_features = partial(_action_head_get_action_with_rtc, action_head)
    action_head._newton_rtc_patch_status = "enabled"


class GrootRtcPolicy:
    """Minimal checkpoint policy wrapper copied from the validated L10 RTC probe path."""

    def __init__(
        self,
        *,
        isaac_groot_root: Path,
        model_path: Path,
        vlm_model_path: Path,
        device: str,
        strict: bool,
    ) -> None:
        _validate_groot_python_runtime()
        root = isaac_groot_root.expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        import gr00t.model  # noqa: F401
        import torch
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import MessageType, VLAStepData
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.policy.policy import BasePolicy
        from transformers import AutoConfig, AutoModel, AutoProcessor

        BasePolicy.__init__(self, strict=strict)
        self._torch = torch
        self._message_type = MessageType
        self._vla_step_data = VLAStepData
        self.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
        self.device = str(device)

        model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        model_config.model_name = str(vlm_model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            config=model_config,
            local_files_only=True,
            transformers_loading_kwargs={"local_files_only": True},
        )
        self.model.eval().to(device=self.device, dtype=torch.bfloat16)
        _patch_pytorch_action_head_rtc(self.model)

        self.processor = AutoProcessor.from_pretrained(
            _processor_path(model_path),
            model_name=str(vlm_model_path),
            transformers_loading_kwargs={"local_files_only": True},
            local_files_only=True,
        )
        self.processor.eval()
        all_configs = self.processor.get_modality_configs()
        self.modality_configs = {
            key: value for key, value in all_configs[self.embodiment_tag.value].items() if key != "rl_info"
        }
        self.language_key = self.modality_configs["language"].modality_keys[0]
        self.collate_fn = self.processor.collator
        self._unbatch_observation = Gr00tPolicy._unbatch_observation.__get__(self)
        self.check_observation = Gr00tPolicy.check_observation.__get__(self)
        self.check_action = Gr00tPolicy.check_action.__get__(self)

    def get_action(
        self,
        observation: dict[str, Any],
        *,
        previous_action: dict[str, np.ndarray] | None,
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.strict:
            self.check_observation(observation)
        unbatched = self._unbatch_observation(observation)
        if previous_action is not None and len(unbatched) != 1:
            raise ValueError("RTC previous_action supports batch size 1 only")
        processed_inputs = []
        states = []
        for item in unbatched:
            states.append(item["state"])
            step = self._vla_step_data(
                images=item["video"],
                states=item["state"],
                actions={} if previous_action is None else previous_action,
                text=item["language"][self.language_key][0],
                embodiment=self.embodiment_tag,
            )
            messages = [{"type": self._message_type.EPISODE_STEP.value, "content": step}]
            processed_inputs.append(self.processor(messages))
        inputs = self.collate_fn(processed_inputs)
        inputs = _to_device_dtype(inputs, device=self.device, dtype=self._torch.bfloat16)
        with self._torch.inference_mode():
            prediction = self.model.get_action(**inputs, options=options)
        normalized_action = prediction["action_pred"].float().cpu().numpy()
        batched_states = {
            key: np.stack([state[key] for state in states], axis=0)
            for key in self.modality_configs["state"].modality_keys
        }
        action = self.processor.decode_action(normalized_action, self.embodiment_tag, batched_states)
        action = {key: np.asarray(value, dtype=np.float32) for key, value in action.items()}
        if self.strict:
            self.check_action(action)
        return action, {
            "backend": "pytorch",
            "rtc_patch": str(getattr(self.model.action_head, "_newton_rtc_patch_status", "unknown")),
        }


class HoldPolicy:
    """Deterministic no-model policy used to validate the Newton control loop."""

    def __init__(self, modalities: CheckpointModalities) -> None:
        self.modalities = modalities

    def get_action(
        self,
        observation: dict[str, Any],
        *,
        previous_action: dict[str, np.ndarray] | None,
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del previous_action, options
        state = observation["state"]
        horizon = len(self.modalities.action.delta_indices)
        source_by_action = {
            "eef_9d": np.asarray(state["eef_9d"])[0, -1],
            "hand_joint_target": np.asarray(state["hand_joint_pos"])[0, -1],
            "arm_joint_target": np.asarray(state["arm_joint_pos"])[0, -1],
        }
        action = {
            key: np.repeat(source_by_action[key][None, None, :], horizon, axis=1).astype(np.float32)
            for key in self.modalities.action.keys
        }
        return action, {"backend": "hold"}


class SmoothEpisodeSource:
    def __init__(self, smooth_dir: Path, episode_index: int, *, loop: bool) -> None:
        self.smooth_dir = smooth_dir.expanduser().resolve()
        self.episode_index = int(episode_index)
        self.loop = bool(loop)
        self.info = _load_json(self.smooth_dir / "meta" / "info.json")
        self.modality = _load_json(self.smooth_dir / "meta" / "modality.json")
        self.episode = self._episode_metadata()
        self.length = int(self.episode["length"])
        self.task = str(self.episode.get("tasks", [DEFAULT_INSTRUCTION])[0])
        self._captures: dict[str, Any] = {}
        self._state_rows: np.ndarray | None = None

    def _episode_metadata(self) -> dict[str, Any]:
        with (self.smooth_dir / "meta" / "episodes.jsonl").open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                if int(row["episode_index"]) == self.episode_index:
                    return row
        raise KeyError(f"Smooth episode {self.episode_index} is not listed in meta/episodes.jsonl")

    def _frame_index(self, frame_index: int) -> int:
        if self.loop:
            return int(frame_index) % self.length
        return int(np.clip(int(frame_index), 0, self.length - 1))

    def _video_path(self, key: str) -> Path:
        return (
            self.smooth_dir
            / "videos"
            / f"chunk-{self.episode_index // 1000:03d}"
            / f"observation.images.{key}"
            / f"episode_{self.episode_index:06d}.mp4"
        )

    def _capture(self, key: str) -> Any:
        if key in self._captures:
            return self._captures[key]
        import cv2

        path = self._video_path(key)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open smooth video: {path}")
        self._captures[key] = capture
        return capture

    def _read_rgb(self, key: str, frame_index: int) -> np.ndarray:
        import cv2

        index = self._frame_index(frame_index)
        capture = self._capture(key)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read {key} frame={index} from {self._video_path(key)}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)

    def video_observation(self, frame_index: int, spec: ModalitySpec) -> dict[str, np.ndarray]:
        return {
            key: np.stack([self._read_rgb(key, frame_index + delta) for delta in spec.delta_indices], axis=0)[None, ...]
            for key in spec.keys
        }

    def _load_state_rows(self) -> np.ndarray:
        if self._state_rows is not None:
            return self._state_rows
        import pyarrow.parquet as parquet

        path = (
            self.smooth_dir
            / "data"
            / f"chunk-{self.episode_index // 1000:03d}"
            / f"episode_{self.episode_index:06d}.parquet"
        )
        table = parquet.read_table(path, columns=["observation.state"])
        values = table["observation.state"].combine_chunks().to_pylist()
        self._state_rows = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
        return self._state_rows

    def state_observation(self, frame_index: int, spec: ModalitySpec) -> dict[str, np.ndarray]:
        rows = self._load_state_rows()
        output = {}
        for key in spec.keys:
            metadata = self.modality["state"].get(key)
            if metadata is None:
                raise KeyError(f"Smooth modality metadata does not contain state key {key!r}")
            start, end = int(metadata["start"]), int(metadata["end"])
            values = [rows[self._frame_index(frame_index + delta), start:end] for delta in spec.delta_indices]
            output[key] = np.stack(values, axis=0)[None, ...].astype(np.float32)
        return output

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()


class SimVideoHistory:
    def __init__(self, spec: ModalitySpec) -> None:
        history = max([abs(value) for value in spec.delta_indices] + [0]) + 1
        self.spec = spec
        self.frames: deque[dict[str, np.ndarray]] = deque(maxlen=history)

    def append(self, images: dict[str, np.ndarray]) -> None:
        self.frames.append({key: np.asarray(value, dtype=np.uint8) for key, value in images.items()})

    def observation(self) -> dict[str, np.ndarray]:
        if not self.frames:
            raise RuntimeError("Simulator image history is empty")
        buffered = list(self.frames)
        output = {}
        for key in self.spec.keys:
            selected = []
            for delta in self.spec.delta_indices:
                index = len(buffered) - 1 + min(int(delta), 0)
                selected.append(buffered[max(0, index)][key])
            output[key] = np.stack(selected, axis=0)[None, ...].astype(np.uint8)
        return output


class NewtonPolicyController:
    """Bridge decoded policy targets into Newton EEF IK and L10 drive control."""

    def __init__(
        self,
        example: scene_runtime.Example,
        *,
        arm_control_mode: str,
        eef_transform_mode: str,
        eef_frame_update: str,
        eef_body_suffix: str,
        action_dt_s: float,
        max_arm_joint_step: float,
        max_hand_joint_step: float,
        ik_finite_difference_rad: float,
        ik_max_task_step_m: float,
        ik_max_rotation_step_rad: float,
        ik_position_weight: float,
        ik_orientation_weight: float,
        ik_damping_lambda: float,
        arm_joint_fallback: bool,
    ) -> None:
        self.example = example
        self.arm_control_mode = str(arm_control_mode)
        self.eef_transform_mode = str(eef_transform_mode)
        self.eef_frame_update = str(eef_frame_update)
        self.action_dt_s = max(float(action_dt_s), 1.0e-6)
        self.max_arm_joint_step = max(0.0, float(max_arm_joint_step))
        self.max_hand_joint_step = max(0.0, float(max_hand_joint_step))
        self.arm_joint_fallback = bool(arm_joint_fallback)
        self.q_index_by_label, self.qd_index_by_label = self._joint_index_maps()
        self.arm_labels = tuple(f"right_joint{index}" for index in range(1, 8))
        self.arm_q_indices = tuple(self._required_q_index(label) for label in self.arm_labels)
        self.arm_qd_indices = tuple(self._required_qd_index(label) for label in self.arm_labels)
        self.hand_spec = load_linker_l10_right_hand_spec()
        self._target_q = self.example.control.joint_target_q.numpy().copy()
        self._target_qd = self.example.control.joint_target_qd.numpy().copy()
        self._joint_limit_lower = self.example.model.joint_limit_lower.numpy().copy()
        self._joint_limit_upper = self.example.model.joint_limit_upper.numpy().copy()
        self._policy_from_world_rotation = np.eye(3, dtype=np.float64)
        self._policy_from_world_translation = np.zeros(3, dtype=np.float64)
        self._state_to_genesis_transform = _rigid_transform_matrix(
            NODE0_STATE_TO_GENESIS_TRANSLATION_XYZ,
            NODE0_STATE_TO_GENESIS_QUATERNION_XYZW,
        )
        self._eef_offset_transform = _rigid_transform_matrix(
            NODE0_STATE_TO_GENESIS_EEF_OFFSET_TRANSLATION_XYZ,
            NODE0_STATE_TO_GENESIS_EEF_OFFSET_QUATERNION_XYZW,
        )
        self._eef_frame_calibrated = False

        self.kinematics = NewtonLinkKinematicsModel(
            model=self.example.model,
            side="right",
            arm_joint_q_indices=self.arm_q_indices,
            eef_body_suffix=str(eef_body_suffix),
            finite_difference_rad=float(ik_finite_difference_rad),
        )
        newton_world_from_base = self._body_pose_matrix("/right_base_link")
        self._genesis_to_world_transform = newton_world_from_base @ _invert_rigid_transform(
            self._state_to_genesis_transform
        )
        initial_q = tuple(float(value) for value in self.arm_q())
        lower_limits = tuple(float(self._joint_limit_lower[index]) for index in self.arm_qd_indices)
        upper_limits = tuple(float(self._joint_limit_upper[index]) for index in self.arm_qd_indices)
        self.ik = FullPoseDifferentialIkController(
            FullPoseDifferentialIkControllerConfig(
                seed_joint_positions_rad=initial_q,
                joint_lower_limits_rad=lower_limits,
                joint_upper_limits_rad=upper_limits,
                neutral_joint_positions_rad=initial_q,
                kinematics_model=self.kinematics,
                max_task_step_m=float(ik_max_task_step_m),
                max_rotation_step_rad=float(ik_max_rotation_step_rad),
                position_weight=float(ik_position_weight),
                orientation_weight=float(ik_orientation_weight),
                max_joint_step_rad=self.max_arm_joint_step,
                max_joint_velocity_rad_s=self.max_arm_joint_step / self.action_dt_s,
                damping_lambda=float(ik_damping_lambda),
                posture_bias_gain=0.04,
                joint_limit_bias_gain=0.35,
                bias_weight=0.08,
                joint_limit_soft_margin_rad=0.25,
            )
        )
        self.ik.reset(self._robot_state(timestamp_s=0.0))
        print(
            "[groot-control] "
            f"arm_mode={self.arm_control_mode} eef_transform={self.eef_transform_mode} "
            f"eef_frame_update={self.eef_frame_update} "
            f"eef_body={eef_body_suffix} arm_joint_fallback={self.arm_joint_fallback} "
            f"genesis_to_newton_xyz={np.round(self._genesis_to_world_transform[:3, 3], 6).tolist()}",
            flush=True,
        )

    def _joint_index_maps(self) -> tuple[dict[str, int], dict[str, int]]:
        model = self.example.model
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        q_map: dict[str, int] = {}
        qd_map: dict[str, int] = {}
        for joint_index, full_label in enumerate(model.joint_label):
            label = str(full_label).rsplit("/", maxsplit=1)[-1]
            q0 = int(q_start[joint_index])
            q1 = int(q_start[joint_index + 1]) if joint_index + 1 < len(q_start) else model.joint_coord_count
            qd0 = int(qd_start[joint_index])
            qd1 = int(qd_start[joint_index + 1]) if joint_index + 1 < len(qd_start) else model.joint_dof_count
            if q1 - q0 == 1:
                q_map[label] = q0
            if qd1 - qd0 == 1:
                qd_map[label] = qd0
        return q_map, qd_map

    def _required_q_index(self, label: str) -> int:
        if label not in self.q_index_by_label:
            raise KeyError(f"Newton model is missing scalar joint {label!r}")
        return int(self.q_index_by_label[label])

    def _required_qd_index(self, label: str) -> int:
        if label not in self.qd_index_by_label:
            raise KeyError(f"Newton model is missing scalar joint DOF {label!r}")
        return int(self.qd_index_by_label[label])

    def _body_pose_matrix(self, label_suffix: str) -> np.ndarray:
        body_index = next(
            (index for index, label in enumerate(self.example.model.body_label) if label.endswith(label_suffix)),
            None,
        )
        if body_index is None:
            raise ValueError(f"Newton model is missing body ending with {label_suffix!r}")
        body_q = _require_finite_values(
            self.example.state_0.body_q.numpy()[body_index],
            name=f"Newton body pose {label_suffix!r}",
        )
        return _pose7_to_matrix(
            Pose7(
                position_xyz=tuple(float(value) for value in body_q[:3]),
                quaternion_xyzw=tuple(float(value) for value in body_q[3:7]),
            )
        )

    def _clip_joint(self, label: str, value: float) -> float:
        qd_index = self.qd_index_by_label.get(label)
        if qd_index is None:
            return float(value)
        lower = float(self._joint_limit_lower[qd_index])
        upper = float(self._joint_limit_upper[qd_index])
        return float(np.clip(value, min(lower, upper), max(lower, upper)))

    def arm_q(self) -> np.ndarray:
        joint_q = self.example.state_0.joint_q.numpy()
        return _require_finite_values(
            np.asarray([joint_q[index] for index in self.arm_q_indices], dtype=np.float32),
            name="Newton arm joint positions",
        )

    def arm_qd(self) -> np.ndarray:
        joint_qd = self.example.state_0.joint_qd.numpy()
        return _require_finite_values(
            np.asarray([joint_qd[index] for index in self.arm_qd_indices], dtype=np.float32),
            name="Newton arm joint velocities",
        )

    def _robot_state(self, *, timestamp_s: float) -> RobotStateSnapshot:
        full_joint_q = _require_finite_values(
            self.example.state_0.joint_q.numpy(),
            name="Newton joint positions before EEF IK",
        )
        self.kinematics.sync_joint_q(full_joint_q)
        arm_q = tuple(float(value) for value in self.arm_q())
        arm_qd = tuple(float(value) for value in self.arm_qd())
        return RobotStateSnapshot(
            timestamp_s=float(timestamp_s),
            joint_positions_rad=arm_q,
            joint_velocities_rad_s=arm_qd,
            ee_pose=self.kinematics.forward_pose(arm_q),
        )

    def calibrate_eef_frame(self, observation_eef_9d: np.ndarray, *, timestamp_s: float) -> dict[str, Any]:
        if self.arm_control_mode != "eef_ik":
            return {"enabled": False, "reason": "arm_control_mode_joint_target"}
        if self.eef_transform_mode == "node0_fixed":
            if self._eef_frame_calibrated:
                return {
                    "enabled": True,
                    "updated": False,
                    "reason": "node0_fixed_transform",
                    "state_to_genesis_translation": np.asarray(
                        NODE0_STATE_TO_GENESIS_TRANSLATION_XYZ, dtype=np.float64
                    ),
                    "state_to_genesis_rot6d": _rotmat_to_rot6d(self._state_to_genesis_transform[:3, :3]),
                    "eef_offset_translation": np.asarray(
                        NODE0_STATE_TO_GENESIS_EEF_OFFSET_TRANSLATION_XYZ, dtype=np.float64
                    ),
                    "eef_offset_rot6d": _rotmat_to_rot6d(self._eef_offset_transform[:3, :3]),
                    "genesis_to_newton_translation": self._genesis_to_world_transform[:3, 3].copy(),
                    "genesis_to_newton_rot6d": _rotmat_to_rot6d(self._genesis_to_world_transform[:3, :3]),
                }

            policy_pose = _eef_9d_to_pose(observation_eef_9d)
            robot_state = self._robot_state(timestamp_s=timestamp_s)
            assert robot_state.ee_pose is not None
            current_world_pose = _pose7_to_matrix(robot_state.ee_pose)
            self._eef_frame_calibrated = True
            self.ik.reset(robot_state)
            mapped_world_pose = self._policy_pose_to_world_pose(policy_pose)
            residual = mapped_world_pose[:3, 3] - current_world_pose[:3, 3]
            print(
                "[groot-eef-frame] node0 fixed A*T_policy*B with Genesis-to-Newton alignment "
                f"policy_xyz={np.round(policy_pose[:3, 3], 6).tolist()} "
                f"mapped_world_xyz={np.round(mapped_world_pose[:3, 3], 6).tolist()} "
                f"current_world_xyz={np.round(current_world_pose[:3, 3], 6).tolist()} "
                f"initial_residual={np.round(residual, 6).tolist()}",
                flush=True,
            )
            return {
                "enabled": True,
                "updated": True,
                "reason": "node0_fixed_transform",
                "policy_xyz": policy_pose[:3, 3].copy(),
                "mapped_world_xyz": mapped_world_pose[:3, 3].copy(),
                "current_world_xyz": current_world_pose[:3, 3].copy(),
                "initial_residual_xyz": residual.copy(),
                "state_to_genesis_translation": np.asarray(NODE0_STATE_TO_GENESIS_TRANSLATION_XYZ, dtype=np.float64),
                "state_to_genesis_rot6d": _rotmat_to_rot6d(self._state_to_genesis_transform[:3, :3]),
                "eef_offset_translation": np.asarray(
                    NODE0_STATE_TO_GENESIS_EEF_OFFSET_TRANSLATION_XYZ, dtype=np.float64
                ),
                "eef_offset_rot6d": _rotmat_to_rot6d(self._eef_offset_transform[:3, :3]),
                "genesis_to_newton_translation": self._genesis_to_world_transform[:3, 3].copy(),
                "genesis_to_newton_rot6d": _rotmat_to_rot6d(self._genesis_to_world_transform[:3, :3]),
            }

        should_update = not self._eef_frame_calibrated or self.eef_frame_update == "replan"
        if not should_update:
            return {
                "enabled": True,
                "updated": False,
                "reason": "fixed_frame_already_calibrated",
                "policy_from_world_translation": self._policy_from_world_translation.copy(),
                "policy_from_world_rot6d": _rotmat_to_rot6d(self._policy_from_world_rotation),
            }

        policy_pose = _eef_9d_to_pose(observation_eef_9d)
        robot_state = self._robot_state(timestamp_s=timestamp_s)
        assert robot_state.ee_pose is not None
        world_pose = _pose7_to_matrix(robot_state.ee_pose)
        rotation = policy_pose[:3, :3] @ world_pose[:3, :3].T
        translation = policy_pose[:3, 3] - rotation @ world_pose[:3, 3]
        self._policy_from_world_rotation = rotation
        self._policy_from_world_translation = translation
        self._eef_frame_calibrated = True
        self.ik.reset(robot_state)

        check = self._world_pose_to_policy_pose(world_pose)
        residual = check[:3, 3] - policy_pose[:3, 3]
        print(
            "[groot-eef-frame] calibrated "
            f"update={self.eef_frame_update} "
            f"policy_xyz={np.round(policy_pose[:3, 3], 6).tolist()} "
            f"world_xyz={np.round(world_pose[:3, 3], 6).tolist()} "
            f"residual={np.round(residual, 9).tolist()}",
            flush=True,
        )
        return {
            "enabled": True,
            "updated": True,
            "reason": "initial_calibration" if self.eef_frame_update == "once" else "replan_calibration",
            "policy_xyz": policy_pose[:3, 3].copy(),
            "world_xyz": world_pose[:3, 3].copy(),
            "policy_from_world_translation": translation.copy(),
            "policy_from_world_rot6d": _rotmat_to_rot6d(rotation),
            "residual_xyz": residual.copy(),
        }

    def _world_pose_to_policy_pose(self, world_pose: np.ndarray) -> np.ndarray:
        world = np.asarray(world_pose, dtype=np.float64).reshape(4, 4)
        if self.eef_transform_mode == "node0_fixed":
            return (
                _invert_rigid_transform(self._state_to_genesis_transform)
                @ _invert_rigid_transform(self._genesis_to_world_transform)
                @ world
                @ _invert_rigid_transform(self._eef_offset_transform)
            )
        policy = np.eye(4, dtype=np.float64)
        policy[:3, 3] = self._policy_from_world_rotation @ world[:3, 3] + self._policy_from_world_translation
        policy[:3, :3] = self._policy_from_world_rotation @ world[:3, :3]
        return policy

    def _policy_pose_to_world_pose(self, policy_pose: np.ndarray) -> np.ndarray:
        if not self._eef_frame_calibrated:
            raise RuntimeError("EEF action frame has not been calibrated from the current observation")
        policy = np.asarray(policy_pose, dtype=np.float64).reshape(4, 4)
        if self.eef_transform_mode == "node0_fixed":
            return (
                self._genesis_to_world_transform
                @ self._state_to_genesis_transform
                @ policy
                @ self._eef_offset_transform
            )
        world = np.eye(4, dtype=np.float64)
        inverse_rotation = self._policy_from_world_rotation.T
        world[:3, 3] = inverse_rotation @ (policy[:3, 3] - self._policy_from_world_translation)
        world[:3, :3] = inverse_rotation @ policy[:3, :3]
        return world

    def hand_command_q(self) -> np.ndarray:
        joint_q = self.example.state_0.joint_q.numpy()
        values = []
        for name in POLICY_HAND_JOINT_NAMES:
            label = f"right_l10_{name}"
            values.append(float(joint_q[self._required_q_index(label)]))
        return _require_finite_values(
            np.asarray(values, dtype=np.float32),
            name="Newton hand joint positions",
        )

    def state_groups(self) -> dict[str, np.ndarray]:
        arm_q = self.arm_q()
        eef_pose = nero_can_flange_pose_from_joints(arm_q)
        eef_9d = np.concatenate([np.asarray(eef_pose[:3, 3], dtype=np.float32), _rotmat_to_rot6d(eef_pose[:3, :3])])
        return {
            "eef_9d": eef_9d.astype(np.float32),
            "hand_joint_pos": _reported_hand_q_from_command(self.hand_command_q()),
            "arm_joint_pos": arm_q,
        }

    @staticmethod
    def _action_row(action: dict[str, np.ndarray], key: str, index: int) -> np.ndarray | None:
        if key not in action:
            return None
        chunk = np.asarray(action[key], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            raise ValueError(f"Action {key!r} must be a nonempty [T,D] chunk, got {chunk.shape}")
        row = chunk[min(max(0, int(index)), chunk.shape[0] - 1)]
        return _require_finite_values(row, name=f"Action {key!r}", error_type=ValueError)

    def _apply_arm_joint_target(self, arm_target: np.ndarray, current_q: np.ndarray) -> None:
        for slot, label in enumerate(self.arm_labels):
            if slot >= arm_target.size:
                break
            q_index = self.arm_q_indices[slot]
            desired = self._clip_joint(label, float(arm_target[slot]))
            if self.max_arm_joint_step > 0.0:
                desired = float(
                    np.clip(
                        desired,
                        float(current_q[q_index]) - self.max_arm_joint_step,
                        float(current_q[q_index]) + self.max_arm_joint_step,
                    )
                )
            self._target_q[q_index] = desired
            self._target_qd[self.arm_qd_indices[slot]] = 0.0

    def _apply_eef_target(self, eef_target: np.ndarray, *, timestamp_s: float) -> dict[str, Any]:
        policy_pose = _eef_9d_to_pose(eef_target)
        world_pose = self._policy_pose_to_world_pose(policy_pose)
        robot_state = self._robot_state(timestamp_s=timestamp_s)
        assert robot_state.ee_pose is not None
        current_world_pose = _pose7_to_matrix(robot_state.ee_pose)
        self.ik.set_target(
            TaskSpaceTarget(
                arm_side="right",
                source_name="groot_rtc",
                timestamp_s=float(timestamp_s),
                frame_id=int(round(timestamp_s / self.action_dt_s)),
                ee_target=_matrix_to_pose7(world_pose),
                orientation_mode="track_full_orientation",
                target_frame="world",
            )
        )
        result = self.ik.step(robot_state, dt_s=self.action_dt_s)
        q_command = np.asarray(result.q_cmd, dtype=np.float64).reshape(-1)
        if q_command.size < 7 or not np.isfinite(q_command[:7]).all() or result.status.startswith("fault"):
            raise RuntimeError(f"EEF IK returned status={result.status} q_cmd={q_command}")
        qd_command = None if result.dq_cmd is None else np.asarray(result.dq_cmd, dtype=np.float64).reshape(-1)
        if qd_command is not None:
            _require_finite_values(qd_command, name="EEF IK joint velocity command")
        for slot, label in enumerate(self.arm_labels):
            self._target_q[self.arm_q_indices[slot]] = self._clip_joint(label, float(q_command[slot]))
            self._target_qd[self.arm_qd_indices[slot]] = (
                0.0 if qd_command is None or slot >= qd_command.size else float(qd_command[slot])
            )
        return {
            "status": result.status,
            "events": result.events,
            "policy_target_xyz": policy_pose[:3, 3].copy(),
            "world_target_xyz": world_pose[:3, 3].copy(),
            "world_current_xyz": current_world_pose[:3, 3].copy(),
            "world_target_error_m": float(np.linalg.norm(world_pose[:3, 3] - current_world_pose[:3, 3])),
            "target_position_error_m": result.target_position_error_m,
            "target_orientation_error_rad": result.target_orientation_error_rad,
            "residual_position_error_m": result.residual_position_error_m,
            "residual_orientation_error_rad": result.residual_orientation_error_rad,
            "singularity_metric": result.singularity_metric,
            "damping_scale": result.damping_scale,
            "q_command": q_command[:7].copy(),
        }

    def apply(self, action: dict[str, np.ndarray], action_index: int, *, timestamp_s: float) -> dict[str, Any]:
        current_q = _require_finite_values(
            self.example.state_0.joint_q.numpy(),
            name="Newton joint positions before applying policy action",
        )
        _require_finite_values(
            self.example.state_0.joint_qd.numpy(),
            name="Newton joint velocities before applying policy action",
        )
        arm_target = self._action_row(action, "arm_joint_target", action_index)
        eef_target = self._action_row(action, "eef_9d", action_index)
        arm_source = "hold"
        eef_applied: dict[str, Any] | None = None
        eef_error: str | None = None
        if self.arm_control_mode == "eef_ik" and eef_target is not None:
            try:
                eef_applied = self._apply_eef_target(eef_target, timestamp_s=timestamp_s)
                arm_source = "eef_ik"
            except Exception as exc:
                eef_error = str(exc)
                if not self.arm_joint_fallback:
                    raise
        if arm_source != "eef_ik" and arm_target is not None:
            self._apply_arm_joint_target(arm_target, current_q)
            arm_source = "arm_joint_target_fallback" if eef_error is not None else "arm_joint_target"

        hand_target = self._action_row(action, "hand_joint_target", action_index)
        command_target = None
        if hand_target is not None:
            command_target = _reported_hand_q_to_command(hand_target[: len(POLICY_HAND_JOINT_NAMES)])
            current_hand = self.hand_command_q()
            if self.max_hand_joint_step > 0.0:
                command_target = np.clip(
                    command_target,
                    current_hand - self.max_hand_joint_step,
                    current_hand + self.max_hand_joint_step,
                )
            base_values = NamedJointValues(
                joint_names=POLICY_HAND_JOINT_NAMES,
                joint_positions=tuple(float(value) for value in command_target),
            )
            expanded = self.hand_spec.expand_mimic_joint_values(base_values)
            for name, value in zip(expanded.joint_names, expanded.joint_positions, strict=True):
                label = f"right_l10_{name}"
                q_index = self.q_index_by_label.get(label)
                if q_index is None:
                    continue
                self._target_q[q_index] = self._clip_joint(label, float(value))
                qd_index = self.qd_index_by_label.get(label)
                if qd_index is not None:
                    self._target_qd[qd_index] = 0.0

        _require_finite_values(self._target_q, name="Newton joint position control targets")
        _require_finite_values(self._target_qd, name="Newton joint velocity control targets")
        self.example.control.joint_target_q.assign(self._target_q)
        self.example.control.joint_target_qd.assign(self._target_qd)
        return {
            "arm_source": arm_source,
            "arm_target": None if arm_target is None else arm_target[:7].copy(),
            "arm_applied": self._target_q[np.asarray(self.arm_q_indices, dtype=np.int32)].copy(),
            "eef_target": None if eef_target is None else eef_target[:9].copy(),
            "eef_applied": eef_applied,
            "eef_error": eef_error,
            "hand_reported_target": None if hand_target is None else hand_target[:10].copy(),
            "hand_command_target": None if command_target is None else command_target.copy(),
        }


class GrootRtcExample(scene_runtime.Example):
    def __init__(self, viewer: Any, args: argparse.Namespace) -> None:
        self.groot_args = args
        self.modalities = _validate_asset_layout(args)
        super().__init__(viewer, args)
        self.viewer_fifo_preview = (
            ViewerFifoPreview(
                viewer,
                args.viewer_fifo_preview,
                width=int(args.viewer_fifo_preview_width),
                height=int(args.viewer_fifo_preview_height),
                fps=float(args.viewer_fifo_preview_fps),
                input_width=int(args.viewer_fifo_preview_input_width),
            )
            if args.viewer_fifo_preview is not None
            else None
        )
        _initialize_right_hand_pose(self, tuple(args.groot_initial_hand_q))
        if self.d455_preview is None or not self.d455_preview.enabled:
            raise ValueError("GR00T simulator images require --d455-preview")
        if self.d405_preview is None or not self.d405_preview.enabled:
            raise ValueError("GR00T simulator images require --d405-preview")

        self.smooth = SmoothEpisodeSource(args.smooth_dir, args.episode_index, loop=args.smooth_loop)
        self.instruction = str(args.instruction).strip() or self.smooth.task or DEFAULT_INSTRUCTION
        self.image_source = str(args.image_source)
        self.state_source = str(args.state_source)
        self.action_dt_s = 1.0 / max(float(args.action_fps), 1.0e-6)
        self.replan_horizon = max(1, int(args.replan_horizon))
        self.action_horizon = len(self.modalities.action.delta_indices)
        self.controller = NewtonPolicyController(
            self,
            arm_control_mode=str(args.arm_control_mode),
            eef_transform_mode=str(args.eef_transform_mode),
            eef_frame_update=str(args.eef_frame_update),
            eef_body_suffix=str(args.eef_body_suffix),
            action_dt_s=self.action_dt_s,
            max_arm_joint_step=float(args.max_arm_joint_step),
            max_hand_joint_step=float(args.max_hand_joint_step),
            ik_finite_difference_rad=float(args.ik_finite_difference_rad),
            ik_max_task_step_m=float(args.ik_max_task_step_m),
            ik_max_rotation_step_rad=math.radians(float(args.ik_max_rotation_step_deg)),
            ik_position_weight=float(args.ik_position_weight),
            ik_orientation_weight=float(args.ik_orientation_weight),
            ik_damping_lambda=float(args.ik_damping_lambda),
            arm_joint_fallback=bool(args.arm_joint_fallback),
        )
        self.sim_video_history = SimVideoHistory(self.modalities.video)
        self.seed_manager = TeleopRtcSeedManager(
            action_keys=self.modalities.action.keys,
            action_dt_s=self.action_dt_s,
        )
        self.policy_step = 0
        self.replan_index = 0
        self._first_observation_dumped = False
        self._model_input_preview_images: dict[str, np.ndarray] = {}
        self._model_input_preview_logged = False
        self._sim_ego_preprocess_logged = False
        self.action_index = 0
        self.action_chunk: dict[str, np.ndarray] | None = None
        self.next_policy_time_s = 0.0
        self.policy_enabled = bool(args.start_policy)
        self.async_policy = bool(args.async_policy and not args.dry_run_policy)
        self._policy_executor: ThreadPoolExecutor | None = None
        self._replan_future: Future[PolicyReplanResult] | None = None
        self._policy_torch: Any | None = None
        self._policy_cuda_stream: Any | None = None
        self.trace_path = None if args.no_policy_trace else args.trace_jsonl.expanduser().resolve()
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.write_text("", encoding="utf-8")

        if args.dry_run_policy:
            self.policy: Any = HoldPolicy(self.modalities)
        else:
            policy_device = _resolve_policy_device(args.policy_device)
            print(
                f"[groot] loading checkpoint={args.policy_checkpoint.resolve()} "
                f"vlm={args.vlm_model.resolve()} device={policy_device}",
                flush=True,
            )
            self.policy = GrootRtcPolicy(
                isaac_groot_root=args.isaac_groot_root,
                model_path=args.policy_checkpoint.resolve(),
                vlm_model_path=args.vlm_model.resolve(),
                device=policy_device,
                strict=bool(args.strict_policy),
            )
            if self.async_policy:
                import torch

                self._policy_torch = torch
                self._policy_cuda_stream = torch.cuda.Stream(device=policy_device)
                torch.cuda.current_stream(device=policy_device).synchronize()
                self._policy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="groot-gpu")

        # Seed live image buffers once; subsequent policy calls consume the prior rendered frame.
        self.render_camera_previews()
        self._closed = False
        atexit.register(self.close_policy_resources)
        print(
            "[groot] ready "
            f"image_source={self.image_source} state_source={self.state_source} "
            f"episode={self.smooth.episode_index} instruction={self.instruction!r} "
            f"action_hz={args.action_fps:g} replan={self.replan_horizon} "
            f"rtc={bool(args.rtc)} arm_control={args.arm_control_mode} "
            f"async_policy={self.async_policy} capture_graph={bool(args.capture_graph)} "
            f"start_policy={self.policy_enabled}",
            flush=True,
        )

    def _sim_video_observation(self) -> dict[str, np.ndarray]:
        assert self.d455_preview is not None and self.d405_preview is not None
        ego_source = _packed_color_to_rgb(self.d455_preview.color_image)
        roi_zoom = float(self.d455_roi_zoom) if self.groot_args.sim_ego_roi else 1.0
        ego_view = _node0_ego_view_preprocess(
            ego_source,
            zoom=roi_zoom,
            center_x=float(self.d455_roi_center_x),
            center_y=float(self.d455_roi_center_y),
        )
        if not self._sim_ego_preprocess_logged:
            source_height, source_width, _ = ego_source.shape
            crop_rect = scene_runtime._roi_crop_rect(  # noqa: SLF001
                source_width,
                source_height,
                zoom=roi_zoom,
                center_x=float(self.d455_roi_center_x),
                center_y=float(self.d455_roi_center_y),
            )
            print(
                "[groot-ego-view] node0 pipeline "
                f"source={source_width}x{source_height} crop_xywh={crop_rect} "
                f"zoom={roi_zoom:g} center=({self.d455_roi_center_x:g},{self.d455_roi_center_y:g}) "
                f"model_input={NODE0_EGO_INPUT_WIDTH}x{NODE0_EGO_INPUT_HEIGHT} rgb=True",
                flush=True,
            )
            self._sim_ego_preprocess_logged = True
        self.sim_video_history.append(
            {
                "ego_view": ego_view,
                "wrist_view": _packed_color_to_rgb(self.d405_preview.color_image),
            }
        )
        return self.sim_video_history.observation()

    def _sim_state_observation(self) -> dict[str, np.ndarray]:
        source = self.controller.state_groups()
        return {key: np.asarray(source[key], dtype=np.float32)[None, None, ...] for key in self.modalities.state.keys}

    def _build_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        smooth_frame = int(self.groot_args.smooth_frame_offset) + self.policy_step
        if self.image_source == "smooth":
            video = self.smooth.video_observation(smooth_frame, self.modalities.video)
        else:
            video = self._sim_video_observation()
        if self.state_source == "smooth":
            state = self.smooth.state_observation(smooth_frame, self.modalities.state)
        else:
            state = self._sim_state_observation()
        language = {key: [[self.instruction]] for key in self.modalities.language.keys}
        observation = {"video": video, "state": state, "language": language}
        self._model_input_preview_images = {
            key: np.ascontiguousarray(np.asarray(value, dtype=np.uint8)[0, -1])
            for key, value in video.items()
            if key in {"ego_view", "wrist_view"}
        }
        if not self._model_input_preview_logged:
            image_shapes = " ".join(
                f"{key}={tuple(image.shape)}" for key, image in self._model_input_preview_images.items()
            )
            print(
                f"[groot-input-preview] exact model inputs source={self.image_source} {image_shapes}",
                flush=True,
            )
            self._model_input_preview_logged = True
        if self.groot_args.dump_first_observation_dir is not None and not self._first_observation_dumped:
            from PIL import Image

            dump_dir = self.groot_args.dump_first_observation_dir.expanduser().resolve()
            dump_dir.mkdir(parents=True, exist_ok=True)
            for key, value in video.items():
                frame = np.asarray(value, dtype=np.uint8).reshape(-1, *np.asarray(value).shape[-3:])[-1]
                Image.fromarray(frame).save(dump_dir / f"{key}.png")
            self._first_observation_dumped = True
            print(f"[groot] dumped first model observation to {dump_dir}", flush=True)
        metadata = {
            "image_source": self.image_source,
            "state_source": self.state_source,
            "episode_index": self.smooth.episode_index,
            "smooth_frame": self.smooth._frame_index(smooth_frame),
            "video_shapes": {key: list(value.shape) for key, value in video.items()},
            "state_shapes": {key: list(value.shape) for key, value in state.items()},
            "state_latest": {key: np.asarray(value)[0, -1].copy() for key, value in state.items()},
            "image_preprocessing": "raw_rgb_no_manual_resize_or_letterbox",
        }
        return observation, metadata

    def _prepare_replan(self) -> PolicyReplanRequest:
        observation, source_metadata = self._build_observation()
        timeline_s = float(self.sim_time)
        observation_eef = np.asarray(observation["state"]["eef_9d"], dtype=np.float32).reshape(-1, 9)[-1]
        source_metadata["eef_frame"] = self.controller.calibrate_eef_frame(
            observation_eef,
            timestamp_s=timeline_s,
        )
        rtc_seed, seed_start_s, seed_metadata = self.seed_manager.seed_window(
            anchor_start_s=timeline_s,
            anchor_frame_id=self.policy_step,
            horizon=self.action_horizon,
        )
        options, rtc_metadata = _rtc_options(
            enabled=bool(self.groot_args.rtc),
            mode=str(self.groot_args.rtc_mode),
            previous_action=rtc_seed,
            previous_start_s=seed_start_s,
            current_s=timeline_s,
            action_dt_s=self.action_dt_s,
            fallback_replan_horizon=self.replan_horizon,
            max_overlap_steps=int(self.groot_args.rtc_max_overlap_steps),
            frozen_steps=int(self.groot_args.rtc_frozen_steps),
            ramp_rate=float(self.groot_args.rtc_ramp_rate),
        )
        return PolicyReplanRequest(
            replan_index=self.replan_index,
            policy_step=self.policy_step,
            timeline_s=timeline_s,
            observation=observation,
            source_metadata=source_metadata,
            rtc_seed=rtc_seed,
            seed_metadata=seed_metadata,
            options=options,
            rtc_metadata=rtc_metadata,
        )

    def _infer_replan(self, request: PolicyReplanRequest) -> PolicyReplanResult:
        started = time.perf_counter()
        if self._policy_cuda_stream is None:
            policy_action, policy_metadata = self.policy.get_action(
                request.observation,
                previous_action=request.rtc_seed if request.options is not None else None,
                options=request.options,
            )
        else:
            assert self._policy_torch is not None
            with self._policy_torch.cuda.stream(self._policy_cuda_stream):
                policy_action, policy_metadata = self.policy.get_action(
                    request.observation,
                    previous_action=request.rtc_seed if request.options is not None else None,
                    options=request.options,
                )
            self._policy_cuda_stream.synchronize()
        inference_s = time.perf_counter() - started
        return PolicyReplanResult(
            request=request,
            policy_action=policy_action,
            policy_metadata=policy_metadata,
            inference_s=inference_s,
        )

    def _install_replan(self, result: PolicyReplanResult) -> bool:
        request = result.request
        unbatched = _unbatch_action(result.policy_action, self.modalities.action.keys)
        frozen_steps = 0 if request.options is None else int(request.options["rtc_frozen_steps"])
        action_chunk = _stored_rtc_action(
            policy_action=unbatched,
            rtc_seed_action=request.rtc_seed,
            action_keys=self.modalities.action.keys,
            frozen_steps=frozen_steps,
        )
        horizon = min(value.shape[0] for value in action_chunk.values())
        elapsed_steps = _elapsed_action_steps(
            start_s=request.timeline_s,
            current_s=float(self.sim_time),
            action_dt_s=self.action_dt_s,
        )
        stale = elapsed_steps >= horizon
        if not stale:
            self.seed_manager.push(action_chunk, start_s=request.timeline_s, frame_id=request.policy_step)
            self.action_chunk = action_chunk
            self.action_index = elapsed_steps
        _append_jsonl(
            self.trace_path,
            {
                "schema_version": "newton.groot_rtc.replan.v1",
                "event": "replan",
                "replan_index": request.replan_index,
                "policy_step": request.policy_step,
                "timeline_s": request.timeline_s,
                "inference_s": result.inference_s,
                "async_policy": self.async_policy,
                "elapsed_steps_before_install": elapsed_steps,
                "stale": stale,
                "source": request.source_metadata,
                "rtc": request.rtc_metadata,
                "rtc_seed": request.seed_metadata,
                "policy": result.policy_metadata,
                "action": action_chunk,
            },
        )
        print(
            f"[groot] replan={request.replan_index} step={request.policy_step} "
            f"inference={result.inference_s:.3f}s async={self.async_policy} "
            f"install_index={elapsed_steps}/{horizon} stale={stale} "
            f"image={self.image_source} state={self.state_source} "
            f"rtc={request.rtc_metadata.get('reason')} "
            f"overlap={request.rtc_metadata.get('overlap_steps', 0)}",
            flush=True,
        )
        self.replan_index += 1
        return not stale

    def _submit_replan(self) -> None:
        request = self._prepare_replan()
        if self._policy_executor is None:
            self._install_replan(self._infer_replan(request))
            return
        self._replan_future = self._policy_executor.submit(self._infer_replan, request)
        print(
            f"[groot] submitted async replan={request.replan_index} "
            f"step={request.policy_step} timeline={request.timeline_s:.3f}s",
            flush=True,
        )

    def _consume_replan(self) -> bool:
        future = self._replan_future
        if future is None or not future.done():
            return False
        self._replan_future = None
        return self._install_replan(future.result())

    def _policy_tick(self) -> None:
        self._consume_replan()
        needs_replan = self.action_chunk is None or self.action_index >= self.replan_horizon
        if needs_replan and self._replan_future is None:
            self._submit_replan()
        if self.action_chunk is None:
            return
        action_horizon = min(value.shape[0] for value in self.action_chunk.values())
        if self.action_index >= action_horizon:
            return
        timeline_s = float(self.sim_time)
        applied = self.controller.apply(self.action_chunk, self.action_index, timestamp_s=timeline_s)
        _append_jsonl(
            self.trace_path,
            {
                "schema_version": "newton.groot_rtc.execute.v1",
                "event": "execute",
                "policy_step": self.policy_step,
                "action_index": self.action_index,
                "applied": applied,
            },
        )
        log_every = max(0, int(self.groot_args.control_log_every))
        if log_every > 0 and self.policy_step % log_every == 0:
            eef = applied.get("eef_applied")
            if isinstance(eef, dict):
                print(
                    "[groot-control] "
                    f"step={self.policy_step} source={applied['arm_source']} status={eef['status']} "
                    f"policy_xyz={np.round(eef['policy_target_xyz'], 5).tolist()} "
                    f"world_xyz={np.round(eef['world_target_xyz'], 5).tolist()} "
                    f"current_xyz={np.round(eef['world_current_xyz'], 5).tolist()} "
                    f"error_m={eef['world_target_error_m']:.5f}",
                    flush=True,
                )
            else:
                print(
                    f"[groot-control] step={self.policy_step} source={applied['arm_source']} "
                    f"eef_error={applied.get('eef_error')}",
                    flush=True,
                )
        self.action_index += 1
        self.policy_step += 1
        if int(self.groot_args.max_policy_steps) > 0 and self.policy_step >= int(self.groot_args.max_policy_steps):
            print(f"[groot] reached --max-policy-steps={self.groot_args.max_policy_steps}", flush=True)
            self.policy_enabled = False
            if hasattr(self.viewer, "close"):
                self.viewer.close()

    def step(self) -> None:
        if self.policy_enabled and self.sim_time + 1.0e-9 >= self.next_policy_time_s:
            self._policy_tick()
            self.next_policy_time_s += self.action_dt_s
        super().step()

    def render(self) -> None:
        super().render()
        if self.viewer_fifo_preview is not None:
            self.viewer_fifo_preview.capture(self.viewer, self._model_input_preview_images)

    def close_policy_resources(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._replan_future is not None:
            self._replan_future.cancel()
        if self._policy_executor is not None:
            self._policy_executor.shutdown(wait=True, cancel_futures=True)
            self._policy_executor = None
        if self.viewer_fifo_preview is not None:
            self.viewer_fifo_preview.close()
        self.smooth.close()

    def test_final(self) -> None:
        self.close_policy_resources()
        super().test_final()


def _resolve_policy_device(device: str) -> str:
    value = str(device)
    if value != "auto":
        return value
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _vec10(text: str) -> tuple[float, ...]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 10:
        raise argparse.ArgumentTypeError("expected 10 comma-separated numbers")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric comma-separated values") from exc


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--isaac-groot-root", type=Path, default=DEFAULT_ISAAC_GROOT_ROOT)
    parser.add_argument("--policy-checkpoint", type=Path, default=DEFAULT_POLICY_CHECKPOINT)
    parser.add_argument("--vlm-model", type=Path, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--smooth-dir", type=Path, default=DEFAULT_SMOOTH_DIR)
    parser.add_argument("--episode-index", type=int, default=8)
    parser.add_argument("--smooth-frame-offset", type=int, default=0)
    parser.add_argument("--smooth-loop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--image-source", choices=("sim", "smooth"), default="sim")
    parser.add_argument("--state-source", choices=("sim", "smooth"), default="sim")
    parser.add_argument(
        "--sim-ego-roi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply node0's 2x D455 ROI before the 320x180 frame-tap resize.",
    )
    parser.add_argument("--groot-initial-hand-q", type=_vec10, default=GROOT_INITIAL_HAND_COMMAND_Q)
    parser.add_argument("--instruction", default="", help="Empty uses the selected smooth episode task.")
    parser.add_argument("--start-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--async-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run GR00T inference on a dedicated CUDA worker so the viewer remains responsive.",
    )
    parser.add_argument("--dry-run-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--strict-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-fps", type=float, default=10.0)
    parser.add_argument("--replan-horizon", type=int, default=8)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtc-mode", choices=("compat", "off"), default="compat")
    parser.add_argument("--rtc-max-overlap-steps", type=int, default=24)
    parser.add_argument("--rtc-frozen-steps", type=int, default=4)
    parser.add_argument("--rtc-ramp-rate", type=float, default=3.0)
    parser.add_argument("--arm-control-mode", choices=("eef_ik", "joint_target"), default="eef_ik")
    parser.add_argument(
        "--eef-transform-mode",
        choices=("node0_fixed", "initial_calibration"),
        default="node0_fixed",
        help="Use node0's fixed A*T*B transform or the older observation-based alignment.",
    )
    parser.add_argument(
        "--eef-frame-update",
        choices=("once", "replan"),
        default="once",
        help="Frame update cadence used only with --eef-transform-mode initial_calibration.",
    )
    parser.add_argument("--eef-body-suffix", default="/right_revo2_flange")
    parser.add_argument("--arm-joint-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-arm-joint-step", type=float, default=0.045)
    parser.add_argument("--max-hand-joint-step", type=float, default=0.08)
    parser.add_argument("--ik-finite-difference-rad", type=float, default=1.0e-4)
    parser.add_argument("--ik-max-task-step-m", type=float, default=0.03)
    parser.add_argument("--ik-max-rotation-step-deg", type=float, default=5.0)
    parser.add_argument("--ik-position-weight", type=float, default=3.0)
    parser.add_argument("--ik-orientation-weight", type=float, default=1.0)
    parser.add_argument("--ik-damping-lambda", type=float, default=0.02)
    parser.add_argument("--control-log-every", type=int, default=8)
    parser.add_argument("--max-policy-steps", type=int, default=0)
    parser.add_argument("--trace-jsonl", type=Path, default=DEFAULT_TRACE_JSONL)
    parser.add_argument("--no-policy-trace", action="store_true")
    parser.add_argument("--dump-first-observation-dir", type=Path, default=None)
    parser.add_argument(
        "--viewer-fifo-preview",
        type=Path,
        default=None,
        help="Write throttled RGB viewer frames to a host display FIFO.",
    )
    parser.add_argument("--viewer-fifo-preview-width", type=int, default=1600)
    parser.add_argument("--viewer-fifo-preview-height", type=int, default=720)
    parser.add_argument("--viewer-fifo-preview-fps", type=float, default=15.0)
    parser.add_argument("--viewer-fifo-preview-input-width", type=int, default=320)


def create_parser() -> argparse.ArgumentParser:
    parser = scene_runtime.Example.create_parser()
    parser.description = __doc__
    parser.set_defaults(
        capture_graph=True,
        quest_teleop=False,
        d455_preview=True,
        d405_preview=True,
        d455_opencv_window=False,
        d405_opencv_window=False,
        camera_preview_fps=15.0,
        d455_render_width=NODE0_EGO_SOURCE_WIDTH,
        d455_render_height=NODE0_EGO_SOURCE_HEIGHT,
        d405_width=640,
        d405_height=480,
        initial_right_arm_q=GROOT_INITIAL_RIGHT_ARM_Q,
        d405_fov=GROOT_D405_FOV_DEG,
        d405_connector_rel_euler=GROOT_D405_CONNECTOR_REL_EULER_DEG,
    )
    _add_policy_args(parser)
    return parser


def _validate_only(args: argparse.Namespace) -> None:
    modalities = _validate_asset_layout(args)
    smooth = SmoothEpisodeSource(args.smooth_dir, args.episode_index, loop=args.smooth_loop)
    try:
        frame = int(args.smooth_frame_offset)
        video = smooth.video_observation(frame, modalities.video)
        state = smooth.state_observation(frame, modalities.state)
    finally:
        smooth.close()
    print(
        "[validate] assets ready "
        f"checkpoint={args.policy_checkpoint.resolve()} vlm={args.vlm_model.resolve()} "
        f"smooth={args.smooth_dir.resolve()} episode={args.episode_index}",
        flush=True,
    )
    print(f"[validate] video_shapes={ {key: list(value.shape) for key, value in video.items()} }", flush=True)
    print(f"[validate] state_shapes={ {key: list(value.shape) for key, value in state.items()} }", flush=True)
    print(
        f"[validate] action_keys={modalities.action.keys} horizon={len(modalities.action.delta_indices)} "
        "image_preprocessing=raw_rgb_no_manual_resize_or_letterbox",
        flush=True,
    )


def main() -> None:
    parser = create_parser()
    preliminary_args = parser.parse_args()
    if preliminary_args.validate_only:
        _validate_only(preliminary_args)
        return
    viewer, args = newton.examples.init(parser)
    newton.examples.run(GrootRtcExample(viewer, args), args)


if __name__ == "__main__":
    main()
