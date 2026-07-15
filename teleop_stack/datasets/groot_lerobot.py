# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Windowed reader for the existing Nero + L10 LeRobot v2 dataset."""

from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from teleop_stack.camera_preprocessing import preprocess_ego_rgb

STATE_KEY = "observation.state"
EGO_KEY = "observation.images.ego_view"
WRIST_KEY = "observation.images.wrist_view"
ACTION_KEY = "action"
STATE_SIZE = 26
ACTION_SIZE = 19


@dataclass(frozen=True)
class GrootWindowDatasetStats:
    """Min/max statistics used to normalize DP state and actions on the GPU."""

    state_min: np.ndarray
    state_max: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray


@dataclass(frozen=True)
class GrootLeRobotBCSplit:
    """Reproducible successful-demonstration split for behavior cloning."""

    episodes_sha256: str
    split_seed: int
    validation_fraction: float
    source_episode_count: int
    source_frame_count: int
    train_episode_indices: tuple[int, ...]
    validation_episode_indices: tuple[int, ...]
    excluded_unsuccessful_episode_indices: tuple[int, ...]
    excluded_duplicate_episode_indices: tuple[int, ...]
    train_raw_episode_ids: tuple[str, ...]
    validation_raw_episode_ids: tuple[str, ...]


def _load_episode_metadata(root: Path, expected_count: int) -> tuple[list[dict[str, Any]], str]:
    episodes_path = root / "meta" / "episodes.jsonl"
    contents = episodes_path.read_bytes()
    rows = [json.loads(line) for line in contents.decode("utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["episode_index"]))
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} episodes, found {len(rows)}")
    for expected_index, row in enumerate(rows):
        if int(row["episode_index"]) != expected_index:
            raise ValueError("Dataset episode indices must be contiguous and zero based")
    return rows, hashlib.sha256(contents).hexdigest()


def create_groot_lerobot_bc_split(
    root: str | Path,
    *,
    validation_fraction: float = 0.1,
    split_seed: int = 0,
) -> GrootLeRobotBCSplit:
    """Select successful unique clips and split them by raw episode.

    Exact duplicate clips are identified by raw episode ID and inclusive source
    frame range. Distinct clips from one raw episode are retained but assigned
    to the same split to prevent temporal leakage.

    Args:
        root: LeRobot dataset root.
        validation_fraction: Fraction of raw episode groups reserved for validation.
        split_seed: Seed used to shuffle raw episode groups deterministically.

    Returns:
        Successful, deduplicated train and validation episode indices.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    dataset_root = Path(root).expanduser().resolve()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    source_episode_count = int(info["total_episodes"])
    episodes, episodes_sha256 = _load_episode_metadata(dataset_root, source_episode_count)

    selected: list[dict[str, Any]] = []
    excluded_unsuccessful: list[int] = []
    excluded_duplicates: list[int] = []
    seen_clips: dict[tuple[str, int, int], int] = {}
    for row in episodes:
        episode_index = int(row["episode_index"])
        metadata = row.get("teleop_stack_metadata", {})
        if metadata.get("success") is not True or metadata.get("outcome") != "success":
            excluded_unsuccessful.append(episode_index)
            continue

        raw_episode_id = metadata.get("raw_episode_id")
        start_frame = metadata.get("source_start_frame")
        end_frame = metadata.get("source_end_frame")
        if not isinstance(raw_episode_id, str) or not raw_episode_id:
            raise ValueError(f"Episode {episode_index} does not have a valid raw_episode_id")
        if start_frame is None or end_frame is None:
            raise ValueError(f"Episode {episode_index} does not have a source frame range")
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        length = int(row["length"])
        if start_frame < 0 or end_frame < start_frame or length != end_frame - start_frame + 1:
            raise ValueError(
                f"Episode {episode_index} has an invalid source frame range: "
                f"start={start_frame} end={end_frame} length={length}"
            )

        clip_key = (raw_episode_id, start_frame, end_frame)
        if clip_key in seen_clips:
            excluded_duplicates.append(episode_index)
            continue
        seen_clips[clip_key] = episode_index
        selected.append(row)

    grouped_indices: defaultdict[str, list[int]] = defaultdict(list)
    for row in selected:
        metadata = row["teleop_stack_metadata"]
        grouped_indices[str(metadata["raw_episode_id"])].append(int(row["episode_index"]))
    raw_episode_ids = sorted(grouped_indices)
    if len(raw_episode_ids) < 2:
        raise ValueError("At least two successful raw episode groups are required for train/validation splitting")

    shuffled_ids = raw_episode_ids.copy()
    random.Random(int(split_seed)).shuffle(shuffled_ids)
    validation_group_count = round(len(shuffled_ids) * float(validation_fraction))
    validation_group_count = max(1, min(len(shuffled_ids) - 1, validation_group_count))
    validation_ids = set(shuffled_ids[:validation_group_count])
    train_ids = set(shuffled_ids[validation_group_count:])
    train_indices = sorted(index for raw_id in train_ids for index in grouped_indices[raw_id])
    validation_indices = sorted(index for raw_id in validation_ids for index in grouped_indices[raw_id])

    return GrootLeRobotBCSplit(
        episodes_sha256=episodes_sha256,
        split_seed=int(split_seed),
        validation_fraction=float(validation_fraction),
        source_episode_count=source_episode_count,
        source_frame_count=int(info["total_frames"]),
        train_episode_indices=tuple(train_indices),
        validation_episode_indices=tuple(validation_indices),
        excluded_unsuccessful_episode_indices=tuple(excluded_unsuccessful),
        excluded_duplicate_episode_indices=tuple(excluded_duplicates),
        train_raw_episode_ids=tuple(sorted(train_ids)),
        validation_raw_episode_ids=tuple(sorted(validation_ids)),
    )


class GrootLeRobotWindowDataset:
    """Read observation histories and future absolute-action windows.

    Numeric Parquet columns are loaded once because the current dataset is
    small. Existing ``.mp4.frames.npy`` RGB caches are read through bounded
    memory maps when available; otherwise H.264 frames are decoded in
    DataLoader workers. Returned pinned CPU tensors can then be copied to CUDA
    with ``non_blocking=True``. The head camera uses the exact ROI/resize path
    used by current GR00T simulator inference.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        preprocess_ego: bool = True,
        episode_indices: Sequence[int] | None = None,
        stats: GrootWindowDatasetStats | None = None,
        video_cache_size: int = 8,
        video_decode_threads: int = 1,
        require_frame_cache: bool = False,
    ) -> None:
        if obs_horizon < 1 or pred_horizon < 1:
            raise ValueError("obs_horizon and pred_horizon must be positive")
        if video_cache_size < 1 or video_decode_threads < 1:
            raise ValueError("video_cache_size and video_decode_threads must be positive")
        self.root = Path(root).expanduser().resolve()
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.preprocess_ego = bool(preprocess_ego)
        self.video_cache_size = int(video_cache_size)
        self.video_decode_threads = int(video_decode_threads)
        self.require_frame_cache = bool(require_frame_cache)
        self.info = self._load_json(self.root / "meta" / "info.json")
        self._validate_metadata()
        self.all_episodes, self.episodes_sha256 = _load_episode_metadata(self.root, int(self.info["total_episodes"]))
        if episode_indices is None:
            selected_indices = tuple(range(len(self.all_episodes)))
        else:
            selected_indices = tuple(int(index) for index in episode_indices)
            if not selected_indices:
                raise ValueError("episode_indices must not be empty")
            if len(set(selected_indices)) != len(selected_indices):
                raise ValueError("episode_indices must be unique")
            if min(selected_indices) < 0 or max(selected_indices) >= len(self.all_episodes):
                raise IndexError("episode_indices contains an out-of-range source episode")
        self.episode_indices = selected_indices
        self.episodes = [self.all_episodes[index] for index in selected_indices]
        self._numeric = self._load_numeric_episodes()
        self._samples = [
            (episode_index, frame_index)
            for episode_index, episode in enumerate(self.episodes)
            for frame_index in range(int(episode["length"]))
        ]
        self.stats = stats if stats is not None else self._compute_stats()
        self._validate_stats()
        self._captures: OrderedDict[tuple[int, str], Any] = OrderedDict()
        self._frame_arrays: OrderedDict[tuple[int, str], np.ndarray] = OrderedDict()
        self._video_resource_order: OrderedDict[tuple[int, str], str] = OrderedDict()
        self.frame_cache_file_count = self._validate_frame_cache_files()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _validate_metadata(self) -> None:
        features = self.info.get("features", {})
        expected = {
            STATE_KEY: ("float32", [STATE_SIZE]),
            ACTION_KEY: ("float32", [ACTION_SIZE]),
            EGO_KEY: ("video", [800, 1280, 3]),
            WRIST_KEY: ("video", [480, 640, 3]),
        }
        for key, (dtype, shape) in expected.items():
            feature = features.get(key)
            if feature is None:
                raise KeyError(f"Dataset metadata does not contain {key!r}")
            if feature.get("dtype") != dtype or feature.get("shape") != shape:
                raise ValueError(
                    f"Unexpected metadata for {key}: dtype={feature.get('dtype')} shape={feature.get('shape')}"
                )
        if int(self.info.get("fps", -1)) != 10:
            raise ValueError(f"Expected a 10 Hz dataset, got fps={self.info.get('fps')}")

    def _parquet_path(self, episode_index: int) -> Path:
        return self.root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"

    def _video_path(self, episode_index: int, key: str) -> Path:
        return self.root / "videos" / f"chunk-{episode_index // 1000:03d}" / key / f"episode_{episode_index:06d}.mp4"

    def _frame_cache_path(self, episode_index: int, key: str) -> Path:
        return Path(f"{self._video_path(episode_index, key)}.frames.npy")

    def _validate_frame_cache_files(self) -> int:
        count = 0
        missing: list[Path] = []
        for metadata in self.episodes:
            episode_index = int(metadata["episode_index"])
            for key in (EGO_KEY, WRIST_KEY):
                path = self._frame_cache_path(episode_index, key)
                if path.is_file():
                    count += 1
                    if self.require_frame_cache:
                        frames = np.load(path, mmap_mode="r", allow_pickle=False)
                        try:
                            self._validate_frame_array(frames, episode_index, key, path)
                        finally:
                            self._close_frame_array(frames)
                elif self.require_frame_cache:
                    missing.append(path)
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(f"Missing {len(missing)} required frame caches; first paths: {preview}")
        return count

    def _load_numeric_episodes(self) -> list[dict[str, np.ndarray]]:
        import pyarrow.parquet as parquet  # noqa: PLC0415

        output = []
        for metadata in self.episodes:
            episode_index = int(metadata["episode_index"])
            table = parquet.read_table(self._parquet_path(episode_index), columns=[STATE_KEY, ACTION_KEY])
            state = np.asarray(table[STATE_KEY].combine_chunks().to_pylist(), dtype=np.float32)
            action = np.asarray(table[ACTION_KEY].combine_chunks().to_pylist(), dtype=np.float32)
            length = int(metadata["length"])
            if state.shape != (length, STATE_SIZE) or action.shape != (length, ACTION_SIZE):
                raise ValueError(
                    f"Episode {episode_index} numeric shapes do not match metadata: state={state.shape} action={action.shape}"
                )
            output.append({STATE_KEY: state, ACTION_KEY: action})
        return output

    def _compute_stats(self) -> GrootWindowDatasetStats:
        state = np.concatenate([episode[STATE_KEY] for episode in self._numeric], axis=0)
        action = np.concatenate([episode[ACTION_KEY] for episode in self._numeric], axis=0)
        return GrootWindowDatasetStats(
            state_min=state.min(axis=0),
            state_max=state.max(axis=0),
            action_min=action.min(axis=0),
            action_max=action.max(axis=0),
        )

    def _validate_stats(self) -> None:
        expected_shapes = {
            "state_min": (STATE_SIZE,),
            "state_max": (STATE_SIZE,),
            "action_min": (ACTION_SIZE,),
            "action_max": (ACTION_SIZE,),
        }
        for name, shape in expected_shapes.items():
            value = np.asarray(getattr(self.stats, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Dataset statistics {name} must be finite with shape {shape}, got {value.shape}")
        if np.any(np.asarray(self.stats.state_min) > np.asarray(self.stats.state_max)):
            raise ValueError("Dataset state_min must not exceed state_max")
        if np.any(np.asarray(self.stats.action_min) > np.asarray(self.stats.action_max)):
            raise ValueError("Dataset action_min must not exceed action_max")

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def _clamped_indices(start: int, count: int, length: int) -> np.ndarray:
        return np.clip(np.arange(start, start + count, dtype=np.int64), 0, length - 1)

    def _capture(self, episode_index: int, key: str) -> Any:
        cache_key = (episode_index, key)
        if cache_key in self._captures:
            self._captures.move_to_end(cache_key)
            self._touch_video_resource(cache_key, "capture")
            return self._captures[cache_key]
        import cv2  # noqa: PLC0415

        path = self._video_path(episode_index, key)
        if hasattr(cv2, "CAP_PROP_N_THREADS"):
            capture = cv2.VideoCapture(
                str(path),
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_N_THREADS, self.video_decode_threads],
            )
        else:
            capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self._touch_video_resource(cache_key, "capture")
        self._captures[cache_key] = capture
        return capture

    @staticmethod
    def _close_frame_array(array: np.ndarray) -> None:
        memory_map = getattr(array, "_mmap", None)
        if memory_map is not None and not memory_map.closed:
            memory_map.close()

    def _touch_video_resource(self, cache_key: tuple[int, str], backend: str) -> None:
        if cache_key in self._video_resource_order:
            self._video_resource_order.move_to_end(cache_key)
            return
        while len(self._video_resource_order) >= self.video_cache_size:
            evicted_key, evicted_backend = self._video_resource_order.popitem(last=False)
            if evicted_backend == "capture":
                self._captures.pop(evicted_key).release()
            else:
                self._close_frame_array(self._frame_arrays.pop(evicted_key))
        self._video_resource_order[cache_key] = backend

    def _validate_frame_array(self, frames: np.ndarray, episode_index: int, key: str, path: Path) -> None:
        length = int(self.all_episodes[episode_index]["length"])
        shape = tuple(int(value) for value in self.info["features"][key]["shape"])
        if frames.dtype != np.uint8 or frames.shape != (length, *shape) or not frames.flags.c_contiguous:
            raise ValueError(
                f"Frame cache {path} does not match metadata: dtype={frames.dtype} shape={frames.shape}, "
                f"expected contiguous uint8 {(length, *shape)}"
            )

    def _frame_array(self, episode_index: int, key: str) -> np.ndarray | None:
        cache_key = (episode_index, key)
        if cache_key in self._frame_arrays:
            self._frame_arrays.move_to_end(cache_key)
            self._touch_video_resource(cache_key, "frames")
            return self._frame_arrays[cache_key]
        path = self._frame_cache_path(episode_index, key)
        if not path.is_file():
            return None
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            self._validate_frame_array(frames, episode_index, key, path)
        except Exception:
            self._close_frame_array(frames)
            raise
        self._touch_video_resource(cache_key, "frames")
        self._frame_arrays[cache_key] = frames
        return frames

    def _read_rgb_frames(self, episode_index: int, key: str, indices: np.ndarray) -> np.ndarray:
        frame_array = self._frame_array(episode_index, key)
        if frame_array is not None:
            rgb_frames = np.array(frame_array[indices], dtype=np.uint8, order="C", copy=True)
            if key == EGO_KEY and self.preprocess_ego:
                rgb_frames = np.stack([preprocess_ego_rgb(frame) for frame in rgb_frames], axis=0)
            return np.ascontiguousarray(rgb_frames)

        import cv2  # noqa: PLC0415

        capture = self._capture(episode_index, key)
        unique = np.unique(indices)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(unique[0]))
        decoded: dict[int, np.ndarray] = {}
        for frame_index in range(int(unique[0]), int(unique[-1]) + 1):
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Failed to decode {key} episode={episode_index} frame={frame_index}")
            if frame_index in unique:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)
                if key == EGO_KEY and self.preprocess_ego:
                    rgb = preprocess_ego_rgb(rgb)
                decoded[frame_index] = np.ascontiguousarray(rgb)
        return np.stack([decoded[int(index)] for index in indices], axis=0)

    def __getitem__(self, sample_index: int) -> dict[str, Any]:
        import torch

        episode_slot, frame_index = self._samples[int(sample_index)]
        metadata = self.episodes[episode_slot]
        episode_index = int(metadata["episode_index"])
        length = int(metadata["length"])
        obs_indices = self._clamped_indices(frame_index - self.obs_horizon + 1, self.obs_horizon, length)
        action_indices = self._clamped_indices(frame_index, self.pred_horizon, length)
        action_is_pad = np.arange(frame_index, frame_index + self.pred_horizon) >= length
        numeric = self._numeric[episode_slot]
        return {
            STATE_KEY: torch.from_numpy(numeric[STATE_KEY][obs_indices].copy()),
            EGO_KEY: torch.from_numpy(self._read_rgb_frames(episode_index, EGO_KEY, obs_indices)),
            WRIST_KEY: torch.from_numpy(self._read_rgb_frames(episode_index, WRIST_KEY, obs_indices)),
            ACTION_KEY: torch.from_numpy(numeric[ACTION_KEY][action_indices].copy()),
            "action_is_pad": torch.from_numpy(action_is_pad),
        }

    def close(self) -> None:
        """Release video handles and frame-cache memory maps owned by this process."""
        for capture in getattr(self, "_captures", {}).values():
            capture.release()
        if hasattr(self, "_captures"):
            self._captures.clear()
        for frame_array in getattr(self, "_frame_arrays", {}).values():
            self._close_frame_array(frame_array)
        if hasattr(self, "_frame_arrays"):
            self._frame_arrays.clear()
        if hasattr(self, "_video_resource_order"):
            self._video_resource_order.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_captures"] = OrderedDict()
        state["_frame_arrays"] = OrderedDict()
        state["_video_resource_order"] = OrderedDict()
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
