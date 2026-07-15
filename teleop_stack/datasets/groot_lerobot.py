# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Windowed reader for the existing Nero + L10 LeRobot v2 dataset."""

from __future__ import annotations

import json
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


class GrootLeRobotWindowDataset:
    """Read observation histories and future absolute-action windows.

    Numeric Parquet columns are loaded once because the current dataset is
    small. H.264 frames remain compressed and are decoded in DataLoader
    workers; returned pinned CPU tensors can then be copied to CUDA with
    ``non_blocking=True``. The head camera uses the exact ROI/resize path used
    by current GR00T simulator inference.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        preprocess_ego: bool = True,
    ) -> None:
        if obs_horizon < 1 or pred_horizon < 1:
            raise ValueError("obs_horizon and pred_horizon must be positive")
        self.root = Path(root).expanduser().resolve()
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.preprocess_ego = bool(preprocess_ego)
        self.info = self._load_json(self.root / "meta" / "info.json")
        self._validate_metadata()
        self.episodes = self._load_episodes()
        self._numeric = self._load_numeric_episodes()
        self._samples = [
            (episode_index, frame_index)
            for episode_index, episode in enumerate(self.episodes)
            for frame_index in range(int(episode["length"]))
        ]
        self.stats = self._compute_stats()
        self._captures: dict[tuple[int, str], Any] = {}

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

    def _load_episodes(self) -> list[dict[str, Any]]:
        rows = []
        with (self.root / "meta" / "episodes.jsonl").open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        rows.sort(key=lambda row: int(row["episode_index"]))
        if len(rows) != int(self.info["total_episodes"]):
            raise ValueError(f"Expected {self.info['total_episodes']} episodes, found {len(rows)}")
        for expected_index, row in enumerate(rows):
            if int(row["episode_index"]) != expected_index:
                raise ValueError("Dataset episode indices must be contiguous and zero based")
        return rows

    def _parquet_path(self, episode_index: int) -> Path:
        return self.root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"

    def _video_path(self, episode_index: int, key: str) -> Path:
        return self.root / "videos" / f"chunk-{episode_index // 1000:03d}" / key / f"episode_{episode_index:06d}.mp4"

    def _load_numeric_episodes(self) -> list[dict[str, np.ndarray]]:
        import pyarrow.parquet as parquet  # noqa: PLC0415

        output = []
        for episode_index, metadata in enumerate(self.episodes):
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

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def _clamped_indices(start: int, count: int, length: int) -> np.ndarray:
        return np.clip(np.arange(start, start + count, dtype=np.int64), 0, length - 1)

    def _capture(self, episode_index: int, key: str) -> Any:
        cache_key = (episode_index, key)
        if cache_key in self._captures:
            return self._captures[cache_key]
        import cv2  # noqa: PLC0415

        path = self._video_path(episode_index, key)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self._captures[cache_key] = capture
        return capture

    def _read_rgb_frames(self, episode_index: int, key: str, indices: np.ndarray) -> np.ndarray:
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

        episode_index, frame_index = self._samples[int(sample_index)]
        length = int(self.episodes[episode_index]["length"])
        obs_indices = self._clamped_indices(frame_index - self.obs_horizon + 1, self.obs_horizon, length)
        action_indices = self._clamped_indices(frame_index, self.pred_horizon, length)
        action_is_pad = np.arange(frame_index, frame_index + self.pred_horizon) >= length
        numeric = self._numeric[episode_index]
        return {
            STATE_KEY: torch.from_numpy(numeric[STATE_KEY][obs_indices].copy()),
            EGO_KEY: torch.from_numpy(self._read_rgb_frames(episode_index, EGO_KEY, obs_indices)),
            WRIST_KEY: torch.from_numpy(self._read_rgb_frames(episode_index, WRIST_KEY, obs_indices)),
            ACTION_KEY: torch.from_numpy(numeric[ACTION_KEY][action_indices].copy()),
            "action_is_pad": torch.from_numpy(action_is_pad),
        }

    def close(self) -> None:
        """Release OpenCV video handles owned by this process."""
        for capture in getattr(self, "_captures", {}).values():
            capture.release()
        if hasattr(self, "_captures"):
            self._captures.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_captures"] = {}
        return state

    def __del__(self) -> None:
        self.close()
