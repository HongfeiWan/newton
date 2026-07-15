# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for GR00T LeRobot behavior-cloning dataset splits."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from teleop_stack.datasets import GrootLeRobotWindowDataset, create_groot_lerobot_bc_split
from teleop_stack.datasets.groot_lerobot import WRIST_KEY


def _episode(
    episode_index: int,
    raw_episode_id: str,
    start_frame: int,
    end_frame: int,
    *,
    success: bool = True,
) -> dict[str, object]:
    return {
        "episode_index": episode_index,
        "length": end_frame - start_frame + 1,
        "teleop_stack_metadata": {
            "success": success,
            "outcome": "success" if success else "safety_stop",
            "raw_episode_id": raw_episode_id,
            "source_start_frame": start_frame,
            "source_end_frame": end_frame,
        },
    }


class TestGrootLeRobotBCSplit(unittest.TestCase):
    def _write_metadata(self, root: Path, episodes: list[dict[str, object]]) -> None:
        meta = root / "meta"
        meta.mkdir(parents=True)
        total_frames = sum(int(episode["length"]) for episode in episodes)
        (meta / "info.json").write_text(
            json.dumps({"total_episodes": len(episodes), "total_frames": total_frames}),
            encoding="utf-8",
        )
        contents = "".join(f"{json.dumps(episode)}\n" for episode in episodes)
        (meta / "episodes.jsonl").write_text(contents, encoding="utf-8")

    def test_filters_deduplicates_and_groups_raw_episodes(self) -> None:
        episodes = [
            _episode(0, "raw-a", 0, 4),
            _episode(1, "raw-a", 0, 4),
            _episode(2, "raw-a", 5, 7),
            _episode(3, "raw-b", 0, 3),
            _episode(4, "raw-c", 0, 2),
            _episode(5, "raw-failed", 0, 1, success=False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_metadata(root, episodes)

            split = create_groot_lerobot_bc_split(root, validation_fraction=0.5, split_seed=17)
            repeated = create_groot_lerobot_bc_split(root, validation_fraction=0.5, split_seed=17)

        self.assertEqual(split, repeated)
        self.assertEqual(split.excluded_unsuccessful_episode_indices, (5,))
        self.assertEqual(split.excluded_duplicate_episode_indices, (1,))

        train_indices = set(split.train_episode_indices)
        validation_indices = set(split.validation_episode_indices)
        self.assertEqual(train_indices | validation_indices, {0, 2, 3, 4})
        self.assertFalse(train_indices & validation_indices)
        self.assertTrue({0, 2} <= train_indices or {0, 2} <= validation_indices)
        self.assertFalse(set(split.train_raw_episode_ids) & set(split.validation_raw_episode_ids))

    def test_rejects_invalid_source_frame_range(self) -> None:
        episodes = [
            _episode(0, "raw-a", 0, 2),
            _episode(1, "raw-b", 0, 2),
        ]
        episodes[0]["length"] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_metadata(root, episodes)

            with self.assertRaisesRegex(ValueError, "invalid source frame range"):
                create_groot_lerobot_bc_split(root, validation_fraction=0.5, split_seed=0)


class TestGrootLeRobotFrameCache(unittest.TestCase):
    def test_reads_rgb_frames_from_memory_mapped_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = GrootLeRobotWindowDataset.__new__(GrootLeRobotWindowDataset)
            dataset.root = Path(directory)
            dataset.preprocess_ego = False
            dataset.video_cache_size = 1
            dataset.video_decode_threads = 1
            dataset.info = {"features": {WRIST_KEY: {"shape": [2, 3, 3]}}}
            dataset.all_episodes = [{"length": 3}, {"length": 3}]
            dataset._captures = OrderedDict()
            dataset._frame_arrays = OrderedDict()
            dataset._video_resource_order = OrderedDict()
            cache_path = dataset._frame_cache_path(0, WRIST_KEY)
            cache_path.parent.mkdir(parents=True)
            frames = np.arange(3 * 2 * 3 * 3, dtype=np.uint8).reshape(3, 2, 3, 3)
            np.save(cache_path, frames)
            second_cache_path = dataset._frame_cache_path(1, WRIST_KEY)
            np.save(second_cache_path, frames + 1)

            selected = dataset._read_rgb_frames(0, WRIST_KEY, np.asarray([2, 0, 2]))
            first_mapping = dataset._frame_arrays[(0, WRIST_KEY)]

            np.testing.assert_array_equal(selected, frames[[2, 0, 2]])
            self.assertTrue(selected.flags.c_contiguous)
            self.assertTrue(selected.flags.writeable)
            self.assertEqual(len(dataset._frame_arrays), 1)
            dataset._read_rgb_frames(1, WRIST_KEY, np.asarray([0]))
            self.assertTrue(first_mapping._mmap.closed)
            self.assertEqual(tuple(dataset._frame_arrays), ((1, WRIST_KEY),))
            dataset.close()
            self.assertFalse(dataset._frame_arrays)
            np.testing.assert_array_equal(selected, frames[[2, 0, 2]])


if __name__ == "__main__":
    unittest.main()
