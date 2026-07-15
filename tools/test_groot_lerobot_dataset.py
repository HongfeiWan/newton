# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for GR00T LeRobot behavior-cloning dataset splits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from teleop_stack.datasets import create_groot_lerobot_bc_split


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


if __name__ == "__main__":
    unittest.main()
