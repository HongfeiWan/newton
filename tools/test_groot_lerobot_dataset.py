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
import pyarrow as pa
import pyarrow.parquet as parquet

from teleop_stack.datasets import GrootLeRobotWindowDataset, create_groot_lerobot_bc_split
from teleop_stack.datasets.groot_lerobot import ACTION_KEY, EGO_KEY, STATE_KEY, WRIST_KEY


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
            "rot6d_raw_truth_migration": {
                "schema": "teleop_stack.rot6d_physical_truth_migration.v1",
            },
        },
    }


class TestGrootLeRobotBCSplit(unittest.TestCase):
    def _write_metadata(self, root: Path, episodes: list[dict[str, object]]) -> None:
        meta = root / "meta"
        meta.mkdir(parents=True)
        total_frames = sum(int(episode["length"]) for episode in episodes)
        state_names = [f"state.{index}" for index in range(26)]
        action_names = [f"action.{index}" for index in range(19)]
        state_names[10:16] = [f"arm_eef_rot6d.{name}" for name in ("r00", "r01", "r02", "r10", "r11", "r12")]
        action_names[3:9] = [f"arm_eef_rot6d_target.{name}" for name in ("r00", "r01", "r02", "r10", "r11", "r12")]
        (meta / "info.json").write_text(
            json.dumps(
                {
                    "total_episodes": len(episodes),
                    "total_frames": total_frames,
                    "features": {
                        "observation.state": {"dtype": "float32", "shape": [26], "names": state_names},
                        "action": {"dtype": "float32", "shape": [19], "names": action_names},
                    },
                    "teleop_stack": {
                        "rot6d_convention": "row_major_first_two_rows_[r00,r01,r02,r10,r11,r12]",
                        "arm_action_semantics": "absolute_flange_pose_xyz_rot6d_target_in_state_frame",
                        "dp_action_semantics": "absolute_flange_pose_and_hand_from_observation_state_same_frame",
                        "dp_action_source_slices": {"eef": [7, 16], "hand": [16, 26]},
                        "dp_action_provenance": {"mode": "state_copy"},
                        "rot6d_raw_truth_migration": {
                            "schema": "teleop_stack.rot6d_physical_truth_migration.v1",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        contents = "".join(f"{json.dumps(episode)}\n" for episode in episodes)
        (meta / "episodes.jsonl").write_text(contents, encoding="utf-8")
        data = root / "data" / "chunk-000"
        data.mkdir(parents=True)
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            state = np.full((length, 26), float(episode_index), dtype=np.float32)
            action = np.full((length, 19), float(episode_index), dtype=np.float32)
            table = pa.table(
                {
                    "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 26)),
                    "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 19)),
                }
            )
            parquet.write_table(table, data / f"episode_{episode_index:06d}.parquet")

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

    def test_numeric_fingerprint_changes_when_parquet_values_change(self) -> None:
        episodes = [_episode(0, "raw-a", 0, 2), _episode(1, "raw-b", 0, 2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_metadata(root, episodes)
            initial = create_groot_lerobot_bc_split(root, validation_fraction=0.5, split_seed=0)

            path = root / "data" / "chunk-000" / "episode_000000.parquet"
            state = np.zeros((3, 26), dtype=np.float32)
            state[0, 10] = 0.25
            action = np.zeros((3, 19), dtype=np.float32)
            action[0, 3] = 0.25
            parquet.write_table(
                pa.table(
                    {
                        "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 26)),
                        "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 19)),
                    }
                ),
                path,
            )
            changed = create_groot_lerobot_bc_split(root, validation_fraction=0.5, split_seed=0)

        self.assertEqual(initial.episodes_sha256, changed.episodes_sha256)
        self.assertNotEqual(initial.numeric_data_sha256, changed.numeric_data_sha256)

    def test_rejects_action_that_is_not_same_frame_state_target(self) -> None:
        episodes = [_episode(0, "raw-a", 0, 2), _episode(1, "raw-b", 0, 2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_metadata(root, episodes)
            path = root / "data" / "chunk-000" / "episode_000000.parquet"
            table = parquet.read_table(path)
            action = np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
            action[1, 3] = 0.01
            parquet.write_table(
                pa.table(
                    {
                        "observation.state": table["observation.state"],
                        "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 19)),
                    }
                ),
                path,
            )

            with self.assertRaisesRegex(ValueError, "same-frame state EEF/hand target"):
                create_groot_lerobot_bc_split(root, validation_fraction=0.5, split_seed=0)

    def test_rejects_non_finite_numeric_value(self) -> None:
        episodes = [_episode(0, "raw-a", 0, 2), _episode(1, "raw-b", 0, 2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_metadata(root, episodes)
            path = root / "data" / "chunk-000" / "episode_000001.parquet"
            table = parquet.read_table(path)
            state = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
            state[1, 0] = np.nan
            parquet.write_table(
                pa.table(
                    {
                        "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 26)),
                        "action": table["action"],
                    }
                ),
                path,
            )

            with self.assertRaisesRegex(ValueError, "non-finite"):
                create_groot_lerobot_bc_split(root, validation_fraction=0.5, split_seed=0)

    def test_nonstatic_window_keeps_same_frame_then_next_frame_actions(self) -> None:
        frame_count = 3
        state = np.zeros((frame_count, 26), dtype=np.float32)
        for frame_index, angle in enumerate((0.15, 0.35, 0.65)):
            cosine = np.float32(np.cos(angle))
            sine = np.float32(np.sin(angle))
            state[frame_index, 7:10] = (0.25 + 0.01 * frame_index, -0.2, 0.5 + 0.02 * frame_index)
            state[frame_index, 10:16] = (cosine, -sine, 0.0, sine, cosine, 0.0)
            state[frame_index, 16:26] = np.linspace(
                0.01 * frame_index,
                0.2 + 0.01 * frame_index,
                10,
                dtype=np.float32,
            )
        action = np.concatenate((state[:, 7:16], state[:, 16:26]), axis=1)

        dataset = GrootLeRobotWindowDataset.__new__(GrootLeRobotWindowDataset)
        dataset.obs_horizon = 1
        dataset.pred_horizon = frame_count
        dataset.episodes = [{"episode_index": 0, "length": frame_count}]
        dataset._samples = [(0, 0)]
        dataset._numeric = [{STATE_KEY: state, ACTION_KEY: action}]
        dataset._read_rgb_frames = lambda _episode, _key, indices: np.zeros((len(indices), 2, 2, 3), dtype=np.uint8)

        sample = dataset[0]
        sampled_action = sample[ACTION_KEY].numpy()
        np.testing.assert_allclose(sampled_action[0], action[0], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(sampled_action[1], action[1], atol=0.0, rtol=0.0)
        self.assertFalse(np.array_equal(sampled_action[0], sampled_action[1]))
        np.testing.assert_allclose(sample[STATE_KEY][0, 7:16].numpy(), sampled_action[0, :9])
        self.assertEqual(tuple(sample[EGO_KEY].shape), (1, 2, 2, 3))
        self.assertEqual(tuple(sample[WRIST_KEY].shape), (1, 2, 2, 3))


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
            self.assertTrue(selected.flags["WRITE" + "ABLE"])
            self.assertEqual(len(dataset._frame_arrays), 1)
            dataset._read_rgb_frames(1, WRIST_KEY, np.asarray([0]))
            self.assertTrue(first_mapping._mmap.closed)
            self.assertEqual(tuple(dataset._frame_arrays), ((1, WRIST_KEY),))
            dataset.close()
            self.assertFalse(dataset._frame_arrays)
            np.testing.assert_array_equal(selected, frames[[2, 0, 2]])


if __name__ == "__main__":
    unittest.main()
