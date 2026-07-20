# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the frozen-DP rollout comparison helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from tools.compare_newton_groot_dp_rollouts import (
    _accumulate_collision_buffer,
    _action_row,
    _advance_first_terminal,
    _build_summary,
    _done_reset_mask,
    _evaluate_gates,
    _mask_inactive_action,
    _masked_sum,
    _needs_policy_prediction,
    _validate_modes,
    _write_summary,
)


def _result(
    mode: str,
    *,
    episodes: int = 4,
    xyz_m: float = 0.001,
    contacts: int = 1,
    grasps: int = 0,
    lifts: int = 0,
) -> dict[str, object]:
    return {
        "mode": mode,
        "episodes_completed": episodes,
        "mean_xyz_command_displacement_m": xyz_m,
        "contact_episode_count": contacts,
        "grasp_episode_count": grasps,
        "grasp_confirmed_episode_count": grasps,
        "lift_episode_count": lifts,
        "finite": True,
        "collision_buffers": {
            "rigid_contacts": {
                "available": True,
                "capacity": 16,
                "max_observed_count": 4,
                "overflow_step_count": 0,
                "overflow_frame_count": 0,
                "overflow_excess_count": 0,
            },
            "triangle_pairs": {
                "available": True,
                "capacity": 32,
                "max_observed_count": 8,
                "overflow_step_count": 0,
                "overflow_frame_count": 0,
                "overflow_excess_count": 0,
            },
        },
    }


class TestFrozenDPRolloutComparison(unittest.TestCase):
    def test_mode_schedule_replans_and_selects_expected_rows(self) -> None:
        self.assertEqual([_action_row("index0", step) for step in range(10)], [0] * 10)
        self.assertEqual([_action_row("index1", step) for step in range(10)], [1] * 10)
        self.assertEqual([_action_row("chunk8", step) for step in range(10)], [0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
        self.assertTrue(all(_needs_policy_prediction("index0", step) for step in range(10)))
        self.assertTrue(all(_needs_policy_prediction("index1", step) for step in range(10)))
        self.assertEqual(
            [_needs_policy_prediction("chunk8", step) for step in range(10)],
            [True, False, False, False, False, False, False, False, True, False],
        )

    def test_mode_horizon_validation_is_fail_fast(self) -> None:
        _validate_modes(["index0"], pred_horizon=1)
        _validate_modes(["index1"], pred_horizon=2)
        _validate_modes(["chunk8"], pred_horizon=8)
        with self.assertRaisesRegex(ValueError, "pred_horizon >= 2"):
            _validate_modes(["index1"], pred_horizon=1)
        with self.assertRaisesRegex(ValueError, "pred_horizon >= 8"):
            _validate_modes(["chunk8"], pred_horizon=7)

    def test_first_terminal_quota_counts_each_lane_once(self) -> None:
        active = np.ones(4, dtype=np.bool_)
        counts = np.zeros(4, dtype=np.int64)

        first, active = _advance_first_terminal(
            active,
            np.asarray((True, False, False, False)),
            np.asarray((False, False, False, False)),
        )
        counts += first
        first, active = _advance_first_terminal(
            active,
            np.asarray((True, False, True, False)),
            np.asarray((False, True, False, False)),
        )
        counts += first
        first, active = _advance_first_terminal(
            active,
            np.asarray((True, True, True, True)),
            np.asarray((False, False, False, True)),
        )
        counts += first

        np.testing.assert_array_equal(counts, np.ones(4, dtype=np.int64))
        self.assertFalse(active.any())

    def test_every_done_lane_is_reset_but_only_active_lane_is_counted(self) -> None:
        active = np.asarray((False, True, True), dtype=np.bool_)
        terminated = np.asarray((True, True, False), dtype=np.bool_)
        truncated = np.asarray((False, False, False), dtype=np.bool_)

        first_terminal, next_active = _advance_first_terminal(active, terminated, truncated)
        reset_mask = _done_reset_mask(terminated, truncated)

        np.testing.assert_array_equal(first_terminal, (False, True, False))
        np.testing.assert_array_equal(reset_mask, (True, True, False))
        np.testing.assert_array_equal(next_active, (False, False, True))

    @unittest.skipUnless(torch is not None, "requires PyTorch")
    def test_inactive_action_is_hold_and_masked_sum_ignores_nan(self) -> None:
        state = torch.arange(52, dtype=torch.float32).reshape(2, 26)
        action = torch.full((2, 19), 123.0)
        action[1].fill_(float("nan"))
        active = torch.tensor((True, False))

        masked = _mask_inactive_action(action, state, active)
        expected_hold = torch.cat((state[1, 7:16], state[1, 16:26]))

        torch.testing.assert_close(masked[0], action[0])
        torch.testing.assert_close(masked[1], expected_hold)
        value = torch.tensor((3.0, float("nan")))
        self.assertEqual(float(_masked_sum(value, active)), 3.0)

    @unittest.skipUnless(torch is not None, "requires PyTorch")
    def test_collision_buffer_accumulator_tracks_steps_excess_and_max(self) -> None:
        accumulators = {
            "test_overflow_step_count": torch.zeros((), dtype=torch.int64),
            "test_overflow_frame_count": torch.zeros((), dtype=torch.int64),
            "test_overflow_excess_count": torch.zeros((), dtype=torch.int64),
            "test_max_observed_count": torch.zeros((), dtype=torch.int64),
        }
        _accumulate_collision_buffer(
            accumulators,
            prefix="test",
            frame_max=torch.tensor(7),
            overflow_frame_count=torch.tensor(2),
            overflow_excess_count=torch.tensor(3),
        )
        _accumulate_collision_buffer(
            accumulators,
            prefix="test",
            frame_max=torch.tensor(4),
            overflow_frame_count=torch.tensor(0),
            overflow_excess_count=torch.tensor(0),
        )

        self.assertEqual(int(accumulators["test_overflow_step_count"]), 1)
        self.assertEqual(int(accumulators["test_overflow_frame_count"]), 2)
        self.assertEqual(int(accumulators["test_overflow_excess_count"]), 3)
        self.assertEqual(int(accumulators["test_max_observed_count"]), 7)

    def test_strict_gates_cover_motion_contact_and_chunk_fallback(self) -> None:
        passing = {
            "index0": _result("index0", xyz_m=0.0001, contacts=0),
            "index1": _result("index1", xyz_m=0.0006, contacts=1, grasps=1, lifts=1),
            "chunk8": _result("chunk8", xyz_m=0.002, contacts=2, grasps=1, lifts=1),
        }
        self.assertTrue(_evaluate_gates(passing, expected_episodes=4)["passed"])

        no_motion = {mode: dict(result) for mode, result in passing.items()}
        no_motion["index1"]["mean_xyz_command_displacement_m"] = 0.0005
        self.assertFalse(_evaluate_gates(no_motion, expected_episodes=4)["index1_motion"]["passed"])

        no_contact = {mode: dict(result) for mode, result in passing.items()}
        no_contact["index1"]["contact_episode_count"] = 0
        self.assertFalse(_evaluate_gates(no_contact, expected_episodes=4)["index1_contact"]["passed"])

        needs_chunk = {mode: dict(result) for mode, result in passing.items()}
        needs_chunk["index1"]["grasp_confirmed_episode_count"] = 0
        needs_chunk["index1"]["lift_episode_count"] = 0
        self.assertFalse(_evaluate_gates(needs_chunk, expected_episodes=4)["receding_chunk_not_required"]["passed"])

        overflow = deepcopy(passing)
        overflow["chunk8"]["collision_buffers"]["triangle_pairs"]["overflow_step_count"] = 1
        self.assertFalse(_evaluate_gates(overflow, expected_episodes=4)["collision_buffers_clean"]["passed"])

        unavailable = deepcopy(passing)
        unavailable["index1"]["collision_buffers"]["triangle_pairs"]["available"] = False
        self.assertFalse(_evaluate_gates(unavailable, expected_episodes=4)["collision_buffers_clean"]["passed"])

    def test_summary_round_trip_and_exact_episode_gate(self) -> None:
        results = {
            "index0": _result("index0", xyz_m=0.0001, contacts=0),
            "index1": _result("index1", xyz_m=0.0007, contacts=1),
            "chunk8": _result("chunk8", xyz_m=0.001, contacts=1),
        }
        results["chunk8"]["episodes_completed"] = 3
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "dp.pt"
            checkpoint.write_bytes(b"checkpoint")
            summary = _build_summary(
                checkpoint=checkpoint,
                checkpoint_sha256="abc123",
                config={"num_envs": 4},
                results=results,
            )
            path = _write_summary(summary, Path(directory) / "output")
            decoded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(decoded["schema_version"], 1)
        self.assertFalse(decoded["gates"]["exact_episode_count"]["passed"])
        self.assertFalse(decoded["passed"])


if __name__ == "__main__":
    unittest.main()
