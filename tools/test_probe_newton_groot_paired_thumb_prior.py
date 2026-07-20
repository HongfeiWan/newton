#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the paired frozen-DP thumb-prior probe."""

from __future__ import annotations

import unittest

from tools.probe_newton_groot_paired_thumb_prior import (
    _DEFAULT_THUMB_OFFSETS,
    _build_paired_actions,
    _evaluate_gates,
    _expected_row_counts,
    _paired_deltas,
    _validate_args,
    _variant_totals,
    create_parser,
)


class TestPairedThumbPriorProbe(unittest.TestCase):
    def test_defaults_form_sixteen_zero_prior_pairs(self) -> None:
        args = create_parser().parse_args(["dp.pt"])
        _validate_args(args)
        self.assertEqual(args.num_pairs, 16)
        self.assertEqual(args.max_steps, 300)
        self.assertEqual(tuple(args.thumb_offsets_normalized), _DEFAULT_THUMB_OFFSETS)

    def test_offsets_must_be_strictly_positive_and_finite(self) -> None:
        args = create_parser().parse_args(["dp.pt", "--thumb-offsets-normalized", "0", "0.3", "0.4"])
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_args(args)

    def test_pair_base_is_exactly_shared_and_only_prior_thumb_changes(self) -> None:
        import torch

        base = torch.ones((2, 19))
        minimum = torch.zeros(19)
        maximum = torch.full((19,), 2.0)
        offsets = torch.tensor((0.1, 0.4, 0.8))
        action, paired_base, clamped = _build_paired_actions(base, offsets, minimum, maximum)
        torch.testing.assert_close(paired_base[0], paired_base[1])
        torch.testing.assert_close(paired_base[2], paired_base[3])
        torch.testing.assert_close(action[0], base[0])
        torch.testing.assert_close(action[2], base[1])
        changed = action[1] - action[0]
        torch.testing.assert_close(changed[[9, 10, 18]], offsets)
        torch.testing.assert_close(changed[:9], torch.zeros(9))
        torch.testing.assert_close(changed[11:18], torch.zeros(7))
        self.assertFalse(bool(clamped.any()))

    def test_prior_thumb_offsets_clamp_without_altering_zero_lane(self) -> None:
        import torch

        base = torch.full((1, 19), 1.8)
        minimum = torch.zeros(19)
        maximum = torch.full((19,), 2.0)
        action, _, clamped = _build_paired_actions(
            base,
            torch.tensor((0.3, 0.4, 0.5)),
            minimum,
            maximum,
        )
        torch.testing.assert_close(action[0], base[0])
        torch.testing.assert_close(action[1, (9, 10, 18)], torch.full((3,), 2.0))
        torch.testing.assert_close(clamped, torch.ones((1, 3), dtype=torch.bool))

    def test_row_schedule_executes_cached_rows_zero_through_seven(self) -> None:
        self.assertEqual(_expected_row_counts(8, 2), [2] * 8)
        self.assertEqual(_expected_row_counts(10, 2), [4, 4, 2, 2, 2, 2, 2, 2])
        counts = _expected_row_counts(300, 16)
        self.assertEqual(counts[:4], [16 * 38] * 4)
        self.assertEqual(counts[4:], [16 * 37] * 4)
        self.assertEqual(sum(counts), 300 * 16)

    @staticmethod
    def _lane(any_steps: int, physical_lift: float, *, thumb: bool) -> dict:
        return {
            "any_contact_steps": any_steps,
            "live_opposed_steps": 0,
            "confirmed_grasp_steps": 0,
            "carrying_steps": 0,
            "physical_max_lift_height_m": physical_lift,
            "gated_max_lift_height_m": 0.0,
            "ever": {
                "any_contact": any_steps > 0,
                "thumb_contact": thumb,
                "live_opposed": False,
                "confirmed_grasp": False,
                "carrying": False,
            },
        }

    def test_variant_totals_and_paired_deltas_preserve_pairing(self) -> None:
        results = [
            {"zero": self._lane(1, 0.01, thumb=False), "positive_prior": self._lane(3, 0.03, thumb=True)},
            {"zero": self._lane(2, 0.02, thumb=True), "positive_prior": self._lane(1, 0.02, thumb=False)},
        ]
        totals = _variant_totals(results, "positive_prior")
        self.assertEqual(totals["lanes_with_thumb_contact"], 1)
        self.assertEqual(totals["mean_any_contact_steps"], 2.0)
        delta = _paired_deltas(results)["any_contact_steps"]
        self.assertEqual(delta["mean_prior_minus_zero"], 0.5)
        self.assertEqual((delta["prior_wins"], delta["ties"], delta["zero_wins"]), (1, 0, 1))

    def test_clean_gates_require_exact_samples_rows_and_buffers(self) -> None:
        rows = _expected_row_counts(300, 16)
        clean = _evaluate_gates(
            num_pairs=16,
            num_envs=32,
            control_steps=300,
            expected_steps=300,
            finite_action_count=9600,
            finite_state_count=9600,
            expected_lane_samples=9600,
            finite_results=True,
            initial_state_max_abs_lane_delta=0.0,
            shared_pair_base_max_abs_delta=0.0,
            row_counts=rows,
            expected_row_counts=rows,
            triangle_buffer_available=True,
            rigid_overflow_frames=0,
            rigid_overflow_excess=0,
            triangle_overflow_frames=0,
            triangle_overflow_excess=0,
        )
        self.assertTrue(clean["passed"])
        dirty = _evaluate_gates(
            num_pairs=16,
            num_envs=32,
            control_steps=300,
            expected_steps=300,
            finite_action_count=9600,
            finite_state_count=9600,
            expected_lane_samples=9600,
            finite_results=True,
            initial_state_max_abs_lane_delta=0.0,
            shared_pair_base_max_abs_delta=1.0e-6,
            row_counts=rows,
            expected_row_counts=rows,
            triangle_buffer_available=True,
            rigid_overflow_frames=0,
            rigid_overflow_excess=0,
            triangle_overflow_frames=0,
            triangle_overflow_excess=0,
        )
        self.assertFalse(dirty["passed"])
        self.assertFalse(dirty["shared_base_within_pair"]["passed"])


if __name__ == "__main__":
    unittest.main()
