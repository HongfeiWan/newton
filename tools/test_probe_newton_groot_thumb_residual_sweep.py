#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the frozen-DP thumb residual sweep."""

from __future__ import annotations

import unittest

from tools.probe_newton_groot_thumb_residual_sweep import (
    _apply_thumb_offsets,
    _build_variants,
    _evaluate_gates,
    _validate_args,
    create_parser,
)


class TestThumbResidualSweep(unittest.TestCase):
    def test_signed_finite_offset_grid_is_supported(self) -> None:
        args = create_parser().parse_args(
            [
                "dp.pt",
                "--pitch-offsets",
                "-0.2",
                "0.0",
                "0.2",
                "--yaw-offsets",
                "-0.8",
                "0.0",
                "0.8",
                "--roll-offsets",
                "-0.8",
                "0.0",
                "0.8",
            ]
        )
        _validate_args(args)
        self.assertEqual(len(_build_variants(args.pitch_offsets, args.yaw_offsets, args.roll_offsets)), 27)

    def test_builds_exact_pitch_yaw_roll_product(self) -> None:
        variants = _build_variants((0.0, 0.1, 0.2), (0.0, 0.4, 0.8), (0.0, 0.4, 0.8))
        self.assertEqual(len(variants), 27)
        self.assertEqual(variants[0], (0.0, 0.0, 0.0))
        self.assertEqual(variants[-1], (0.2, 0.8, 0.8))
        self.assertEqual(variants[13], (0.1, 0.4, 0.4))

    def test_offsets_only_three_thumb_coordinates_and_clamps(self) -> None:
        import torch

        base = torch.ones((2, 19))
        minimum = torch.zeros(19)
        maximum = torch.full((19,), 2.0)
        offsets = torch.tensor(((0.1, 0.4, 0.8), (0.2, 0.8, 1.2)))
        action, clamped = _apply_thumb_offsets(base, offsets, minimum, maximum)
        changed = action - base
        torch.testing.assert_close(changed[0, (9, 10, 18)], offsets[0])
        torch.testing.assert_close(changed[0, :9], torch.zeros(9))
        torch.testing.assert_close(changed[0, 11:18], torch.zeros(7))
        torch.testing.assert_close(action[1, (9, 10, 18)], torch.tensor((1.2, 1.8, 2.0)))
        torch.testing.assert_close(clamped[0], torch.tensor((False, False, False)))
        torch.testing.assert_close(clamped[1], torch.tensor((False, False, True)))

    def test_strict_gates_require_finite_and_clean_buffers(self) -> None:
        clean = _evaluate_gates(
            num_variants=27,
            control_steps=300,
            expected_steps=300,
            finite_action_count=8100,
            finite_state_count=8100,
            expected_action_count=8100,
            finite_results=True,
            initial_state_max_abs_lane_delta=0.0,
            triangle_buffer_available=True,
            rigid_overflow_frames=0,
            rigid_overflow_excess=0,
            triangle_overflow_frames=0,
            triangle_overflow_excess=0,
        )
        self.assertTrue(clean["passed"])
        overflow = _evaluate_gates(
            num_variants=27,
            control_steps=300,
            expected_steps=300,
            finite_action_count=8099,
            finite_state_count=8100,
            expected_action_count=8100,
            finite_results=True,
            initial_state_max_abs_lane_delta=0.0,
            triangle_buffer_available=True,
            rigid_overflow_frames=0,
            rigid_overflow_excess=0,
            triangle_overflow_frames=1,
            triangle_overflow_excess=2,
        )
        self.assertFalse(overflow["passed"])
        self.assertFalse(overflow["finite_actions"]["passed"])
        self.assertFalse(overflow["triangle_pair_buffer_clean"]["passed"])


if __name__ == "__main__":
    unittest.main()
