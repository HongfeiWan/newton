#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the frozen-DP pregrasp geometry probe."""

from __future__ import annotations

import unittest

from tools.probe_newton_groot_pregrasp_geometry import (
    _PREGRASP_DISTANCE_SCALE_M,
    _diagnose_zero_score,
    _evaluate_gates,
    _summarize_geometry_samples,
    _validate_args,
    create_parser,
)


class TestPregraspGeometryProbe(unittest.TestCase):
    def test_parser_defaults_to_requested_gpu_rollout(self) -> None:
        args = create_parser().parse_args(["dp.pt"])
        _validate_args(args)
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.num_envs, 32)
        self.assertEqual(args.max_steps, 300)
        self.assertAlmostEqual(_PREGRASP_DISTANCE_SCALE_M, 0.08)

    def test_gap_and_score_distributions_include_bilateral_pair(self) -> None:
        import torch

        gaps = torch.tensor(
            (
                ((0.01, 0.04, 0.03, 0.02, 0.05), (0.02, 0.01, 0.04, 0.05, 0.03)),
                ((0.03, 0.05, 0.02, 0.04, 0.01), (0.04, 0.03, 0.05, 0.01, 0.02)),
            )
        )
        scores = torch.tensor(((0.0, 0.25), (0.5, 1.0)))
        summary = _summarize_geometry_samples(gaps, scores)
        self.assertEqual(summary["sample_count"], 4)
        self.assertAlmostEqual(summary["distance_scale_m"], 0.08)
        self.assertAlmostEqual(summary["per_finger_gap_m"]["thumb"]["mean"], 0.025)
        self.assertAlmostEqual(summary["per_finger_gap_m"]["thumb"]["min"], 0.01)
        self.assertAlmostEqual(summary["bilateral_gap_m"]["best_non_thumb"]["mean"], 0.0125)
        self.assertAlmostEqual(summary["bilateral_gap_m"]["worse_side"]["p50"], 0.025)
        self.assertAlmostEqual(summary["opposed_pregrasp_score"]["mean"], 0.4375)
        self.assertAlmostEqual(summary["opposed_pregrasp_score"]["p50"], 0.375)
        self.assertAlmostEqual(summary["opposed_pregrasp_score"]["p95"], 0.925)
        self.assertAlmostEqual(summary["opposed_pregrasp_score"]["max"], 1.0)
        self.assertAlmostEqual(summary["opposed_pregrasp_score"]["nonzero_fraction"], 0.75)

    def test_zero_attribution_separates_distance_from_pair_gate(self) -> None:
        import torch

        far_gaps = torch.full((4, 5), 1.0)
        zero_score = torch.zeros(4)
        far_summary = _summarize_geometry_samples(far_gaps, zero_score)
        self.assertEqual(_diagnose_zero_score(far_summary)["classification"], "distance_proximity_saturation_likely")
        close_gaps = torch.full((4, 5), 0.001)
        close_summary = _summarize_geometry_samples(close_gaps, zero_score)
        diagnosis = _diagnose_zero_score(close_summary)
        self.assertEqual(diagnosis["classification"], "downstream_pair_gate_likely")
        self.assertAlmostEqual(diagnosis["downstream_pair_gate_zero_fraction"], 1.0)

    def test_strict_gates_require_finite_samples_and_clean_buffers(self) -> None:
        clean = _evaluate_gates(
            num_envs=32,
            control_steps=300,
            expected_control_steps=300,
            finite_action_samples=9600,
            finite_state_samples=9600,
            finite_gap_samples=9600,
            finite_score_samples=9600,
            nonnegative_gap_samples=9600,
            bounded_score_samples=9600,
            expected_lane_samples=9600,
            cache_rows_in_range=True,
            triangle_buffer_available=True,
            rigid_overflow_frames=0,
            rigid_overflow_excess=0,
            triangle_overflow_frames=0,
            triangle_overflow_excess=0,
        )
        self.assertTrue(clean["passed"])
        dirty = _evaluate_gates(
            num_envs=32,
            control_steps=300,
            expected_control_steps=300,
            finite_action_samples=9599,
            finite_state_samples=9600,
            finite_gap_samples=9600,
            finite_score_samples=9600,
            nonnegative_gap_samples=9600,
            bounded_score_samples=9600,
            expected_lane_samples=9600,
            cache_rows_in_range=True,
            triangle_buffer_available=True,
            rigid_overflow_frames=0,
            rigid_overflow_excess=0,
            triangle_overflow_frames=1,
            triangle_overflow_excess=2,
        )
        self.assertFalse(dirty["passed"])
        self.assertFalse(dirty["sample_integrity"]["passed"])
        self.assertFalse(dirty["triangle_pair_buffer_clean"]["passed"])


if __name__ == "__main__":
    unittest.main()
