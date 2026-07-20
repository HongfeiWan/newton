#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the paired frozen-DP thumb-gap probe."""

from __future__ import annotations

import unittest

from tools.probe_newton_groot_paired_thumb_gap import (
    _DEFAULT_THUMB_RESIDUALS,
    _assign_pair_variants,
    _build_paired_actions,
    _diagnose_variants,
    _evaluate_gates,
    _expected_row_counts,
    _paired_metric_samples,
    _resolve_thumb_residuals,
    _summarize_variant_deltas,
    _validate_args,
    create_parser,
)


class TestPairedThumbGapProbe(unittest.TestCase):
    def test_defaults_use_sixteen_pairs_and_both_reference_signs(self) -> None:
        args = create_parser().parse_args(["dp.pt"])
        _validate_args(args)
        self.assertEqual(args.num_pairs, 16)
        residuals = _resolve_thumb_residuals(args)
        self.assertEqual(residuals, _DEFAULT_THUMB_RESIDUALS)
        for positive, negative in zip(residuals[0], residuals[1], strict=True):
            self.assertEqual(positive, -negative)
        indices, assigned = _assign_pair_variants(residuals, args.num_pairs)
        self.assertEqual(indices.count(0), 8)
        self.assertEqual(indices.count(1), 8)
        self.assertEqual(assigned[0], residuals[0])
        self.assertEqual(assigned[1], residuals[1])

    def test_repeated_cli_residuals_are_configurable_and_distinct(self) -> None:
        args = create_parser().parse_args(
            [
                "dp.pt",
                "--thumb-residual-normalized",
                "0.1",
                "0.2",
                "0.3",
                "--thumb-residual-normalized",
                "-0.1",
                "-0.2",
                "-0.3",
                "--thumb-residual-normalized",
                "0.2",
                "-0.2",
                "0.1",
            ]
        )
        _validate_args(args)
        self.assertEqual(len(_resolve_thumb_residuals(args)), 3)
        duplicate = create_parser().parse_args(
            [
                "dp.pt",
                "--thumb-residual-normalized",
                "0.1",
                "0.2",
                "0.3",
                "--thumb-residual-normalized",
                "0.1",
                "0.2",
                "0.3",
            ]
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            _validate_args(duplicate)

    def test_actions_share_base_and_apply_signed_thumb_residuals_only(self) -> None:
        import torch

        base = torch.ones((2, 19))
        minimum = torch.zeros(19)
        maximum = torch.full((19,), 2.0)
        residuals = torch.tensor(((0.1, 0.4, 0.8), (-0.2, -0.5, -0.9)))
        action, paired_base, clamped = _build_paired_actions(base, residuals, minimum, maximum)
        torch.testing.assert_close(paired_base[0], paired_base[1])
        torch.testing.assert_close(paired_base[2], paired_base[3])
        torch.testing.assert_close(action[0], base[0])
        torch.testing.assert_close(action[2], base[1])
        torch.testing.assert_close(action[1, (9, 10, 18)] - action[0, (9, 10, 18)], residuals[0])
        torch.testing.assert_close(action[3, (9, 10, 18)] - action[2, (9, 10, 18)], residuals[1])
        torch.testing.assert_close(action[1, :9], action[0, :9])
        torch.testing.assert_close(action[1, 11:18], action[0, 11:18])
        self.assertFalse(bool(clamped.any()))

    def test_actions_report_absolute_clamp_without_changing_zero_lane(self) -> None:
        import torch

        base = torch.full((1, 19), 1.8)
        action, _, clamped = _build_paired_actions(
            base,
            torch.tensor(((0.3, 0.4, 0.5),)),
            torch.zeros(19),
            torch.full((19,), 2.0),
        )
        torch.testing.assert_close(action[0], base[0])
        torch.testing.assert_close(action[1, (9, 10, 18)], torch.full((3,), 2.0))
        torch.testing.assert_close(clamped[0], torch.zeros(3, dtype=torch.bool))
        torch.testing.assert_close(clamped[1], torch.ones(3, dtype=torch.bool))

    @staticmethod
    def _samples():
        import torch

        steps, lanes = 3, 4
        zeros3 = torch.zeros((steps, lanes, 3))
        zeros1 = torch.zeros((steps, lanes))
        samples = {
            "requested_thumb_action": zeros3.clone(),
            "post_thumb_state": zeros3.clone(),
            "post_minus_pre_state": zeros3.clone(),
            "dynamic_rate_limit": torch.zeros_like(zeros3, dtype=torch.bool),
            "absolute_clamp": torch.zeros_like(zeros3, dtype=torch.bool),
            "thumb_surface_gap": torch.full((steps, lanes), 0.10),
            "best_non_thumb_gap": torch.full((steps, lanes), 0.02),
            "pregrasp_score": zeros1.clone(),
            "finger_contact_any_frame": torch.zeros((steps, lanes, 5), dtype=torch.bool),
            "opposed_any_frame": torch.zeros((steps, lanes), dtype=torch.bool),
            "opposed_streak": torch.zeros((steps, lanes), dtype=torch.int32),
        }
        samples["requested_thumb_action"][:, 1] = 0.2
        samples["requested_thumb_action"][:, 3] = -0.2
        samples["post_thumb_state"][:, 1] = 0.1
        samples["post_thumb_state"][:, 3] = -0.1
        samples["thumb_surface_gap"][:, 1] = 0.07
        samples["thumb_surface_gap"][:, 3] = 0.12
        samples["pregrasp_score"][:, 1] = 0.3
        samples["pregrasp_score"][:, 3] = -0.1
        return samples

    def test_paired_deltas_identify_the_sign_that_reduces_thumb_gap(self) -> None:
        import torch

        paired = _paired_metric_samples(self._samples())
        variants = _summarize_variant_deltas(paired, torch.tensor((0, 1)), 2)
        self.assertAlmostEqual(variants[0]["treatment_minus_zero"]["thumb_surface_gap_m"]["mean"], -0.03)
        self.assertAlmostEqual(variants[1]["treatment_minus_zero"]["thumb_surface_gap_m"]["mean"], 0.02)
        diagnosis = _diagnose_variants(variants)
        self.assertEqual(diagnosis["best_thumb_gap_variant_index"], 0)
        self.assertEqual(diagnosis["variant_effects"][0]["classification"], "residual_sign_reduces_thumb_gap")

    def test_diagnosis_distinguishes_rate_limit_and_absolute_clamp(self) -> None:
        import torch

        paired = _paired_metric_samples(self._samples())
        variants = _summarize_variant_deltas(paired, torch.tensor((0, 1)), 2)
        for name in ("thumb_cmc_pitch", "thumb_cmc_yaw", "thumb_cmc_roll"):
            variants[0]["treatment_minus_zero"]["treatment_dynamic_rate_limit"][name]["mean"] = 1.0
            variants[0]["treatment_minus_zero"]["post_thumb_state_rad"][name]["mean_abs"] = 0.0
            variants[1]["treatment_minus_zero"]["treatment_absolute_clamp"][name]["mean"] = 1.0
        diagnosis = _diagnose_variants(variants)
        self.assertEqual(
            diagnosis["variant_effects"][0]["classification"],
            "dynamic_rate_limit_or_tracking_dominant",
        )
        self.assertEqual(diagnosis["variant_effects"][1]["classification"], "absolute_bound_clamp_dominant")

    def test_row_schedule_and_strict_finite_buffer_gates(self) -> None:
        rows = _expected_row_counts(300, 16)
        self.assertEqual(rows[:4], [16 * 38] * 4)
        self.assertEqual(rows[4:], [16 * 37] * 4)
        counts = dict.fromkeys(("actions", "states", "gaps"), 9600)
        clean = _evaluate_gates(
            num_pairs=16,
            num_envs=32,
            variant_pair_counts=[8, 8],
            control_steps=300,
            expected_steps=300,
            finite_counts=counts,
            expected_lane_samples=9600,
            paired_deltas_finite=True,
            initial_pair_state_max_abs_delta=0.0,
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
            variant_pair_counts=[16, 0],
            control_steps=300,
            expected_steps=300,
            finite_counts={**counts, "gaps": 9599},
            expected_lane_samples=9600,
            paired_deltas_finite=False,
            initial_pair_state_max_abs_delta=0.0,
            shared_pair_base_max_abs_delta=0.0,
            row_counts=rows,
            expected_row_counts=rows,
            triangle_buffer_available=True,
            rigid_overflow_frames=0,
            rigid_overflow_excess=0,
            triangle_overflow_frames=1,
            triangle_overflow_excess=2,
        )
        self.assertFalse(dirty["passed"])
        self.assertFalse(dirty["all_variants_have_pairs"]["passed"])
        self.assertFalse(dirty["finite_rollout_samples"]["passed"])
        self.assertFalse(dirty["finite_paired_deltas"]["passed"])
        self.assertFalse(dirty["triangle_pair_buffer_clean"]["passed"])


if __name__ == "__main__":
    unittest.main()
