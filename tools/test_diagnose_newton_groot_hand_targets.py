#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Groot hand-target diagnostic."""

from __future__ import annotations

import unittest

from tools.diagnose_newton_groot_hand_targets import (
    _nearest_reference_rows,
    _summarize_condition,
    _target_comparison,
    _validate_args,
    create_parser,
)


class TestHandTargetDiagnostic(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch
        self.minimum = torch.zeros(19)
        self.maximum = torch.full((19,), 2.0)

    def test_synchronized_reference_replay_uses_larger_triangle_pair_capacity(self) -> None:
        args = create_parser().parse_args(("dp.pt", "residual.pt", "actions.npy"))
        self.assertEqual(args.triangle_pairs_per_env, 196608)
        _validate_args(args)
        args.triangle_pairs_per_env = 0
        with self.assertRaisesRegex(ValueError, "triangle-pairs-per-env"):
            _validate_args(args)

    def test_nearest_reference_excludes_thumb_coordinates(self) -> None:
        torch = self.torch
        base = torch.ones((1, 19))
        references = torch.ones((2, 19))
        references[0, 11] = 0.0
        references[1, 9] = 2.0
        match = _nearest_reference_rows(base, references, self.minimum, self.maximum)
        torch.testing.assert_close(match, torch.tensor((1,)))

    def test_priority_reports_uncovered_thumb_coordinate(self) -> None:
        torch = self.torch
        base = torch.ones((2, 19))
        candidate = base.clone()
        executed = base[:, 9:19].clone()
        reference = base.clone()
        reference[:, 9] += 0.20
        reference[:, 10] += 0.05
        candidate[:, 9] += 0.10
        executed[:, 0] += 0.04
        summary = _target_comparison(
            base,
            candidate,
            executed,
            reference,
            self.minimum,
            self.maximum,
            hand_scale_normalized=0.1,
        )
        self.assertEqual(summary["thumb_priority"][0]["joint_name"], "thumb_cmc_pitch")
        self.assertAlmostEqual(
            summary["per_joint"]["thumb_cmc_pitch"]["coverage_fraction_at_configured_scale"],
            0.0,
        )
        self.assertAlmostEqual(
            summary["per_joint"]["thumb_cmc_yaw"]["coverage_fraction_at_configured_scale"],
            1.0,
        )
        self.assertAlmostEqual(summary["thumb_mae_normalized"]["candidate_to_reference"], 0.05)
        self.assertGreater(
            summary["per_joint"]["thumb_cmc_pitch"]["candidate_execution_gap_rad"]["p50"],
            0.0,
        )

    def test_condition_summary_reports_dynamic_rate_limit(self) -> None:
        torch = self.torch
        base = torch.ones((1, 2, 19))
        candidate = base.clone()
        candidate[..., 9] += 0.10
        rollout = {
            "base": base,
            "candidate": candidate,
            "current_hand": torch.ones((1, 2, 10)),
            "executed_hand": torch.ones((1, 2, 10)),
            "raw_latent": torch.zeros((1, 2, 16)),
            "active": torch.ones((1, 2), dtype=torch.bool),
        }
        reference = torch.ones((1, 19))
        reference[:, 9] += 0.20
        summary = _summarize_condition(
            rollout,
            torch.tensor(((True, False),)),
            reference,
            self.minimum,
            self.maximum,
            hand_scale_normalized=0.1,
            hand_max_joint_step_rad=0.08,
        )
        self.assertEqual(summary["sample_count"], 1)
        self.assertAlmostEqual(summary["per_joint"]["thumb_cmc_pitch"]["dynamic_rate_limit_fraction"], 1.0)
        self.assertAlmostEqual(summary["per_joint"]["thumb_cmc_yaw"]["dynamic_rate_limit_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
