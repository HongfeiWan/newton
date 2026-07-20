# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the L10 finger-root load probe statistics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from tools.probe_newton_l10_finger_root_load import _binary_auc, _summarize_samples
from tools.probe_newton_l10_finger_root_load_replay import _load_episode_actions


@unittest.skipIf(torch is None, "PyTorch is required")
class TestFingerRootLoadProbe(unittest.TestCase):
    def test_replay_loads_preextracted_numpy_actions(self):
        expected = np.arange(57, dtype=np.float32).reshape(3, 19)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "actions.npy"
            np.save(path, expected)
            actual = _load_episode_actions(path, episode_index=0)
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_binary_auc_handles_order_and_ties(self):
        labels = torch.tensor((False, False, True, True))
        self.assertAlmostEqual(float(_binary_auc(torch.tensor((0.0, 0.1, 0.8, 0.9)), labels)), 1.0)
        self.assertAlmostEqual(float(_binary_auc(torch.tensor((0.8, 0.9, 0.0, 0.1)), labels)), 0.0)
        self.assertAlmostEqual(float(_binary_auc(torch.full((4,), 0.5), labels)), 0.5)

    def test_summary_passes_separated_two_finger_load(self):
        loads = torch.tensor(
            (
                (0.00, 0.02, 0.01, 0.00, 0.00),
                (0.01, 0.01, 0.02, 0.00, 0.00),
                (0.30, 0.20, 0.05, 0.00, 0.00),
                (0.40, 0.25, 0.04, 0.00, 0.00),
                (0.90, 0.80, 0.10, 0.00, 0.00),
                (0.85, 0.75, 0.10, 0.00, 0.00),
            )
        )
        contact = torch.tensor((False, False, True, True, True, True))
        grasp = torch.tensor((False, False, False, False, True, True))
        summary = _summarize_samples(
            loads,
            contact,
            grasp,
            reset_zero=torch.tensor(True),
            episodes_completed=torch.tensor(3),
            episodes_expected=3,
        )
        self.assertTrue(summary["passed"])
        self.assertGreater(summary["second_largest_load"]["grasp_minus_free"], 0.1)
        self.assertGreater(summary["second_largest_load"]["grasp_auc"], 0.75)
        json.dumps(summary, allow_nan=False)

    def test_missing_grasp_samples_fails_without_nan_json(self):
        loads = torch.zeros((4, 5))
        summary = _summarize_samples(
            loads,
            torch.zeros(4, dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool),
            reset_zero=torch.tensor(True),
            episodes_completed=torch.tensor(4),
            episodes_expected=4,
        )
        self.assertIsNone(summary["second_largest_load"]["grasp_auc"])
        self.assertFalse(summary["passed"])
        json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
