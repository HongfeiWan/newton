# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the L10 finger-root actuator-load observation."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from teleop_stack.envs import groot_newton_env as env_module


class TestGrootFingerRootLoad(unittest.TestCase):
    device = "cpu"

    def test_joint_order_matches_five_active_root_flexion_motors(self):
        self.assertEqual(
            env_module._FINGER_ROOT_JOINT_NAMES,
            (
                "thumb_cmc_pitch",
                "index_mcp_pitch",
                "middle_mcp_pitch",
                "ring_mcp_pitch",
                "pinky_mcp_pitch",
            ),
        )

    def test_qfrc_normalization_uses_per_world_dof_offsets(self):
        qfrc = np.zeros(20, dtype=np.float32)
        qfrc[1:6] = (-1.0, 0.0, 1.5, 3.0, 6.0)
        qfrc[11:16] = (-4.0, 1.0, -1.0, 1.5, 0.75)
        raw = wp.zeros((2, 5), dtype=wp.float32, device=self.device)
        load = wp.zeros((2, 5), dtype=wp.float32, device=self.device)
        wp.launch(
            env_module._extract_finger_root_load,
            dim=(2, 5),
            inputs=[
                wp.array(qfrc, dtype=wp.float32, device=self.device),
                wp.array((0, 10, 20), dtype=wp.int32, device=self.device),
                wp.array((1, 2, 3, 4, 5), dtype=wp.int32, device=self.device),
                wp.array((-1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.float32, device=self.device),
                wp.array((0.0, 0.0, 0.5, 0.0, 0.0), dtype=wp.float32, device=self.device),
                wp.array((2.0, 2.0, 2.0, 3.0, 3.0), dtype=wp.float32, device=self.device),
                raw,
                load,
            ],
            device=self.device,
        )
        np.testing.assert_allclose(raw.numpy()[0], (-1.0, 0.0, 1.5, 3.0, 6.0))
        np.testing.assert_allclose(load.numpy()[0], (0.5, 0.0, 0.5, 1.0, 1.0))
        np.testing.assert_allclose(load.numpy()[1], (1.0, 0.5, 0.0, 0.5, 0.25))

    def test_partial_reset_clears_only_selected_lane(self):
        raw = wp.full((3, 5), 2.0, dtype=wp.float32, device=self.device)
        load = wp.full((3, 5), 0.75, dtype=wp.float32, device=self.device)
        wp.launch(
            env_module._clear_finger_root_load_rows,
            dim=(3, 5),
            inputs=[wp.array((False, True, False), dtype=wp.bool, device=self.device), raw, load],
            device=self.device,
        )
        np.testing.assert_allclose(raw.numpy(), ((2.0,) * 5, (0.0,) * 5, (2.0,) * 5))
        np.testing.assert_allclose(load.numpy(), ((0.75,) * 5, (0.0,) * 5, (0.75,) * 5))

    def test_load_calibration_config_is_validated(self):
        with self.assertRaisesRegex(ValueError, "finger_root_load_bias"):
            env_module.GrootNewtonEnvConfig(finger_root_load_bias=(0.0,))
        with self.assertRaisesRegex(ValueError, "finger_root_load_scale"):
            env_module.GrootNewtonEnvConfig(finger_root_load_scale=(1.0, 1.0, 0.0, 1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "finger_root_closing_sign"):
            env_module.GrootNewtonEnvConfig(finger_root_closing_sign=(1.0, 1.0, 0.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
