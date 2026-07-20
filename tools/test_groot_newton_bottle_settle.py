# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Groot Newton construction-time bottle settling pass."""

from __future__ import annotations

import types
import unittest

import numpy as np
import warp as wp

from teleop_stack.envs import groot_newton_env as env_module


class TestGrootNewtonBottleSettle(unittest.TestCase):
    device = "cpu"

    def test_settle_frames_cannot_be_negative(self):
        with self.assertRaisesRegex(ValueError, "bottle_settle_frames"):
            env_module.GrootNewtonEnvConfig(bottle_settle_frames=-1)

    def test_settle_requires_even_substeps_without_graph_capture(self):
        with self.assertRaisesRegex(ValueError, "bottle settling requires an even substeps_per_frame"):
            env_module.GrootNewtonEnvConfig(
                bottle_settle_frames=1,
                substeps_per_frame=3,
                capture_graph=False,
            )

    def test_settle_metadata_describes_cuda_graph_contract(self):
        env = object.__new__(env_module.GrootNewtonEnv)
        env.config = env_module.GrootNewtonEnvConfig(bottle_settle_frames=60, simulation_hz=60)
        env._scene = types.SimpleNamespace(graph=object())

        self.assertEqual(
            env.bottle_settle_metadata,
            {
                "enabled": True,
                "frames": 60,
                "duration_seconds": 1.0,
                "backend": "cuda_graph",
                "copied_free_joint_coordinates_per_env": 7,
                "zeroed_free_joint_velocities_per_env": 6,
            },
        )

    def test_capture_defaults_changes_only_bottle_free_joint(self):
        settled_q_np = np.arange(24, dtype=np.float32)
        settled_qd_np = np.arange(22, dtype=np.float32) + 100.0
        default_q_np = np.full(24, -1.0, dtype=np.float32)
        default_qd_np = np.full(22, -2.0, dtype=np.float32)
        q_start_np = np.asarray([2, 14], dtype=np.int32)
        qd_start_np = np.asarray([3, 14], dtype=np.int32)

        settled_q = wp.array(settled_q_np, dtype=wp.float32, device=self.device)
        settled_qd = wp.array(settled_qd_np, dtype=wp.float32, device=self.device)
        default_q = wp.array(default_q_np, dtype=wp.float32, device=self.device)
        default_qd = wp.array(default_qd_np, dtype=wp.float32, device=self.device)
        q_start = wp.array(q_start_np, dtype=wp.int32, device=self.device)
        qd_start = wp.array(qd_start_np, dtype=wp.int32, device=self.device)

        wp.launch(
            env_module._capture_bottle_reset_defaults,
            dim=(2, 7),
            inputs=[settled_q, settled_qd, q_start, qd_start, default_q, default_qd],
            device=self.device,
        )

        expected_q = default_q_np.copy()
        expected_q[2:9] = settled_q_np[2:9]
        expected_q[14:21] = settled_q_np[14:21]
        expected_qd = default_qd_np.copy()
        expected_qd[3:9] = 0.0
        expected_qd[14:20] = 0.0
        expected_settled_qd = settled_qd_np.copy()
        expected_settled_qd[3:9] = 0.0
        expected_settled_qd[14:20] = 0.0
        np.testing.assert_array_equal(default_q.numpy(), expected_q)
        np.testing.assert_array_equal(default_qd.numpy(), expected_qd)
        np.testing.assert_array_equal(settled_qd.numpy(), expected_settled_qd)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_cuda_graph_settle_becomes_full_and_partial_reset_baseline(self):
        import torch

        env = env_module.GrootNewtonEnv(
            num_envs=2,
            device="cuda:0",
            obs_mode="state_dict",
            render_images=False,
            load_scene_visuals=False,
            bottle_settle_frames=60,
            capture_graph=True,
        )
        try:
            self.assertEqual(env.bottle_settle_metadata["backend"], "cuda_graph")
            defaults = wp.to_torch(env.model.joint_q)
            state_q = wp.to_torch(env.state_0.joint_q)
            q_start = torch.as_tensor(env._bottle_q_start_np, device="cuda:0", dtype=torch.long)
            offsets = torch.arange(7, device="cuda:0")
            indices = q_start[:, None] + offsets[None, :]
            torch.testing.assert_close(state_q[indices], defaults[indices])

            state_q[q_start + 2] += torch.tensor([0.1, 0.2], device="cuda:0")
            lane_one_before = state_q[indices[1]].clone()
            mask = torch.tensor([True, False], device="cuda:0")
            _, info = env.reset(world_mask=mask)
            torch.testing.assert_close(state_q[indices[0]], defaults[indices[0]])
            torch.testing.assert_close(state_q[indices[1]], lane_one_before)
            torch.testing.assert_close(info["lift_height"][0], torch.zeros((), device="cuda:0"))

            env.reset()
            torch.testing.assert_close(state_q[indices], defaults[indices])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
