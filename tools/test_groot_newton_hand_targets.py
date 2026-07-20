# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for active L10 and mimic-follower control targets."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from teleop_stack.envs import groot_newton_env as env_module


class TestGrootNewtonHandTargets(unittest.TestCase):
    device = "cpu"
    num_envs = 3
    coords_per_world = 20

    def _array(self, values, dtype):
        return wp.array(values, dtype=dtype, device=self.device)

    def _stub_env(self, control_mode: str):
        env = object.__new__(env_module.GrootNewtonEnv)
        env.device = wp.get_device(self.device)
        env.num_envs = self.num_envs
        env.control_mode = control_mode
        env._control_mode_id = env_module._CONTROL_MODE_IDS[control_mode]

        starts = np.arange(self.num_envs + 1, dtype=np.int32) * self.coords_per_world
        local_active = np.arange(env_module.JOINT_ACTION_SIZE, dtype=np.int32)
        total = self.num_envs * self.coords_per_world
        lower = np.full(total, -2.0, dtype=np.float32)
        upper = np.full(total, 2.0, dtype=np.float32)
        for world in range(self.num_envs):
            lower[world * self.coords_per_world + 17] = 0.0
            upper[world * self.coords_per_world + 17] = 0.7
            lower[world * self.coords_per_world + 18] = -0.25
            upper[world * self.coords_per_world + 18] = 0.25

        env.model = SimpleNamespace(
            joint_coord_world_start=self._array(starts, wp.int32),
            joint_dof_world_start=self._array(starts, wp.int32),
            joint_limit_lower=self._array(lower, wp.float32),
            joint_limit_upper=self._array(upper, wp.float32),
        )
        state_q = np.zeros(total, dtype=np.float32)
        for world in range(self.num_envs):
            state_q[world * self.coords_per_world : world * self.coords_per_world + env_module.JOINT_ACTION_SIZE] = (
                np.linspace(-0.5, 0.5, env_module.JOINT_ACTION_SIZE, dtype=np.float32) + 0.1 * world
            )
        env.state_0 = SimpleNamespace(joint_q=self._array(state_q, wp.float32))
        env.control = SimpleNamespace(
            joint_target_q=self._array(np.full(total, -9.0, dtype=np.float32), wp.float32),
            joint_target_qd=self._array(np.full(total, 7.0, dtype=np.float32), wp.float32),
        )
        env._action = self._array(
            np.zeros(
                (
                    self.num_envs,
                    env_module.ACTION_SIZE if control_mode == "pd_eef_pose_abs" else env_module.JOINT_ACTION_SIZE,
                ),
                dtype=np.float32,
            ),
            wp.float32,
        )
        env._action_local_q = self._array(local_active, wp.int32)
        env._action_local_qd = self._array(local_active, wp.int32)
        env._action_scale = self._array(np.full(env_module.JOINT_ACTION_SIZE, 0.4, dtype=np.float32), wp.float32)

        q_indices = starts[:-1, None] + np.asarray((7, 8), dtype=np.int32)[None, :]
        follower_indices = starts[:-1, None] + np.asarray((17, 18), dtype=np.int32)[None, :]
        env._hand_mimic_count = 2
        env._hand_mimic_leader_q_indices = self._array(q_indices, wp.int32)
        env._hand_mimic_follower_q_indices = self._array(follower_indices, wp.int32)
        env._hand_mimic_follower_qd_indices = self._array(follower_indices, wp.int32)
        env._hand_mimic_multiplier = self._array((1.5, -0.5), wp.float32)
        env._hand_mimic_offset = self._array((0.1, 0.2), wp.float32)
        return env

    def _assert_mimic_targets(self, env, lanes=None) -> None:
        target_q = env.control.joint_target_q.numpy()
        target_qd = env.control.joint_target_qd.numpy()
        leader = env._hand_mimic_leader_q_indices.numpy()
        follower = env._hand_mimic_follower_q_indices.numpy()
        follower_qd = env._hand_mimic_follower_qd_indices.numpy()
        multiplier = env._hand_mimic_multiplier.numpy()
        offset = env._hand_mimic_offset.numpy()
        lower = env.model.joint_limit_lower.numpy()
        upper = env.model.joint_limit_upper.numpy()
        for world in range(self.num_envs) if lanes is None else lanes:
            for mimic in range(env._hand_mimic_count):
                expected = np.clip(
                    offset[mimic] + multiplier[mimic] * target_q[leader[world, mimic]],
                    lower[follower_qd[world, mimic]],
                    upper[follower_qd[world, mimic]],
                )
                self.assertAlmostEqual(float(target_q[follower[world, mimic]]), float(expected), places=6)
                self.assertEqual(float(target_qd[follower_qd[world, mimic]]), 0.0)

    def test_sync_clamps_nonunit_multipliers_for_every_lane(self):
        env = self._stub_env("pd_eef_pose_abs")
        target_q = env.control.joint_target_q.numpy()
        leader = env._hand_mimic_leader_q_indices.numpy()
        leader_values = np.asarray(((0.2, -0.4), (0.8, 0.6), (-1.2, 1.5)), dtype=np.float32)
        for world in range(self.num_envs):
            target_q[leader[world]] = leader_values[world]
        env.control.joint_target_q = self._array(target_q, wp.float32)

        env._sync_hand_mimic_targets()

        self._assert_mimic_targets(env)
        result = env.control.joint_target_q.numpy().reshape(self.num_envs, self.coords_per_world)
        np.testing.assert_allclose(result[:, 17], (0.4, 0.7, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(result[:, 18], (0.25, -0.1, -0.25), atol=1.0e-6)
        np.testing.assert_array_equal(result[:, 19], np.full(self.num_envs, -9.0, dtype=np.float32))

    def test_apply_action_syncs_eef_and_both_joint_control_modes(self):
        for control_mode in ("pd_eef_pose_abs", "pd_joint_pos", "pd_joint_delta_pos"):
            with self.subTest(control_mode=control_mode):
                env = self._stub_env(control_mode)
                if control_mode == "pd_eef_pose_abs":
                    target_q = env.control.joint_target_q.numpy()
                    leader = env._hand_mimic_leader_q_indices.numpy()
                    for world in range(self.num_envs):
                        target_q[leader[world]] = (0.15 + 0.2 * world, -0.4 + 0.3 * world)
                    env.control.joint_target_q = self._array(target_q, wp.float32)
                    env._apply_eef_pose_action_torch = lambda: None
                else:
                    action = env._action.numpy()
                    action[:, 7] = (-0.5, 0.25, 0.75)
                    action[:, 8] = (0.8, -0.6, 0.4)
                    env._action = self._array(action, wp.float32)

                env._apply_action()

                self._assert_mimic_targets(env)

    def test_joint_hold_and_masked_reset_keep_mimic_targets_consistent(self):
        for control_mode in ("pd_joint_pos", "pd_joint_delta_pos"):
            with self.subTest(control_mode=control_mode):
                env = self._stub_env(control_mode)
                env.hold_action()
                env._apply_action()
                self._assert_mimic_targets(env)

        env = self._stub_env("pd_joint_pos")
        total = self.num_envs * self.coords_per_world
        default_q = np.full(total, -3.0, dtype=np.float32)
        default_qd = np.full(total, 4.0, dtype=np.float32)
        default_q[env._hand_mimic_leader_q_indices.numpy()[1]] = (0.3, -0.2)
        target_before = env.control.joint_target_q.numpy().copy()
        target_qd_before = env.control.joint_target_qd.numpy().copy()
        mask = self._array((False, True, False), wp.bool)
        wp.launch(
            env_module._reset_control_targets,
            dim=(self.num_envs, self.coords_per_world),
            inputs=[
                mask,
                env.model.joint_coord_world_start,
                env.model.joint_dof_world_start,
                self._array(default_q, wp.float32),
                self._array(default_qd, wp.float32),
                env.control.joint_target_q,
                env.control.joint_target_qd,
                self.coords_per_world,
                self.coords_per_world,
            ],
            device=self.device,
        )
        env._sync_hand_mimic_targets(mask)

        self._assert_mimic_targets(env, lanes=(1,))
        reset_q = env.control.joint_target_q.numpy().reshape(self.num_envs, self.coords_per_world)
        reset_qd = env.control.joint_target_qd.numpy().reshape(self.num_envs, self.coords_per_world)
        np.testing.assert_array_equal(reset_q[0], target_before.reshape(self.num_envs, self.coords_per_world)[0])
        np.testing.assert_array_equal(reset_q[2], target_before.reshape(self.num_envs, self.coords_per_world)[2])
        np.testing.assert_array_equal(reset_qd[0], target_qd_before.reshape(self.num_envs, self.coords_per_world)[0])
        np.testing.assert_array_equal(reset_qd[2], target_qd_before.reshape(self.num_envs, self.coords_per_world)[2])


if __name__ == "__main__":
    unittest.main()
