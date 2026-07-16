# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Groot Newton staged reward and transfer state machine."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from teleop_stack.envs import groot_newton_env as env_module


class TestGrootNewtonReward(unittest.TestCase):
    device = "cpu"

    def _array(self, values, dtype):
        return wp.array(values, dtype=dtype, device=self.device)

    def _zeros(self, count, dtype):
        return wp.zeros(count, dtype=dtype, device=self.device)

    def _launch_reward(self, reward_mode: int, *, terminate_on_success: bool = True):
        phases = self._array(
            [
                env_module._TASK_PHASE_APPROACH,
                env_module._TASK_PHASE_CARRYING,
                env_module._TASK_PHASE_CARRYING,
                env_module._TASK_PHASE_RELEASED,
                env_module._TASK_PHASE_SUCCESS,
                env_module._TASK_PHASE_FAIL,
            ],
            wp.int32,
        )
        count = phases.shape[0]
        episode_step = self._zeros(count, wp.int32)
        episode_return = self._zeros(count, wp.float32)
        success_once = self._zeros(count, wp.bool)
        reaching = self._array([0.5] * count, wp.float32)
        lift = self._array([0.25] * count, wp.float32)
        place = self._array([0.75] * count, wp.float32)
        static = self._array([0.5] * count, wp.float32)
        reached_lift = self._array([False, False, True, True, True, False], wp.bool)
        placed = self._array([False, False, False, True, True, False], wp.bool)
        success = self._array([False, False, False, False, True, False], wp.bool)
        fail = self._array([False, False, False, False, False, True], wp.bool)
        dense = self._zeros(count, wp.float32)
        reward = self._zeros(count, wp.float32)
        terminated = self._zeros(count, wp.bool)
        truncated = self._zeros(count, wp.bool)
        wp.launch(
            env_module._advance_episode,
            dim=count,
            inputs=[
                episode_step,
                episode_return,
                success_once,
                reaching,
                lift,
                place,
                static,
                phases,
                reached_lift,
                placed,
                success,
                fail,
                dense,
                reward,
                terminated,
                truncated,
                100,
                reward_mode,
                terminate_on_success,
                True,
            ],
            device=self.device,
        )
        return dense.numpy(), reward.numpy(), terminated.numpy()

    def test_stage_reward_overrides_and_normalization(self):
        dense, reward, terminated = self._launch_reward(env_module._REWARD_MODE_NORMALIZED_DENSE)
        expected = np.asarray([1.0, 3.25, 4.75, 6.5, 8.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(dense, expected, atol=1.0e-6)
        np.testing.assert_allclose(reward, expected / env_module._STAGE_REWARD_MAX, atol=1.0e-6)
        np.testing.assert_array_equal(terminated, [False, False, False, False, True, True])

    def test_sparse_reward_marks_failure(self):
        _, reward, _ = self._launch_reward(env_module._REWARD_MODE_SPARSE)
        np.testing.assert_array_equal(reward, [0.0, 0.0, 0.0, 0.0, 1.0, -1.0])

    def test_success_stage_can_fill_the_fixed_horizon(self):
        _, _, terminated = self._launch_reward(env_module._REWARD_MODE_NORMALIZED_DENSE, terminate_on_success=False)
        np.testing.assert_array_equal(terminated, [False, False, False, False, False, True])
        self.assertFalse(env_module.GrootNewtonEnvConfig().terminate_on_success)

    def test_lift_threshold_must_be_below_lift_height(self):
        with self.assertRaisesRegex(ValueError, "goal_threshold"):
            env_module.GrootNewtonEnvConfig(bottle_lift_height=0.1, goal_threshold=0.1)

    def _phase_arrays(self, count: int = 1):
        return {
            "obj_pose": wp.zeros((count, 7), dtype=wp.float32, device=self.device),
            "initial_pose": wp.zeros((count, 7), dtype=wp.float32, device=self.device),
            "is_grasped": self._zeros(count, wp.bool),
            "has_contact": self._zeros(count, wp.bool),
            "pose_valid": self._zeros(count, wp.bool),
            "is_static": self._zeros(count, wp.bool),
            "phase": self._zeros(count, wp.int32),
            "grasp_frames": self._zeros(count, wp.int32),
            "gap_frames": self._zeros(count, wp.int32),
            "settle_frames": self._zeros(count, wp.int32),
            "grasp_confirmed": self._zeros(count, wp.bool),
            "transport_started": self._zeros(count, wp.bool),
            "reached_lift": self._zeros(count, wp.bool),
            "release_armed": self._zeros(count, wp.bool),
            "released": self._zeros(count, wp.bool),
            "early_release": self._zeros(count, wp.bool),
            "max_z": self._zeros(count, wp.float32),
            "success": self._zeros(count, wp.bool),
            "fail": self._zeros(count, wp.bool),
        }

    def _set(self, destination, values, dtype):
        wp.copy(destination, self._array(values, dtype))

    def _advance_phase(self, arrays, *, grasp_frames=2, release_frames=2, settle_frames=2):
        wp.launch(
            env_module._advance_transfer_phase,
            dim=arrays["phase"].shape[0],
            inputs=[
                arrays["obj_pose"],
                arrays["initial_pose"],
                arrays["is_grasped"],
                arrays["has_contact"],
                arrays["pose_valid"],
                arrays["is_static"],
                0.1,
                0.005,
                0.01,
                grasp_frames,
                release_frames,
                settle_frames,
                arrays["phase"],
                arrays["grasp_frames"],
                arrays["gap_frames"],
                arrays["settle_frames"],
                arrays["grasp_confirmed"],
                arrays["transport_started"],
                arrays["reached_lift"],
                arrays["release_armed"],
                arrays["released"],
                arrays["early_release"],
                arrays["max_z"],
                arrays["success"],
                arrays["fail"],
            ],
            device=self.device,
        )

    def test_grasp_carry_release_and_settle(self):
        arrays = self._phase_arrays()
        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        self._advance_phase(arrays)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_APPROACH)
        self._advance_phase(arrays)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_CARRYING)

        pose = np.zeros((1, 7), dtype=np.float32)
        pose[0, 2] = 0.1
        wp.copy(arrays["obj_pose"], wp.array(pose, dtype=wp.float32, device=self.device))
        self._advance_phase(arrays)
        self.assertTrue(arrays["transport_started"].numpy()[0])
        self.assertTrue(arrays["reached_lift"].numpy()[0])

        self._set(arrays["pose_valid"], [True], wp.bool)
        self._advance_phase(arrays)
        self.assertTrue(arrays["release_armed"].numpy()[0])
        self._set(arrays["has_contact"], [False], wp.bool)
        self._advance_phase(arrays)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_CARRYING)
        self._advance_phase(arrays)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_RELEASED)

        self._set(arrays["is_static"], [True], wp.bool)
        self._advance_phase(arrays)
        self.assertFalse(arrays["success"].numpy()[0])
        self._advance_phase(arrays)
        self.assertTrue(arrays["success"].numpy()[0])
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_SUCCESS)

    def test_release_before_lift_fails(self):
        arrays = self._phase_arrays()
        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=1)
        pose = np.zeros((1, 7), dtype=np.float32)
        pose[0, 0] = 0.02
        wp.copy(arrays["obj_pose"], wp.array(pose, dtype=wp.float32, device=self.device))
        self._advance_phase(arrays, grasp_frames=1, release_frames=1)
        self._set(arrays["has_contact"], [False], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=1)
        self.assertTrue(arrays["early_release"].numpy()[0])
        self.assertTrue(arrays["fail"].numpy()[0])
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_FAIL)

    def test_release_outside_valid_pose_fails(self):
        arrays = self._phase_arrays()
        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=1)
        pose = np.zeros((1, 7), dtype=np.float32)
        pose[0, 2] = 0.1
        wp.copy(arrays["obj_pose"], wp.array(pose, dtype=wp.float32, device=self.device))
        self._advance_phase(arrays, grasp_frames=1, release_frames=1)
        self.assertTrue(arrays["reached_lift"].numpy()[0])
        self._set(arrays["has_contact"], [False], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=1)
        self.assertTrue(arrays["early_release"].numpy()[0])
        self.assertTrue(arrays["fail"].numpy()[0])
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_FAIL)

    def test_throwing_into_valid_pose_during_release_debounce_fails(self):
        arrays = self._phase_arrays()
        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=2)
        pose = np.zeros((1, 7), dtype=np.float32)
        pose[0, 2] = 0.1
        wp.copy(arrays["obj_pose"], wp.array(pose, dtype=wp.float32, device=self.device))
        self._advance_phase(arrays, grasp_frames=1, release_frames=2)
        self.assertTrue(arrays["reached_lift"].numpy()[0])
        self.assertFalse(arrays["release_armed"].numpy()[0])

        self._set(arrays["has_contact"], [False], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=2)
        self._set(arrays["pose_valid"], [True], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=2)
        self.assertTrue(arrays["fail"].numpy()[0])
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_FAIL)

    def test_masked_task_reset(self):
        arrays = self._phase_arrays(count=2)
        self._set(arrays["phase"], [env_module._TASK_PHASE_SUCCESS, env_module._TASK_PHASE_FAIL], wp.int32)
        self._set(arrays["success"], [True, False], wp.bool)
        self._set(arrays["fail"], [False, True], wp.bool)
        mask = self._array([True, False], wp.bool)
        wp.launch(
            env_module._reset_transfer_task,
            dim=2,
            inputs=[
                mask,
                arrays["phase"],
                arrays["grasp_frames"],
                arrays["gap_frames"],
                arrays["settle_frames"],
                arrays["grasp_confirmed"],
                arrays["transport_started"],
                arrays["reached_lift"],
                arrays["release_armed"],
                arrays["released"],
                arrays["early_release"],
                arrays["success"],
                arrays["fail"],
            ],
            device=self.device,
        )
        np.testing.assert_array_equal(
            arrays["phase"].numpy(), [env_module._TASK_PHASE_APPROACH, env_module._TASK_PHASE_FAIL]
        )
        np.testing.assert_array_equal(arrays["success"].numpy(), [False, False])
        np.testing.assert_array_equal(arrays["fail"].numpy(), [False, True])


if __name__ == "__main__":
    unittest.main()
