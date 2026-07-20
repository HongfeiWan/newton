# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Groot Newton staged reward and transfer state machine."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from teleop_stack.envs import groot_newton_env as env_module
from teleop_stack.retargeting.hand_config import LINKER_L10_FINGERTIP_LOCAL_OFFSETS_M
from tools.run_newton_groot_rl_env import create_parser


@wp.kernel
def _evaluate_lift_metrics(
    current_z: wp.array[wp.float32],
    initial_z: wp.array[wp.float32],
    lift_height: wp.float32,
    height: wp.array[wp.float32],
    progress: wp.array[wp.float32],
):
    index = wp.tid()
    height[index] = env_module._positive_lift_height(current_z[index], initial_z[index])
    progress[index] = env_module._normalized_lift_progress(current_z[index], initial_z[index], lift_height)


@wp.kernel
def _evaluate_opposed_grasp_rows(
    finger_contact_counts: wp.array2d[wp.int32],
    grasp_finger_count: wp.int32,
    result: wp.array[wp.bool],
):
    world = wp.tid()
    touching_fingers = int(0)
    non_thumb_contact = False
    for finger in range(5):
        if finger_contact_counts[world, finger] > 0:
            touching_fingers = touching_fingers + 1
            if finger > 0:
                non_thumb_contact = True
    result[world] = env_module._is_opposed_grasp(
        finger_contact_counts[world, 0] > 0,
        non_thumb_contact,
        touching_fingers,
        grasp_finger_count,
    )


class TestGrootNewtonReward(unittest.TestCase):
    device = "cpu"

    def _array(self, values, dtype):
        return wp.array(values, dtype=dtype, device=self.device)

    def _zeros(self, count, dtype):
        return wp.zeros(count, dtype=dtype, device=self.device)

    def _finger_contacts(self, values):
        return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=self.device)

    def _launch_pregrasp_geometry(self, local_fingertips, *, rotate_last=False, return_pairs=False):
        radius = 0.0317
        half_height = 0.0948
        body_transforms = []
        shape_transforms = []
        shape_scales = []
        body_world_start = []
        shape_world_start = []
        offsets = np.asarray(LINKER_L10_FINGERTIP_LOCAL_OFFSETS_M, dtype=np.float32)
        for world, fingertips in enumerate(local_fingertips):
            body_world_start.append(len(body_transforms))
            shape_world_start.append(len(shape_transforms))
            rotated = rotate_last and world == len(local_fingertips) - 1
            if rotated:
                angle = np.pi / 2.0
                rotation = np.asarray(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)), dtype=np.float32)
                quaternion = wp.quat(0.0, float(np.sin(angle / 2.0)), 0.0, float(np.cos(angle / 2.0)))
                bottle_position = np.asarray((0.4, -0.2, 0.3), dtype=np.float32)
            else:
                rotation = np.eye(3, dtype=np.float32)
                quaternion = wp.quat_identity()
                bottle_position = np.zeros(3, dtype=np.float32)
            body_transforms.append(wp.transform(wp.vec3(*bottle_position), quaternion))
            for point, offset in zip(fingertips, offsets, strict=True):
                world_point = bottle_position + rotation @ np.asarray(point, dtype=np.float32)
                body_transforms.append(wp.transform(wp.vec3(*(world_point - offset)), wp.quat_identity()))
            shape_transforms.append(wp.transform_identity())
            shape_scales.append(wp.vec3(radius, half_height, radius))

        count = len(local_fingertips)
        gaps = wp.zeros((count, 5), dtype=wp.float32, device=self.device)
        opposition = wp.zeros((count, 4), dtype=wp.float32, device=self.device)
        z_score = wp.zeros((count, 4), dtype=wp.float32, device=self.device)
        score = wp.zeros(count, dtype=wp.float32, device=self.device)
        wp.launch(
            env_module._evaluate_opposed_pregrasp_geometry,
            dim=count,
            inputs=[
                wp.array(body_transforms, dtype=wp.transform, device=self.device),
                self._array(body_world_start, wp.int32),
                0,
                self._array(shape_world_start, wp.int32),
                0,
                wp.array(shape_transforms, dtype=wp.transform, device=self.device),
                wp.array(shape_scales, dtype=wp.vec3, device=self.device),
                self._array([1, 2, 3, 4, 5], wp.int32),
                wp.array(offsets, dtype=wp.vec3, device=self.device),
                gaps,
                opposition,
                z_score,
                score,
            ],
            device=self.device,
        )
        if return_pairs:
            return gaps.numpy(), opposition.numpy(), z_score.numpy(), score.numpy()
        return gaps.numpy(), score.numpy()

    def _launch_approach_rewards(
        self,
        reaching_values,
        finger_contact_values,
        is_grasped_values,
        pregrasp_values=None,
        *,
        finger_contact_any_frame_values=None,
        opposed_grasp_any_frame_values=None,
        opposed_grasp_max_consecutive_frames_values=None,
        non_thumb_anchor_contact_fraction_values=None,
        non_thumb_missing_thumb_geometry_progress_values=None,
        thumb_anchor_contact_fraction_values=None,
        thumb_missing_non_thumb_geometry_progress_values=None,
        task_phase_values=None,
        return_components=False,
    ):
        count = len(reaching_values)
        finger_contact_values_np = np.asarray(finger_contact_values, dtype=np.int32)
        is_grasped_values_np = np.asarray(is_grasped_values, dtype=np.bool_)
        if finger_contact_any_frame_values is None:
            finger_contact_any_frame_values = finger_contact_values_np > 0
        if opposed_grasp_any_frame_values is None:
            opposed_grasp_any_frame_values = is_grasped_values_np
        if opposed_grasp_max_consecutive_frames_values is None:
            opposed_grasp_max_consecutive_frames_values = is_grasped_values_np.astype(np.int32)
        if task_phase_values is None:
            task_phase_values = [env_module._TASK_PHASE_APPROACH] * count
        if non_thumb_anchor_contact_fraction_values is None:
            non_thumb_anchor_contact_fraction_values = [0.0] * count
        if non_thumb_missing_thumb_geometry_progress_values is None:
            non_thumb_missing_thumb_geometry_progress_values = [0.0] * count
        if thumb_anchor_contact_fraction_values is None:
            thumb_anchor_contact_fraction_values = [0.0] * count
        if thumb_missing_non_thumb_geometry_progress_values is None:
            thumb_missing_non_thumb_geometry_progress_values = [0.0] * count
        zeros_float = self._zeros(count, wp.float32)
        zeros_bool = self._zeros(count, wp.bool)
        approach_base = self._zeros(count, wp.float32)
        unilateral_gain = self._zeros(count, wp.float32)
        unilateral_reward = self._zeros(count, wp.float32)
        dense = self._zeros(count, wp.float32)
        reward = self._zeros(count, wp.float32)
        wp.launch(
            env_module._advance_episode,
            dim=count,
            inputs=[
                self._zeros(count, wp.int32),
                self._zeros(count, wp.float32),
                self._zeros(count, wp.bool),
                self._array(reaching_values, wp.float32),
                self._array(pregrasp_values, wp.float32) if pregrasp_values is not None else zeros_float,
                zeros_float,
                0.1,
                zeros_float,
                zeros_float,
                self._finger_contacts(finger_contact_values_np),
                self._array(is_grasped_values, wp.bool),
                self._array(finger_contact_any_frame_values, wp.bool),
                self._array(opposed_grasp_any_frame_values, wp.bool),
                self._array(opposed_grasp_max_consecutive_frames_values, wp.int32),
                self._array(non_thumb_anchor_contact_fraction_values, wp.float32),
                self._array(non_thumb_missing_thumb_geometry_progress_values, wp.float32),
                self._array(thumb_anchor_contact_fraction_values, wp.float32),
                self._array(thumb_missing_non_thumb_geometry_progress_values, wp.float32),
                self._array(task_phase_values, wp.int32),
                zeros_bool,
                zeros_bool,
                zeros_bool,
                zeros_bool,
                zeros_bool,
                approach_base,
                unilateral_gain,
                unilateral_reward,
                dense,
                reward,
                self._zeros(count, wp.bool),
                self._zeros(count, wp.bool),
                100,
                env_module._REWARD_MODE_DENSE,
                True,
                True,
            ],
            device=self.device,
        )
        if return_components:
            return dense.numpy(), approach_base.numpy(), unilateral_gain.numpy(), unilateral_reward.numpy()
        return dense.numpy()

    def _accumulate_unilateral_guidance(
        self,
        contact_frames,
        *,
        finger_surface_gap=None,
        thumb_partner_opposition=None,
        thumb_partner_z_score=None,
    ):
        frames = np.asarray(contact_frames, dtype=np.int32)
        if frames.ndim != 3 or frames.shape[2] != 5:
            raise ValueError("contact_frames must have shape [frame, world, 5]")
        frame_count, world_count, _ = frames.shape
        if finger_surface_gap is None:
            finger_surface_gap = np.zeros((world_count, 5), dtype=np.float32)
        if thumb_partner_opposition is None:
            thumb_partner_opposition = np.ones((world_count, 4), dtype=np.float32)
        if thumb_partner_z_score is None:
            thumb_partner_z_score = np.ones((world_count, 4), dtype=np.float32)

        finger_contacts = self._finger_contacts(frames[0])
        is_grasped = self._zeros(world_count, wp.bool)
        finger_any = self._zeros((world_count, 5), wp.bool)
        opposed_any = self._zeros(world_count, wp.bool)
        current_streak = self._zeros(world_count, wp.int32)
        max_streak = self._zeros(world_count, wp.int32)
        non_thumb_fraction = self._zeros(world_count, wp.float32)
        non_thumb_geometry = self._zeros(world_count, wp.float32)
        non_thumb_opposition = self._zeros(world_count, wp.float32)
        non_thumb_z = self._zeros(world_count, wp.float32)
        thumb_fraction = self._zeros(world_count, wp.float32)
        thumb_geometry = self._zeros(world_count, wp.float32)
        gaps = self._array(finger_surface_gap, wp.float32)
        opposition = self._array(thumb_partner_opposition, wp.float32)
        z_score = self._array(thumb_partner_z_score, wp.float32)
        for frame_contacts in frames:
            wp.copy(finger_contacts, self._finger_contacts(frame_contacts))
            wp.launch(
                env_module._accumulate_control_step_contact_topology,
                dim=world_count,
                inputs=[
                    finger_contacts,
                    is_grasped,
                    gaps,
                    opposition,
                    z_score,
                    1.0 / float(frame_count),
                    finger_any,
                    opposed_any,
                    current_streak,
                    max_streak,
                    non_thumb_fraction,
                    non_thumb_geometry,
                    non_thumb_opposition,
                    non_thumb_z,
                    thumb_fraction,
                    thumb_geometry,
                ],
                device=self.device,
            )
        return {
            "cN": non_thumb_fraction.numpy(),
            "GN": non_thumb_geometry.numpy(),
            "oN": non_thumb_opposition.numpy(),
            "zN": non_thumb_z.numpy(),
            "cT": thumb_fraction.numpy(),
            "GT": thumb_geometry.numpy(),
        }

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
        zeros_float = self._zeros(count, wp.float32)
        episode_step = self._zeros(count, wp.int32)
        episode_return = self._zeros(count, wp.float32)
        success_once = self._zeros(count, wp.bool)
        reaching = self._array([0.5] * count, wp.float32)
        current_lift = self._array([0.025] * count, wp.float32)
        place = self._array([0.75] * count, wp.float32)
        static = self._array([0.5] * count, wp.float32)
        finger_contacts = self._finger_contacts(np.zeros((count, 5), dtype=np.int32))
        is_grasped = self._zeros(count, wp.bool)
        reached_lift = self._array([False, False, True, True, True, False], wp.bool)
        release_ready = self._array([False, False, False, False, False, False], wp.bool)
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
                zeros_float,
                current_lift,
                0.1,
                place,
                static,
                finger_contacts,
                is_grasped,
                self._zeros((count, 5), wp.bool),
                self._zeros(count, wp.bool),
                self._zeros(count, wp.int32),
                zeros_float,
                zeros_float,
                zeros_float,
                zeros_float,
                phases,
                reached_lift,
                release_ready,
                placed,
                success,
                fail,
                self._zeros(count, wp.float32),
                self._zeros(count, wp.float32),
                self._zeros(count, wp.float32),
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
        expected = np.asarray([0.5, 2.875, 4.75, 6.5, 8.0, -8.0], dtype=np.float32)
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

    def test_dense_reward_stage_boundaries_are_monotonic(self):
        phases = self._array(
            [
                env_module._TASK_PHASE_APPROACH,
                env_module._TASK_PHASE_APPROACH,
                env_module._TASK_PHASE_APPROACH,
                env_module._TASK_PHASE_APPROACH,
                env_module._TASK_PHASE_CARRYING,
                env_module._TASK_PHASE_CARRYING,
                env_module._TASK_PHASE_CARRYING,
                env_module._TASK_PHASE_CARRYING,
                env_module._TASK_PHASE_CARRYING,
                env_module._TASK_PHASE_CARRYING,
                env_module._TASK_PHASE_RELEASED,
                env_module._TASK_PHASE_RELEASED,
                env_module._TASK_PHASE_SUCCESS,
                env_module._TASK_PHASE_FAIL,
            ],
            wp.int32,
        )
        count = phases.shape[0]
        zeros_float = self._zeros(count, wp.float32)
        episode_step = self._zeros(count, wp.int32)
        episode_return = self._zeros(count, wp.float32)
        success_once = self._zeros(count, wp.bool)
        reaching = self._array([0.0, 1.0, 1.0, 0.0] + [0.0] * 10, wp.float32)
        current_lift = self._array([0.0] * 4 + [0.0, 0.1] + [0.0] * 8, wp.float32)
        place = self._array([0.0] * 6 + [0.0, 1.0] + [0.0] * 6, wp.float32)
        static = self._array([0.0] * 9 + [1.0, 0.0, 1.0, 0.0, 0.0], wp.float32)
        finger_contacts_np = np.zeros((count, 5), dtype=np.int32)
        finger_contacts_np[2, 1] = 1
        finger_contacts_np[3:, :2] = 1
        finger_contacts = self._finger_contacts(finger_contacts_np)
        is_grasped = self._array([False, False, False, True] + [False] * 10, wp.bool)
        reached_lift = self._array([False] * 6 + [True] * 8, wp.bool)
        release_ready = self._array([False] * 8 + [True, True] + [False] * 4, wp.bool)
        placed = self._array([False] * 10 + [True, True] + [False] * 2, wp.bool)
        success = self._array([False] * 12 + [True, False], wp.bool)
        fail = self._array([False] * 13 + [True], wp.bool)
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
                zeros_float,
                current_lift,
                0.1,
                place,
                static,
                finger_contacts,
                is_grasped,
                self._zeros((count, 5), wp.bool),
                self._zeros(count, wp.bool),
                self._zeros(count, wp.int32),
                zeros_float,
                zeros_float,
                zeros_float,
                zeros_float,
                phases,
                reached_lift,
                release_ready,
                placed,
                success,
                fail,
                self._zeros(count, wp.float32),
                self._zeros(count, wp.float32),
                self._zeros(count, wp.float32),
                dense,
                reward,
                terminated,
                truncated,
                100,
                env_module._REWARD_MODE_DENSE,
                True,
                True,
            ],
            device=self.device,
        )

        expected = np.asarray([0.0, 1.0, 1.0, 1.625, 2.0, 4.0, 4.0, 5.0, 6.0, 7.0, 6.0, 7.0, 8.0, -8.0])
        dense_np = dense.numpy()
        np.testing.assert_allclose(dense_np, expected, atol=1.0e-6)
        np.testing.assert_allclose(reward.numpy(), expected, atol=1.0e-6)
        self.assertTrue(np.all(np.diff(dense_np[:10]) >= 0.0))
        self.assertTrue(np.all(np.diff(dense_np[10:13]) >= 0.0))
        self.assertGreater(dense_np[3], dense_np[2])
        self.assertGreater(dense_np[8], dense_np[7])

    def test_pre_lift_reward_resolves_takeoff_progress(self):
        count = 3
        zeros_float = self._zeros(count, wp.float32)
        zeros_bool = self._zeros(count, wp.bool)
        episode_step = self._zeros(count, wp.int32)
        episode_return = self._zeros(count, wp.float32)
        success_once = self._zeros(count, wp.bool)
        current_lift = self._array([0.0, 0.001, 0.01], wp.float32)
        phases = self._array([env_module._TASK_PHASE_CARRYING] * count, wp.int32)
        finger_contacts = self._finger_contacts([[1, 1, 0, 0, 0]] * count)
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
                zeros_float,
                zeros_float,
                current_lift,
                0.1,
                zeros_float,
                zeros_float,
                finger_contacts,
                zeros_bool,
                self._zeros((count, 5), wp.bool),
                self._zeros(count, wp.bool),
                self._zeros(count, wp.int32),
                zeros_float,
                zeros_float,
                zeros_float,
                zeros_float,
                phases,
                zeros_bool,
                zeros_bool,
                zeros_bool,
                zeros_bool,
                zeros_bool,
                self._zeros(count, wp.float32),
                self._zeros(count, wp.float32),
                self._zeros(count, wp.float32),
                dense,
                reward,
                terminated,
                truncated,
                100,
                env_module._REWARD_MODE_NORMALIZED_DENSE,
                True,
                True,
            ],
            device=self.device,
        )

        expected_dense = np.asarray([2.0, 2.065, 2.65], dtype=np.float32)
        np.testing.assert_allclose(dense.numpy(), expected_dense, atol=1.0e-6)
        np.testing.assert_allclose(reward.numpy(), expected_dense / env_module._STAGE_REWARD_MAX, atol=1.0e-6)

    def test_lift_reward_uses_gated_max_and_configured_full_height(self):
        count = 3
        zeros_float = self._zeros(count, wp.float32)
        zeros_bool = self._zeros(count, wp.bool)
        gated_max_lift = self._array([0.0, 0.04, 0.08], wp.float32)
        dense = self._zeros(count, wp.float32)
        reward = self._zeros(count, wp.float32)
        wp.launch(
            env_module._advance_episode,
            dim=count,
            inputs=[
                self._zeros(count, wp.int32),
                self._zeros(count, wp.float32),
                self._zeros(count, wp.bool),
                zeros_float,
                zeros_float,
                gated_max_lift,
                0.08,
                zeros_float,
                zeros_float,
                self._finger_contacts([[0, 0, 0, 0, 0]] * count),
                zeros_bool,
                self._zeros((count, 5), wp.bool),
                self._zeros(count, wp.bool),
                self._zeros(count, wp.int32),
                zeros_float,
                zeros_float,
                zeros_float,
                zeros_float,
                self._array([env_module._TASK_PHASE_CARRYING] * count, wp.int32),
                zeros_bool,
                zeros_bool,
                zeros_bool,
                zeros_bool,
                zeros_bool,
                self._zeros(count, wp.float32),
                self._zeros(count, wp.float32),
                self._zeros(count, wp.float32),
                dense,
                reward,
                self._zeros(count, wp.bool),
                self._zeros(count, wp.bool),
                100,
                env_module._REWARD_MODE_DENSE,
                True,
                True,
            ],
            device=self.device,
        )

        np.testing.assert_allclose(dense.numpy(), [2.0, 3.25, 4.0], atol=1.0e-6)

    def test_lift_threshold_must_be_below_lift_height(self):
        with self.assertRaisesRegex(ValueError, "goal_threshold"):
            env_module.GrootNewtonEnvConfig(bottle_lift_height=0.1, goal_threshold=0.1)

    def test_lift_height_progress_uses_positive_delta_z(self):
        current_z = self._array([0.4, 0.5, 0.55, 0.6, 0.7], wp.float32)
        initial_z = self._array([0.5] * 5, wp.float32)
        height = self._zeros(5, wp.float32)
        progress = self._zeros(5, wp.float32)
        wp.launch(
            _evaluate_lift_metrics,
            dim=5,
            inputs=[current_z, initial_z, 0.1, height, progress],
            device=self.device,
        )
        np.testing.assert_allclose(height.numpy(), [0.0, 0.0, 0.05, 0.1, 0.2], atol=1.0e-6)
        np.testing.assert_allclose(progress.numpy(), [0.0, 0.0, 0.5, 1.0, 1.0], atol=1.0e-6)

    def test_opposed_grasp_requires_thumb_and_non_thumb_contact(self):
        finger_contacts = self._finger_contacts(
            (
                (0, 1, 1, 0, 0),
                (1, 1, 0, 0, 0),
                (1, 0, 0, 0, 0),
                (1, 1, 1, 0, 0),
            )
        )
        result = self._zeros(4, wp.bool)
        wp.launch(
            _evaluate_opposed_grasp_rows,
            dim=4,
            inputs=[finger_contacts, 2, result],
            device=self.device,
        )
        np.testing.assert_array_equal(result.numpy(), [False, True, False, True])

        wp.launch(
            _evaluate_opposed_grasp_rows,
            dim=4,
            inputs=[finger_contacts, 3, result],
            device=self.device,
        )
        np.testing.assert_array_equal(result.numpy(), [False, False, False, True])

    def test_pregrasp_geometry_has_continuous_radial_opposition(self):
        radius = 0.0317
        far = (radius + 0.1, 0.0, 0.0)
        thumb = (radius, 0.0, 0.0)

        def pair(angle):
            partner = (radius * np.cos(angle), radius * np.sin(angle), 0.0)
            return (thumb, partner, far, far, far)

        same_side = pair(0.0)
        diagonal = pair(np.pi / 4.0)
        orthogonal = pair(np.pi / 2.0)
        opposed = pair(np.pi)
        single_side = ((radius, 0.0, 0.0), (-radius - 0.1, 0.0, 0.0), far, far, far)
        axial_mismatch = (
            (radius, 0.0, -0.08),
            (-radius, 0.0, 0.08),
            far,
            far,
            far,
        )
        radial_singularity = ((0.0, 0.0, 0.0), (-radius, 0.0, 0.0), far, far, far)
        gaps, opposition, z_score, score = self._launch_pregrasp_geometry(
            (
                same_side,
                diagonal,
                orthogonal,
                opposed,
                single_side,
                axial_mismatch,
                radial_singularity,
                opposed,
            ),
            rotate_last=True,
            return_pairs=True,
        )

        np.testing.assert_allclose(gaps[:4, :2], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            score[:4],
            (0.0, 0.5 * (1.0 - np.cos(np.pi / 4.0)), 0.5, 1.0),
            atol=1.0e-5,
        )
        self.assertTrue(np.all(np.diff(score[:4]) > 0.0))
        self.assertGreater(float(score[4]), 0.0)
        self.assertLess(float(score[4]), 0.2)
        self.assertLess(float(score[5]), 1.0e-5)
        self.assertTrue(np.isfinite(score[6]))
        self.assertEqual(float(score[6]), 0.0)
        np.testing.assert_allclose(score[7], score[3], atol=1.0e-5)
        np.testing.assert_allclose(
            opposition[:4, 0],
            (0.0, 0.5 * (1.0 - np.cos(np.pi / 4.0)), 0.5, 1.0),
            atol=1.0e-5,
        )
        np.testing.assert_allclose(z_score[:4, 0], 1.0, atol=1.0e-6)
        self.assertEqual(opposition.shape, (8, 4))
        self.assertEqual(z_score.shape, (8, 4))

    def test_pregrasp_distance_proximity_is_monotonic_at_v11_scale(self):
        radius = 0.0317
        gap_values = np.asarray((0.0, 0.02, 0.08, 0.12), dtype=np.float32)
        same_side_far = (radius + 0.5, 0.0, 0.0)
        fingertip_rows = tuple(
            (
                (radius + float(gap), 0.0, 0.0),
                (-radius - float(gap), 0.0, 0.0),
                same_side_far,
                same_side_far,
                same_side_far,
            )
            for gap in gap_values
        )

        gaps, score = self._launch_pregrasp_geometry(fingertip_rows)
        expected = 1.0 - np.tanh(gap_values / 0.08)

        self.assertAlmostEqual(env_module._PREGRASP_DISTANCE_SCALE_M, 0.08)
        np.testing.assert_allclose(gaps[:, :2], np.repeat(gap_values[:, None], 2, axis=1), atol=1.0e-6)
        np.testing.assert_allclose(score, expected, atol=1.0e-6)
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertTrue(np.all(np.diff(score) < 0.0))

    def test_reaching_tcp_uses_real_fingertips_instead_of_distal_origins(self):
        offsets = np.asarray(LINKER_L10_FINGERTIP_LOCAL_OFFSETS_M, dtype=np.float32)
        expected_tcp = offsets.mean(axis=0)
        self.assertGreater(float(np.linalg.norm(expected_tcp)), 0.02)

        identity = wp.transform_identity()
        bottle = wp.transform(wp.vec3(*expected_tcp), wp.quat_identity())
        body_q = wp.array([bottle, identity, identity, identity, identity, identity, identity], dtype=wp.transform)
        body_qd = wp.zeros(7, dtype=wp.spatial_vector, device=self.device)
        goal = np.asarray([[*expected_tcp[:2], expected_tcp[2] + 0.1]], dtype=np.float32)
        initial_pose = np.asarray([[*expected_tcp, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        obj_pose = wp.zeros((1, 7), dtype=wp.float32, device=self.device)
        tcp_pose = wp.zeros((1, 7), dtype=wp.float32, device=self.device)
        tcp_to_obj = wp.zeros((1, 3), dtype=wp.float32, device=self.device)
        obj_to_goal = wp.zeros((1, 3), dtype=wp.float32, device=self.device)
        reaching = self._zeros(1, wp.float32)
        scalar_outputs = [self._zeros(1, wp.float32) for _ in range(10)]
        bool_outputs = [self._zeros(1, wp.bool) for _ in range(7)]

        wp.launch(
            env_module._evaluate_transfer_bottle,
            dim=1,
            inputs=[
                body_q,
                body_qd,
                self._array([0], wp.int32),
                0,
                1,
                self._array([2, 3, 4, 5, 6], wp.int32),
                wp.array(offsets, dtype=wp.vec3, device=self.device),
                self._finger_contacts([[0, 0, 0, 0, 0]]),
                wp.array(goal, dtype=wp.float32, device=self.device),
                wp.array(initial_pose, dtype=wp.float32, device=self.device),
                self._array([env_module._TASK_PHASE_APPROACH], wp.int32),
                self._zeros(1, wp.bool),
                self._zeros(env_module.JOINT_ACTION_SIZE, wp.float32),
                self._array([0], wp.int32),
                self._array(list(range(env_module.JOINT_ACTION_SIZE)), wp.int32),
                0.1,
                0.1,
                0.01,
                0.25,
                0.2,
                0.02,
                0.5,
                2,
                obj_pose,
                tcp_pose,
                tcp_to_obj,
                obj_to_goal,
                self._zeros(1, wp.int32),
                *bool_outputs,
                *scalar_outputs[:5],
                reaching,
                *scalar_outputs[5:],
            ],
            device=self.device,
        )

        np.testing.assert_allclose(tcp_pose.numpy()[0, :3], expected_tcp, atol=1.0e-7)
        np.testing.assert_allclose(tcp_to_obj.numpy()[0], 0.0, atol=1.0e-7)
        self.assertAlmostEqual(float(reaching.numpy()[0]), 1.0, places=6)

    def test_partial_contact_reward_stage_truth_table(self):
        dense = self._launch_approach_rewards(
            [0.25] * 9,
            (
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (1, 1, 0, 0, 0),
            ),
            [False, False, False, False, False, False, False, False, True],
            [0.0] * 9,
            finger_contact_any_frame_values=(
                (False, False, False, False, False),
                (False, True, False, False, False),
                (False, True, False, False, False),
                (False, True, False, False, False),
                (True, False, False, False, False),
                # Thumb and index touched in different physics frames.
                (True, True, False, False, False),
                (True, True, False, False, False),
                (True, True, True, True, True),
                (True, True, False, False, False),
            ),
            opposed_grasp_any_frame_values=(False, False, False, False, False, False, True, True, True),
            opposed_grasp_max_consecutive_frames_values=(0, 0, 0, 0, 0, 0, 1, 5, 5),
            non_thumb_anchor_contact_fraction_values=(0.0, 1.0, 1.0, 1.0, 0.0, 0.4, 0.0, 0.0, 1.0),
            non_thumb_missing_thumb_geometry_progress_values=(0.0, 0.0, 0.5, 1.0, 0.0, 0.2, 0.0, 0.0, 1.0),
            thumb_anchor_contact_fraction_values=(0.0, 0.0, 0.0, 0.0, 1.0, 0.8, 0.0, 0.0, 1.0),
            thumb_missing_non_thumb_geometry_progress_values=(0.0, 0.0, 0.0, 0.0, 0.75, 0.6, 0.0, 0.0, 1.0),
        )
        np.testing.assert_allclose(
            dense,
            [0.25, 0.375, 0.9375, 1.50, 1.21875, 1.025, 1.529, 1.60, 1.625],
            atol=1.0e-6,
        )
        self.assertLess(float(dense[1]), float(dense[2]))
        self.assertLess(float(dense[2]), float(dense[3]))
        self.assertGreater(float(dense[5]), 0.525)
        self.assertLess(float(dense[5]), float(dense[6]))
        self.assertLessEqual(float(dense[7]), env_module._UNCONFIRMED_OPPOSED_REWARD_MAX + 1.0e-6)
        self.assertLess(float(dense[7]), float(dense[8]))

    def test_unilateral_guidance_is_contact_gated_monotonic_and_capped(self):
        dense, approach_base, gain, unilateral_reward = self._launch_approach_rewards(
            [0.4] * 6,
            [(0, 0, 0, 0, 0)] * 6,
            [False] * 6,
            [0.5] * 6,
            non_thumb_anchor_contact_fraction_values=(0.0, 0.25, 0.5, 0.5, 1.0, 4.0),
            non_thumb_missing_thumb_geometry_progress_values=(1.0, 0.0, 0.0, 0.25, 1.0, 4.0),
            return_components=True,
        )

        np.testing.assert_allclose(approach_base, 0.575, atol=1.0e-6)
        self.assertEqual(float(dense[0]), float(approach_base[0]))
        self.assertEqual(float(gain[0]), 0.0)
        self.assertEqual(float(unilateral_reward[0]), 0.0)
        self.assertTrue(np.all(np.diff(dense[1:5]) > 0.0))
        np.testing.assert_allclose(dense[4:], env_module._PARTIAL_CONTACT_REWARD_MAX, atol=1.0e-6)
        np.testing.assert_allclose(gain[1:], dense[1:] - approach_base[1:], atol=1.0e-6)
        np.testing.assert_allclose(unilateral_reward[1:], dense[1:], atol=1.0e-6)
        self.assertTrue(np.all(dense <= env_module._PARTIAL_CONTACT_REWARD_MAX + 1.0e-6))

    def test_asynchronous_side_guidance_uses_max_not_sum(self):
        dense = self._launch_approach_rewards(
            [0.5] * 4,
            [(0, 0, 0, 0, 0)] * 4,
            [False] * 4,
            [0.0] * 4,
            finger_contact_any_frame_values=[
                (False, True, False, False, False),
                (True, False, False, False, False),
                (True, True, False, False, False),
                (True, True, False, False, False),
            ],
            non_thumb_anchor_contact_fraction_values=(0.5, 0.0, 0.5, 0.5),
            non_thumb_missing_thumb_geometry_progress_values=(0.4, 0.0, 0.4, 0.4),
            thumb_anchor_contact_fraction_values=(0.0, 0.75, 0.75, 0.75),
            thumb_missing_non_thumb_geometry_progress_values=(0.0, 0.3, 0.3, 0.3),
            opposed_grasp_any_frame_values=(False, False, False, True),
            opposed_grasp_max_consecutive_frames_values=(0, 0, 0, 1),
        )

        self.assertAlmostEqual(float(dense[2]), max(float(dense[0]), float(dense[1])), places=6)
        self.assertLess(float(dense[2]), env_module._PARTIAL_CONTACT_REWARD_MAX)
        self.assertGreater(float(dense[3]), env_module._PARTIAL_CONTACT_REWARD_MAX)

    def test_unilateral_accumulator_uses_missing_side_geometry_and_contact_gate(self):
        contacts = np.asarray(
            (
                ((0, 1, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
                ((0, 1, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
            ),
            dtype=np.int32,
        )
        opposition = np.zeros((3, 4), dtype=np.float32)
        z_score = np.zeros((3, 4), dtype=np.float32)
        opposition[0, 0] = 0.4
        z_score[0, 0] = 0.2
        opposition[1, 0] = 0.4
        z_score[1, 0] = 0.2
        result = self._accumulate_unilateral_guidance(
            contacts,
            thumb_partner_opposition=opposition,
            thumb_partner_z_score=z_score,
        )

        expected_factor = 0.60 + 0.25 * 0.4 + 0.15 * 0.2
        np.testing.assert_allclose(result["cN"], (1.0, 0.0, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(result["GN"], (expected_factor, 0.0, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(result["oN"], (0.4, 0.0, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(result["zN"], (0.2, 0.0, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(result["cT"], (0.0, 1.0, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(result["GT"], (0.0, expected_factor, 0.0), atol=1.0e-6)
        self.assertTrue(np.all(result["GN"] <= result["cN"] + 1.0e-6))
        self.assertTrue(np.all(result["GT"] <= result["cT"] + 1.0e-6))

    def test_unilateral_accumulator_is_invariant_to_physics_frame_count(self):
        contacts_four = np.zeros((4, 1, 5), dtype=np.int32)
        contacts_four[:2, 0, 1] = 1
        contacts_eight = np.zeros((8, 1, 5), dtype=np.int32)
        contacts_eight[:4, 0, 1] = 1

        result_four = self._accumulate_unilateral_guidance(contacts_four)
        result_eight = self._accumulate_unilateral_guidance(contacts_eight)

        for key in ("cN", "GN", "oN", "zN", "cT", "GT"):
            np.testing.assert_allclose(result_four[key], result_eight[key], atol=1.0e-6)
        np.testing.assert_allclose(result_four["cN"], 0.5, atol=1.0e-6)
        np.testing.assert_allclose(result_four["GN"], 0.5, atol=1.0e-6)

    def test_unconfirmed_opposed_reward_is_monotonic_and_capped_below_strict(self):
        streaks = (1, 2, 3, 4, 5, 6, 100)
        count = len(streaks)
        dense = self._launch_approach_rewards(
            [1.0] * count,
            [(0, 0, 0, 0, 0)] * count,
            [False] * count,
            [1.0] * count,
            finger_contact_any_frame_values=[(True, True, True, True, True)] * count,
            opposed_grasp_any_frame_values=[True] * count,
            opposed_grasp_max_consecutive_frames_values=streaks,
        )

        np.testing.assert_allclose(dense[:5], [1.544, 1.558, 1.572, 1.586, 1.60], atol=1.0e-6)
        np.testing.assert_allclose(dense[4:], env_module._UNCONFIRMED_OPPOSED_REWARD_MAX, atol=1.0e-6)
        self.assertTrue(np.all(np.diff(dense) >= 0.0))
        self.assertLess(float(dense.max()), 1.0 + env_module._NON_THUMB_CONTACT_REWARD_PER_FINGER + 0.5)

    def test_pregrasp_reward_is_capped_below_strict_opposition(self):
        dense = self._launch_approach_rewards(
            [1.0, 0.9, 0.0],
            ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (1, 1, 0, 0, 0)),
            [False, False, True],
            [1.0, 1.0, 1.0],
        )

        np.testing.assert_allclose(dense, [1.35, 1.25, 1.625], atol=1.0e-6)
        self.assertLess(float(dense[:2].max()), float(dense[2]))

    def test_release_confirmation_defaults_to_one_control_interval(self):
        self.assertEqual(env_module.GrootNewtonEnvConfig().grasp_confirm_frames, 6)
        self.assertEqual(create_parser().parse_args([]).grasp_confirm_frames, 6)
        self.assertEqual(env_module.GrootNewtonEnvConfig().release_confirm_frames, 6)
        self.assertEqual(create_parser().parse_args([]).release_confirm_frames, 6)

    def test_contact_buffer_capacities_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "contact buffer"):
            env_module.GrootNewtonEnvConfig(triangle_pairs_per_env=0)

    def test_control_step_contact_is_sticky_and_masked_reset_clears_it(self):
        current = self._array([False, True, False], wp.bool)
        sticky = self._zeros(3, wp.bool)
        wp.launch(
            env_module._accumulate_control_step_contact,
            dim=3,
            inputs=[current, sticky],
            device=self.device,
        )
        self._set(current, [True, False, False], wp.bool)
        wp.launch(
            env_module._accumulate_control_step_contact,
            dim=3,
            inputs=[current, sticky],
            device=self.device,
        )
        np.testing.assert_array_equal(sticky.numpy(), [True, True, False])

        mask = self._array([True, False, False], wp.bool)
        wp.launch(
            env_module._clear_control_step_contact,
            dim=3,
            inputs=[mask, sticky],
            device=self.device,
        )
        np.testing.assert_array_equal(sticky.numpy(), [False, True, False])

    def test_masked_episode_reset_clears_unilateral_reward_components(self):
        mask = self._array([True, False], wp.bool)
        episode_step = self._array([7, 8], wp.int32)
        episode_return = self._array([1.0, 2.0], wp.float32)
        success_once = self._array([True, True], wp.bool)
        approach_base = self._array([0.7, 0.8], wp.float32)
        unilateral_gain = self._array([0.2, 0.3], wp.float32)
        unilateral_reward = self._array([0.9, 1.1], wp.float32)
        dense = self._array([0.9, 1.1], wp.float32)
        reward = self._array([0.1, 0.2], wp.float32)
        terminated = self._array([True, True], wp.bool)
        truncated = self._array([True, True], wp.bool)

        wp.launch(
            env_module._reset_episode_arrays,
            dim=2,
            inputs=[
                mask,
                episode_step,
                episode_return,
                success_once,
                approach_base,
                unilateral_gain,
                unilateral_reward,
                dense,
                reward,
                terminated,
                truncated,
            ],
            device=self.device,
        )

        np.testing.assert_array_equal(episode_step.numpy(), (0, 8))
        np.testing.assert_allclose(approach_base.numpy(), (0.0, 0.8), atol=1.0e-6)
        np.testing.assert_allclose(unilateral_gain.numpy(), (0.0, 0.3), atol=1.0e-6)
        np.testing.assert_allclose(unilateral_reward.numpy(), (0.0, 1.1), atol=1.0e-6)

    def test_control_step_contact_topology_accumulates_transients_and_consecutive_opposition(self):
        finger_contacts = self._finger_contacts(np.zeros((2, 5), dtype=np.int32))
        is_grasped = self._zeros(2, wp.bool)
        finger_any = self._zeros((2, 5), wp.bool)
        opposed_any = self._zeros(2, wp.bool)
        current_streak = self._zeros(2, wp.int32)
        max_streak = self._zeros(2, wp.int32)
        gaps = self._zeros((2, 5), wp.float32)
        pair_scores = wp.full((2, 4), value=1.0, dtype=wp.float32, device=self.device)
        non_thumb_fraction = self._zeros(2, wp.float32)
        non_thumb_geometry = self._zeros(2, wp.float32)
        non_thumb_opposition = self._zeros(2, wp.float32)
        non_thumb_z = self._zeros(2, wp.float32)
        thumb_fraction = self._zeros(2, wp.float32)
        thumb_geometry = self._zeros(2, wp.float32)

        def accumulate(contacts, opposed):
            wp.copy(finger_contacts, self._finger_contacts(contacts))
            self._set(is_grasped, opposed, wp.bool)
            wp.launch(
                env_module._accumulate_control_step_contact_topology,
                dim=2,
                inputs=[
                    finger_contacts,
                    is_grasped,
                    gaps,
                    pair_scores,
                    pair_scores,
                    0.25,
                    finger_any,
                    opposed_any,
                    current_streak,
                    max_streak,
                    non_thumb_fraction,
                    non_thumb_geometry,
                    non_thumb_opposition,
                    non_thumb_z,
                    thumb_fraction,
                    thumb_geometry,
                ],
                device=self.device,
            )

        accumulate(((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)), (False, True))
        accumulate(((0, 1, 0, 0, 0), (0, 0, 0, 0, 0)), (True, True))
        accumulate(((0, 0, 1, 0, 0), (1, 0, 0, 0, 0)), (True, False))
        accumulate(((0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), (False, True))

        np.testing.assert_array_equal(
            finger_any.numpy(),
            ((True, True, True, False, False), (True, True, False, False, False)),
        )
        np.testing.assert_array_equal(opposed_any.numpy(), (True, True))
        np.testing.assert_array_equal(current_streak.numpy(), (0, 1))
        np.testing.assert_array_equal(max_streak.numpy(), (2, 2))
        np.testing.assert_allclose(non_thumb_fraction.numpy(), (0.5, 0.25), atol=1.0e-6)
        np.testing.assert_allclose(non_thumb_geometry.numpy(), (0.5, 0.25), atol=1.0e-6)
        np.testing.assert_allclose(non_thumb_opposition.numpy(), (0.5, 0.25), atol=1.0e-6)
        np.testing.assert_allclose(non_thumb_z.numpy(), (0.5, 0.25), atol=1.0e-6)
        np.testing.assert_allclose(thumb_fraction.numpy(), (0.25, 0.25), atol=1.0e-6)
        np.testing.assert_allclose(thumb_geometry.numpy(), (0.25, 0.25), atol=1.0e-6)

        mask = self._array((True, False), wp.bool)
        wp.launch(
            env_module._clear_control_step_contact_topology,
            dim=2,
            inputs=[
                mask,
                finger_any,
                opposed_any,
                current_streak,
                max_streak,
                non_thumb_fraction,
                non_thumb_geometry,
                non_thumb_opposition,
                non_thumb_z,
                thumb_fraction,
                thumb_geometry,
            ],
            device=self.device,
        )
        np.testing.assert_array_equal(
            finger_any.numpy(),
            ((False, False, False, False, False), (True, True, False, False, False)),
        )
        np.testing.assert_array_equal(opposed_any.numpy(), (False, True))
        np.testing.assert_array_equal(current_streak.numpy(), (0, 1))
        np.testing.assert_array_equal(max_streak.numpy(), (0, 2))
        np.testing.assert_allclose(non_thumb_fraction.numpy(), (0.0, 0.25), atol=1.0e-6)
        np.testing.assert_allclose(non_thumb_geometry.numpy(), (0.0, 0.25), atol=1.0e-6)
        np.testing.assert_allclose(non_thumb_opposition.numpy(), (0.0, 0.25), atol=1.0e-6)
        np.testing.assert_allclose(non_thumb_z.numpy(), (0.0, 0.25), atol=1.0e-6)
        np.testing.assert_allclose(thumb_fraction.numpy(), (0.0, 0.25), atol=1.0e-6)
        np.testing.assert_allclose(thumb_geometry.numpy(), (0.0, 0.25), atol=1.0e-6)

    def test_evaluate_and_info_expose_five_finger_contact_counts(self):
        class TaskInfoStub:
            def __getattr__(self, name):
                return name

        stub = TaskInfoStub()
        finger_contacts = object()
        stub._finger_contacts = finger_contacts
        stub.evaluate_warp = lambda: env_module.GrootNewtonEnv.evaluate_warp(stub)

        evaluation = stub.evaluate_warp()
        info = env_module.GrootNewtonEnv._info_warp(stub)

        self.assertIs(evaluation["finger_contact_counts"], finger_contacts)
        self.assertIs(info["finger_contact_counts"], finger_contacts)
        self.assertEqual(
            evaluation["finger_contact_any_frame_this_control_step"],
            "_finger_contact_any_frame_this_control_step",
        )
        self.assertEqual(
            evaluation["opposed_grasp_any_frame_this_control_step"],
            "_opposed_grasp_any_frame_this_control_step",
        )
        self.assertEqual(
            evaluation["opposed_grasp_max_consecutive_physics_frames_this_control_step"],
            "_opposed_grasp_max_consecutive_frames_this_control_step",
        )
        self.assertEqual(evaluation["thumb_partner_opposition"], "_thumb_partner_opposition")
        self.assertEqual(evaluation["thumb_partner_z_score"], "_thumb_partner_z_score")
        self.assertEqual(
            evaluation["non_thumb_anchor_contact_fraction_this_control_step"],
            "_non_thumb_anchor_contact_fraction_this_control_step",
        )
        self.assertEqual(
            evaluation["non_thumb_missing_thumb_geometry_progress_this_control_step"],
            "_non_thumb_missing_thumb_geometry_progress_this_control_step",
        )
        self.assertEqual(
            evaluation["non_thumb_guidance_opposition_progress_this_control_step"],
            "_non_thumb_guidance_opposition_progress_this_control_step",
        )
        self.assertEqual(
            evaluation["non_thumb_guidance_z_progress_this_control_step"],
            "_non_thumb_guidance_z_progress_this_control_step",
        )
        self.assertEqual(
            evaluation["thumb_anchor_contact_fraction_this_control_step"],
            "_thumb_anchor_contact_fraction_this_control_step",
        )
        self.assertEqual(
            evaluation["thumb_missing_non_thumb_geometry_progress_this_control_step"],
            "_thumb_missing_non_thumb_geometry_progress_this_control_step",
        )
        self.assertEqual(info["reward_components"]["approach_base"], "_approach_base_reward")
        self.assertEqual(info["reward_components"]["unilateral_guidance_gain"], "_unilateral_guidance_gain")
        self.assertEqual(info["reward_components"]["unilateral_contact_reward"], "_unilateral_contact_reward")
        self.assertEqual(evaluation["grasp_support_gap_frames"], "_grasp_support_gap_frames")
        self.assertEqual(evaluation["current_lift_height"], "_current_lift_height")
        self.assertIs(evaluation["current_lift_height"], evaluation["lift_height"])

    def test_control_step_collision_diagnostics_include_intermediate_frames(self):
        count = self._array([4], wp.int32)
        frame_max = self._zeros(1, wp.int32)
        overflow_frames = self._zeros(1, wp.int32)
        overflow_excess = self._zeros(1, wp.int32)

        for observed in (4, 7, 6):
            self._set(count, [observed], wp.int32)
            wp.launch(
                env_module._accumulate_control_step_collision_buffer,
                dim=1,
                inputs=[count, 5, frame_max, overflow_frames, overflow_excess],
                device=self.device,
            )

        self.assertEqual(int(frame_max.numpy()[0]), 7)
        self.assertEqual(int(overflow_frames.numpy()[0]), 2)
        self.assertEqual(int(overflow_excess.numpy()[0]), 3)

    def _phase_arrays(self, count: int = 1):
        return {
            "obj_pose": wp.zeros((count, 7), dtype=wp.float32, device=self.device),
            "initial_pose": wp.zeros((count, 7), dtype=wp.float32, device=self.device),
            "is_grasped": self._zeros(count, wp.bool),
            "has_contact": self._zeros(count, wp.bool),
            "pose_valid": self._zeros(count, wp.bool),
            "is_static": self._zeros(count, wp.bool),
            "current_lift": self._zeros(count, wp.float32),
            "physical_max_lift": self._zeros(count, wp.float32),
            "phase": self._zeros(count, wp.int32),
            "grasp_frames": self._zeros(count, wp.int32),
            "support_gap_frames": self._zeros(count, wp.int32),
            "gap_frames": self._zeros(count, wp.int32),
            "settle_frames": self._zeros(count, wp.int32),
            "grasp_confirmed": self._zeros(count, wp.bool),
            "transport_started": self._zeros(count, wp.bool),
            "reached_lift": self._zeros(count, wp.bool),
            "release_armed": self._zeros(count, wp.bool),
            "released": self._zeros(count, wp.bool),
            "early_release": self._zeros(count, wp.bool),
            "max_z": self._zeros(count, wp.float32),
            "max_lift": self._zeros(count, wp.float32),
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
                arrays["current_lift"],
                0.1,
                0.005,
                0.01,
                grasp_frames,
                release_frames,
                settle_frames,
                arrays["phase"],
                arrays["grasp_frames"],
                arrays["support_gap_frames"],
                arrays["gap_frames"],
                arrays["settle_frames"],
                arrays["grasp_confirmed"],
                arrays["transport_started"],
                arrays["reached_lift"],
                arrays["release_armed"],
                arrays["released"],
                arrays["early_release"],
                arrays["max_z"],
                arrays["max_lift"],
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
        self._set(arrays["current_lift"], [0.1], wp.float32)
        self._advance_phase(arrays)
        self.assertTrue(arrays["transport_started"].numpy()[0])
        self.assertTrue(arrays["reached_lift"].numpy()[0])
        self.assertAlmostEqual(float(arrays["max_lift"].numpy()[0]), 0.1, places=6)

        self._set(arrays["pose_valid"], [True], wp.bool)
        self._advance_phase(arrays)
        self.assertTrue(arrays["release_armed"].numpy()[0])
        self._set(arrays["is_grasped"], [False], wp.bool)
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
        pose[0, 2] = 0.02
        wp.copy(arrays["obj_pose"], wp.array(pose, dtype=wp.float32, device=self.device))
        self._set(arrays["current_lift"], [0.02], wp.float32)
        self._advance_phase(arrays, grasp_frames=1, release_frames=1)
        self._set(arrays["is_grasped"], [False], wp.bool)
        self._set(arrays["has_contact"], [False], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=1)
        self.assertTrue(arrays["early_release"].numpy()[0])
        self.assertTrue(arrays["fail"].numpy()[0])
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_FAIL)

    def test_pre_transport_contact_loss_fails_after_debounce(self):
        arrays = self._phase_arrays()
        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=2)
        pose = np.zeros((1, 7), dtype=np.float32)
        pose[0, 0] = 0.02
        wp.copy(arrays["obj_pose"], wp.array(pose, dtype=wp.float32, device=self.device))
        self._advance_phase(arrays, grasp_frames=1, release_frames=2)
        self.assertFalse(arrays["transport_started"].numpy()[0])

        self._set(arrays["is_grasped"], [False], wp.bool)
        self._set(arrays["has_contact"], [False], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=2)
        self.assertFalse(arrays["fail"].numpy()[0])
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_CARRYING)
        self._advance_phase(arrays, grasp_frames=1, release_frames=2)
        self.assertTrue(arrays["fail"].numpy()[0])
        self.assertTrue(arrays["early_release"].numpy()[0])
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_FAIL)

    def test_grasp_confirmation_frame_accumulates_contacted_lift(self):
        arrays = self._phase_arrays()
        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        self._set(arrays["current_lift"], [0.04], wp.float32)

        self._advance_phase(arrays, grasp_frames=1)

        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_CARRYING)
        self.assertTrue(arrays["transport_started"].numpy()[0])
        self.assertAlmostEqual(float(arrays["max_lift"].numpy()[0]), 0.04, places=6)

    def test_opposed_grasp_confirms_on_sixth_frame_and_interruption_restarts(self):
        arrays = self._phase_arrays()
        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        for _ in range(5):
            self._advance_phase(arrays, grasp_frames=6)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_APPROACH)
        self.assertEqual(int(arrays["grasp_frames"].numpy()[0]), 5)

        self._set(arrays["is_grasped"], [False], wp.bool)
        self._advance_phase(arrays, grasp_frames=6)
        self.assertEqual(int(arrays["grasp_frames"].numpy()[0]), 0)
        self.assertFalse(arrays["grasp_confirmed"].numpy()[0])

        self._set(arrays["is_grasped"], [True], wp.bool)
        for _ in range(5):
            self._advance_phase(arrays, grasp_frames=6)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_APPROACH)
        self._advance_phase(arrays, grasp_frames=6)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_CARRYING)
        self.assertTrue(arrays["grasp_confirmed"].numpy()[0])
        dense = self._launch_approach_rewards(
            [0.0],
            [(1, 1, 0, 0, 0)],
            [True],
            [0.0],
            task_phase_values=arrays["phase"].numpy(),
        )
        np.testing.assert_allclose(dense, [2.0], atol=1.0e-6)

    def test_degraded_but_contacted_carry_accumulates_lift_before_transport(self):
        arrays = self._phase_arrays()
        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        self._advance_phase(arrays, grasp_frames=1)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_CARRYING)

        self._set(arrays["is_grasped"], [False], wp.bool)
        self._set(arrays["current_lift"], [0.009], wp.float32)
        self._set(arrays["pose_valid"], [True], wp.bool)
        for _ in range(5):
            self._advance_phase(arrays, grasp_frames=6)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_CARRYING)
        self.assertEqual(int(arrays["support_gap_frames"].numpy()[0]), 0)
        self.assertAlmostEqual(float(arrays["max_lift"].numpy()[0]), 0.009, places=6)
        self.assertFalse(arrays["transport_started"].numpy()[0])
        self.assertFalse(arrays["release_armed"].numpy()[0])

        self._advance_phase(arrays, grasp_frames=6)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_CARRYING)
        self.assertEqual(int(arrays["support_gap_frames"].numpy()[0]), 0)
        self.assertTrue(arrays["grasp_confirmed"].numpy()[0])
        self.assertFalse(arrays["release_armed"].numpy()[0])
        self.assertFalse(arrays["fail"].numpy()[0])

    def test_transport_allows_degraded_contact_to_reach_lift_and_arm_release(self):
        arrays = self._phase_arrays()
        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        self._set(arrays["current_lift"], [0.02], wp.float32)
        self._advance_phase(arrays, grasp_frames=1)
        self.assertTrue(arrays["transport_started"].numpy()[0])
        self.assertAlmostEqual(float(arrays["max_lift"].numpy()[0]), 0.02, places=6)

        self._set(arrays["is_grasped"], [False], wp.bool)
        self._set(arrays["current_lift"], [0.1], wp.float32)
        self._set(arrays["pose_valid"], [True], wp.bool)
        for _ in range(8):
            self._advance_phase(arrays, grasp_frames=6)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_CARRYING)
        self.assertAlmostEqual(float(arrays["max_lift"].numpy()[0]), 0.1, places=6)
        self.assertTrue(arrays["reached_lift"].numpy()[0])
        self.assertTrue(arrays["release_armed"].numpy()[0])

    def test_max_lift_height_only_accumulates_during_contact(self):
        arrays = self._phase_arrays()
        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        self._advance_phase(arrays, grasp_frames=1)
        self._set(arrays["current_lift"], [0.004], wp.float32)
        self._advance_phase(arrays)
        self.assertAlmostEqual(float(arrays["max_lift"].numpy()[0]), 0.004, places=6)

        self._set(arrays["is_grasped"], [False], wp.bool)
        self._set(arrays["has_contact"], [False], wp.bool)
        self._set(arrays["current_lift"], [0.08], wp.float32)
        self._advance_phase(arrays, release_frames=2)
        self.assertAlmostEqual(float(arrays["max_lift"].numpy()[0]), 0.004, places=6)
        self.assertFalse(arrays["reached_lift"].numpy()[0])
        self.assertFalse(arrays["transport_started"].numpy()[0])
        self.assertFalse(arrays["release_armed"].numpy()[0])

    def test_default_release_confirmation_boundary_and_contact_flicker(self):
        arrays = self._phase_arrays()
        release_frames = env_module.GrootNewtonEnvConfig().release_confirm_frames
        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=release_frames)

        self._set(arrays["is_grasped"], [False], wp.bool)
        self._set(arrays["has_contact"], [False], wp.bool)
        for _ in range(release_frames - 1):
            self._advance_phase(arrays, grasp_frames=1, release_frames=release_frames)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_CARRYING)

        self._set(arrays["is_grasped"], [True], wp.bool)
        self._set(arrays["has_contact"], [True], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=release_frames)
        self.assertEqual(int(arrays["gap_frames"].numpy()[0]), 0)

        self._set(arrays["is_grasped"], [False], wp.bool)
        self._set(arrays["has_contact"], [False], wp.bool)
        for _ in range(release_frames - 1):
            self._advance_phase(arrays, grasp_frames=1, release_frames=release_frames)
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_CARRYING)
        self._advance_phase(arrays, grasp_frames=1, release_frames=release_frames)
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
        self._set(arrays["current_lift"], [0.1], wp.float32)
        self._advance_phase(arrays, grasp_frames=1, release_frames=1)
        self.assertTrue(arrays["reached_lift"].numpy()[0])
        self._set(arrays["is_grasped"], [False], wp.bool)
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
        self._set(arrays["current_lift"], [0.1], wp.float32)
        self._advance_phase(arrays, grasp_frames=1, release_frames=2)
        self.assertTrue(arrays["reached_lift"].numpy()[0])
        self.assertFalse(arrays["release_armed"].numpy()[0])

        self._set(arrays["is_grasped"], [False], wp.bool)
        self._set(arrays["has_contact"], [False], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=2)
        self._set(arrays["pose_valid"], [True], wp.bool)
        self._advance_phase(arrays, grasp_frames=1, release_frames=2)
        self.assertTrue(arrays["fail"].numpy()[0])
        self.assertEqual(arrays["phase"].numpy()[0], env_module._TASK_PHASE_FAIL)

    def test_masked_task_reset(self):
        arrays = self._phase_arrays(count=2)
        self._set(arrays["current_lift"], [0.02, 0.03], wp.float32)
        self._set(arrays["physical_max_lift"], [0.06, 0.07], wp.float32)
        self._set(arrays["max_lift"], [0.04, 0.05], wp.float32)
        self._set(arrays["phase"], [env_module._TASK_PHASE_SUCCESS, env_module._TASK_PHASE_FAIL], wp.int32)
        self._set(arrays["support_gap_frames"], [4, 5], wp.int32)
        self._set(arrays["success"], [True, False], wp.bool)
        self._set(arrays["fail"], [False, True], wp.bool)
        mask = self._array([True, False], wp.bool)
        wp.launch(
            env_module._reset_transfer_task,
            dim=2,
            inputs=[
                mask,
                arrays["current_lift"],
                arrays["physical_max_lift"],
                arrays["max_lift"],
                arrays["phase"],
                arrays["grasp_frames"],
                arrays["support_gap_frames"],
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
        np.testing.assert_allclose(arrays["current_lift"].numpy(), [0.0, 0.03], atol=1.0e-6)
        np.testing.assert_allclose(arrays["physical_max_lift"].numpy(), [0.0, 0.07], atol=1.0e-6)
        np.testing.assert_allclose(arrays["max_lift"].numpy(), [0.0, 0.05], atol=1.0e-6)
        np.testing.assert_array_equal(arrays["support_gap_frames"].numpy(), [0, 5])

        finger_contacts = self._finger_contacts(((1, 2, 3, 4, 5), (5, 4, 3, 2, 1)))
        wp.launch(
            env_module._clear_finger_contact_rows,
            dim=(2, 5),
            inputs=[mask, finger_contacts],
            device=self.device,
        )
        np.testing.assert_array_equal(
            finger_contacts.numpy(),
            ((0, 0, 0, 0, 0), (5, 4, 3, 2, 1)),
        )


if __name__ == "__main__":
    unittest.main()
