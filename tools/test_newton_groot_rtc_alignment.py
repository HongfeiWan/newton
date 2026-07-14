#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the node0 GR00T conventions used by Newton."""

# ruff: noqa: SLF001

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import run_newton_groot_rtc_control as groot_runtime


class TestNode0GrootAlignment(unittest.TestCase):
    def test_rot6d_uses_first_two_rows(self) -> None:
        rotation = np.asarray(
            (
                (0.0, -1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )

        rot6d = groot_runtime._rotmat_to_rot6d(rotation)

        np.testing.assert_array_equal(rot6d, (0.0, -1.0, 0.0, 1.0, 0.0, 0.0))
        np.testing.assert_allclose(groot_runtime._rot6d_to_rotmat(rot6d), rotation, atol=1.0e-7)

    def test_node0_eef_transform_is_scene_a_policy_b(self) -> None:
        controller = groot_runtime.NewtonPolicyController.__new__(groot_runtime.NewtonPolicyController)
        controller.eef_transform_mode = "node0_fixed"
        controller._eef_frame_calibrated = True
        controller._state_to_genesis_transform = groot_runtime._rigid_transform_matrix(
            groot_runtime.NODE0_STATE_TO_GENESIS_TRANSLATION_XYZ,
            groot_runtime.NODE0_STATE_TO_GENESIS_QUATERNION_XYZW,
        )
        controller._eef_offset_transform = groot_runtime._rigid_transform_matrix(
            groot_runtime.NODE0_STATE_TO_GENESIS_EEF_OFFSET_TRANSLATION_XYZ,
            groot_runtime.NODE0_STATE_TO_GENESIS_EEF_OFFSET_QUATERNION_XYZW,
        )
        controller._genesis_to_world_transform = groot_runtime._rigid_transform_matrix(
            (-0.003, 0.003, 0.16),
            (0.0, 0.0, 0.0, 1.0),
        )
        policy_pose = np.eye(4, dtype=np.float64)
        policy_pose[:3, 3] = (0.41, -0.12, 0.73)
        policy_pose[:3, :3] = np.asarray(
            groot_runtime.quat_xyzw_to_matrix((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
        )

        world_pose = controller._policy_pose_to_world_pose(policy_pose)
        expected = (
            controller._genesis_to_world_transform
            @ controller._state_to_genesis_transform
            @ policy_pose
            @ controller._eef_offset_transform
        )

        np.testing.assert_allclose(world_pose, expected, atol=1.0e-12)
        np.testing.assert_allclose(
            controller._world_pose_to_policy_pose(world_pose),
            policy_pose,
            atol=1.0e-12,
        )

    def test_node0_recorded_action_maps_to_recorded_genesis_command(self) -> None:
        policy_eef_9d = np.asarray(
            (
                -0.31398898363113403,
                -0.35250094532966614,
                0.11842889338731766,
                0.1266147345304489,
                0.9485205411911011,
                -0.2903059124946594,
                -0.9918148517608643,
                0.11618837714195251,
                -0.052948661148548126,
            ),
            dtype=np.float64,
        )
        state_to_genesis = groot_runtime._rigid_transform_matrix(
            groot_runtime.NODE0_STATE_TO_GENESIS_TRANSLATION_XYZ,
            groot_runtime.NODE0_STATE_TO_GENESIS_QUATERNION_XYZW,
        )
        eef_offset = groot_runtime._rigid_transform_matrix(
            groot_runtime.NODE0_STATE_TO_GENESIS_EEF_OFFSET_TRANSLATION_XYZ,
            groot_runtime.NODE0_STATE_TO_GENESIS_EEF_OFFSET_QUATERNION_XYZW,
        )

        command_pose = state_to_genesis @ groot_runtime._eef_9d_to_pose(policy_eef_9d) @ eef_offset

        np.testing.assert_allclose(
            command_pose[:3, 3],
            (-0.3829947270248274, 0.15444763057334623, 0.6148848766495125),
            atol=1.0e-7,
        )
        expected_rotation = groot_runtime.quat_xyzw_to_matrix(
            (-0.6543440978913473, 0.0923386986444439, 0.7413283715375681, -0.11721609036675296)
        )
        np.testing.assert_allclose(command_pose[:3, :3], expected_rotation, atol=1.0e-7)

    def test_sim_image_defaults_match_node0(self) -> None:
        args = groot_runtime.create_parser().parse_args([])

        self.assertEqual((args.d455_render_width, args.d455_render_height), (1280, 800))
        self.assertEqual((args.d405_width, args.d405_height), (640, 480))
        self.assertTrue(args.sim_ego_roi)
        self.assertEqual(args.eef_transform_mode, "node0_fixed")
        self.assertTrue(args.async_policy)
        self.assertTrue(args.capture_graph)
        self.assertFalse(args.enforce_bottle_above_scene_collision)
        self.assertEqual(args.camera_preview_fps, 15.0)
        self.assertIsNone(args.viewer_fifo_preview)
        self.assertEqual(
            (args.viewer_fifo_preview_width, args.viewer_fifo_preview_height),
            (1600, 720),
        )
        self.assertEqual(args.viewer_fifo_preview_input_width, 320)

    def test_action_row_rejects_nonfinite_policy_values(self) -> None:
        action = {"arm_joint_target": np.asarray(((0.0, np.nan),), dtype=np.float32)}

        with self.assertRaisesRegex(ValueError, "finite"):
            groot_runtime.NewtonPolicyController._action_row(action, "arm_joint_target", 0)

    def test_l10_friction_defaults_to_vr_value(self) -> None:
        args = groot_runtime.create_parser().parse_args([])

        self.assertEqual(args.l10_friction, 3.0)

    def test_l10_contact_gap_defaults_to_point_one_mm(self) -> None:
        args = groot_runtime.create_parser().parse_args([])

        self.assertEqual(args.l10_contact_gap, 1.0e-4)

    def test_scene_physics_config_supplies_contact_defaults(self) -> None:
        args = groot_runtime.create_parser().parse_args([])

        self.assertEqual(args.scene_physics_config, groot_runtime.scene_runtime.DEFAULT_SCENE_PHYSICS_CONFIG)
        self.assertEqual(args.scene_friction, 0.8)
        self.assertEqual(args.dynamic_bottle_friction, 0.45)
        self.assertEqual(args.dynamic_bottle_contact_gap, 5.0e-4)
        self.assertEqual(args.l10_contact_margin, -1.0e-3)
        self.assertTrue(args.hydroelastic_contacts)

    def test_cli_overrides_scene_physics_config(self) -> None:
        args = groot_runtime.create_parser().parse_args(
            ["--l10-friction", "2.5", "--dynamic-bottle-contact-gap", "0.0002"]
        )

        self.assertEqual(args.l10_friction, 2.5)
        self.assertEqual(args.dynamic_bottle_contact_gap, 2.0e-4)

    def test_node0_ego_view_crops_before_frame_tap_resize(self) -> None:
        image = np.full((800, 1280, 3), (255, 0, 0), dtype=np.uint8)
        image[320:720, 320:960] = (0, 127, 0)

        processed = groot_runtime._node0_ego_view_preprocess(
            image,
            zoom=2.0,
            center_x=0.5,
            center_y=0.65,
        )

        self.assertEqual(processed.shape, (180, 320, 3))
        np.testing.assert_array_equal(processed, np.full((180, 320, 3), (0, 127, 0), dtype=np.uint8))

    def test_async_replan_skips_elapsed_actions(self) -> None:
        self.assertEqual(
            groot_runtime._elapsed_action_steps(start_s=1.0, current_s=1.96, action_dt_s=0.1),
            10,
        )
        self.assertEqual(
            groot_runtime._elapsed_action_steps(start_s=2.0, current_s=1.0, action_dt_s=0.1),
            0,
        )

    def test_viewer_fifo_preview_writes_rgb_frame(self) -> None:
        frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

        class FakeFrame:
            shape = frame.shape

            def numpy(self) -> np.ndarray:
                return frame

        class FakeWindow:
            def set_size(self, width: int, height: int) -> None:
                self.size = (width, height)

        class FakeViewer:
            def __init__(self) -> None:
                self.renderer = type("Renderer", (), {"window": FakeWindow()})()

            def get_frame(self, *, target_image, render_ui: bool):
                self.target_image = target_image
                self.render_ui = render_ui
                return FakeFrame()

        viewer = FakeViewer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "viewer.rgb"
            preview = groot_runtime.ViewerFifoPreview(viewer, path, width=3, height=2, fps=15.0)
            preview.capture(viewer)
            preview.close()

            self.assertEqual(viewer.renderer.window.size, (3, 2))
            self.assertFalse(viewer.render_ui)
            self.assertEqual(path.read_bytes(), frame.tobytes())

    def test_viewer_fifo_preview_composes_exact_model_inputs(self) -> None:
        scene = np.full((4, 4, 3), 7, dtype=np.uint8)
        ego = np.full((1, 2, 3), (10, 20, 30), dtype=np.uint8)
        wrist = np.full((1, 2, 3), (40, 50, 60), dtype=np.uint8)

        preview = groot_runtime.ViewerFifoPreview.__new__(groot_runtime.ViewerFifoPreview)
        preview.width = 6
        preview.height = 4
        preview.input_width = 2
        preview.scene_width = 4

        composed = preview._compose_frame(scene, {"ego_view": ego, "wrist_view": wrist})

        np.testing.assert_array_equal(composed[:, :4], scene)
        self.assertTrue(np.any(np.all(composed[:, 4:] == ego[0, 0], axis=2)))
        self.assertTrue(np.any(np.all(composed[:, 4:] == wrist[0, 0], axis=2)))


if __name__ == "__main__":
    unittest.main()
