# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end regression tests for the GR00T row-first rotation contract."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from teleop_stack.datasets.groot_lerobot import (
    ACTION_KEY,
    EGO_KEY,
    STATE_KEY,
    WRIST_KEY,
    GrootLeRobotWindowDataset,
)
from teleop_stack.envs import groot_newton_env
from teleop_stack.policies import groot_diffusion_policy

try:
    import pyarrow as pa
    import pyarrow.parquet as parquet
except ImportError:
    pa = None
    parquet = None

try:
    import torch
except ImportError:
    torch = None


ROW_FIRST_NAMES = ("r00", "r01", "r02", "r10", "r11", "r12")


def _axis_angle_rotation(axis: tuple[float, float, float], angle_rad: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    axis_array /= np.linalg.norm(axis_array)
    x, y, z = axis_array
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)
    return np.eye(3) + math.sin(angle_rad) * skew + (1.0 - math.cos(angle_rad)) * (skew @ skew)


def _exporter_eef_9d(position: tuple[float, float, float], rotation: np.ndarray) -> np.ndarray:
    """Represent the canonical exporter boundary without importing another repo."""
    return np.concatenate((np.asarray(position), np.asarray(rotation)[:2, :].reshape(6))).astype(np.float32)


def _state_row(arm: np.ndarray, eef_9d: np.ndarray, hand: np.ndarray) -> np.ndarray:
    return np.concatenate((arm, eef_9d[:3], eef_9d[3:9], hand)).astype(np.float32)


def _action_row(eef_9d: np.ndarray, hand: np.ndarray) -> np.ndarray:
    return np.concatenate((eef_9d, hand)).astype(np.float32)


def _write_contract_dataset(root: Path, state: np.ndarray, action: np.ndarray) -> None:
    state_names = (
        [f"arm_joint_pos.joint{index}" for index in range(7)]
        + ["arm_eef_pos.x", "arm_eef_pos.y", "arm_eef_pos.z"]
        + [f"arm_eef_rot6d.{name}" for name in ROW_FIRST_NAMES]
        + [f"hand_joint_pos.joint{index}" for index in range(10)]
    )
    action_names = (
        ["arm_eef_pos_target.x", "arm_eef_pos_target.y", "arm_eef_pos_target.z"]
        + [f"arm_eef_rot6d_target.{name}" for name in ROW_FIRST_NAMES]
        + [f"hand_joint_target.joint{index}" for index in range(10)]
    )
    info = {
        "total_episodes": 1,
        "total_frames": int(state.shape[0]),
        "fps": 10,
        "features": {
            STATE_KEY: {"dtype": "float32", "shape": [26], "names": state_names},
            ACTION_KEY: {"dtype": "float32", "shape": [19], "names": action_names},
            EGO_KEY: {"dtype": "video", "shape": [800, 1280, 3]},
            WRIST_KEY: {"dtype": "video", "shape": [480, 640, 3]},
        },
        "teleop_stack": {
            "rot6d_convention": "row_major_first_two_rows_[r00,r01,r02,r10,r11,r12]",
            "arm_action_semantics": "absolute_flange_pose_xyz_rot6d_target_in_state_frame",
            "dp_action_semantics": "absolute_flange_pose_and_hand_from_observation_state_same_frame",
            "dp_action_source_slices": {"eef": [7, 16], "hand": [16, 26]},
            "dp_action_provenance": {"mode": "state_copy"},
            "rot6d_raw_truth_migration": {
                "schema": "teleop_stack.rot6d_physical_truth_migration.v1",
            },
        },
    }
    episode = {
        "episode_index": 0,
        "length": int(state.shape[0]),
        "teleop_stack_metadata": {
            "rot6d_raw_truth_migration": {
                "schema": "teleop_stack.rot6d_physical_truth_migration.v1",
            }
        },
    }
    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta_dir / "episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
    table = pa.table(
        {
            STATE_KEY: pa.array(state.tolist(), type=pa.list_(pa.float32(), 26)),
            ACTION_KEY: pa.array(action.tolist(), type=pa.list_(pa.float32(), 19)),
        }
    )
    parquet.write_table(table, data_dir / "episode_000000.parquet")


class _UnusedModule(torch.nn.Module if torch is not None else object):
    def forward(self, *_args, **_kwargs):
        raise AssertionError("The contract test must not run image encoders or the denoiser")


@unittest.skipUnless(torch is not None and parquet is not None, "requires PyTorch and PyArrow")
class TestGrootRotationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rotation = _axis_angle_rotation((1.0, -2.0, 3.0), 0.73)
        cls.other_rotation = _axis_angle_rotation((-3.0, 1.0, 2.0), -0.41)
        cls.eef = _exporter_eef_9d((0.31, -0.17, 0.62), cls.rotation)
        other_eef = _exporter_eef_9d((0.29, -0.11, 0.68), cls.other_rotation)
        arm = np.linspace(-0.6, 0.6, 7, dtype=np.float32)
        hand = np.linspace(0.02, 0.2, 10, dtype=np.float32)
        cls.state = np.stack((_state_row(arm, cls.eef, hand), _state_row(arm + 0.05, other_eef, hand + 0.01)))
        cls.action = np.stack((_action_row(cls.eef, hand), _action_row(other_eef, hand + 0.01)))

    def _dataset_sample(self) -> tuple[GrootLeRobotWindowDataset, dict[str, object]]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        _write_contract_dataset(root, self.state, self.action)
        dataset = GrootLeRobotWindowDataset(root, obs_horizon=1, pred_horizon=1, preprocess_ego=False)
        self.addCleanup(dataset.close)
        dataset._read_rgb_frames = lambda _episode, _key, indices: np.zeros((len(indices), 2, 2, 3), dtype=np.uint8)
        return dataset, dataset[0]

    def test_dataset_rejects_missing_state_target_contract(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        _write_contract_dataset(root, self.state, self.action)
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        del info["teleop_stack"]["dp_action_semantics"]
        info_path.write_text(json.dumps(info), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "state-target DP contract"):
            GrootLeRobotWindowDataset(root, obs_horizon=1, pred_horizon=1, preprocess_ego=False)

    def _dp_round_trip(self, dataset: GrootLeRobotWindowDataset, sample: dict[str, object]):
        config = groot_diffusion_policy.GrootDiffusionPolicyConfig(
            obs_horizon=1,
            pred_horizon=1,
            camera_feature_dim=8,
            state_feature_dim=8,
            denoiser_width=8,
        )
        with (
            mock.patch.object(groot_diffusion_policy, "_CameraEncoder", side_effect=lambda *_args: _UnusedModule()),
            mock.patch.object(
                groot_diffusion_policy,
                "_TemporalDenoiser",
                side_effect=lambda *_args: _UnusedModule(),
            ),
        ):
            policy = groot_diffusion_policy.GrootDiffusionPolicy(
                state_min=dataset.stats.state_min,
                state_max=dataset.stats.state_max,
                action_min=dataset.stats.action_min,
                action_max=dataset.stats.action_max,
                config=config,
            )

        state = sample[STATE_KEY].float()
        action = sample[ACTION_KEY].float()
        normalized_state = policy._normalize(state, policy.state_min, policy.state_max)
        normalized_action = policy._normalize(action, policy.action_min, policy.action_max)
        decoded_state = policy._unnormalize(normalized_state, policy.state_min, policy.state_max)
        decoded_action = policy._unnormalize(normalized_action, policy.action_min, policy.action_max)
        return decoded_state, decoded_action

    def test_exporter_dataset_dp_and_ik_preserve_row_first_rotation(self) -> None:
        self.assertFalse(np.allclose(self.rotation, np.eye(3)))
        self.assertFalse(np.allclose(self.rotation, self.rotation.T))
        row_first = self.rotation[:2, :].reshape(6)
        column_first = self.rotation[:, :2].T.reshape(6)
        self.assertGreater(float(np.linalg.norm(row_first - column_first)), 0.5)

        dataset, sample = self._dataset_sample()
        self.assertEqual(
            tuple(name.rsplit(".", 1)[-1] for name in dataset.info["features"][STATE_KEY]["names"][10:16]),
            ROW_FIRST_NAMES,
        )
        self.assertEqual(
            tuple(name.rsplit(".", 1)[-1] for name in dataset.info["features"][ACTION_KEY]["names"][3:9]),
            ROW_FIRST_NAMES,
        )
        np.testing.assert_allclose(sample[STATE_KEY][0, 10:16].numpy(), row_first, atol=1.0e-7)
        np.testing.assert_allclose(sample[ACTION_KEY][0, 3:9].numpy(), row_first, atol=1.0e-7)
        np.testing.assert_allclose(sample[STATE_KEY][0, 7:10].numpy(), self.eef[:3], atol=1.0e-7)
        np.testing.assert_allclose(sample[ACTION_KEY][0, 0:3].numpy(), self.eef[:3], atol=1.0e-7)

        decoded_state, decoded_action = self._dp_round_trip(dataset, sample)
        np.testing.assert_allclose(decoded_state[0, 7:10].numpy(), self.eef[:3], atol=1.0e-6)
        np.testing.assert_allclose(decoded_action[0, 0:3].numpy(), self.eef[:3], atol=1.0e-6)
        np.testing.assert_allclose(decoded_state[0, 10:16].numpy(), row_first, atol=1.0e-6)
        np.testing.assert_allclose(decoded_action[0, 3:9].numpy(), row_first, atol=1.0e-6)

        fallback = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        ik_target = groot_newton_env.GrootNewtonEnv._rotation_6d_to_matrix_torch(decoded_action[:, 3:9], fallback)
        np.testing.assert_allclose(ik_target.numpy(), self.rotation[None], atol=1.0e-6)

    def test_state_frame_action_is_the_rotation_target_used_by_ik(self) -> None:
        dataset, sample = self._dataset_sample()
        _, decoded_action = self._dp_round_trip(dataset, sample)
        env = groot_newton_env.GrootNewtonEnv.__new__(groot_newton_env.GrootNewtonEnv)
        env._action = decoded_action.clone()
        env.state_0 = SimpleNamespace(joint_q=torch.zeros(17, dtype=torch.float32))
        env.control = SimpleNamespace(
            joint_target_q=torch.zeros(17, dtype=torch.float32),
            joint_target_qd=torch.zeros(17, dtype=torch.float32),
        )
        env._arm_q_indices_torch = torch.arange(7).reshape(1, 7)
        env._arm_qd_indices_torch = torch.arange(7).reshape(1, 7)
        env._hand_q_indices_torch = torch.arange(7, 17).reshape(1, 10)
        env._hand_qd_indices_torch = torch.arange(7, 17).reshape(1, 10)
        env._arm_lower_torch = torch.full((1, 7), -10.0)
        env._arm_upper_torch = torch.full((1, 7), 10.0)
        env._hand_lower_torch = torch.full((1, 10), -10.0)
        env._hand_upper_torch = torch.full((1, 10), 10.0)
        env._ik_identity_torch = torch.eye(7).unsqueeze(0)
        env._hand_joint_pos = torch.zeros((1, 10), dtype=torch.float32)
        env.control_dt = 0.1
        env.config = SimpleNamespace(
            ik_iterations=1,
            ik_max_task_step_m=10.0,
            ik_max_rotation_step_rad=10.0,
            ik_position_weight=1.0,
            ik_orientation_weight=1.0,
            ik_damping_lambda=0.01,
            ik_max_joint_step_rad=10.0,
            hand_max_joint_step_rad=10.0,
        )

        identity = torch.eye(3).unsqueeze(0)
        jacobian = torch.zeros((1, 6, 7), dtype=torch.float32)
        jacobian[0, 0:3, 0:3] = torch.eye(3)
        jacobian[0, 3:6, 3:6] = torch.eye(3)
        env._eef_fk_jacobian_torch = lambda _q: (torch.zeros((1, 3)), identity, jacobian)

        with mock.patch.object(groot_newton_env.wp, "to_torch", side_effect=lambda value: value):
            env._apply_eef_pose_action_torch_fp32(torch)

        rotation = torch.as_tensor(self.rotation, dtype=torch.float32).unsqueeze(0)
        orientation_error = 0.5 * (
            torch.linalg.cross(identity[:, :, 0], rotation[:, :, 0], dim=-1)
            + torch.linalg.cross(identity[:, :, 1], rotation[:, :, 1], dim=-1)
            + torch.linalg.cross(identity[:, :, 2], rotation[:, :, 2], dim=-1)
        )
        expected_task_error = torch.cat((decoded_action[:, 0:3], orientation_error), dim=-1)
        expected_step = expected_task_error / (1.0 + env.config.ik_damping_lambda**2)
        np.testing.assert_allclose(
            env.control.joint_target_q[:6].numpy(),
            expected_step[0, :6].numpy(),
            atol=1.0e-6,
        )

    def test_hold_action_encodes_rows_and_decodes_back_to_fk_rotation(self) -> None:
        env = groot_newton_env.GrootNewtonEnv.__new__(groot_newton_env.GrootNewtonEnv)
        env.control_mode = "pd_eef_pose_abs"
        env.num_envs = 1
        env._action = torch.zeros((1, 19), dtype=torch.float32)
        env.state_0 = SimpleNamespace(joint_q=torch.zeros(7, dtype=torch.float32))
        env._arm_q_indices_torch = torch.arange(7).reshape(1, 7)
        env._hand_joint_pos = torch.zeros((1, 10), dtype=torch.float32)
        rotation = torch.as_tensor(self.rotation, dtype=torch.float32).unsqueeze(0)
        position = torch.as_tensor(((0.31, -0.17, 0.62),), dtype=torch.float32)
        env._eef_fk_jacobian_torch = lambda _q: (position, rotation, torch.zeros((1, 6, 7)))

        with mock.patch.object(groot_newton_env.wp, "to_torch", side_effect=lambda value: value):
            action = env.hold_action().clone()

        np.testing.assert_allclose(action[0, 0:3].numpy(), position[0].numpy(), atol=1.0e-6)
        np.testing.assert_allclose(action[0, 3:9].numpy(), self.rotation[:2, :].reshape(6), atol=1.0e-6)
        decoded = env._rotation_6d_to_matrix_torch(action[:, 3:9], torch.eye(3).unsqueeze(0))
        np.testing.assert_allclose(decoded.numpy(), self.rotation[None], atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
