# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the GR00T residual PPO policy components."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from teleop_stack.policies.groot_residual_ppo import (
    GrootResidualActorCritic,
    GrootResidualActorCriticConfig,
    compose_residual_action,
    compute_gae,
    normalize_physical_action,
    validate_world_from_action_rotation,
)
from tools.train_newton_groot_residual_ppo import (
    _ACTION_DIAGNOSTIC_NAMES,
    _ACTOR_CONDITION_SOURCE,
    _BASE_ACTION_HORIZON,
    _BASE_ACTION_MODE,
    _CRITIC_PRIVILEGED_SOURCE,
    _EEF_POSITION_FRAME,
    _HAND_TARGET_SEMANTICS,
    _PRIVILEGED_STATE_DIM,
    _R_WORLD_FROM_ACTION,
    _RESET_CACHE_POLICY,
    _RESIDUAL_POSITION_FRAME,
    _RESUME_CACHE_POLICY,
    _RESUME_TRAIN_ARG_NAMES,
    _REWARD_CONTRACT_VERSION,
    _TRAINING_CONTRACT_VERSION,
    _action_diagnostics,
    _action_position_to_world,
    _control_step_contact_topology,
    _evaluate,
    _EvaluationQuota,
    _event_flags,
    _finger_contact_topology,
    _hand_residual_scale_contract,
    _is_better_eval,
    _is_better_return,
    _normalize_current_state,
    _partial_contact_reward_stages,
    _PerLaneActionChunkCache,
    _ppo_update,
    _pre_action_task_flags,
    _prepare_policy_step,
    _privileged_task_state,
    _resolve_hand_residual_scales,
    _reward_v13_signals,
    _rollout_diagnostic_metrics,
    _validate_args,
    _validate_frozen_dp_training_contract,
    _validate_resume_train_args,
    _validate_resume_training_contract,
    create_parser,
)


def _axis_angle_matrix(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    x, y, z = direction
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _rpy_matrix(*, roll: float, pitch: float, yaw: float) -> np.ndarray:
    sin_roll, cos_roll = math.sin(roll), math.cos(roll)
    sin_pitch, cos_pitch = math.sin(pitch), math.cos(pitch)
    sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)
    rotation_x = np.asarray(((1.0, 0.0, 0.0), (0.0, cos_roll, -sin_roll), (0.0, sin_roll, cos_roll)))
    rotation_y = np.asarray(((cos_pitch, 0.0, sin_pitch), (0.0, 1.0, 0.0), (-sin_pitch, 0.0, cos_pitch)))
    rotation_z = np.asarray(((cos_yaw, -sin_yaw, 0.0), (sin_yaw, cos_yaw, 0.0), (0.0, 0.0, 1.0)))
    return rotation_z @ rotation_y @ rotation_x


def _training_contract_payload(**overrides) -> dict[str, object]:
    payload = {
        "training_contract_version": _TRAINING_CONTRACT_VERSION,
        "reward_contract_version": _REWARD_CONTRACT_VERSION,
        "base_action_mode": _BASE_ACTION_MODE,
        "base_action_horizon": _BASE_ACTION_HORIZON,
        "actor_condition_source": _ACTOR_CONDITION_SOURCE,
        "critic_privileged_source": _CRITIC_PRIVILEGED_SOURCE,
        "reset_cache_policy": _RESET_CACHE_POLICY,
        "resume_cache_policy": _RESUME_CACHE_POLICY,
        "eef_position_frame": _EEF_POSITION_FRAME,
        "R_world_from_action": _R_WORLD_FROM_ACTION,
        "residual_position_frame": _RESIDUAL_POSITION_FRAME,
        "hand_target_semantics": _HAND_TARGET_SEMANTICS,
        "hand_residual_scale": {
            "mode": "uniform",
            "default_scale_normalized": 0.1,
            "thumb_hand_indices": [0, 1, 9],
            "thumb_latent_indices": [6, 7, 15],
            "thumb_joint_names": ["thumb_cmc_pitch", "thumb_cmc_yaw", "thumb_cmc_roll"],
            "effective_scale_normalized": [0.1] * 10,
        },
    }
    payload.update(overrides)
    return payload


def _physical_bounds() -> tuple[torch.Tensor, torch.Tensor]:
    minimum = torch.tensor([-1.0, -2.0, -3.0] + [-1.0] * 6 + [-0.5] * 10)
    maximum = torch.tensor([1.0, 2.0, 3.0] + [1.0] * 6 + [0.5] * 10)
    return minimum, maximum


def _contract_observation(condition: torch.Tensor) -> dict[str, torch.Tensor]:
    state = torch.zeros(condition.shape[0], 2, 26)
    state[:, :, 10] = 1.0
    state[:, :, 14] = 1.0
    return {
        "condition": condition,
        "observation.state": state,
        "observation.finger_root_load": torch.zeros(condition.shape[0], 2, 5),
    }


def _contract_chunk(condition: torch.Tensor, *, horizon: int = _BASE_ACTION_HORIZON) -> torch.Tensor:
    batch_size = condition.shape[0]
    rows = torch.arange(horizon, dtype=torch.float32)[None, :]
    offsets = condition[:, :1].float() * 0.01
    chunk = torch.zeros(batch_size, horizon, 19)
    chunk[:, :, 0] = -0.7 + 0.1 * rows + offsets
    chunk[:, :, 1] = -1.4 + 0.2 * rows + offsets
    chunk[:, :, 2] = -2.1 + 0.3 * rows + offsets
    chunk[:, :, 3] = 1.0
    chunk[:, :, 7] = 1.0
    chunk[:, :, 9:19] = (-0.35 + 0.1 * rows + offsets)[:, :, None]
    return chunk


class _ContractDP:
    def __init__(self, *, pred_horizon: int = _BASE_ACTION_HORIZON) -> None:
        self.config = SimpleNamespace(pred_horizon=pred_horizon, action_dim=19)
        self.state_min = torch.full((26,), -2.0)
        self.state_max = torch.full((26,), 2.0)
        self.encoded_conditions: list[torch.Tensor] = []

    def encode_observation(self, observation):
        condition = observation["condition"].float()
        self.encoded_conditions.append(condition.clone())
        return condition

    def predict_action_from_condition(self, condition, *_args, **_kwargs):
        return _contract_chunk(condition, horizon=self.config.pred_horizon)


class _FakeEvalDP:
    def __init__(self) -> None:
        self.config = SimpleNamespace(pred_horizon=_BASE_ACTION_HORIZON, action_dim=19)
        self.state_min = torch.full((26,), -2.0)
        self.state_max = torch.full((26,), 2.0)

    def encode_observation(self, observation):
        return torch.zeros(observation["lane"].shape[0], 4)

    def predict_action_from_condition(self, condition, *_args, **_kwargs):
        base = torch.zeros(condition.shape[0], 19)
        base[:, 3] = 1.0
        base[:, 7] = 1.0
        return base[:, None].expand(-1, _BASE_ACTION_HORIZON, -1).clone()


class _FakeEvalActor:
    def __init__(self) -> None:
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def act(self, policy_input, _privileged, **_kwargs):
        batch_size = policy_input.shape[0]
        raw_latent = torch.zeros(batch_size, 16)
        zeros = torch.zeros(batch_size)
        return raw_latent, zeros, zeros, zeros


class _FakeMultiLaneEvalEnv:
    def __init__(self, episode_lengths: tuple[int, ...]) -> None:
        self.episode_lengths = torch.tensor(episode_lengths)
        self.num_envs = len(episode_lengths)
        self.unwrapped = self
        self.config = SimpleNamespace(
            bottle_lift_height=0.1,
            bottle_min_xy_displacement=0.1,
            final_z_threshold=0.1,
            final_orientation_threshold_rad=0.1,
            max_episode_steps=max(episode_lengths),
            hand_max_joint_step_rad=0.08,
        )
        self.frames_per_action = 6
        self.steps = torch.zeros(self.num_envs, dtype=torch.int64)
        self.returns = torch.zeros(self.num_envs)
        self.full_reset_count = 0
        self.partial_reset_masks: list[tuple[bool, ...]] = []

    def _observation(self):
        state = torch.zeros(self.num_envs, 2, 26)
        state[:, :, 10] = 1.0
        state[:, :, 14] = 1.0
        return {
            "lane": torch.arange(self.num_envs),
            "observation.state": state,
            "observation.finger_root_load": torch.zeros(self.num_envs, 2, 5),
        }

    def _info(self, done):
        lane = torch.arange(self.num_envs)
        success = done & (lane == 1)
        fail = done & (lane == 2)
        false = torch.zeros(self.num_envs, dtype=torch.bool)
        has_stepped = self.steps > 0
        finger_contact_counts = torch.zeros(self.num_envs, 5, dtype=torch.int64)
        finger_contact_counts[:, 0] = has_stepped & ((lane == 0) | (lane == 2))
        finger_contact_counts[:, 1] = has_stepped & ((lane == 1) | (lane == 2))
        thumb_contact = finger_contact_counts[:, 0] > 0
        non_thumb_contact = (finger_contact_counts[:, 1:] > 0).any(dim=-1)
        has_contact = thumb_contact | non_thumb_contact
        opposed_grasp = thumb_contact & non_thumb_contact
        finger_contact_any_frame = finger_contact_counts > 0
        opposed_grasp_any_frame = opposed_grasp.clone()
        opposed_grasp_max_consecutive_frames = 2 * opposed_grasp.long()
        phase = torch.where(success, 3, torch.where(fail, 4, 0)).long()
        lift_height = 0.01 * self.steps.float()
        pregrasp_score = torch.clamp(0.25 * self.steps.float(), 0.0, 1.0)
        reaching_reward = torch.clamp(0.1 * self.steps.float(), 0.0, 1.0)
        non_thumb_contact_fraction = non_thumb_contact.float()
        thumb_contact_fraction = thumb_contact.float()
        finger_surface_gap = torch.full((self.num_envs, 5), 0.08)
        return {
            "task_phase": phase,
            "has_hand_contact": has_contact,
            "had_hand_contact_this_control_step": has_contact,
            "touching_finger_count": (finger_contact_counts > 0).sum(dim=-1),
            "finger_contact_counts": finger_contact_counts,
            "finger_contact_any_frame_this_control_step": finger_contact_any_frame,
            "non_thumb_anchor_contact_fraction_this_control_step": non_thumb_contact_fraction,
            "non_thumb_missing_thumb_geometry_progress_this_control_step": 0.25 * non_thumb_contact_fraction,
            "thumb_anchor_contact_fraction_this_control_step": thumb_contact_fraction,
            "thumb_missing_non_thumb_geometry_progress_this_control_step": 0.5 * thumb_contact_fraction,
            "non_thumb_guidance_opposition_progress_this_control_step": 0.8 * non_thumb_contact_fraction,
            "non_thumb_guidance_z_progress_this_control_step": 0.6 * non_thumb_contact_fraction,
            "opposed_grasp_any_frame_this_control_step": opposed_grasp_any_frame,
            "opposed_grasp_max_consecutive_physics_frames_this_control_step": (opposed_grasp_max_consecutive_frames),
            "is_grasped": opposed_grasp,
            "grasp_confirmed": false.clone(),
            "transport_started": false.clone(),
            "is_lifted": false.clone(),
            "release_armed": false.clone(),
            "release_ready": false.clone(),
            "released": false.clone(),
            "early_release": false.clone(),
            "is_obj_placed": false.clone(),
            "is_obj_static": false.clone(),
            "is_robot_static": false.clone(),
            "success": success,
            "fail": fail,
            "xy_displacement": torch.zeros(self.num_envs),
            "final_z_error": torch.zeros(self.num_envs),
            "orientation_error": torch.zeros(self.num_envs),
            "finger_surface_gap": finger_surface_gap,
            "opposed_pregrasp_score": pregrasp_score,
            "current_lift_height": lift_height,
            "physical_max_lift_height": lift_height,
            "max_contacted_carry_lift_height": lift_height,
            "max_lift_height": lift_height,
            "reward_components": {
                "reaching": reaching_reward,
                "opposed_pregrasp": pregrasp_score,
                "approach_base": torch.clamp(reaching_reward + 0.35 * pregrasp_score, max=1.35),
                "unilateral_guidance_gain": 0.05 * non_thumb_contact_fraction,
                "unilateral_contact_reward": 1.0 * non_thumb_contact_fraction,
            },
            "episode": {
                "length": self.steps.clone(),
                "return": self.returns.clone(),
                "success_once": success,
                "fail_at_end": fail,
            },
        }

    def reset(self, *, world_mask=None):
        if world_mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool)
            self.full_reset_count += 1
        else:
            mask = world_mask.bool().clone()
            self.partial_reset_masks.append(tuple(bool(value) for value in mask))
        self.steps[mask] = 0
        self.returns[mask] = 0.0
        return self._observation(), self._info(torch.zeros(self.num_envs, dtype=torch.bool))

    def step(self, _action):
        reward = torch.arange(1, self.num_envs + 1, dtype=torch.float32)
        self.steps += 1
        self.returns += reward
        done = self.steps >= self.episode_lengths
        return self._observation(), reward, done, torch.zeros_like(done), self._info(done)


@unittest.skipUnless(torch is not None, "requires PyTorch")
class TestGrootResidualPPO(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = GrootResidualActorCriticConfig(condition_dim=11, hidden_dim=32)
        self.policy = GrootResidualActorCritic(self.config)
        self.policy_input = torch.randn(5, self.config.policy_input_dim)
        self.privileged = torch.randn(5, self.config.privileged_dim)

    def _synthetic_rollout(self, rollout_steps: int = 3, num_envs: int = 2):
        policy_input = torch.randn(rollout_steps, num_envs, self.config.policy_input_dim)
        privileged = torch.randn(rollout_steps, num_envs, self.config.privileged_dim)
        with torch.no_grad():
            raw_latent, old_log_prob, _, old_value = self.policy.act(policy_input, privileged)
        advantages = torch.tensor(((1.0, -0.5), (0.25, 0.75), (-1.0, 0.4)))
        returns = old_value + advantages
        rollout = {
            "policy_input": policy_input,
            "privileged": privileged,
            "raw_latent": raw_latent,
            "log_prob": old_log_prob,
            "value": old_value,
        }
        return rollout, advantages, returns

    def test_policy_input_dimension_includes_all_cached_and_live_features(self) -> None:
        self.assertEqual(self.config.policy_input_dim, 11 + 19 + 26 + 8 + 26 + 5)
        v13_config = GrootResidualActorCriticConfig(condition_dim=11, privileged_dim=_PRIVILEGED_STATE_DIM)
        self.assertEqual(v13_config.policy_input_dim, self.config.policy_input_dim)
        self.assertEqual(v13_config.privileged_dim, 25)
        with self.assertRaisesRegex(ValueError, "state_delta_dim must equal"):
            GrootResidualActorCriticConfig(condition_dim=11, state_delta_dim=25)
        with self.assertRaisesRegex(ValueError, "finger_root_load_dim must be 5"):
            GrootResidualActorCriticConfig(condition_dim=11, finger_root_load_dim=4)

    def test_reward_v13_guidance_is_normalized_and_appended_only_to_critic(self) -> None:
        zeros = torch.zeros(2)
        false = torch.zeros(2, dtype=torch.bool)
        finger_surface_gap = torch.full((2, 5), 0.2)
        finger_surface_gap[:, 0] = torch.tensor((0.0, 0.08))
        info = {
            "task_phase": torch.tensor((0, 1)),
            **{
                name: false.clone()
                for name in (
                    "has_hand_contact",
                    "is_grasped",
                    "grasp_confirmed",
                    "transport_started",
                    "is_lifted",
                    "release_armed",
                    "release_ready",
                    "released",
                    "is_obj_placed",
                    "is_obj_static",
                )
            },
            "touching_finger_count": zeros.clone(),
            "xy_displacement": zeros.clone(),
            "current_lift_height": zeros.clone(),
            "orientation_error": zeros.clone(),
            "episode": {"length": torch.tensor((0, 50))},
            "finger_surface_gap": finger_surface_gap,
            "non_thumb_anchor_contact_fraction_this_control_step": torch.tensor((1.0, 0.5)),
            "non_thumb_missing_thumb_geometry_progress_this_control_step": torch.tensor((0.2, 0.4)),
            "thumb_anchor_contact_fraction_this_control_step": torch.tensor((0.0, 0.25)),
            "thumb_missing_non_thumb_geometry_progress_this_control_step": torch.tensor((0.0, 0.1)),
            "non_thumb_guidance_opposition_progress_this_control_step": torch.tensor((0.8, 0.25)),
            "non_thumb_guidance_z_progress_this_control_step": torch.tensor((0.6, 0.1)),
            "reward_components": {
                "reaching": torch.tensor((0.5, 1.0)),
                "opposed_pregrasp": torch.tensor((0.4, 1.0)),
                "unilateral_guidance_gain": torch.tensor((0.1, 0.2)),
                "unilateral_contact_reward": torch.tensor((0.9, 1.1)),
            },
        }
        signals = _reward_v13_signals(info)
        torch.testing.assert_close(signals["reward_v13_r0"], torch.tensor((0.64, 1.35)))
        torch.testing.assert_close(signals["reward_v13_cN"], torch.tensor((1.0, 0.5)))
        torch.testing.assert_close(signals["reward_v13_GN"], torch.tensor((0.2, 0.4)))
        torch.testing.assert_close(signals["reward_v13_guidance_opposition"], torch.tensor((0.8, 0.5)))
        torch.testing.assert_close(signals["reward_v13_guidance_z"], torch.tensor((0.6, 0.2)))
        torch.testing.assert_close(
            signals["reward_v13_thumb_proximity"],
            torch.tensor((1.0, 1.0 - math.tanh(1.0))),
        )

        config = SimpleNamespace(
            bottle_min_xy_displacement=0.1,
            bottle_lift_height=0.1,
            final_orientation_threshold_rad=0.1,
            max_episode_steps=100,
        )
        privileged = _privileged_task_state(info, config)
        self.assertEqual(privileged.shape, (2, _PRIVILEGED_STATE_DIM))
        torch.testing.assert_close(
            privileged[:, -5:],
            torch.stack(
                (
                    signals["reward_v13_r0"],
                    signals["reward_v13_cN"],
                    signals["reward_v13_cT"],
                    signals["reward_v13_GN"],
                    signals["reward_v13_GT"],
                ),
                dim=-1,
            ),
        )

    def test_policy_step_accepts_direct_live_load_and_requires_temporal_state(self) -> None:
        minimum, maximum = _physical_bounds()
        condition = torch.tensor(((1.0, 10.0),))
        observation = _contract_observation(condition)
        observation["observation.state"][:, -1, 0] = 1.0
        observation["observation.finger_root_load"] = torch.tensor(((0.1, 0.2, 0.3, 0.4, 0.5),))
        prepared = _prepare_policy_step(
            observation,
            _ContractDP(),
            scheduler=None,
            cache=_PerLaneActionChunkCache(num_lanes=1, action_dim=19, device="cpu"),
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
        )
        torch.testing.assert_close(prepared.normalized_state_delta[:, 0], torch.tensor((0.5,)))
        torch.testing.assert_close(
            prepared.finger_root_load,
            torch.tensor(((0.1, 0.2, 0.3, 0.4, 0.5),)),
        )

        missing_load = _contract_observation(condition)
        del missing_load["observation.finger_root_load"]
        with self.assertRaisesRegex(ValueError, "requires live 'observation.finger_root_load'"):
            _prepare_policy_step(
                missing_load,
                _ContractDP(),
                scheduler=None,
                cache=_PerLaneActionChunkCache(num_lanes=1, action_dim=19, device="cpu"),
                action_min=minimum,
                action_max=maximum,
                inference_steps=1,
                use_bfloat16=False,
            )

        one_frame = _contract_observation(condition)
        one_frame["observation.state"] = one_frame["observation.state"][:, -1:]
        with self.assertRaisesRegex(ValueError, "at least two state-history frames"):
            _prepare_policy_step(
                one_frame,
                _ContractDP(),
                scheduler=None,
                cache=_PerLaneActionChunkCache(num_lanes=1, action_dim=19, device="cpu"),
                action_min=minimum,
                action_max=maximum,
                inference_steps=1,
                use_bfloat16=False,
            )

    def test_cached_base_executes_rows_zero_through_seven_before_replan(self) -> None:
        minimum, maximum = _physical_bounds()
        condition = torch.tensor(((2.0, 20.0),))
        observation = _contract_observation(condition)
        frozen_dp = _ContractDP()
        cache = _PerLaneActionChunkCache(num_lanes=1, action_dim=19, device="cpu")

        for expected_row in range(_BASE_ACTION_HORIZON):
            prepared = _prepare_policy_step(
                observation,
                frozen_dp,
                scheduler=None,
                cache=cache,
                action_min=minimum,
                action_max=maximum,
                inference_steps=1,
                use_bfloat16=False,
            )
            self.assertEqual(int(prepared.row_index), expected_row)
            torch.testing.assert_close(
                prepared.base_action,
                _contract_chunk(condition)[:, expected_row],
            )
            torch.testing.assert_close(prepared.policy_input[:, : condition.shape[1]], condition)
            self.assertEqual(bool(prepared.replanned), expected_row == 0)
            cache.advance(torch.ones(1, dtype=torch.bool))

        self.assertFalse(bool(cache.valid[0]))
        self.assertEqual(len(frozen_dp.encoded_conditions), 1)
        replanned = _prepare_policy_step(
            observation,
            frozen_dp,
            scheduler=None,
            cache=cache,
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
        )
        self.assertEqual(int(replanned.row_index), 0)
        self.assertTrue(bool(replanned.replanned))
        self.assertEqual(len(frozen_dp.encoded_conditions), 2)

    def test_bootstrap_peek_reuses_same_row_without_extra_plan(self) -> None:
        minimum, maximum = _physical_bounds()
        condition = torch.tensor(((5.0, 50.0),))
        observation = _contract_observation(condition)
        frozen_dp = _ContractDP()
        cache = _PerLaneActionChunkCache(num_lanes=1, action_dim=19, device="cpu")

        bootstrap = _prepare_policy_step(
            observation,
            frozen_dp,
            scheduler=None,
            cache=cache,
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
        )
        first_control_step = _prepare_policy_step(
            observation,
            frozen_dp,
            scheduler=None,
            cache=cache,
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
        )

        self.assertEqual(len(frozen_dp.encoded_conditions), 1)
        self.assertEqual(int(bootstrap.row_index), 0)
        self.assertEqual(int(first_control_step.row_index), 0)
        self.assertTrue(bool(bootstrap.replanned))
        self.assertTrue(bool(first_control_step.replanned))
        torch.testing.assert_close(first_control_step.policy_input, bootstrap.policy_input)
        torch.testing.assert_close(first_control_step.base_action, bootstrap.base_action)

        cache.advance(torch.ones(1, dtype=torch.bool))
        next_step = _prepare_policy_step(
            observation,
            frozen_dp,
            scheduler=None,
            cache=cache,
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
        )
        self.assertEqual(int(next_step.row_index), 1)
        self.assertFalse(bool(next_step.replanned))

    def test_cached_chunk_actor_input_refreshes_all_live_features_each_step(self) -> None:
        minimum, maximum = _physical_bounds()
        condition = torch.tensor(((3.0, 30.0),))
        frozen_dp = _ContractDP()
        cache = _PerLaneActionChunkCache(num_lanes=1, action_dim=19, device="cpu")
        observation = _contract_observation(condition)
        observation["observation.state"][:, -1, 0] = -1.0
        observation["observation.finger_root_load"][:, -1] = 0.1

        row_zero = _prepare_policy_step(
            observation,
            frozen_dp,
            scheduler=None,
            cache=cache,
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
        )
        cache.advance(torch.ones(1, dtype=torch.bool))
        next_observation = _contract_observation(torch.tensor(((99.0, 990.0),)))
        next_observation["observation.state"][:, -1, 0] = 1.0
        next_observation["observation.finger_root_load"][:, -1] = 0.8
        row_one = _prepare_policy_step(
            next_observation,
            frozen_dp,
            scheduler=None,
            cache=cache,
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
        )

        state_start = condition.shape[1] + 19
        row_start = state_start + 26
        delta_start = row_start + _BASE_ACTION_HORIZON
        load_start = delta_start + 26
        self.assertEqual(len(frozen_dp.encoded_conditions), 1)
        torch.testing.assert_close(row_zero.policy_input[:, :2], condition)
        torch.testing.assert_close(row_one.policy_input[:, :2], condition)
        torch.testing.assert_close(row_zero.policy_input[:, state_start], torch.tensor((-0.5,)))
        torch.testing.assert_close(row_one.policy_input[:, state_start], torch.tensor((0.5,)))
        torch.testing.assert_close(
            row_zero.policy_input[:, row_start:delta_start],
            torch.nn.functional.one_hot(torch.tensor((0,)), 8).float(),
        )
        torch.testing.assert_close(
            row_one.policy_input[:, row_start:delta_start],
            torch.nn.functional.one_hot(torch.tensor((1,)), 8).float(),
        )
        torch.testing.assert_close(row_zero.policy_input[:, delta_start], torch.tensor((-0.5,)))
        torch.testing.assert_close(row_one.policy_input[:, delta_start], torch.tensor((0.5,)))
        torch.testing.assert_close(row_zero.policy_input[:, load_start:], torch.full((1, 5), 0.1))
        torch.testing.assert_close(row_one.policy_input[:, load_start:], torch.full((1, 5), 0.8))

    def test_cache_reset_replans_only_reset_lane_and_preserves_other_lane_row(self) -> None:
        minimum, maximum = _physical_bounds()
        frozen_dp = _ContractDP()
        cache = _PerLaneActionChunkCache(num_lanes=2, action_dim=19, device="cpu")
        initial_condition = torch.tensor(((1.0, 10.0), (2.0, 20.0)))
        initial_observation = _contract_observation(initial_condition)

        first = _prepare_policy_step(
            initial_observation,
            frozen_dp,
            scheduler=None,
            cache=cache,
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
        )
        torch.testing.assert_close(first.row_index, torch.tensor((0, 0)))
        cache.advance(torch.ones(2, dtype=torch.bool))
        cache.invalidate(torch.tensor((True, False)))

        reset_condition = torch.tensor(((11.0, 110.0), (99.0, 990.0)))
        reset_observation = _contract_observation(reset_condition)
        reset_observation["observation.state"][1, -1, 0] = 1.0
        reset_observation["observation.finger_root_load"][0, -1] = 0.2
        reset_observation["observation.finger_root_load"][1, -1] = 0.9
        after_reset = _prepare_policy_step(
            reset_observation,
            frozen_dp,
            scheduler=None,
            cache=cache,
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
        )

        torch.testing.assert_close(after_reset.row_index, torch.tensor((0, 1)))
        torch.testing.assert_close(after_reset.replanned, torch.tensor((True, False)))
        torch.testing.assert_close(after_reset.policy_input[0, :2], reset_condition[0])
        torch.testing.assert_close(after_reset.policy_input[1, :2], initial_condition[1])
        torch.testing.assert_close(after_reset.base_action[0], _contract_chunk(reset_condition[:1])[0, 0])
        torch.testing.assert_close(after_reset.base_action[1], _contract_chunk(initial_condition[1:])[0, 1])
        torch.testing.assert_close(after_reset.normalized_state_delta[:, 0], torch.tensor((0.0, 0.5)))
        torch.testing.assert_close(
            after_reset.finger_root_load,
            torch.tensor(((0.2,) * 5, (0.9,) * 5)),
        )
        self.assertEqual([tuple(value[:, 0].tolist()) for value in frozen_dp.encoded_conditions], [(1.0, 2.0), (11.0,)])

        cache.advance(torch.ones(2, dtype=torch.bool))
        continued = _prepare_policy_step(
            reset_observation,
            frozen_dp,
            scheduler=None,
            cache=cache,
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
        )
        torch.testing.assert_close(continued.row_index, torch.tensor((1, 2)))

    def test_inactive_eval_lanes_hold_without_planning_or_advancing(self) -> None:
        minimum, maximum = _physical_bounds()
        condition = torch.tensor(((1.0, 10.0), (2.0, 20.0), (3.0, 30.0)))
        observation = _contract_observation(condition)
        observation["observation.state"][:, -1, 7:10] = torch.tensor(
            ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9))
        )
        frozen_dp = _ContractDP()
        cache = _PerLaneActionChunkCache(num_lanes=3, action_dim=19, device="cpu")
        active = torch.tensor((True, False, False))

        prepared = _prepare_policy_step(
            observation,
            frozen_dp,
            scheduler=None,
            cache=cache,
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
            eligible=active,
        )

        self.assertEqual([tuple(value[:, 0].tolist()) for value in frozen_dp.encoded_conditions], [(1.0,)])
        torch.testing.assert_close(prepared.row_index, torch.tensor((0, -1, -1)))
        expected_hold = torch.cat(
            (
                observation["observation.state"][:, -1, 7:16],
                observation["observation.state"][:, -1, 16:26],
            ),
            dim=-1,
        )
        torch.testing.assert_close(prepared.base_action[1:], expected_hold[1:])
        cache.advance(active)
        torch.testing.assert_close(cache.row, torch.tensor((1, 0, 0)))
        torch.testing.assert_close(cache.valid, torch.tensor((True, False, False)))

    def test_refill_is_lane_local_and_validation_failure_is_atomic(self) -> None:
        cache = _PerLaneActionChunkCache(num_lanes=3, action_dim=19, device="cpu")
        selected_lanes = torch.tensor((0, 2))
        selected_condition = torch.tensor(((1.0, 10.0), (3.0, 30.0)))
        selected_chunk = _contract_chunk(selected_condition)
        cache.refill(selected_lanes, selected_chunk, selected_condition)

        torch.testing.assert_close(cache.valid, torch.tensor((True, False, True)))
        torch.testing.assert_close(cache.plan_count, torch.tensor((1, 0, 1)))
        torch.testing.assert_close(cache.chunk[0], selected_chunk[0])
        torch.testing.assert_close(cache.chunk[1], torch.zeros_like(cache.chunk[1]))
        torch.testing.assert_close(cache.chunk[2], selected_chunk[1])

        snapshot = {
            "chunk": cache.chunk.clone(),
            "condition": cache.condition.clone(),
            "row": cache.row.clone(),
            "valid": cache.valid.clone(),
            "fresh": cache.fresh.clone(),
            "plan_count": cache.plan_count.clone(),
        }
        with self.assertRaisesRegex(ValueError, "pred_horizon >= 8"):
            cache.refill(
                torch.tensor((0,)),
                _contract_chunk(torch.tensor(((9.0, 90.0),)), horizon=7),
                torch.tensor(((9.0, 90.0),)),
            )
        with self.assertRaisesRegex(ValueError, "condition width changed"):
            cache.refill(
                torch.tensor((2,)),
                _contract_chunk(torch.tensor(((9.0, 90.0),))),
                torch.tensor(((9.0, 90.0, 900.0),)),
            )
        for name, expected in snapshot.items():
            torch.testing.assert_close(getattr(cache, name), expected)

    def test_normalized_base_and_residual_composition_use_same_cached_row(self) -> None:
        minimum, maximum = _physical_bounds()
        condition = torch.tensor(((4.0, 40.0),))
        chunk = _contract_chunk(condition)
        cache = _PerLaneActionChunkCache(num_lanes=1, action_dim=19, device="cpu")
        cache.refill(torch.tensor((0,)), chunk, condition)
        for _ in range(3):
            cache.advance(torch.ones(1, dtype=torch.bool))

        class UnexpectedDP:
            state_min = torch.full((26,), -2.0)
            state_max = torch.full((26,), 2.0)

            def encode_observation(self, _observation):
                raise AssertionError("a valid cached lane must not replan")

        prepared = _prepare_policy_step(
            _contract_observation(condition),
            UnexpectedDP(),
            scheduler=None,
            cache=cache,
            action_min=minimum,
            action_max=maximum,
            inference_steps=1,
            use_bfloat16=False,
        )
        expected_base = chunk[:, 3]
        self.assertEqual(int(prepared.row_index), 3)
        torch.testing.assert_close(prepared.base_action, expected_base)
        base_start = condition.shape[1]
        base_end = base_start + 19
        torch.testing.assert_close(
            prepared.policy_input[:, base_start:base_end],
            normalize_physical_action(expected_base, minimum, maximum),
        )
        current_state = _contract_observation(condition)["observation.state"][:, -1]
        torch.testing.assert_close(
            prepared.policy_input[:, base_end : base_end + 26],
            _normalize_current_state(current_state, UnexpectedDP.state_min, UnexpectedDP.state_max),
        )
        row_start = base_end + 26
        delta_start = row_start + _BASE_ACTION_HORIZON
        torch.testing.assert_close(
            prepared.policy_input[:, row_start:delta_start],
            torch.nn.functional.one_hot(torch.tensor((3,)), _BASE_ACTION_HORIZON).float(),
        )
        torch.testing.assert_close(prepared.policy_input[:, delta_start : delta_start + 26], torch.zeros(1, 26))
        torch.testing.assert_close(prepared.policy_input[:, delta_start + 26 :], torch.zeros(1, 5))
        composed = compose_residual_action(
            prepared.base_action,
            torch.zeros(1, self.config.residual_dim),
            minimum,
            maximum,
        )
        torch.testing.assert_close(composed, expected_base, atol=1.0e-6, rtol=0.0)

    def test_frozen_dp_horizon_seven_fails_without_populating_cache(self) -> None:
        minimum, maximum = _physical_bounds()
        frozen_dp = _ContractDP(pred_horizon=7)
        cache = _PerLaneActionChunkCache(num_lanes=1, action_dim=19, device="cpu")
        with self.assertRaisesRegex(ValueError, "pred_horizon >= 8"):
            _prepare_policy_step(
                _contract_observation(torch.tensor(((1.0, 10.0),))),
                frozen_dp,
                scheduler=None,
                cache=cache,
                action_min=minimum,
                action_max=maximum,
                inference_steps=1,
                use_bfloat16=False,
            )
        self.assertIsNone(cache.condition)
        torch.testing.assert_close(cache.valid, torch.tensor((False,)))
        torch.testing.assert_close(cache.plan_count, torch.tensor((0,)))
        with self.assertRaisesRegex(ValueError, "pred_horizon must be >= 8"):
            _validate_frozen_dp_training_contract(SimpleNamespace(pred_horizon=7))
        _validate_frozen_dp_training_contract(SimpleNamespace(pred_horizon=8))
        with self.assertRaisesRegex(ValueError, "obs_horizon must be >= 2"):
            _validate_frozen_dp_training_contract(SimpleNamespace(pred_horizon=8, obs_horizon=1))

    def test_evaluation_quota_accepts_each_lane_once_per_wave_exactly(self) -> None:
        quota = _EvaluationQuota(episodes=3, num_envs=2, device="cpu")
        torch.testing.assert_close(quota.start_wave(), torch.tensor((True, True)))
        torch.testing.assert_close(quota.accept(torch.tensor((True, False))), torch.tensor((True, False)))
        torch.testing.assert_close(quota.accept(torch.tensor((True, False))), torch.tensor((False, False)))
        torch.testing.assert_close(quota.accept(torch.tensor((False, True))), torch.tensor((False, True)))
        self.assertTrue(quota.wave_complete)
        self.assertFalse(quota.complete)

        partial_wave = quota.start_wave()
        self.assertEqual(int(partial_wave.sum()), 1)
        accepted = quota.accept(torch.tensor((True, True)))
        self.assertEqual(int(accepted.sum()), 1)
        self.assertTrue(quota.complete)
        self.assertEqual(quota.completed, 3)

        small_quota = _EvaluationQuota(episodes=2, num_envs=4, device="cpu")
        self.assertEqual(int(small_quota.start_wave().sum()), 2)
        self.assertEqual(int(small_quota.accept(torch.ones(4, dtype=torch.bool)).sum()), 2)
        self.assertEqual(small_quota.completed, 2)

    def test_evaluate_collects_exact_first_terminal_episodes_across_waves(self) -> None:
        def evaluate(episode_lengths: tuple[int, ...], episodes: int):
            env = _FakeMultiLaneEvalEnv(episode_lengths)
            actor = _FakeEvalActor()
            args = SimpleNamespace(
                device="cpu",
                seed=17,
                num_envs=len(episode_lengths),
                inference_steps=1,
                bfloat16=False,
                position_residual_scale_m=0.015,
                vertical_residual_scale_m=0.05,
                rotation_residual_scale_deg=5.0,
                hand_residual_scale_normalized=0.1,
                max_episode_steps=max(episode_lengths),
            )
            minimum, maximum = _physical_bounds()
            state_min = torch.full((26,), -2.0)
            state_max = torch.full((26,), 2.0)
            metrics, _, _ = _evaluate(
                env,
                _FakeEvalDP(),
                scheduler=None,
                actor_critic=actor,
                action_min=minimum,
                action_max=maximum,
                state_min=state_min,
                state_max=state_max,
                hand_residual_scale=torch.full((10,), 0.1),
                args=args,
                episodes=episodes,
            )
            self.assertTrue(actor.training)
            return metrics, env

        metrics, env = evaluate((1, 3, 4), episodes=5)
        self.assertEqual(metrics["episodes"], 5.0)
        self.assertEqual(metrics["success_count"], 1.0)
        self.assertEqual(metrics["fail_count"], 2.0)
        self.assertAlmostEqual(metrics["mean_return"], 6.4)
        self.assertAlmostEqual(metrics["mean_length"], 2.6)
        self.assertAlmostEqual(metrics["mean_episode_max_lift_height_m"], 0.026)
        self.assertAlmostEqual(metrics["max_episode_max_lift_height_m"], 0.04)
        self.assertAlmostEqual(metrics["mean_episode_physical_max_lift_height_m"], 0.026)
        self.assertAlmostEqual(metrics["max_episode_physical_max_lift_height_m"], 0.04)
        self.assertAlmostEqual(metrics["episode_physical_max_lift_ge_1mm_ever_rate"], 1.0)
        self.assertAlmostEqual(metrics["episode_physical_max_lift_ge_10mm_ever_rate"], 1.0)
        self.assertAlmostEqual(metrics["episode_physical_max_lift_ge_50mm_ever_rate"], 0.0)
        self.assertAlmostEqual(metrics["episode_contacted_carry_max_lift_ge_1mm_ever_rate"], 1.0)
        self.assertAlmostEqual(metrics["episode_contacted_carry_max_lift_ge_10mm_ever_rate"], 1.0)
        self.assertAlmostEqual(metrics["episode_contacted_carry_max_lift_ge_50mm_ever_rate"], 0.0)
        self.assertAlmostEqual(metrics["event_success_ever_rate"], 0.2)
        self.assertAlmostEqual(metrics["event_fail_ever_rate"], 0.4)
        self.assertAlmostEqual(metrics["thumb_contact_episode_ever_rate"], 0.8)
        self.assertAlmostEqual(metrics["non_thumb_contact_episode_ever_rate"], 0.6)
        self.assertAlmostEqual(metrics["opposed_grasp_episode_ever_rate"], 0.4)
        self.assertAlmostEqual(metrics["thumb_only_contact_episode_ever_rate"], 0.4)
        self.assertAlmostEqual(metrics["opposed_grasp_any_frame_episode_ever_rate"], 0.4)
        self.assertAlmostEqual(metrics["finger_thumb_contact_any_frame_episode_ever_rate"], 0.8)
        self.assertGreater(metrics["thumb_contact_any_frame_step_fraction"], 0.0)
        self.assertGreater(metrics["non_thumb_contact_any_frame_step_fraction"], 0.0)
        self.assertGreater(metrics["opposed_grasp_any_frame_step_fraction"], 0.0)
        for stage_name in ("non_thumb_only", "thumb_only", "bilateral_unconfirmed"):
            self.assertGreaterEqual(metrics[f"partial_contact_stage_{stage_name}_step_fraction"], 0.0)
            self.assertLessEqual(metrics[f"partial_contact_stage_{stage_name}_step_fraction"], 1.0)
            self.assertGreaterEqual(metrics[f"partial_contact_stage_{stage_name}_episode_ever_rate"], 0.0)
            self.assertLessEqual(metrics[f"partial_contact_stage_{stage_name}_episode_ever_rate"], 1.0)
        self.assertEqual(metrics["opposed_grasp_max_consecutive_physics_frames_max"], 2.0)
        self.assertAlmostEqual(
            sum(
                metrics[f"opposed_grasp_max_consecutive_physics_frames_eq_{frames}_step_fraction"]
                for frames in range(7)
            ),
            1.0,
        )
        self.assertGreater(metrics["opposed_pregrasp_score_ge_0_1_step_fraction"], 0.0)
        self.assertLessEqual(metrics["opposed_pregrasp_score_ge_0_1_step_fraction"], 1.0)
        self.assertGreater(metrics["reward_v13_r0_mean"], 0.0)
        self.assertGreater(metrics["reward_v13_cN_mean"], 0.0)
        self.assertGreater(metrics["reward_v13_GN_mean"], 0.0)
        self.assertGreater(metrics["reward_v13_cN_positive_episode_ever_rate"], 0.0)
        self.assertAlmostEqual(metrics["reward_v13_non_thumb_conditioned_thumb_gap_m_mean"], 0.08)
        self.assertAlmostEqual(
            metrics["reward_v13_non_thumb_conditioned_thumb_proximity_mean"],
            1.0 - math.tanh(1.0),
            places=6,
        )
        self.assertAlmostEqual(metrics["reward_v13_non_thumb_conditioned_guidance_opposition_mean"], 0.8, places=6)
        self.assertAlmostEqual(metrics["reward_v13_non_thumb_conditioned_guidance_z_mean"], 0.6, places=6)
        self.assertGreater(metrics["reward_v13_unilateral_guidance_gain_mean"], 0.0)
        self.assertGreater(metrics["reward_v13_unilateral_contact_reward_mean"], 0.0)
        self.assertEqual(env.full_reset_count, 3)
        self.assertGreaterEqual(sum(mask == (True, False, False) for mask in env.partial_reset_masks), 2)
        self.assertAlmostEqual(
            sum(metrics[f"phase_{name}_fraction"] for name in ("approach", "carrying", "released", "success", "fail")),
            1.0,
        )
        self.assertAlmostEqual(sum(metrics[f"base_action_row_{row}_fraction"] for row in range(8)), 1.0)

        partial_metrics, partial_env = evaluate((1, 3, 4, 2), episodes=2)
        self.assertEqual(partial_metrics["episodes"], 2.0)
        self.assertEqual(partial_metrics["success_count"], 1.0)
        self.assertEqual(partial_metrics["fail_count"], 0.0)
        self.assertAlmostEqual(partial_metrics["mean_return"], 3.5)
        self.assertAlmostEqual(partial_metrics["mean_length"], 2.0)
        self.assertAlmostEqual(partial_metrics["thumb_contact_episode_ever_rate"], 0.5)
        self.assertAlmostEqual(partial_metrics["non_thumb_contact_episode_ever_rate"], 0.5)
        self.assertAlmostEqual(partial_metrics["opposed_grasp_episode_ever_rate"], 0.0)
        self.assertEqual(partial_env.full_reset_count, 2)
        self.assertTrue(any(mask[0] and not mask[1] for mask in partial_env.partial_reset_masks))

    def test_best_checkpoint_ranking_uses_fail_and_return_tiebreaks(self) -> None:
        baseline = {"success_rate": 0.0, "fail_rate": 0.2, "mean_return": 10.0}
        safer = {"success_rate": 0.0, "fail_rate": 0.1, "mean_return": 9.0}
        higher_return = {"success_rate": 0.0, "fail_rate": 0.2, "mean_return": 11.0}
        successful = {"success_rate": 0.1, "fail_rate": 0.9, "mean_return": 1.0}

        self.assertTrue(_is_better_eval(safer, baseline))
        self.assertTrue(_is_better_eval(higher_return, baseline))
        self.assertTrue(_is_better_eval(successful, baseline))
        self.assertTrue(_is_better_return(higher_return, baseline))
        self.assertFalse(_is_better_return(safer, baseline))

    def test_resume_requires_training_v9_reward_v13_and_rejects_older_contracts(self) -> None:
        self.assertEqual(_TRAINING_CONTRACT_VERSION, 9)
        self.assertEqual(_REWARD_CONTRACT_VERSION, 13)
        _validate_resume_training_contract(_training_contract_payload())
        with self.assertRaisesRegex(ValueError, "obsolete"):
            _validate_resume_training_contract(
                {
                    "training_contract_version": 8,
                    "reward_contract_version": _REWARD_CONTRACT_VERSION,
                }
            )

        with self.assertRaisesRegex(ValueError, "base_action_mode"):
            _validate_resume_training_contract(_training_contract_payload(base_action_mode="replan_each_step_index1"))
        with self.assertRaisesRegex(ValueError, "base_action_horizon"):
            _validate_resume_training_contract(_training_contract_payload(base_action_horizon=7))
        with self.assertRaisesRegex(ValueError, "actor_condition_source"):
            _validate_resume_training_contract(_training_contract_payload(actor_condition_source="current_observation"))
        with self.assertRaisesRegex(ValueError, "critic_privileged_source"):
            _validate_resume_training_contract(_training_contract_payload(critic_privileged_source="task_state"))

        with self.assertRaisesRegex(ValueError, "reward_contract_version"):
            _validate_resume_training_contract(_training_contract_payload(reward_contract_version=12))
        with self.assertRaisesRegex(ValueError, "eef_position_frame"):
            _validate_resume_training_contract(_training_contract_payload(eef_position_frame="world"))
        with self.assertRaisesRegex(ValueError, "R_world_from_action"):
            _validate_resume_training_contract(_training_contract_payload(R_world_from_action=np.eye(3).tolist()))
        with self.assertRaisesRegex(ValueError, "residual_position_frame"):
            _validate_resume_training_contract(_training_contract_payload(residual_position_frame="action_xyz"))
        with self.assertRaisesRegex(ValueError, "hand_target_semantics"):
            _validate_resume_training_contract(_training_contract_payload(hand_target_semantics="active_only"))
        missing_hand_semantics = _training_contract_payload()
        del missing_hand_semantics["hand_target_semantics"]
        with self.assertRaisesRegex(ValueError, "hand_target_semantics"):
            _validate_resume_training_contract(missing_hand_semantics)
        missing_scale_contract = _training_contract_payload()
        del missing_scale_contract["hand_residual_scale"]
        with self.assertRaisesRegex(ValueError, "hand_residual_scale"):
            _validate_resume_training_contract(missing_scale_contract)

    def test_training_parser_reserves_triangle_pairs_for_reachable_residuals(self) -> None:
        args = create_parser().parse_args(("frozen_dp.pt",))
        self.assertEqual(args.triangle_pairs_per_env, 131_072)
        _validate_args(args)
        args.triangle_pairs_per_env = 0
        with self.assertRaisesRegex(ValueError, "triangle_pairs_per_env"):
            _validate_args(args)

    def test_thumb_scale_cli_resolves_contract_vector_without_changing_other_joints(self) -> None:
        defaults = create_parser().parse_args(("frozen_dp.pt",))
        self.assertIsNone(defaults.thumb_residual_scales_normalized)
        mode, effective = _resolve_hand_residual_scales(defaults)
        self.assertEqual(mode, "uniform")
        self.assertEqual(effective, (0.1,) * 10)

        overridden = create_parser().parse_args(
            (
                "frozen_dp.pt",
                "--hand-residual-scale-normalized",
                "0.15",
                "--thumb-residual-scales-normalized",
                "0.3",
                "0.7",
                "0.9",
            )
        )
        _validate_args(overridden)
        mode, effective = _resolve_hand_residual_scales(overridden)
        self.assertEqual(mode, "thumb_override")
        self.assertEqual(tuple(effective[index] for index in (0, 1, 9)), (0.3, 0.7, 0.9))
        self.assertTrue(all(effective[index] == 0.15 for index in range(2, 9)))
        contract = _hand_residual_scale_contract(overridden)
        self.assertEqual(contract["thumb_hand_indices"], [0, 1, 9])
        self.assertEqual(contract["thumb_latent_indices"], [6, 7, 15])
        self.assertEqual(contract["thumb_joint_names"], ["thumb_cmc_pitch", "thumb_cmc_yaw", "thumb_cmc_roll"])
        self.assertEqual(contract["effective_scale_normalized"], list(effective))

        overridden.thumb_residual_scales_normalized[1] = 0.0
        with self.assertRaisesRegex(ValueError, "residual action scales"):
            _validate_args(overridden)

    def test_resume_rejects_optimizer_contract_changes(self) -> None:
        saved_args = {name: index for index, name in enumerate(_RESUME_TRAIN_ARG_NAMES)}
        args = SimpleNamespace(**saved_args, total_timesteps=2_000_000)
        _validate_resume_train_args(saved_args, args)

        args.learning_rate = -1
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            _validate_resume_train_args(saved_args, args)

        args = SimpleNamespace(**saved_args, total_timesteps=2_000_000)
        args.triangle_pairs_per_env = -1
        with self.assertRaisesRegex(ValueError, "triangle_pairs_per_env"):
            _validate_resume_train_args(saved_args, args)

    def test_act_and_evaluate_actions_preserve_raw_gaussian_log_prob(self) -> None:
        generator = torch.Generator().manual_seed(123)
        raw_latent, old_log_prob, old_entropy, old_value = self.policy.act(
            self.policy_input,
            self.privileged,
            generator=generator,
        )
        log_prob, entropy, value = self.policy.evaluate_actions(
            self.policy_input,
            self.privileged,
            raw_latent,
        )

        self.assertEqual(tuple(raw_latent.shape), (5, 16))
        torch.testing.assert_close(log_prob, old_log_prob)
        torch.testing.assert_close(entropy, old_entropy)
        torch.testing.assert_close(value, old_value)

        deterministic, _, _, _ = self.policy.act(
            self.policy_input,
            self.privileged,
            deterministic=True,
        )
        torch.testing.assert_close(deterministic, torch.zeros_like(deterministic))

    def test_actor_critic_and_log_std_receive_gradients(self) -> None:
        raw_latent = torch.randn(5, self.config.residual_dim)
        log_prob, entropy, value = self.policy.evaluate_actions(self.policy_input, self.privileged, raw_latent)
        loss = -log_prob.mean() - 0.01 * entropy.mean() + value.square().mean()
        loss.backward()

        actor_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.policy.actor.parameters()
            if parameter.grad is not None
        )
        critic_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.policy.critic.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(actor_grad, 0.0)
        self.assertGreater(critic_grad, 0.0)
        self.assertIsNotNone(self.policy.log_std.grad)
        self.assertGreater(float(self.policy.log_std.grad.abs().sum()), 0.0)

    def test_synthetic_ppo_update_changes_trainable_parameters(self) -> None:
        rollout_steps = 3
        num_envs = 2
        rollout, advantages, returns = self._synthetic_rollout(rollout_steps, num_envs)
        args = SimpleNamespace(
            num_envs=num_envs,
            rollout_steps=rollout_steps,
            update_epochs=2,
            minibatch_size=3,
            clip_coef=0.2,
            clip_vloss=True,
            value_coef=0.5,
            entropy_coef=0.001,
            max_grad_norm=0.5,
            target_kl=0.0,
        )
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1.0e-3)
        before = [parameter.detach().clone() for parameter in self.policy.parameters()]
        metrics = _ppo_update(
            self.policy,
            optimizer,
            rollout,
            advantages,
            returns,
            args,
            torch.Generator().manual_seed(99),
        )
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, self.policy.parameters(), strict=True))
        )

    def test_ppo_reports_post_update_full_batch_kl_and_stops_early(self) -> None:
        rollout, advantages, returns = self._synthetic_rollout()
        args = SimpleNamespace(
            num_envs=2,
            rollout_steps=3,
            update_epochs=4,
            minibatch_size=3,
            clip_coef=0.2,
            clip_vloss=True,
            value_coef=0.5,
            entropy_coef=0.001,
            max_grad_norm=0.5,
            target_kl=1.0e-6,
        )
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.05)
        metrics = _ppo_update(
            self.policy,
            optimizer,
            rollout,
            advantages,
            returns,
            args,
            torch.Generator().manual_seed(101),
        )

        with torch.no_grad():
            new_log_prob, _, _ = self.policy.evaluate_actions(
                rollout["policy_input"], rollout["privileged"], rollout["raw_latent"]
            )
            log_ratio = new_log_prob - rollout["log_prob"]
            ratio = log_ratio.exp()
            expected_kl = float(((ratio - 1.0) - log_ratio).mean())
            flat_kl = ((ratio - 1.0) - log_ratio).reshape(-1)
            expected_max_minibatch_kl = max(float(flat_kl[:3].mean()), float(flat_kl[3:].mean()))
        self.assertEqual(metrics["optimizer_steps"], 1.0)
        self.assertEqual(metrics["kl_early_stop"], 1.0)
        self.assertGreater(metrics["approx_kl"], args.target_kl)
        self.assertAlmostEqual(metrics["approx_kl"], expected_kl, delta=1.0e-5)
        self.assertAlmostEqual(metrics["max_minibatch_kl"], expected_max_minibatch_kl, delta=1.0e-5)
        self.assertGreater(metrics["max_update_step_kl"], args.target_kl)

    def test_ppo_clips_actor_and_critic_gradients_separately(self) -> None:
        rollout, advantages, returns = self._synthetic_rollout()
        returns = returns + 1000.0
        args = SimpleNamespace(
            num_envs=2,
            rollout_steps=3,
            update_epochs=1,
            minibatch_size=6,
            clip_coef=0.2,
            clip_vloss=False,
            value_coef=0.5,
            entropy_coef=0.001,
            max_grad_norm=1.0e-6,
            target_kl=0.0,
        )
        metrics = _ppo_update(
            self.policy,
            torch.optim.Adam(self.policy.parameters(), lr=1.0e-4),
            rollout,
            advantages,
            returns,
            args,
            torch.Generator().manual_seed(102),
        )

        self.assertGreater(metrics["actor_grad_norm_mean"], args.max_grad_norm)
        self.assertGreater(metrics["critic_grad_norm_mean"], args.max_grad_norm)
        self.assertEqual(metrics["actor_grad_clip_fraction"], 1.0)
        self.assertEqual(metrics["critic_grad_clip_fraction"], 1.0)

    def test_action_and_stage_diagnostics(self) -> None:
        minimum, maximum = _physical_bounds()
        base = torch.zeros(2, 19)
        base[:, :3] = maximum[:3]
        base[:, 9:19] = maximum[9:19]
        raw = torch.full((2, 16), 100.0)
        hand_scales = torch.tensor((0.2, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6))
        diagnostics = _action_diagnostics(
            base,
            raw,
            minimum,
            maximum,
            torch.tensor(((0.0, 0.0, 2.5), (0.0, 0.0, 2.5))),
            torch.zeros(2, 10),
            position_scale_m=0.02,
            vertical_position_scale_m=0.05,
            hand_scale_normalized=hand_scales,
        )
        torch.testing.assert_close(diagnostics["position_residual_tanh_saturation_fraction"], torch.ones(2))
        torch.testing.assert_close(diagnostics["rotation_residual_tanh_saturation_fraction"], torch.ones(2))
        torch.testing.assert_close(diagnostics["hand_residual_tanh_saturation_fraction"], torch.ones(2))
        torch.testing.assert_close(diagnostics["position_action_clamp_fraction"], torch.ones(2))
        torch.testing.assert_close(diagnostics["vertical_action_clamp_fraction"], torch.ones(2))
        torch.testing.assert_close(diagnostics["hand_action_clamp_fraction"], torch.ones(2))
        for name, expected in (("pitch", 0.2), ("yaw", 0.4), ("roll", 0.6)):
            prefix = f"thumb_cmc_{name}"
            torch.testing.assert_close(
                diagnostics[f"{prefix}_residual_signed_normalized"],
                torch.full((2,), expected),
            )
            torch.testing.assert_close(
                diagnostics[f"{prefix}_residual_abs_normalized"],
                torch.full((2,), expected),
            )
            torch.testing.assert_close(diagnostics[f"{prefix}_residual_tanh_saturation_fraction"], torch.ones(2))
            torch.testing.assert_close(diagnostics[f"{prefix}_absolute_bound_clamp_fraction"], torch.ones(2))
            torch.testing.assert_close(diagnostics[f"{prefix}_dynamic_rate_limit_fraction"], torch.ones(2))
        torch.testing.assert_close(diagnostics["vertical_residual_signed_m"], torch.full((2,), 0.05))
        torch.testing.assert_close(diagnostics["vertical_residual_abs_m"], torch.full((2,), 0.05))
        torch.testing.assert_close(
            diagnostics["base_target_z_minus_current_eef_z_m"],
            torch.full((2,), 0.5),
        )
        torch.testing.assert_close(
            diagnostics["composed_target_z_minus_current_eef_z_m"],
            torch.full((2,), 0.5),
        )

        rotation_only = torch.zeros(1, 16)
        rotation_only[:, 3:6] = 1.1
        rotation_diagnostics = _action_diagnostics(
            torch.zeros(1, 19),
            rotation_only,
            minimum,
            maximum,
            torch.zeros(1, 3),
            torch.zeros(1, 10),
            position_scale_m=0.02,
            vertical_position_scale_m=0.05,
            hand_scale_normalized=0.1,
        )
        self.assertTrue(torch.all(torch.tanh(rotation_only[:, 3:6]).abs() < 0.95))
        torch.testing.assert_close(rotation_diagnostics["rotation_residual_tanh_saturation_fraction"], torch.ones(1))
        torch.testing.assert_close(rotation_diagnostics["position_residual_tanh_saturation_fraction"], torch.zeros(1))

        rollout = {
            "task_phase": torch.tensor(((0, 1), (2, 3), (4, 0))),
            "base_action_row": torch.tensor(((0, 1), (2, 3), (4, 5))),
            "base_action_replanned": torch.tensor(((True, True), (False, False), (False, False))),
            "base_action_discarded_rows": torch.zeros(3, 2, dtype=torch.int64),
            "episode_done": torch.tensor(((False, False), (True, False), (False, True))),
            "episode_max_lift_height_m": torch.tensor(((0.0, 0.0), (0.03, 0.0), (0.0, 0.07))),
            "episode_physical_max_lift_height_m": torch.tensor(((0.0, 0.0), (0.04, 0.0), (0.0, 0.09))),
            "current_lift_height_m": torch.tensor(((0.0, 0.01), (0.02, 0.03), (0.04, 0.05))),
            "physical_max_lift_height_m": torch.tensor(((0.01, 0.02), (0.04, 0.05), (0.06, 0.09))),
            "contacted_carry_max_lift_height_m": torch.tensor(((0.0, 0.0005), (0.001, 0.009), (0.01, 0.009))),
            "opposed_pregrasp_score": torch.tensor(((0.0, 0.2), (0.4, 0.6), (0.8, 1.0))),
            "reward_v13_r0": torch.full((3, 2), 0.7),
            "reward_v13_cN": torch.tensor(((1.0, 0.0), (0.5, 0.0), (1.0, 0.0))),
            "reward_v13_cT": torch.tensor(((0.0, 0.5), (0.25, 0.0), (0.0, 0.0))),
            "reward_v13_GN": torch.tensor(((0.2, 0.0), (0.3, 0.0), (0.4, 0.0))),
            "reward_v13_GT": torch.tensor(((0.0, 0.2), (0.1, 0.0), (0.0, 0.0))),
            "reward_v13_thumb_gap_m": torch.tensor(((0.01, 0.02), (0.03, 0.04), (0.05, 0.06))),
            "reward_v13_thumb_proximity": torch.tensor(((0.9, 0.0), (0.8, 0.0), (0.7, 0.0))),
            "reward_v13_guidance_opposition": torch.tensor(((0.8, 0.0), (0.6, 0.0), (0.4, 0.0))),
            "reward_v13_guidance_z": torch.tensor(((0.7, 0.0), (0.5, 0.0), (0.3, 0.0))),
            "reward_v13_unilateral_guidance_gain": torch.full((3, 2), 0.1),
            "reward_v13_unilateral_contact_reward": torch.full((3, 2), 0.2),
            "actual_eef_delta_z_m": torch.tensor(((0.001, 0.002), (0.003, 0.004), (0.005, 0.006))),
            "pre_action_contact": torch.tensor(((True, False), (True, True), (False, False))),
            "pre_action_thumb_contact": torch.tensor(((True, False), (True, True), (False, False))),
            "pre_action_non_thumb_contact": torch.tensor(((False, False), (True, False), (False, False))),
            "pre_action_grasp": torch.tensor(((False, False), (True, False), (False, False))),
            "pre_action_carrying": torch.tensor(((False, True), (False, True), (False, False))),
            "finger_contact_any_frame": torch.tensor(
                (
                    ((True, False, False, False, False), (False, True, False, False, False)),
                    ((True, True, False, False, False), (False, False, True, False, False)),
                    ((False, False, False, False, False), (True, False, False, False, True)),
                )
            ),
            "opposed_grasp_any_frame": torch.tensor(((False, False), (True, False), (False, True))),
            "opposed_grasp_max_consecutive_physics_frames": torch.tensor(((0, 0), (3, 0), (0, 2))),
            "partial_contact_stage_non_thumb_only": torch.tensor(((True, False), (False, False), (False, False))),
            "partial_contact_stage_thumb_only": torch.tensor(((False, True), (False, False), (False, False))),
            "partial_contact_stage_bilateral_unconfirmed": torch.tensor(
                ((False, False), (True, False), (False, False))
            ),
            "finger_root_load": torch.tensor(
                (
                    ((0.0, 0.0, 0.0, 0.0, 0.0), (0.1, 0.1, 0.0, 0.0, 0.0)),
                    ((0.8, 0.7, 0.0, 0.0, 0.0), (0.4, 0.3, 0.0, 0.0, 0.0)),
                    ((0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0)),
                )
            ),
            "position_residual_tanh_saturation_fraction": torch.full((3, 2), 0.5),
            "rotation_residual_tanh_saturation_fraction": torch.full((3, 2), 0.75),
            "hand_residual_tanh_saturation_fraction": torch.full((3, 2), 0.25),
            "position_action_clamp_fraction": torch.full((3, 2), 0.25),
            "vertical_action_clamp_fraction": torch.full((3, 2), 0.5),
            "hand_action_clamp_fraction": torch.zeros(3, 2),
            "vertical_residual_signed_m": torch.full((3, 2), 0.01),
            "vertical_residual_abs_m": torch.full((3, 2), 0.02),
            "base_target_z_minus_current_eef_z_m": torch.full((3, 2), 0.03),
            "composed_target_z_minus_current_eef_z_m": torch.full((3, 2), 0.04),
            **{name: torch.zeros(3, 2) for name in _ACTION_DIAGNOSTIC_NAMES if name.startswith("thumb_cmc_")},
            **{
                f"event_{name}_rise": torch.zeros(3, 2, dtype=torch.bool)
                for name in (
                    "contact",
                    "grasp",
                    "transport",
                    "lift",
                    "release_ready",
                    "release_armed",
                    "released",
                    "early_release",
                    "success",
                    "fail",
                )
            },
        }
        rollout["event_contact_rise"][0, 0] = True
        metrics = _rollout_diagnostic_metrics(rollout)
        self.assertAlmostEqual(
            sum(metrics[f"phase_{name}_fraction"] for name in ("approach", "carrying", "released", "success", "fail")),
            1.0,
        )
        self.assertEqual(metrics["event_contact_rise_count"], 1.0)
        self.assertAlmostEqual(metrics["event_contact_rise_per_1000_steps"], 1000.0 / 6.0)
        self.assertAlmostEqual(sum(metrics[f"base_action_row_{row}_fraction"] for row in range(8)), 1.0)
        self.assertEqual(metrics["base_action_replan_lane_count"], 2.0)
        self.assertAlmostEqual(metrics["mean_episode_max_lift_height_m"], 0.05)
        self.assertAlmostEqual(metrics["max_episode_max_lift_height_m"], 0.07)
        self.assertAlmostEqual(metrics["mean_episode_physical_max_lift_height_m"], 0.065)
        self.assertAlmostEqual(metrics["max_episode_physical_max_lift_height_m"], 0.09)
        self.assertAlmostEqual(metrics["mean_current_lift_height_m"], 0.025)
        self.assertAlmostEqual(metrics["max_rollout_physical_lift_height_m"], 0.09)
        self.assertAlmostEqual(metrics["max_rollout_contacted_carry_lift_height_m"], 0.01)
        self.assertEqual(metrics["rollout_physical_max_lift_ge_1mm_ever"], 1.0)
        self.assertAlmostEqual(metrics["rollout_physical_max_lift_ge_1mm_step_fraction"], 1.0)
        self.assertAlmostEqual(metrics["rollout_physical_max_lift_ge_50mm_step_fraction"], 0.5)
        self.assertAlmostEqual(metrics["rollout_physical_max_lift_ge_50mm_lane_ever_fraction"], 1.0)
        self.assertEqual(metrics["rollout_contacted_carry_max_lift_ge_10mm_ever"], 1.0)
        self.assertAlmostEqual(metrics["rollout_contacted_carry_max_lift_ge_10mm_step_fraction"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["rollout_contacted_carry_max_lift_ge_10mm_lane_ever_fraction"], 0.5)
        self.assertEqual(metrics["rollout_contacted_carry_max_lift_ge_50mm_ever"], 0.0)
        self.assertAlmostEqual(metrics["rollout_contacted_carry_max_lift_ge_50mm_step_fraction"], 0.0)
        self.assertAlmostEqual(metrics["rollout_contacted_carry_max_lift_ge_50mm_lane_ever_fraction"], 0.0)
        self.assertAlmostEqual(metrics["contact_step_fraction"], 0.5)
        self.assertAlmostEqual(metrics["thumb_contact_step_fraction"], 0.5)
        self.assertAlmostEqual(metrics["non_thumb_contact_step_fraction"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["opposed_grasp_step_fraction"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["thumb_only_contact_step_fraction"], 2.0 / 6.0)
        self.assertAlmostEqual(metrics["thumb_to_opposed_grasp_step_conversion"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["thumb_contact_any_frame_step_fraction"], 0.5)
        self.assertAlmostEqual(metrics["non_thumb_contact_any_frame_step_fraction"], 4.0 / 6.0)
        self.assertAlmostEqual(metrics["opposed_grasp_any_frame_step_fraction"], 2.0 / 6.0)
        self.assertAlmostEqual(metrics["partial_contact_stage_non_thumb_only_step_fraction"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["partial_contact_stage_thumb_only_step_fraction"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["partial_contact_stage_bilateral_unconfirmed_step_fraction"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["finger_pinky_contact_any_frame_step_fraction"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["opposed_grasp_max_consecutive_physics_frames_mean"], 5.0 / 6.0)
        self.assertEqual(metrics["opposed_grasp_max_consecutive_physics_frames_p50"], 0.0)
        self.assertEqual(metrics["opposed_grasp_max_consecutive_physics_frames_p95"], 3.0)
        self.assertEqual(metrics["opposed_grasp_max_consecutive_physics_frames_max"], 3.0)
        self.assertAlmostEqual(
            metrics["opposed_grasp_max_consecutive_physics_frames_eq_0_step_fraction"],
            4.0 / 6.0,
        )
        self.assertAlmostEqual(metrics["carrying_step_fraction"], 2.0 / 6.0)
        self.assertAlmostEqual(metrics["opposed_pregrasp_score_mean"], 0.5)
        self.assertAlmostEqual(metrics["opposed_pregrasp_score_p95"], 0.95)
        self.assertAlmostEqual(metrics["opposed_pregrasp_score_ge_0_1_step_fraction"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["opposed_pregrasp_score_ge_0_5_step_fraction"], 0.5)
        self.assertAlmostEqual(metrics["reward_v13_r0_mean"], 0.7)
        self.assertAlmostEqual(metrics["reward_v13_cN_mean"], 2.5 / 6.0)
        self.assertAlmostEqual(metrics["reward_v13_cN_positive_step_fraction"], 0.5)
        self.assertAlmostEqual(metrics["reward_v13_GN_positive_step_fraction"], 0.5)
        self.assertAlmostEqual(metrics["reward_v13_non_thumb_conditioned_thumb_gap_m_mean"], 0.03)
        self.assertAlmostEqual(metrics["reward_v13_non_thumb_conditioned_thumb_gap_m_min"], 0.01)
        self.assertAlmostEqual(metrics["reward_v13_non_thumb_conditioned_thumb_gap_m_p50"], 0.03)
        self.assertAlmostEqual(metrics["reward_v13_non_thumb_conditioned_thumb_proximity_mean"], 0.8)
        self.assertAlmostEqual(metrics["reward_v13_non_thumb_conditioned_guidance_opposition_mean"], 0.6)
        self.assertAlmostEqual(metrics["reward_v13_non_thumb_conditioned_guidance_z_mean"], 0.5)
        self.assertAlmostEqual(metrics["reward_v13_unilateral_guidance_gain_mean"], 0.1)
        self.assertAlmostEqual(metrics["reward_v13_unilateral_contact_reward_mean"], 0.2)
        self.assertAlmostEqual(metrics["contact_to_grasp_step_conversion"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["contact_vertical_residual_signed_m"], 0.01)
        self.assertAlmostEqual(metrics["carrying_actual_eef_delta_z_m"], 0.003)
        self.assertAlmostEqual(metrics["actual_eef_delta_z_m"], 0.0035)
        self.assertAlmostEqual(metrics["finger_root_load_thumb_mean"], 1.3 / 6.0)
        self.assertAlmostEqual(metrics["finger_root_second_load_grasp_mean"], 0.7)
        self.assertAlmostEqual(metrics["vertical_residual_signed_m_two_finger_load"], 0.01)

    def test_contact_event_uses_sticky_control_step_signal(self) -> None:
        info = {
            "has_hand_contact": torch.tensor((True, False)),
            "had_hand_contact_this_control_step": torch.tensor((False, True)),
            **{
                key: torch.zeros(2, dtype=torch.bool)
                for key in (
                    "is_grasped",
                    "grasp_confirmed",
                    "transport_started",
                    "is_lifted",
                    "release_ready",
                    "release_armed",
                    "released",
                    "early_release",
                    "success",
                    "fail",
                )
            },
        }

        torch.testing.assert_close(_event_flags(info)["contact"], torch.tensor((False, True)))

    def test_control_step_contact_topology_keeps_simultaneous_opposition_explicit(self) -> None:
        info = {
            "finger_contact_any_frame_this_control_step": torch.tensor(
                ((True, True, False, False, False), (False, False, True, False, False))
            ),
            "opposed_grasp_any_frame_this_control_step": torch.tensor((False, True)),
            "opposed_grasp_max_consecutive_physics_frames_this_control_step": torch.tensor((0, 4)),
        }

        fingers, thumb, non_thumb, opposed, streak = _control_step_contact_topology(info)

        torch.testing.assert_close(fingers, info["finger_contact_any_frame_this_control_step"])
        torch.testing.assert_close(thumb, torch.tensor((True, False)))
        torch.testing.assert_close(non_thumb, torch.tensor((True, True)))
        # Lane zero touched both sides at some point, but not necessarily in the same physics frame.
        torch.testing.assert_close(opposed, torch.tensor((False, True)))
        torch.testing.assert_close(streak, torch.tensor((0, 4)))

    def test_partial_contact_reward_stages_follow_reward_precedence(self) -> None:
        info = {
            "finger_contact_any_frame_this_control_step": torch.tensor(
                (
                    (False, True, False, False, False),
                    (True, False, False, False, False),
                    (True, True, False, False, False),
                    (True, True, False, False, False),
                    (True, True, False, False, False),
                    (True, True, False, False, False),
                )
            ),
            "opposed_grasp_any_frame_this_control_step": torch.tensor((False, False, False, True, True, True)),
            "opposed_grasp_max_consecutive_physics_frames_this_control_step": torch.tensor((0, 0, 0, 2, 5, 6)),
            "is_grasped": torch.tensor((False, False, False, False, True, False)),
            "task_phase": torch.tensor((0, 0, 0, 0, 0, 1)),
        }

        stages = _partial_contact_reward_stages(info)

        torch.testing.assert_close(
            stages["non_thumb_only"],
            torch.tensor((True, False, False, False, False, False)),
        )
        # Lane two saw thumb/non-thumb in separate frames; without explicit
        # same-frame opposition it stays in the thumb-priority partial stage.
        torch.testing.assert_close(
            stages["thumb_only"],
            torch.tensor((False, True, True, False, False, False)),
        )
        torch.testing.assert_close(
            stages["bilateral_unconfirmed"],
            torch.tensor((False, False, False, True, False, False)),
        )

    def test_world_position_diagnostics_use_current_urdf_frame(self) -> None:
        minimum, maximum = _physical_bounds()
        base = torch.zeros(1, 19)
        base[:, :3] = torch.tensor((0.4, 0.2, 0.1))
        current = torch.tensor(((0.1, 0.15, 0.05),))
        raw = torch.zeros(1, 16)
        raw[:, 2] = math.atanh(0.5)

        diagnostics = _action_diagnostics(
            base,
            raw,
            minimum,
            maximum,
            current,
            torch.zeros(1, 10),
            position_scale_m=0.02,
            vertical_position_scale_m=0.05,
            hand_scale_normalized=0.1,
            world_from_action_rotation=_R_WORLD_FROM_ACTION,
        )

        torch.testing.assert_close(diagnostics["vertical_residual_signed_m"], torch.tensor((0.025,)))
        torch.testing.assert_close(diagnostics["base_target_z_minus_current_eef_z_m"], torch.tensor((0.3,)))
        torch.testing.assert_close(diagnostics["composed_target_z_minus_current_eef_z_m"], torch.tensor((0.325,)))
        eef_delta_action = torch.tensor(((0.012, -0.02, 0.03),))
        torch.testing.assert_close(_action_position_to_world(eef_delta_action)[..., 2], torch.tensor((0.012,)))

    def test_current_urdf_rotation_maps_can_x_to_world_up(self) -> None:
        action_x = torch.tensor(((1.0, 0.0, 0.0),))
        world = _action_position_to_world(action_x)
        torch.testing.assert_close(world, torch.tensor(((0.0, 0.0, 1.0),)))

    def test_world_xyz_residual_maps_through_non_axis_aligned_rotation(self) -> None:
        minimum, maximum = _physical_bounds()
        rotation = _rpy_matrix(
            roll=math.radians(-15.0),
            pitch=math.radians(20.0),
            yaw=math.radians(30.0),
        )
        base = torch.zeros(1, 19)
        base[:, :3] = torch.tensor((0.1, -0.2, 0.3))
        base[:, 3] = 1.0
        base[:, 7] = 1.0
        raw = torch.zeros(1, 16)
        raw[:, :3] = torch.tensor((0.25, -0.5, 0.75))

        composed = compose_residual_action(
            base,
            raw,
            minimum,
            maximum,
            position_scale_m=0.04,
            vertical_position_scale_m=0.08,
            world_from_action_rotation=rotation,
        )

        expected_world = torch.tensor((0.04, 0.04, 0.08)) * torch.tanh(raw[0, :3])
        expected_action = expected_world @ torch.from_numpy(rotation).float()
        torch.testing.assert_close(composed[0, :3] - base[0, :3], expected_action)
        recovered_world = (composed[0, :3] - base[0, :3]) @ torch.from_numpy(rotation).float().T
        torch.testing.assert_close(recovered_world, expected_world)

    def test_position_frame_rotation_validation_rejects_invalid_matrices(self) -> None:
        reference = torch.zeros(1, 19)
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_world_from_action_rotation(torch.eye(2), reference)
        invalid_finite = torch.eye(3)
        invalid_finite[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_world_from_action_rotation(invalid_finite, reference)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validate_world_from_action_rotation(torch.diag(torch.tensor((1.0, 1.0, 2.0))), reference)
        with self.assertRaisesRegex(ValueError, "right-handed"):
            validate_world_from_action_rotation(torch.diag(torch.tensor((1.0, 1.0, -1.0))), reference)

    def test_pre_action_grasp_diagnostic_uses_live_opposed_grasp(self) -> None:
        info = {
            "has_hand_contact": torch.tensor((True, True)),
            "had_hand_contact_this_control_step": torch.tensor((False, False)),
            "is_grasped": torch.tensor((False, True)),
            "finger_contact_counts": torch.tensor(((1, 0, 0, 0, 0), (1, 1, 0, 0, 0))),
            "grasp_confirmed": torch.tensor((True, False)),
            "task_phase": torch.tensor((1, 0)),
        }

        contact, grasp, carrying = _pre_action_task_flags(info)

        torch.testing.assert_close(contact, torch.tensor((True, True)))
        torch.testing.assert_close(grasp, torch.tensor((False, True)))
        torch.testing.assert_close(carrying, torch.tensor((True, False)))

        thumb, non_thumb, opposed = _finger_contact_topology(info)
        torch.testing.assert_close(thumb, torch.tensor((True, True)))
        torch.testing.assert_close(non_thumb, torch.tensor((False, True)))
        torch.testing.assert_close(opposed, torch.tensor((False, True)))

    def test_rotation_residual_is_local_right_multiplication_and_row_first(self) -> None:
        base_rotation = _axis_angle_matrix((1.0, -2.0, 0.5), 0.7)
        local_axis = np.asarray((0.3, 0.7, -0.2), dtype=np.float64)
        local_axis /= np.linalg.norm(local_axis)
        residual_angle = 0.2
        rotation_scale = 0.5
        raw_magnitude = np.arctanh(residual_angle / rotation_scale)

        base = torch.zeros(1, 19)
        base[:, 3:9] = torch.from_numpy(base_rotation[:2, :].reshape(6)).float()
        raw = torch.zeros(1, 16)
        raw[:, 3:6] = torch.from_numpy(local_axis * raw_magnitude).float()
        minimum, maximum = _physical_bounds()
        composed = compose_residual_action(
            base,
            raw,
            minimum,
            maximum,
            rotation_scale_rad=rotation_scale,
        )

        output_row0 = composed[0, 3:6].detach().numpy()
        output_row1 = composed[0, 6:9].detach().numpy()
        output_rotation = np.stack((output_row0, output_row1, np.cross(output_row0, output_row1)))
        local_rotation = _axis_angle_matrix(tuple(local_axis), residual_angle)
        expected = base_rotation @ local_rotation
        left_multiplied = local_rotation @ base_rotation
        np.testing.assert_allclose(output_rotation, expected, atol=2.0e-6)
        self.assertGreater(float(np.linalg.norm(output_rotation - left_multiplied)), 0.05)

    def test_action_composition_respects_position_and_hand_bounds(self) -> None:
        minimum, maximum = _physical_bounds()
        base = torch.zeros(2, 19)
        base[:, 3] = 1.0
        base[:, 7] = 1.0
        base[0, :3] = maximum[:3]
        base[1, :3] = minimum[:3]
        base[0, 9:19] = maximum[9:19]
        base[1, 9:19] = minimum[9:19]
        raw = torch.full((2, 16), 100.0)
        raw[1] = -100.0

        composed = compose_residual_action(
            base,
            raw,
            minimum,
            maximum,
            world_from_action_rotation=_rpy_matrix(
                roll=math.radians(-15.0),
                pitch=math.radians(20.0),
                yaw=math.radians(30.0),
            ),
        )
        self.assertTrue(torch.all(composed[:, :3] >= minimum[:3]))
        self.assertTrue(torch.all(composed[:, :3] <= maximum[:3]))
        self.assertTrue(torch.all(composed[:, 9:19] >= minimum[9:19]))
        self.assertTrue(torch.all(composed[:, 9:19] <= maximum[9:19]))
        rotation = composed[:, 3:9].reshape(2, 2, 3)
        torch.testing.assert_close(torch.linalg.vector_norm(rotation, dim=-1), torch.ones(2, 2), atol=1.0e-5, rtol=0.0)
        torch.testing.assert_close((rotation[:, 0] * rotation[:, 1]).sum(dim=-1), torch.zeros(2), atol=1.0e-5, rtol=0.0)

    def test_action_composition_accepts_scalar_or_per_joint_hand_scales(self) -> None:
        minimum, maximum = _physical_bounds()
        base = torch.zeros(1, 19)
        base[:, 3] = 1.0
        base[:, 7] = 1.0
        raw = torch.zeros(1, 16)
        raw[:, 6:16] = math.atanh(0.5)
        scales = torch.arange(1, 11, dtype=torch.float32) * 0.05

        vector_composed = compose_residual_action(
            base,
            raw,
            minimum,
            maximum,
            hand_scale_normalized=scales,
        )
        scalar_composed = compose_residual_action(
            base,
            raw,
            minimum,
            maximum,
            hand_scale_normalized=0.1,
        )
        torch.testing.assert_close(vector_composed[0, 9:19], 0.25 * scales)
        torch.testing.assert_close(scalar_composed[0, 9:19], torch.full((10,), 0.025))

        with self.assertRaisesRegex(ValueError, "length-10"):
            compose_residual_action(base, raw, minimum, maximum, hand_scale_normalized=torch.ones(3))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            compose_residual_action(base, raw, minimum, maximum, hand_scale_normalized=[0.1] * 9 + [-0.1])

    def test_thumb_diagnostics_use_effective_scale_and_physical_rate_limit(self) -> None:
        minimum, maximum = _physical_bounds()
        base = torch.zeros(1, 19)
        raw = torch.zeros(1, 16)
        raw[:, (6, 7, 15)] = torch.tensor((math.atanh(0.5), math.atanh(0.5), math.atanh(-0.5)))
        scales = torch.tensor((0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6))

        diagnostics = _action_diagnostics(
            base,
            raw,
            minimum,
            maximum,
            torch.zeros(1, 3),
            torch.zeros(1, 10),
            position_scale_m=0.02,
            vertical_position_scale_m=0.05,
            hand_scale_normalized=scales,
            hand_max_joint_step_rad=0.08,
        )

        torch.testing.assert_close(diagnostics["thumb_cmc_pitch_residual_signed_normalized"], torch.tensor((0.2,)))
        torch.testing.assert_close(diagnostics["thumb_cmc_yaw_residual_signed_normalized"], torch.tensor((0.1,)))
        torch.testing.assert_close(diagnostics["thumb_cmc_roll_residual_signed_normalized"], torch.tensor((-0.3,)))
        torch.testing.assert_close(diagnostics["thumb_cmc_roll_residual_abs_normalized"], torch.tensor((0.3,)))
        torch.testing.assert_close(diagnostics["thumb_cmc_pitch_dynamic_rate_limit_fraction"], torch.ones(1))
        torch.testing.assert_close(diagnostics["thumb_cmc_yaw_dynamic_rate_limit_fraction"], torch.zeros(1))
        torch.testing.assert_close(diagnostics["thumb_cmc_roll_dynamic_rate_limit_fraction"], torch.ones(1))
        for name in ("pitch", "yaw", "roll"):
            torch.testing.assert_close(
                diagnostics[f"thumb_cmc_{name}_absolute_bound_clamp_fraction"],
                torch.zeros(1),
            )

    def test_vertical_position_scale_is_independent_and_backward_compatible(self) -> None:
        minimum, maximum = _physical_bounds()
        base = torch.zeros(1, 19)
        base[:, 3] = 1.0
        base[:, 7] = 1.0
        raw = torch.zeros(1, 16)
        raw[:, :3] = 100.0

        independent = compose_residual_action(
            base,
            raw,
            minimum,
            maximum,
            position_scale_m=0.015,
            vertical_position_scale_m=0.05,
        )
        torch.testing.assert_close(independent[:, :3], torch.tensor(((0.015, 0.015, 0.05),)))

        legacy = compose_residual_action(
            base,
            raw,
            minimum,
            maximum,
            position_scale_m=0.015,
        )
        torch.testing.assert_close(legacy[:, :3], torch.full((1, 3), 0.015))

    def test_zero_residual_preserves_a_valid_base_action(self) -> None:
        minimum, maximum = _physical_bounds()
        base = torch.zeros(2, 19)
        base[:, 3] = 1.0
        base[:, 7] = 1.0
        base[:, :3] = torch.tensor((0.2, -0.3, 0.4))
        base[:, 9:19] = torch.linspace(-0.25, 0.25, 10)
        composed = compose_residual_action(
            base,
            torch.zeros(2, 16),
            minimum,
            maximum,
            world_from_action_rotation=_R_WORLD_FROM_ACTION,
        )
        torch.testing.assert_close(composed, base, atol=1.0e-6, rtol=0.0)

    def test_physical_action_normalization_clips_to_unit_interval(self) -> None:
        minimum, maximum = _physical_bounds()
        action = torch.stack((minimum - 1.0, 0.5 * (minimum + maximum), maximum + 1.0))
        normalized = normalize_physical_action(action, minimum, maximum)
        torch.testing.assert_close(normalized[0], -torch.ones(19))
        torch.testing.assert_close(normalized[1], torch.zeros(19))
        torch.testing.assert_close(normalized[2], torch.ones(19))

    def test_gae_bootstraps_truncation_but_not_termination(self) -> None:
        rewards = torch.tensor(((1.0, 1.0), (100.0, 100.0)))
        values = torch.zeros_like(rewards)
        terminated = torch.tensor(((True, False), (False, False)))
        truncated = torch.tensor(((False, True), (False, False)))
        timeout_values = torch.tensor(((5.0, 7.0), (0.0, 0.0)))
        last_value = torch.zeros(2)
        advantages, returns = compute_gae(
            rewards,
            values,
            terminated,
            truncated,
            timeout_values,
            last_value,
            gamma=1.0,
            gae_lambda=1.0,
        )

        torch.testing.assert_close(advantages[0], torch.tensor((1.0, 8.0)))
        torch.testing.assert_close(returns, advantages)

        no_timeout_bootstrap, _ = compute_gae(
            rewards,
            values,
            terminated,
            truncated,
            timeout_values,
            last_value,
            gamma=1.0,
            gae_lambda=1.0,
            bootstrap_time_limit=False,
        )
        torch.testing.assert_close(no_timeout_bootstrap[0], torch.tensor((1.0, 1.0)))

        overlap_advantage, _ = compute_gae(
            torch.tensor(((2.0,),)),
            torch.zeros(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
            torch.full((1, 1), 99.0),
            torch.zeros(1),
            gamma=1.0,
            gae_lambda=1.0,
        )
        torch.testing.assert_close(overlap_advantage, torch.tensor(((2.0,),)))

        both_flags = truncated.clone()
        both_flags[0, 0] = True
        both_advantages, _ = compute_gae(
            rewards,
            values,
            terminated,
            both_flags,
            timeout_values,
            last_value,
            gamma=1.0,
            gae_lambda=1.0,
        )
        self.assertEqual(float(both_advantages[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
