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
)
from tools.train_newton_groot_residual_ppo import _ppo_update


def _axis_angle_matrix(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    x, y, z = direction
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _physical_bounds() -> tuple[torch.Tensor, torch.Tensor]:
    minimum = torch.tensor([-1.0, -2.0, -3.0] + [-1.0] * 6 + [-0.5] * 10)
    maximum = torch.tensor([1.0, 2.0, 3.0] + [1.0] * 6 + [0.5] * 10)
    return minimum, maximum


@unittest.skipUnless(torch is not None, "requires PyTorch")
class TestGrootResidualPPO(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = GrootResidualActorCriticConfig(condition_dim=11, hidden_dim=32)
        self.policy = GrootResidualActorCritic(self.config)
        self.policy_input = torch.randn(5, self.config.policy_input_dim)
        self.privileged = torch.randn(5, self.config.privileged_dim)

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

        composed = compose_residual_action(base, raw, minimum, maximum)
        self.assertTrue(torch.all(composed[:, :3] >= minimum[:3]))
        self.assertTrue(torch.all(composed[:, :3] <= maximum[:3]))
        self.assertTrue(torch.all(composed[:, 9:19] >= minimum[9:19]))
        self.assertTrue(torch.all(composed[:, 9:19] <= maximum[9:19]))
        rotation = composed[:, 3:9].reshape(2, 2, 3)
        torch.testing.assert_close(torch.linalg.vector_norm(rotation, dim=-1), torch.ones(2, 2), atol=1.0e-5, rtol=0.0)
        torch.testing.assert_close((rotation[:, 0] * rotation[:, 1]).sum(dim=-1), torch.zeros(2), atol=1.0e-5, rtol=0.0)

    def test_zero_residual_preserves_a_valid_base_action(self) -> None:
        minimum, maximum = _physical_bounds()
        base = torch.zeros(2, 19)
        base[:, 3] = 1.0
        base[:, 7] = 1.0
        base[:, :3] = torch.tensor((0.2, -0.3, 0.4))
        base[:, 9:19] = torch.linspace(-0.25, 0.25, 10)
        composed = compose_residual_action(base, torch.zeros(2, 16), minimum, maximum)
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
