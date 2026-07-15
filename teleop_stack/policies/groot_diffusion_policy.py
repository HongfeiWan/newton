# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact dual-camera Diffusion Policy for the Nero + L10 task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from teleop_stack.datasets.groot_lerobot import ACTION_KEY, EGO_KEY, STATE_KEY, WRIST_KEY

GROOT_DP_CHECKPOINT_FORMAT = "teleop_stack.groot_l10_diffusion_policy.v2.row_first_state_target"


@dataclass(frozen=True)
class GrootDiffusionPolicyConfig:
    """Network and temporal dimensions matching the current dataset."""

    state_dim: int = 26
    action_dim: int = 19
    obs_horizon: int = 2
    pred_horizon: int = 16
    image_size: int = 224
    camera_feature_dim: int = 256
    state_feature_dim: int = 128
    denoiser_width: int = 512
    diffusion_train_steps: int = 100


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _SinusoidalEmbedding:
    def __new__(cls, dimension: int) -> Any:
        import torch
        from torch import nn

        class SinusoidalEmbedding(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dimension = int(dimension)

            def forward(self, timestep: Any) -> Any:
                half = self.dimension // 2
                exponent = (
                    -math.log(10000.0)
                    * torch.arange(half, dtype=torch.float32, device=timestep.device)
                    / max(half - 1, 1)
                )
                phase = timestep.float()[:, None] * torch.exp(exponent)[None, :]
                embedding = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
                if self.dimension % 2:
                    embedding = torch.nn.functional.pad(embedding, (0, 1))
                return embedding

        return SinusoidalEmbedding()


class _ConditionalResidualBlock1d:
    def __new__(cls, channels: int, condition_dim: int, dilation: int) -> Any:
        from torch import nn

        class ConditionalResidualBlock1d(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                groups = _group_count(channels)
                self.block1 = nn.Sequential(
                    nn.GroupNorm(groups, channels),
                    nn.SiLU(),
                    nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                )
                self.block2 = nn.Sequential(
                    nn.GroupNorm(groups, channels),
                    nn.SiLU(),
                    nn.Conv1d(channels, channels, 3, padding=1),
                )
                self.condition = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, channels * 2))

            def forward(self, value: Any, condition: Any) -> Any:
                scale, bias = self.condition(condition).chunk(2, dim=-1)
                hidden = self.block1(value)
                hidden = hidden * (1.0 + scale[:, :, None]) + bias[:, :, None]
                return value + self.block2(hidden)

        return ConditionalResidualBlock1d()


class _TemporalDenoiser:
    def __new__(cls, action_dim: int, width: int, observation_dim: int) -> Any:
        from torch import nn

        class TemporalDenoiser(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                timestep_dim = 128
                self.time_encoder = nn.Sequential(
                    _SinusoidalEmbedding(timestep_dim),
                    nn.Linear(timestep_dim, timestep_dim * 2),
                    nn.SiLU(),
                    nn.Linear(timestep_dim * 2, timestep_dim),
                )
                condition_dim = observation_dim + timestep_dim
                self.input = nn.Conv1d(action_dim, width, 3, padding=1)
                self.blocks = nn.ModuleList(
                    [_ConditionalResidualBlock1d(width, condition_dim, dilation) for dilation in (1, 2, 4, 8, 4, 2)]
                )
                self.output = nn.Sequential(
                    nn.GroupNorm(_group_count(width), width),
                    nn.SiLU(),
                    nn.Conv1d(width, action_dim, 3, padding=1),
                )

            def forward(self, sample: Any, timestep: Any, observation: Any) -> Any:
                condition = __import__("torch").cat((observation, self.time_encoder(timestep)), dim=-1)
                hidden = self.input(sample.transpose(1, 2))
                for block in self.blocks:
                    hidden = block(hidden, condition)
                return self.output(hidden).transpose(1, 2)

        return TemporalDenoiser()


class _CameraEncoder:
    def __new__(cls, output_dim: int, image_size: int) -> Any:
        from torch import nn
        from torchvision.models import resnet18  # noqa: PLC0415

        class CameraEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.image_size = int(image_size)
                self.backbone = resnet18(weights=None, norm_layer=lambda channels: nn.GroupNorm(32, channels))
                feature_dim = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
                self.projection = nn.Sequential(nn.Linear(feature_dim, output_dim), nn.LayerNorm(output_dim), nn.SiLU())

            def forward(self, image: Any) -> Any:
                import torch.nn.functional as functional

                value = image.permute(0, 3, 1, 2).float().div_(255.0)
                value = functional.interpolate(
                    value,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                value = value.mul_(2.0).sub_(1.0)
                return self.projection(self.backbone(value))

        return CameraEncoder()


class GrootDiffusionPolicy:
    """Dual-encoder visual DP with GPU state/action normalization.

    ``ego_encoder`` and ``wrist_encoder`` never share weights. Inputs and
    outputs use the exact LeRobot feature keys, and :meth:`predict_action`
    returns decoded physical 19-D targets suitable for ``env.step``.
    """

    def __new__(
        cls,
        *,
        state_min: Any,
        state_max: Any,
        action_min: Any,
        action_max: Any,
        config: GrootDiffusionPolicyConfig | None = None,
    ) -> Any:
        import torch
        from torch import nn

        policy_config = config or GrootDiffusionPolicyConfig()

        class Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = policy_config
                self.ego_encoder = _CameraEncoder(policy_config.camera_feature_dim, policy_config.image_size)
                self.wrist_encoder = _CameraEncoder(policy_config.camera_feature_dim, policy_config.image_size)
                self.state_encoder = nn.Sequential(
                    nn.Linear(policy_config.state_dim, policy_config.state_feature_dim),
                    nn.LayerNorm(policy_config.state_feature_dim),
                    nn.SiLU(),
                    nn.Linear(policy_config.state_feature_dim, policy_config.state_feature_dim),
                )
                frame_dim = policy_config.camera_feature_dim * 2 + policy_config.state_feature_dim
                self.denoiser = _TemporalDenoiser(
                    policy_config.action_dim,
                    policy_config.denoiser_width,
                    frame_dim * policy_config.obs_horizon,
                )
                self.register_buffer("state_min", torch.as_tensor(state_min, dtype=torch.float32))
                self.register_buffer("state_max", torch.as_tensor(state_max, dtype=torch.float32))
                self.register_buffer("action_min", torch.as_tensor(action_min, dtype=torch.float32))
                self.register_buffer("action_max", torch.as_tensor(action_max, dtype=torch.float32))

            @staticmethod
            def _normalize(value: Any, minimum: Any, maximum: Any) -> Any:
                span = torch.clamp(maximum - minimum, min=1.0e-6)
                return 2.0 * (value - minimum) / span - 1.0

            @staticmethod
            def _unnormalize(value: Any, minimum: Any, maximum: Any) -> Any:
                return 0.5 * (value + 1.0) * (maximum - minimum) + minimum

            def encode_observation(self, batch: dict[str, Any]) -> Any:
                state = batch[STATE_KEY]
                ego = batch[EGO_KEY]
                wrist = batch[WRIST_KEY]
                if state.ndim != 3 or state.shape[1:] != (
                    self.config.obs_horizon,
                    self.config.state_dim,
                ):
                    raise ValueError(f"Expected {STATE_KEY} [B,{self.config.obs_horizon},26], got {state.shape}")
                batch_size, horizon = state.shape[:2]
                ego_feature = self.ego_encoder(ego.reshape(-1, *ego.shape[2:])).reshape(batch_size, horizon, -1)
                wrist_feature = self.wrist_encoder(wrist.reshape(-1, *wrist.shape[2:])).reshape(batch_size, horizon, -1)
                normalized_state = self._normalize(state.float(), self.state_min, self.state_max)
                state_feature = self.state_encoder(normalized_state)
                return torch.cat((ego_feature, wrist_feature, state_feature), dim=-1).flatten(1)

            def compute_loss(self, batch: dict[str, Any], scheduler: Any) -> Any:
                action = self._normalize(batch[ACTION_KEY].float(), self.action_min, self.action_max)
                noise = torch.randn_like(action)
                timestep = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (action.shape[0],),
                    device=action.device,
                    dtype=torch.long,
                )
                noisy_action = scheduler.add_noise(action, noise, timestep)
                prediction = self.denoiser(noisy_action, timestep, self.encode_observation(batch))
                loss = (prediction - noise).square().mean(dim=-1)
                if "action_is_pad" in batch:
                    valid = ~batch["action_is_pad"].bool()
                    return (loss * valid).sum() / torch.clamp(valid.sum(), min=1)
                return loss.mean()

            @torch.no_grad()
            def predict_action(self, observation: dict[str, Any], scheduler: Any, *, inference_steps: int = 10) -> Any:
                condition = self.encode_observation(observation)
                sample = torch.randn(
                    (condition.shape[0], self.config.pred_horizon, self.config.action_dim),
                    dtype=condition.dtype,
                    device=condition.device,
                )
                scheduler.set_timesteps(inference_steps, device=condition.device)
                for timestep in scheduler.timesteps:
                    timestep_batch = timestep.expand(condition.shape[0])
                    noise = self.denoiser(sample, timestep_batch, condition)
                    sample = scheduler.step(noise, timestep, sample).prev_sample
                sample = torch.clamp(sample, -1.0, 1.0)
                return self._unnormalize(sample, self.action_min, self.action_max)

        return Policy()
