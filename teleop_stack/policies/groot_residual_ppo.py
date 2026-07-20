# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Residual PPO components for a frozen GR00T Diffusion Policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

GROOT_RESIDUAL_PPO_CHECKPOINT_FORMAT = "teleop_stack.groot_l10_residual_ppo.v1"

_EEF_POSITION_DIM = 3
_EEF_ROTATION_DIM = 6
_HAND_ACTION_DIM = 10
_PHYSICAL_ACTION_DIM = _EEF_POSITION_DIM + _EEF_ROTATION_DIM + _HAND_ACTION_DIM
_RESIDUAL_ACTION_DIM = _EEF_POSITION_DIM + 3 + _HAND_ACTION_DIM
_WORLD_FROM_ACTION_ROTATION_CACHE: dict[tuple[Any, ...], Any] = {}


@dataclass(frozen=True)
class GrootResidualActorCriticConfig:
    """Dimensions and initialization for the residual actor-critic.

    ``condition_dim`` is the size of the frozen DP observation embedding. The
    actor receives that embedding followed by the normalized 19-D baseline
    action, the normalized current 26-D robot state, an 8-D one-hot cached
    action-row index, the live normalized 26-D state delta, and five live
    finger-root motor loads. The critic receives the same policy input plus a
    configured privileged task-state vector (20-D by default; trainers may
    explicitly extend it when their checkpoint contract records that shape).

    Args:
        condition_dim: Frozen Diffusion Policy condition dimension.
        base_action_dim: Absolute EEF-pose and hand-target dimension.
        current_state_dim: Current robot-state dimension.
        base_action_row_dim: Number of cached DP rows encoded one-hot.
        state_delta_dim: Live normalized robot-state delta dimension.
        finger_root_load_dim: Number of live finger-root motor loads.
        privileged_dim: Critic-only task-state dimension.
        residual_dim: Raw Gaussian residual dimension.
        hidden_dim: Width of each actor and critic hidden layer.
        initial_log_std: Initial per-dimension Gaussian log standard deviation.
    """

    condition_dim: int
    base_action_dim: int = 19
    current_state_dim: int = 26
    base_action_row_dim: int = 8
    state_delta_dim: int = 26
    finger_root_load_dim: int = 5
    privileged_dim: int = 20
    residual_dim: int = 16
    hidden_dim: int = 512
    initial_log_std: float = -1.5

    def __post_init__(self) -> None:
        if self.condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        if self.base_action_dim != _PHYSICAL_ACTION_DIM:
            raise ValueError(f"base_action_dim must be {_PHYSICAL_ACTION_DIM}")
        if self.current_state_dim <= 0:
            raise ValueError("current_state_dim must be positive")
        if self.base_action_row_dim <= 0:
            raise ValueError("base_action_row_dim must be positive")
        if self.state_delta_dim != self.current_state_dim:
            raise ValueError("state_delta_dim must equal current_state_dim")
        if self.finger_root_load_dim != 5:
            raise ValueError("finger_root_load_dim must be 5")
        if self.privileged_dim <= 0:
            raise ValueError("privileged_dim must be positive")
        if self.residual_dim != _RESIDUAL_ACTION_DIM:
            raise ValueError(f"residual_dim must be {_RESIDUAL_ACTION_DIM}")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not math.isfinite(self.initial_log_std):
            raise ValueError("initial_log_std must be finite")

    @property
    def policy_input_dim(self) -> int:
        """Actor input dimension including all cached and live features."""

        return (
            self.condition_dim
            + self.base_action_dim
            + self.current_state_dim
            + self.base_action_row_dim
            + self.state_delta_dim
            + self.finger_root_load_dim
        )


def _as_action_bound(value: Any, reference: Any, name: str) -> Any:
    import torch

    bound = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if bound.shape[-1:] != (_PHYSICAL_ACTION_DIM,):
        raise ValueError(f"{name} must end in dimension {_PHYSICAL_ACTION_DIM}, got {tuple(bound.shape)}")
    return bound


def _as_hand_residual_scale(value: Any, reference: Any) -> Any:
    """Convert a scalar or per-joint hand scale without reallocating device tensors."""

    import torch

    scale = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if scale.ndim == 0:
        if scale.device.type == "cpu" and (not bool(torch.isfinite(scale)) or bool(scale < 0.0)):
            raise ValueError("hand_scale_normalized must be finite and non-negative")
        return scale
    if scale.shape != (_HAND_ACTION_DIM,):
        raise ValueError(
            f"hand_scale_normalized must be a scalar or length-{_HAND_ACTION_DIM} vector, got {tuple(scale.shape)}"
        )
    # Trainer-owned CUDA vectors are validated once before upload. Validate
    # host inputs here while avoiding a device synchronization in every step.
    if scale.device.type == "cpu" and (not bool(torch.isfinite(scale).all()) or bool((scale < 0.0).any())):
        raise ValueError("hand_scale_normalized must contain only finite non-negative values")
    return scale


def normalize_physical_action(action: Any, action_min: Any, action_max: Any) -> Any:
    """Normalize a physical 19-D action to the bounded actor-input range.

    Values outside the configured physical range are clipped to ``[-1, 1]``.
    The function consists only of Torch operations and preserves the input
    device, allowing it to remain in a batched CUDA rollout path.

    Args:
        action: Physical actions with shape ``[..., 19]``.
        action_min: Broadcastable lower physical bounds with trailing size 19.
        action_max: Broadcastable upper physical bounds with trailing size 19.

    Returns:
        Normalized actions with the same shape as ``action``.
    """

    import torch

    if action.shape[-1:] != (_PHYSICAL_ACTION_DIM,):
        raise ValueError(f"action must end in dimension {_PHYSICAL_ACTION_DIM}, got {tuple(action.shape)}")
    minimum = _as_action_bound(action_min, action, "action_min")
    maximum = _as_action_bound(action_max, action, "action_max")
    span = torch.clamp(maximum - minimum, min=1.0e-6)
    return torch.clamp(2.0 * (action - minimum) / span - 1.0, -1.0, 1.0)


def _row_first_rot6d_to_matrix(rot6d: Any) -> Any:
    import torch

    raw_row0 = rot6d[..., :3]
    raw_row1 = rot6d[..., 3:6]
    norm0 = torch.linalg.vector_norm(raw_row0, dim=-1, keepdim=True)
    row0 = raw_row0 / torch.clamp(norm0, min=1.0e-8)
    orthogonal_row1 = raw_row1 - torch.sum(row0 * raw_row1, dim=-1, keepdim=True) * row0
    norm1 = torch.linalg.vector_norm(orthogonal_row1, dim=-1, keepdim=True)
    row1 = orthogonal_row1 / torch.clamp(norm1, min=1.0e-8)
    row2 = torch.linalg.cross(row0, row1, dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _skew_symmetric(vector: Any) -> Any:
    import torch

    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*vector.shape[:-1], 3, 3)


def _rotation_matrix_from_raw_axis_angle(raw_axis_angle: Any, scale_rad: float) -> Any:
    import torch

    raw_norm = torch.linalg.vector_norm(raw_axis_angle, dim=-1, keepdim=True)
    radial_scale = scale_rad * torch.tanh(raw_norm) / torch.clamp(raw_norm, min=1.0e-8)
    radial_scale = torch.where(raw_norm > 1.0e-8, radial_scale, torch.full_like(raw_norm, scale_rad))
    rotation_vector = raw_axis_angle * radial_scale
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    angle_squared = angle.square()
    safe_angle = torch.clamp(angle, min=1.0e-8)
    safe_angle_squared = torch.clamp(angle_squared, min=1.0e-8)
    sine_ratio = torch.where(
        angle < 1.0e-4,
        1.0 - angle_squared / 6.0 + angle_squared.square() / 120.0,
        torch.sin(angle) / safe_angle,
    )
    cosine_ratio = torch.where(
        angle < 1.0e-4,
        0.5 - angle_squared / 24.0 + angle_squared.square() / 720.0,
        (1.0 - torch.cos(angle)) / safe_angle_squared,
    )
    skew = _skew_symmetric(rotation_vector)
    identity = torch.eye(3, dtype=raw_axis_angle.dtype, device=raw_axis_angle.device)
    return identity + sine_ratio[..., None] * skew + cosine_ratio[..., None] * torch.matmul(skew, skew)


def validate_world_from_action_rotation(rotation: Any | None, reference: Any) -> Any:
    """Return a validated fixed rotation from action position to world XYZ.

    Args:
        rotation: A finite, orthonormal, right-handed 3-by-3 rotation. ``None``
            selects identity for callers whose action position is already in
            the world frame.
        reference: Tensor providing the output device and floating-point dtype.

    Returns:
        The rotation as a tensor on the reference device.

    Raises:
        ValueError: If the supplied matrix is not a proper rotation.
    """

    import torch

    if rotation is None:
        return torch.eye(3, dtype=reference.dtype, device=reference.device)
    is_tensor = torch.is_tensor(rotation)
    validation_matrix = rotation.float() if is_tensor else torch.as_tensor(rotation, dtype=torch.float64)
    if validation_matrix.shape != (3, 3):
        raise ValueError(f"world_from_action_rotation must have shape (3, 3), got {tuple(validation_matrix.shape)}")
    if not bool(torch.isfinite(validation_matrix).all()):
        raise ValueError("world_from_action_rotation must contain only finite values")
    identity = torch.eye(3, dtype=validation_matrix.dtype, device=validation_matrix.device)
    if not bool(
        torch.allclose(
            validation_matrix.transpose(-1, -2) @ validation_matrix,
            identity,
            atol=1.0e-5,
            rtol=1.0e-5,
        )
    ):
        raise ValueError("world_from_action_rotation must be orthonormal")
    determinant = torch.linalg.det(validation_matrix)
    if not bool(torch.isclose(determinant, determinant.new_tensor(1.0), atol=1.0e-5, rtol=1.0e-5)):
        raise ValueError("world_from_action_rotation must be right-handed with determinant +1")
    if is_tensor:
        return rotation.to(dtype=reference.dtype, device=reference.device)

    # Trainer configs use immutable host values. Cache the tiny device tensor so
    # frame validation never introduces a device-to-host synchronization in the
    # rollout loop.
    values = tuple(float(value) for value in validation_matrix.reshape(-1))
    cache_key = (*values, reference.dtype, reference.device.type, reference.device.index)
    matrix = _WORLD_FROM_ACTION_ROTATION_CACHE.get(cache_key)
    if matrix is None:
        matrix = validation_matrix.to(dtype=reference.dtype, device=reference.device)
        _WORLD_FROM_ACTION_ROTATION_CACHE[cache_key] = matrix
    return matrix


def compose_residual_action(
    base_action: Any,
    raw_latent: Any,
    action_min: Any,
    action_max: Any,
    *,
    position_scale_m: float = 0.02,
    vertical_position_scale_m: float | None = None,
    rotation_scale_rad: float = math.radians(10.0),
    hand_scale_normalized: Any = 0.1,
    world_from_action_rotation: Any | None = None,
) -> Any:
    """Compose a bounded physical action from a DP baseline and PPO latent.

    The raw Gaussian latent is mapped as ``xyz[3] + axis-angle[3] + hand[10]``.
    The first three latent values always mean world-frame XYZ. Translation uses
    bounded world-horizontal and world-vertical offsets, then rotates the
    offset back into the action position frame before adding it to the DP
    baseline and applying action-frame bounds. Rotation applies a bounded
    axis-angle in the EEF local frame by right-multiplying the baseline
    rotation. Hand offsets are applied in normalized action coordinates.
    Output rotation remains canonical row-first rot6d.

    Args:
        base_action: Frozen DP physical action with shape ``[..., 19]``.
        raw_latent: Unsquashed Gaussian PPO sample with shape ``[..., 16]``.
        action_min: Broadcastable lower physical bounds with trailing size 19.
        action_max: Broadcastable upper physical bounds with trailing size 19.
        position_scale_m: Maximum x/y translation residual [m], and maximum z
            residual when ``vertical_position_scale_m`` is ``None``.
        vertical_position_scale_m: Maximum z translation residual [m].
        rotation_scale_rad: Maximum local residual rotation angle [rad].
        hand_scale_normalized: Maximum hand offset in ``[-1, 1]`` units,
            supplied as either one scalar or a length-10 per-joint vector.
        world_from_action_rotation: Fixed rotation satisfying
            ``position_world = R @ position_action``. ``None`` selects
            identity for backward-compatible generic use.

    Returns:
        Bounded physical actions with shape ``[..., 19]``.
    """

    import torch

    if base_action.shape[-1:] != (_PHYSICAL_ACTION_DIM,):
        raise ValueError(f"base_action must end in dimension {_PHYSICAL_ACTION_DIM}, got {tuple(base_action.shape)}")
    if raw_latent.shape[-1:] != (_RESIDUAL_ACTION_DIM,):
        raise ValueError(f"raw_latent must end in dimension {_RESIDUAL_ACTION_DIM}, got {tuple(raw_latent.shape)}")
    if base_action.shape[:-1] != raw_latent.shape[:-1]:
        raise ValueError(f"base_action and raw_latent batch shapes differ: {base_action.shape} vs {raw_latent.shape}")
    vertical_scale_m = position_scale_m if vertical_position_scale_m is None else vertical_position_scale_m
    if min(position_scale_m, vertical_scale_m, rotation_scale_rad) < 0.0:
        raise ValueError("residual scales must be non-negative")
    hand_scale = _as_hand_residual_scale(hand_scale_normalized, base_action)

    minimum = _as_action_bound(action_min, base_action, "action_min")
    maximum = _as_action_bound(action_max, base_action, "action_max")

    rotation_world_from_action = validate_world_from_action_rotation(world_from_action_rotation, base_action)
    position_scale = base_action.new_tensor((position_scale_m, position_scale_m, vertical_scale_m))
    position_residual_world = position_scale * torch.tanh(raw_latent[..., :3])
    position_residual_action = torch.matmul(position_residual_world, rotation_world_from_action)
    position = base_action[..., :3] + position_residual_action
    position = torch.maximum(torch.minimum(position, maximum[..., :3]), minimum[..., :3])

    base_rotation = _row_first_rot6d_to_matrix(base_action[..., 3:9])
    local_residual = _rotation_matrix_from_raw_axis_angle(raw_latent[..., 3:6], rotation_scale_rad)
    rotation = torch.matmul(base_rotation, local_residual)
    row_first_rot6d = rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)

    hand_minimum = minimum[..., 9:19]
    hand_maximum = maximum[..., 9:19]
    hand_span = torch.clamp(hand_maximum - hand_minimum, min=1.0e-6)
    normalized_hand = 2.0 * (base_action[..., 9:19] - hand_minimum) / hand_span - 1.0
    normalized_hand = torch.clamp(
        normalized_hand + hand_scale * torch.tanh(raw_latent[..., 6:16]),
        -1.0,
        1.0,
    )
    hand = 0.5 * (normalized_hand + 1.0) * hand_span + hand_minimum
    return torch.cat((position, row_first_rot6d, hand), dim=-1)


def compute_gae(
    rewards: Any,
    values: Any,
    terminated: Any,
    truncated: Any,
    timeout_values: Any,
    last_value: Any,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap_time_limit: bool = True,
) -> tuple[Any, Any]:
    """Compute GAE without leaking across terminated or truncated episodes.

    A true termination suppresses value bootstrapping. A time-limit truncation
    can retain the value of its terminal observation in the TD residual, while
    both conditions stop the recursive GAE chain. Consequently,
    ``timeout_values`` must contain pre-reset terminal-observation values.

    Args:
        rewards: Rewards with time-major shape ``[T, ...]``.
        values: Critic values for current observations, matching ``rewards``.
        terminated: True environment-terminal flags.
        truncated: Time-limit or horizon-truncation flags.
        timeout_values: Terminal-observation values for truncated transitions.
        last_value: Value after the final rollout transition, shape ``[...]``.
        gamma: Discount factor.
        gae_lambda: Generalized advantage-estimation factor.
        bootstrap_time_limit: Whether to bootstrap truncated transitions.

    Returns:
        ``(advantages, returns)`` tensors matching ``rewards``.
    """

    import torch

    expected_shape = rewards.shape
    for name, value in (
        ("values", values),
        ("terminated", terminated),
        ("truncated", truncated),
        ("timeout_values", timeout_values),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}, got {tuple(value.shape)}")
    if rewards.ndim < 1:
        raise ValueError("rewards must have a time dimension")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if last_value.shape != rewards.shape[1:]:
        raise ValueError(f"last_value must have shape {tuple(rewards.shape[1:])}, got {tuple(last_value.shape)}")

    with torch.no_grad():
        terminated_mask = terminated.to(dtype=torch.bool)
        truncated_mask = truncated.to(dtype=torch.bool)
        done_mask = terminated_mask | truncated_mask
        advantages = torch.empty_like(rewards)
        running = torch.zeros_like(last_value)
        next_value = last_value
        for step in range(rewards.shape[0] - 1, -1, -1):
            bootstrap_value = torch.where(terminated_mask[step], torch.zeros_like(next_value), next_value)
            if bootstrap_time_limit:
                timeout_only = truncated_mask[step] & ~terminated_mask[step]
                bootstrap_value = torch.where(timeout_only, timeout_values[step], bootstrap_value)
            else:
                bootstrap_value = torch.where(truncated_mask[step], torch.zeros_like(next_value), bootstrap_value)
            delta = rewards[step] + gamma * bootstrap_value - values[step]
            continuation = (~done_mask[step]).to(dtype=rewards.dtype)
            running = delta + gamma * gae_lambda * continuation * running
            advantages[step] = running
            next_value = values[step]
        returns = advantages + values
    return advantages, returns


class GrootResidualActorCritic:
    """Factory for the asymmetric residual PPO actor-critic module."""

    def __new__(cls, config: GrootResidualActorCriticConfig) -> Any:
        import torch
        from torch import nn

        class ActorCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = config
                self.actor = nn.Sequential(
                    nn.Linear(config.policy_input_dim, config.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(config.hidden_dim, config.residual_dim),
                )
                self.critic = nn.Sequential(
                    nn.Linear(config.policy_input_dim + config.privileged_dim, config.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(config.hidden_dim, 1),
                )
                self.log_std = nn.Parameter(torch.full((config.residual_dim,), config.initial_log_std))
                self._initialize_parameters()

            def _initialize_parameters(self) -> None:
                for network in (self.actor, self.critic):
                    linear_layers = [module for module in network if isinstance(module, nn.Linear)]
                    for layer in linear_layers[:-1]:
                        nn.init.orthogonal_(layer.weight, math.sqrt(2.0))
                        nn.init.zeros_(layer.bias)
                    if network is self.actor:
                        nn.init.zeros_(linear_layers[-1].weight)
                    else:
                        nn.init.orthogonal_(linear_layers[-1].weight, 1.0)
                    nn.init.zeros_(linear_layers[-1].bias)

            def _validate_inputs(self, policy_input: Any, privileged: Any) -> None:
                if policy_input.shape[-1:] != (config.policy_input_dim,):
                    raise ValueError(
                        f"policy_input must end in dimension {config.policy_input_dim}, got {tuple(policy_input.shape)}"
                    )
                if privileged.shape[-1:] != (config.privileged_dim,):
                    raise ValueError(
                        f"privileged must end in dimension {config.privileged_dim}, got {tuple(privileged.shape)}"
                    )
                if policy_input.shape[:-1] != privileged.shape[:-1]:
                    raise ValueError(
                        f"policy_input and privileged batch shapes differ: {policy_input.shape} vs {privileged.shape}"
                    )

            def _statistics(self, policy_input: Any) -> tuple[Any, Any]:
                mean = self.actor(policy_input)
                log_std = torch.clamp(self.log_std, min=-5.0, max=0.0).expand_as(mean)
                return mean, log_std

            @staticmethod
            def _log_prob(raw_latent: Any, mean: Any, log_std: Any) -> Any:
                standardized = (raw_latent - mean) * torch.exp(-log_std)
                return (-0.5 * standardized.square() - log_std - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)

            @staticmethod
            def _entropy(log_std: Any) -> Any:
                return (log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))).sum(dim=-1)

            def value(self, policy_input: Any, privileged: Any) -> Any:
                """Evaluate the asymmetric critic."""

                self._validate_inputs(policy_input, privileged)
                return self.critic(torch.cat((policy_input, privileged), dim=-1)).squeeze(-1)

            def get_value(self, policy_input: Any, privileged: Any) -> Any:
                """Evaluate the asymmetric critic for trainer compatibility."""

                return self.value(policy_input, privileged)

            def act(
                self,
                policy_input: Any,
                privileged: Any,
                generator: Any | None = None,
                deterministic: bool = False,
            ) -> tuple[Any, Any, Any, Any]:
                """Sample an unsquashed residual and evaluate its rollout data."""

                self._validate_inputs(policy_input, privileged)
                mean, log_std = self._statistics(policy_input)
                if deterministic:
                    raw_latent = mean
                else:
                    noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
                    raw_latent = mean + torch.exp(log_std) * noise
                log_prob = self._log_prob(raw_latent, mean, log_std)
                entropy = self._entropy(log_std)
                return raw_latent, log_prob, entropy, self.value(policy_input, privileged)

            def evaluate_actions(
                self,
                policy_input: Any,
                privileged: Any,
                raw_latent: Any,
            ) -> tuple[Any, Any, Any]:
                """Re-evaluate stored raw Gaussian actions for a PPO update."""

                self._validate_inputs(policy_input, privileged)
                if raw_latent.shape[:-1] != policy_input.shape[:-1] or raw_latent.shape[-1:] != (config.residual_dim,):
                    raise ValueError(
                        f"raw_latent must have shape {(*policy_input.shape[:-1], config.residual_dim)}, "
                        f"got {tuple(raw_latent.shape)}"
                    )
                mean, log_std = self._statistics(policy_input)
                log_prob = self._log_prob(raw_latent, mean, log_std)
                entropy = self._entropy(log_std)
                return log_prob, entropy, self.value(policy_input, privileged)

        return ActorCritic()


__all__ = [
    "GROOT_RESIDUAL_PPO_CHECKPOINT_FORMAT",
    "GrootResidualActorCritic",
    "GrootResidualActorCriticConfig",
    "compose_residual_action",
    "compute_gae",
    "normalize_physical_action",
]
