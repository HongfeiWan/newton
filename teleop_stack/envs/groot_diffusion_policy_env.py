# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diffusion Policy adapter for :class:`GrootNewtonEnv`."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

try:
    import gymnasium as gym
except ImportError:
    gym = None

from .groot_newton_env import GrootNewtonEnv


class GrootDiffusionPolicyEnv:
    """Add GPU observation history and action-chunk execution to an environment.

    Observations follow the convention used by ManiSkill's Diffusion Policy
    baseline: the history dimension follows the environment batch dimension.
    The adapter keeps two preallocated CUDA history buffers and never moves a
    rollout observation to the CPU.
    """

    def __init__(self, env: GrootNewtonEnv, *, obs_horizon: int = 2, action_horizon: int = 8):
        if obs_horizon < 1 or action_horizon < 1:
            raise ValueError("obs_horizon and action_horizon must be positive")
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.action_size = env.action_size
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_space = env.action_space
        self.single_action_space = env.single_action_space
        self.single_observation_space, self.observation_space = self._setup_observation_spaces()
        self._history: dict[str, Any] | None = None
        self._next_history: dict[str, Any] | None = None

    @property
    def unwrapped(self) -> GrootNewtonEnv:
        """Return the underlying Newton environment."""
        return self.env

    def _setup_observation_spaces(self) -> tuple[Any, Any]:
        """Describe the policy observation history without dense image bounds."""
        if gym is None:
            return None, None

        class TensorSequenceSpace(gym.Space):
            def __init__(self, low: float | int, high: float | int, shape: tuple[int, ...], dtype: np.dtype):
                super().__init__(shape=shape, dtype=dtype)
                self.low = np.asarray(low, dtype=dtype)
                self.high = np.asarray(high, dtype=dtype)

            def sample(self, mask: Any | None = None, probability: Any | None = None) -> np.ndarray:
                if mask is not None or probability is not None:
                    raise ValueError("TensorSequenceSpace does not support masked sampling")
                if np.issubdtype(self.dtype, np.bool_):
                    return self.np_random.integers(0, 2, size=self.shape, dtype=np.int8).astype(np.bool_)
                if np.issubdtype(self.dtype, np.integer):
                    return self.np_random.integers(self.low, self.high + 1, size=self.shape, dtype=self.dtype)
                low = -1.0 if not np.isfinite(self.low) else self.low
                high = 1.0 if not np.isfinite(self.high) else self.high
                return self.np_random.uniform(low, high, size=self.shape).astype(self.dtype)

            def contains(self, value: Any) -> bool:
                array = np.asarray(value)
                return array.shape == self.shape and np.can_cast(array.dtype, self.dtype)

        def tree_space(value: Any, *, batched: bool) -> gym.Space:
            if isinstance(value, dict):
                return gym.spaces.Dict({key: tree_space(child, batched=batched) for key, child in value.items()})
            prefix = (self.num_envs, self.obs_horizon) if batched else (self.obs_horizon,)
            shape = (*prefix, *value.shape[1:])
            if value.dtype == wp.uint8:
                return TensorSequenceSpace(0, 255, shape, np.dtype(np.uint8))
            if value.dtype == wp.bool:
                return TensorSequenceSpace(False, True, shape, np.dtype(np.bool_))
            if value.dtype == wp.int32:
                return TensorSequenceSpace(
                    np.iinfo(np.int32).min,
                    np.iinfo(np.int32).max,
                    shape,
                    np.dtype(np.int32),
                )
            return TensorSequenceSpace(-np.inf, np.inf, shape, np.dtype(np.float32))

        observation = self.env.policy_observation_warp()
        return tree_space(observation, batched=False), tree_space(observation, batched=True)

    def _allocate_history(self, frame: dict[str, Any]) -> None:
        import torch

        def allocate(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: allocate(child) for key, child in value.items()}
            return torch.empty(
                (value.shape[0], self.obs_horizon, *value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            )

        self._history = allocate(frame)
        self._next_history = allocate(frame)

    @staticmethod
    def _tree_items(tree: dict[str, Any], prefix: tuple[str, ...] = ()):
        for key, value in tree.items():
            path = (*prefix, key)
            if isinstance(value, dict):
                yield from GrootDiffusionPolicyEnv._tree_items(value, path)
            else:
                yield path, value

    @staticmethod
    def _tree_get(tree: dict[str, Any], path: tuple[str, ...]) -> Any:
        value: Any = tree
        for key in path:
            value = value[key]
        return value

    def _fill_history(self, frame: dict[str, Any], env_idx: Any | None = None) -> None:
        if self._history is None or self._next_history is None:
            self._allocate_history(frame)
        assert self._history is not None and self._next_history is not None
        for path, value in self._tree_items(frame):
            current = self._tree_get(self._history, path)
            alternate = self._tree_get(self._next_history, path)
            if env_idx is None:
                for history_step in range(self.obs_horizon):
                    current[:, history_step].copy_(value)
                    alternate[:, history_step].copy_(value)
            else:
                for history_step in range(self.obs_horizon):
                    current[env_idx, history_step] = value[env_idx]
                    alternate[env_idx, history_step] = value[env_idx]

    def _push_history(self, frame: dict[str, Any]) -> dict[str, Any]:
        if self._history is None or self._next_history is None:
            self._fill_history(frame)
            assert self._history is not None
            return self._history
        for path, value in self._tree_items(frame):
            current = self._tree_get(self._history, path)
            alternate = self._tree_get(self._next_history, path)
            if self.obs_horizon > 1:
                alternate[:, :-1].copy_(current[:, 1:])
            alternate[:, -1].copy_(value)
        self._history, self._next_history = self._next_history, self._history
        return self._history

    @staticmethod
    def _reset_indices(world_mask: Any | None, options: dict[str, Any] | None) -> Any | None:
        import torch

        if options is not None and "env_idx" in options:
            return options["env_idx"]
        if world_mask is None:
            return None
        if hasattr(world_mask, "dtype") and isinstance(world_mask, torch.Tensor):
            return torch.where(world_mask)[0]
        if isinstance(world_mask, wp.array):
            return torch.where(wp.to_torch(world_mask))[0]
        return world_mask

    def reset(
        self,
        world_mask: Any | None = None,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset worlds and fill their observation history with the reset frame."""
        _, info = self.env.reset(world_mask, seed=seed, options=options)
        frame = self.env.policy_observation()
        self._fill_history(frame, self._reset_indices(world_mask, options))
        assert self._history is not None
        return self._history, info

    def step(self, action: Any) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
        """Execute one action or a ``[num_envs, horizon, action_size]`` chunk.

        For a chunk, rewards are summed until each world first terminates or
        truncates. The returned observation is the final history after the
        whole chunk. Callers should avoid crossing an episode boundary inside
        a chunk when an exact terminal observation is required.
        """
        import torch

        if action.ndim == 2:
            _, reward, terminated, truncated, info = self.env.step(action)
            return self._push_history(self.env.policy_observation()), reward, terminated, truncated, info
        if action.ndim != 3 or action.shape[0] != self.num_envs or action.shape[2] != self.action_size:
            raise ValueError(
                f"action must have shape ({self.num_envs}, {self.action_size}) or "
                f"({self.num_envs}, horizon, {self.action_size}), got {tuple(action.shape)}"
            )
        if action.shape[1] < 1 or action.shape[1] > self.action_horizon:
            raise ValueError(f"action chunk horizon must be in [1, {self.action_horizon}], got {action.shape[1]}")

        reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=action.device)
        terminated_any = torch.zeros(self.num_envs, dtype=torch.bool, device=action.device)
        truncated_any = torch.zeros(self.num_envs, dtype=torch.bool, device=action.device)
        executed_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=action.device)
        info: dict[str, Any] = {}
        for chunk_step in range(action.shape[1]):
            active = ~(terminated_any | truncated_any)
            _, reward, terminated, truncated, info = self.env.step(action[:, chunk_step])
            reward_sum.add_(reward * active)
            executed_steps.add_(active)
            terminated_any.logical_or_(terminated)
            truncated_any.logical_or_(truncated)
            self._push_history(self.env.policy_observation())
        info = {**info, "action_chunk": {"executed_steps": executed_steps}}
        assert self._history is not None
        return self._history, reward_sum, terminated_any, truncated_any, info

    def close(self) -> None:
        """Close the underlying environment."""
        self.env.close()
