#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune a frozen Groot Diffusion Policy with a residual PPO policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from teleop_stack.envs import GrootDiffusionPolicyEnv, GrootNewtonEnv, GrootNewtonEnvConfig
from teleop_stack.policies import (
    GROOT_DP_CHECKPOINT_FORMAT,
    GROOT_RESIDUAL_PPO_CHECKPOINT_FORMAT,
    GrootDiffusionPolicy,
    GrootDiffusionPolicyConfig,
    GrootResidualActorCritic,
    GrootResidualActorCriticConfig,
    compose_residual_action,
    compute_gae,
    normalize_physical_action,
    validate_world_from_action_rotation,
)

_PRIVILEGED_STATE_DIM = 25
_TASK_PHASE_COUNT = 5
_TRAINING_CONTRACT_VERSION = 9
_REWARD_CONTRACT_VERSION = 13
_BASE_ACTION_MODE = "per_lane_cached_rows_0_7"
_BASE_ACTION_HORIZON = 8
_ACTOR_CONDITION_SOURCE = "chunk_plan_plus_live_state_delta_finger_load_and_row"
_CRITIC_PRIVILEGED_SOURCE = "task_state_plus_reward_v13_r0_cN_cT_GN_GT"
_RESET_CACHE_POLICY = "invalidate_selected_lanes"
_RESUME_CACHE_POLICY = "reset_env_and_invalidate_all"
_EEF_POSITION_FRAME = "right_nero_can_base"
_RESIDUAL_POSITION_FRAME = "world_xyz"
_HAND_TARGET_SEMANTICS = "active_and_clamped_mimic_follower_position_targets"
_R_WORLD_FROM_ACTION = (
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
)
_STATE_KEY = "observation.state"
_FINGER_ROOT_LOAD_KEY = "observation.finger_root_load"
_FINGER_ROOT_LOAD_DIM = 5
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_THUMB_HAND_INDICES = (0, 1, 9)
_THUMB_LATENT_INDICES = (6, 7, 15)
_THUMB_JOINT_NAMES = ("thumb_cmc_pitch", "thumb_cmc_yaw", "thumb_cmc_roll")
_CONTACT_TOPOLOGY_NAMES = ("thumb_contact", "non_thumb_contact", "opposed_grasp", "thumb_only_contact")
_PARTIAL_CONTACT_STAGE_NAMES = ("non_thumb_only", "thumb_only", "bilateral_unconfirmed")
_PREGRASP_DISTANCE_SCALE_M = 0.08
_REWARD_V13_SIGNAL_NAMES = (
    "reward_v13_r0",
    "reward_v13_cN",
    "reward_v13_cT",
    "reward_v13_GN",
    "reward_v13_GT",
    "reward_v13_thumb_gap_m",
    "reward_v13_thumb_proximity",
    "reward_v13_guidance_opposition",
    "reward_v13_guidance_z",
    "reward_v13_unilateral_guidance_gain",
    "reward_v13_unilateral_contact_reward",
)
_NO_LOAD_THRESHOLD = 0.10
_TWO_FINGER_LOAD_THRESHOLD = 0.20
_LIFT_DIAGNOSTIC_THRESHOLDS_M = (
    ("1mm", 0.001),
    ("10mm", 0.010),
    ("50mm", 0.050),
)
_RESUME_TRAIN_ARG_NAMES = (
    "num_envs",
    "rollout_steps",
    "update_epochs",
    "minibatch_size",
    "learning_rate",
    "anneal_lr",
    "gamma",
    "gae_lambda",
    "clip_coef",
    "clip_vloss",
    "value_coef",
    "entropy_coef",
    "max_grad_norm",
    "target_kl",
    "hidden_dim",
    "initial_log_std",
    "seed",
    "inference_steps",
    "position_residual_scale_m",
    "vertical_residual_scale_m",
    "rotation_residual_scale_deg",
    "hand_residual_scale_normalized",
    "thumb_residual_scales_normalized",
    "bootstrap_time_limit",
    "bfloat16",
    "triangle_pairs_per_env",
)
_PHASE_NAMES = ("approach", "carrying", "released", "success", "fail")
_EVENT_INFO_KEYS = {
    "contact": "had_hand_contact_this_control_step",
    "grasp": "grasp_confirmed",
    "transport": "transport_started",
    "lift": "is_lifted",
    "release_ready": "release_ready",
    "release_armed": "release_armed",
    "released": "released",
    "early_release": "early_release",
    "success": "success",
    "fail": "fail",
}
_ACTION_DIAGNOSTIC_NAMES = (
    "position_residual_tanh_saturation_fraction",
    "rotation_residual_tanh_saturation_fraction",
    "hand_residual_tanh_saturation_fraction",
    "position_action_clamp_fraction",
    "vertical_action_clamp_fraction",
    "hand_action_clamp_fraction",
    "vertical_residual_signed_m",
    "vertical_residual_abs_m",
    "base_target_z_minus_current_eef_z_m",
    "composed_target_z_minus_current_eef_z_m",
    *(
        f"{joint_name}_{suffix}"
        for joint_name in _THUMB_JOINT_NAMES
        for suffix in (
            "residual_signed_normalized",
            "residual_abs_normalized",
            "residual_tanh_saturation_fraction",
            "absolute_bound_clamp_fraction",
            "dynamic_rate_limit_fraction",
        )
    ),
)
_PHASE_CONDITIONED_Z_NAMES = (
    "vertical_residual_signed_m",
    "base_target_z_minus_current_eef_z_m",
    "composed_target_z_minus_current_eef_z_m",
    "actual_eef_delta_z_m",
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Frozen Diffusion Policy checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/residual_ppo/groot_l10_transfer"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--initial-log-std", type=float, default=-1.5)
    parser.add_argument("--position-residual-scale-m", type=float, default=0.015)
    parser.add_argument("--vertical-residual-scale-m", type=float, default=0.05)
    parser.add_argument("--rotation-residual-scale-deg", type=float, default=5.0)
    parser.add_argument("--hand-residual-scale-normalized", type=float, default=0.1)
    parser.add_argument(
        "--thumb-residual-scales-normalized",
        type=float,
        nargs=3,
        default=None,
        metavar=("PITCH", "YAW", "ROLL"),
        help="Optional normalized residual scales for thumb pitch/yaw/roll; other hand joints use the scalar scale",
    )
    parser.add_argument("--inference-steps", type=int, default=6)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--bootstrap-time-limit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-every-updates", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--save-every-updates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-textures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hydroelastic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triangle-pairs-per-env", type=int, default=131_072)
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "rollout_steps": args.rollout_steps,
        "update_epochs": args.update_epochs,
        "minibatch_size": args.minibatch_size,
        "hidden_dim": args.hidden_dim,
        "inference_steps": args.inference_steps,
        "max_episode_steps": args.max_episode_steps,
        "save_every_updates": args.save_every_updates,
        "triangle_pairs_per_env": args.triangle_pairs_per_env,
    }
    invalid = [name for name, value in positive_ints.items() if value < 1]
    if invalid:
        raise ValueError(f"Expected positive values for {invalid}")
    if args.eval_every_updates < 0 or args.eval_episodes < 0:
        raise ValueError("evaluation intervals and episode counts cannot be negative")
    if min(args.learning_rate, args.gamma, args.gae_lambda, args.clip_coef, args.max_grad_norm) <= 0.0:
        raise ValueError("PPO learning rate, discounts, clip coefficient, and gradient norm must be positive")
    if args.gamma > 1.0 or args.gae_lambda > 1.0:
        raise ValueError("gamma and gae_lambda cannot exceed one")
    if args.value_coef < 0.0 or args.entropy_coef < 0.0 or args.target_kl < 0.0:
        raise ValueError("PPO loss coefficients and target KL cannot be negative")
    residual_scales = (
        args.position_residual_scale_m,
        args.vertical_residual_scale_m,
        args.rotation_residual_scale_deg,
        args.hand_residual_scale_normalized,
    )
    if args.thumb_residual_scales_normalized is not None:
        residual_scales += tuple(args.thumb_residual_scales_normalized)
    if not all(math.isfinite(value) and value > 0.0 for value in residual_scales):
        raise ValueError("residual action scales must be positive")
    batch_size = args.num_envs * args.rollout_steps
    if args.minibatch_size > batch_size:
        raise ValueError(f"minibatch_size={args.minibatch_size} exceeds rollout batch size {batch_size}")


def _resolve_hand_residual_scales(args: argparse.Namespace) -> tuple[str, tuple[float, ...]]:
    """Resolve CLI hand scales to the contract's fixed ten-joint vector."""

    values = [float(args.hand_residual_scale_normalized)] * 10
    thumb_scales = args.thumb_residual_scales_normalized
    mode = "uniform"
    if thumb_scales is not None:
        if len(thumb_scales) != len(_THUMB_HAND_INDICES):
            raise ValueError("thumb_residual_scales_normalized must contain pitch, yaw, and roll")
        mode = "thumb_override"
        for hand_index, scale in zip(_THUMB_HAND_INDICES, thumb_scales, strict=True):
            values[hand_index] = float(scale)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("effective hand residual scales must be finite and positive")
    return mode, tuple(values)


def _hand_residual_scale_contract(args: argparse.Namespace) -> dict[str, Any]:
    mode, effective = _resolve_hand_residual_scales(args)
    return {
        "mode": mode,
        "default_scale_normalized": float(args.hand_residual_scale_normalized),
        "thumb_hand_indices": list(_THUMB_HAND_INDICES),
        "thumb_latent_indices": list(_THUMB_LATENT_INDICES),
        "thumb_joint_names": list(_THUMB_JOINT_NAMES),
        "effective_scale_normalized": list(effective),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_stats_sha256(stats: dict[str, Any]) -> str:
    canonical = json.dumps(stats, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_frozen_dp(checkpoint_path: Path, device: Any) -> tuple[Any, Any, Any, dict[str, Any], str]:
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: PLC0415

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != GROOT_DP_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Checkpoint format {checkpoint.get('format')!r} is incompatible with {GROOT_DP_CHECKPOINT_FORMAT!r}"
        )
    config = GrootDiffusionPolicyConfig(**checkpoint["config"])
    if config.state_dim != 26 or config.action_dim != 19:
        raise ValueError(f"Expected DP state/action dimensions 26/19, got {config.state_dim}/{config.action_dim}")
    stats = checkpoint.get("train_dataset_stats")
    if not isinstance(stats, dict) or checkpoint.get("train_dataset_stats_sha256") != _canonical_stats_sha256(stats):
        raise ValueError("DP checkpoint training statistics are missing or have an invalid SHA-256")
    state = checkpoint["model"]
    for name, width in (("state_min", 26), ("state_max", 26), ("action_min", 19), ("action_max", 19)):
        values = np.asarray(stats.get(name), dtype=np.float32)
        if values.shape != (width,) or not np.isfinite(values).all():
            raise ValueError(f"Checkpoint {name} must be finite with shape ({width},)")
        model_values = state.get(name)
        if model_values is None or not np.array_equal(np.asarray(model_values.cpu()), values):
            raise ValueError(f"Checkpoint {name} does not match its model normalization buffer")
    if np.any(np.asarray(stats["state_min"]) > np.asarray(stats["state_max"])):
        raise ValueError("Checkpoint state_min must not exceed state_max")
    if np.any(np.asarray(stats["action_min"]) > np.asarray(stats["action_max"])):
        raise ValueError("Checkpoint action_min must not exceed action_max")

    model = GrootDiffusionPolicy(
        state_min=stats["state_min"],
        state_max=stats["state_max"],
        action_min=stats["action_min"],
        action_max=stats["action_max"],
        config=config,
    )
    model.load_state_dict(state)
    model.eval().requires_grad_(False).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion_train_steps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    return model, config, scheduler, stats, _file_sha256(checkpoint_path)


def _select_tree(tree: Any, indices: Any) -> Any:
    if isinstance(tree, dict):
        return {key: _select_tree(value, indices) for key, value in tree.items()}
    return tree[indices]


class _EvaluationQuota:
    """Accept at most one completed episode per lane in each evaluation wave."""

    def __init__(self, episodes: int, num_envs: int, device: Any):
        import torch

        if episodes < 1 or num_envs < 1:
            raise ValueError("evaluation episodes and environment count must be positive")
        self.episodes = episodes
        self.num_envs = num_envs
        self.device = device
        self.completed = 0
        self.wave = 0
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @property
    def complete(self) -> bool:
        return self.completed == self.episodes

    @property
    def wave_complete(self) -> bool:
        return not bool(self.active.any())

    def start_wave(self) -> Any:
        import torch

        if not self.wave_complete:
            raise RuntimeError("cannot start an evaluation wave while lanes remain active")
        remaining = self.episodes - self.completed
        if remaining <= 0:
            raise RuntimeError("evaluation quota is already complete")
        count = min(remaining, self.num_envs)
        self.active.zero_()
        # Rotate the partial wave so repeated evaluations do not always favor lane zero.
        offset = (self.wave * count) % self.num_envs
        indices = (torch.arange(count, device=self.device) + offset) % self.num_envs
        self.active[indices] = True
        self.wave += 1
        return self.active.clone()

    def accept(self, done: Any) -> Any:
        if done.shape != self.active.shape:
            raise ValueError(f"done shape {tuple(done.shape)} does not match {tuple(self.active.shape)}")
        accepted = done.bool() & self.active
        self.active &= ~accepted
        self.completed += int(accepted.sum())
        if self.completed > self.episodes:
            raise RuntimeError("evaluation accepted more episodes than requested")
        return accepted


def _event_flags(info: dict[str, Any]) -> dict[str, Any]:
    return {name: info[key].bool() for name, key in _EVENT_INFO_KEYS.items()}


def _finger_contact_topology(info: dict[str, Any]) -> tuple[Any, Any, Any]:
    counts = info["finger_contact_counts"]
    if counts.ndim != 2 or counts.shape[-1] != len(_FINGER_NAMES):
        raise ValueError(f"finger_contact_counts must have shape [batch,{len(_FINGER_NAMES)}], got {counts.shape}")
    thumb_contact = counts[:, 0] > 0
    non_thumb_contact = (counts[:, 1:] > 0).any(dim=-1)
    opposed_grasp = info["is_grasped"].bool()
    if opposed_grasp.shape != thumb_contact.shape:
        raise ValueError(
            f"is_grasped shape {tuple(opposed_grasp.shape)} does not match contact topology {tuple(thumb_contact.shape)}"
        )
    return thumb_contact, non_thumb_contact, opposed_grasp


def _control_step_contact_topology(info: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    finger_contact = info["finger_contact_any_frame_this_control_step"].bool()
    if finger_contact.ndim != 2 or finger_contact.shape[-1] != len(_FINGER_NAMES):
        raise ValueError(
            "finger_contact_any_frame_this_control_step must have shape "
            f"[batch,{len(_FINGER_NAMES)}], got {finger_contact.shape}"
        )
    opposed_grasp = info["opposed_grasp_any_frame_this_control_step"].bool()
    max_consecutive = info["opposed_grasp_max_consecutive_physics_frames_this_control_step"].long()
    expected_shape = finger_contact.shape[:-1]
    if opposed_grasp.shape != expected_shape or max_consecutive.shape != expected_shape:
        raise ValueError(
            "control-step opposed-grasp diagnostics must match the contact batch shape "
            f"{expected_shape}, got {tuple(opposed_grasp.shape)} and {tuple(max_consecutive.shape)}"
        )
    thumb_contact = finger_contact[:, 0]
    non_thumb_contact = finger_contact[:, 1:].any(dim=-1)
    return finger_contact, thumb_contact, non_thumb_contact, opposed_grasp, max_consecutive


def _partial_contact_reward_stages(info: dict[str, Any]) -> dict[str, Any]:
    _, thumb_contact, non_thumb_contact, opposed_grasp, _ = _control_step_contact_topology(info)
    live_grasp = info["is_grasped"].bool()
    phase = info["task_phase"].long()
    if live_grasp.shape != thumb_contact.shape or phase.shape != thumb_contact.shape:
        raise ValueError("partial-contact reward inputs must share the control-step batch shape")
    partial = (phase == 0) & ~live_grasp
    bilateral_unconfirmed = partial & opposed_grasp
    thumb_only = partial & ~opposed_grasp & thumb_contact
    non_thumb_only = partial & ~opposed_grasp & ~thumb_contact & non_thumb_contact
    return {
        "non_thumb_only": non_thumb_only,
        "thumb_only": thumb_only,
        "bilateral_unconfirmed": bilateral_unconfirmed,
    }


def _pre_action_task_flags(info: dict[str, Any]) -> tuple[Any, Any, Any]:
    contact = info["has_hand_contact"].bool() | info["had_hand_contact_this_control_step"].bool()
    _, _, live_grasp = _finger_contact_topology(info)
    carrying = info["task_phase"].long() == 1
    return contact, live_grasp, carrying


def _action_diagnostics(
    base_action: Any,
    raw_latent: Any,
    action_min: Any,
    action_max: Any,
    current_eef_position: Any,
    current_hand_position: Any,
    *,
    position_scale_m: float,
    vertical_position_scale_m: float,
    hand_scale_normalized: Any,
    hand_max_joint_step_rad: float = 0.08,
    world_from_action_rotation: Any | None = None,
) -> dict[str, Any]:
    import torch

    bounded_latent = torch.tanh(raw_latent)
    rotation_world_from_action = validate_world_from_action_rotation(world_from_action_rotation, base_action)
    position_scale = base_action.new_tensor((position_scale_m, position_scale_m, vertical_position_scale_m))
    position_residual_world = position_scale * bounded_latent[..., :3]
    position_residual_action = torch.matmul(position_residual_world, rotation_world_from_action)
    position_candidate = base_action[..., :3] + position_residual_action
    position_clamped = (position_candidate < action_min[..., :3]) | (position_candidate > action_max[..., :3])
    composed_position = torch.maximum(
        torch.minimum(position_candidate, action_max[..., :3]),
        action_min[..., :3],
    )
    base_position_world = torch.matmul(base_action[..., :3], rotation_world_from_action.transpose(-1, -2))
    composed_position_world = torch.matmul(composed_position, rotation_world_from_action.transpose(-1, -2))
    current_position_world = torch.matmul(current_eef_position, rotation_world_from_action.transpose(-1, -2))

    hand_minimum = action_min[..., 9:19]
    hand_maximum = action_max[..., 9:19]
    hand_span = torch.clamp(hand_maximum - hand_minimum, min=1.0e-6)
    normalized_hand = 2.0 * (base_action[..., 9:19] - hand_minimum) / hand_span - 1.0
    hand_scale = torch.as_tensor(hand_scale_normalized, dtype=base_action.dtype, device=base_action.device)
    if hand_scale.ndim not in (0, 1) or (hand_scale.ndim == 1 and hand_scale.shape != (10,)):
        raise ValueError(f"hand_scale_normalized must be scalar or shape (10,), got {tuple(hand_scale.shape)}")
    normalized_residual = hand_scale * bounded_latent[..., 6:16]
    hand_candidate = normalized_hand + normalized_residual
    hand_clamped = (hand_candidate < -1.0) | (hand_candidate > 1.0)
    composed_normalized_hand = hand_candidate.clamp(-1.0, 1.0)
    composed_hand = 0.5 * (composed_normalized_hand + 1.0) * hand_span + hand_minimum
    if current_hand_position.shape != composed_hand.shape:
        raise ValueError(
            f"current_hand_position shape {tuple(current_hand_position.shape)} must match {tuple(composed_hand.shape)}"
        )
    dynamic_rate_limited = (composed_hand - current_hand_position).abs() > hand_max_joint_step_rad + 1.0e-6
    rotation_saturation = torch.tanh(torch.linalg.vector_norm(raw_latent[..., 3:6], dim=-1)).abs() >= 0.95
    diagnostics = {
        "position_residual_tanh_saturation_fraction": (bounded_latent[..., :3].abs() >= 0.95).float().mean(dim=-1),
        "rotation_residual_tanh_saturation_fraction": rotation_saturation.float(),
        "hand_residual_tanh_saturation_fraction": (bounded_latent[..., 6:16].abs() >= 0.95).float().mean(dim=-1),
        "position_action_clamp_fraction": position_clamped.float().mean(dim=-1),
        "vertical_action_clamp_fraction": position_clamped.any(dim=-1).float(),
        "hand_action_clamp_fraction": hand_clamped.float().mean(dim=-1),
        "vertical_residual_signed_m": position_residual_world[..., 2],
        "vertical_residual_abs_m": position_residual_world[..., 2].abs(),
        "base_target_z_minus_current_eef_z_m": base_position_world[..., 2] - current_position_world[..., 2],
        "composed_target_z_minus_current_eef_z_m": composed_position_world[..., 2] - current_position_world[..., 2],
    }
    for joint_name, hand_index, latent_index in zip(
        _THUMB_JOINT_NAMES,
        _THUMB_HAND_INDICES,
        _THUMB_LATENT_INDICES,
        strict=True,
    ):
        residual = normalized_residual[..., hand_index]
        diagnostics[f"{joint_name}_residual_signed_normalized"] = residual
        diagnostics[f"{joint_name}_residual_abs_normalized"] = residual.abs()
        diagnostics[f"{joint_name}_residual_tanh_saturation_fraction"] = (
            bounded_latent[..., latent_index].abs() >= 0.95
        ).float()
        diagnostics[f"{joint_name}_absolute_bound_clamp_fraction"] = hand_clamped[..., hand_index].float()
        diagnostics[f"{joint_name}_dynamic_rate_limit_fraction"] = dynamic_rate_limited[..., hand_index].float()
    return diagnostics


def _action_position_to_world(position: Any, world_from_action_rotation: Any = _R_WORLD_FROM_ACTION) -> Any:
    rotation = validate_world_from_action_rotation(world_from_action_rotation, position)
    return position @ rotation.transpose(-1, -2)


def _eval_rank(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (metrics["success_rate"], -metrics["fail_rate"], metrics["mean_return"])


def _eval_return_rank(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (metrics["mean_return"], metrics["success_rate"], -metrics["fail_rate"])


def _is_better_eval(metrics: dict[str, float], best: dict[str, float] | None) -> bool:
    return best is None or _eval_rank(metrics) > _eval_rank(best)


def _is_better_return(metrics: dict[str, float], best: dict[str, float] | None) -> bool:
    return best is None or _eval_return_rank(metrics) > _eval_return_rank(best)


def _validate_resume_training_contract(resume: dict[str, Any]) -> None:
    if resume.get("training_contract_version") != _TRAINING_CONTRACT_VERSION:
        raise ValueError(
            "Resume checkpoint uses an obsolete residual PPO training contract; "
            "contract v9 adds reward-v13 guidance to the privileged critic and must be trained from scratch"
        )
    if resume.get("reward_contract_version") != _REWARD_CONTRACT_VERSION:
        raise ValueError(
            f"Resume checkpoint reward_contract_version must be {_REWARD_CONTRACT_VERSION}; train from scratch"
        )
    if resume.get("base_action_mode") != _BASE_ACTION_MODE:
        raise ValueError(f"Resume checkpoint base_action_mode must be {_BASE_ACTION_MODE!r}")
    if resume.get("base_action_horizon") != _BASE_ACTION_HORIZON:
        raise ValueError(f"Resume checkpoint base_action_horizon must be {_BASE_ACTION_HORIZON}")
    if resume.get("actor_condition_source") != _ACTOR_CONDITION_SOURCE:
        raise ValueError(f"Resume checkpoint actor_condition_source must be {_ACTOR_CONDITION_SOURCE!r}")
    if resume.get("critic_privileged_source") != _CRITIC_PRIVILEGED_SOURCE:
        raise ValueError(f"Resume checkpoint critic_privileged_source must be {_CRITIC_PRIVILEGED_SOURCE!r}")
    if resume.get("reset_cache_policy") != _RESET_CACHE_POLICY:
        raise ValueError(f"Resume checkpoint reset_cache_policy must be {_RESET_CACHE_POLICY!r}")
    if resume.get("resume_cache_policy") != _RESUME_CACHE_POLICY:
        raise ValueError(f"Resume checkpoint resume_cache_policy must be {_RESUME_CACHE_POLICY!r}")
    if resume.get("eef_position_frame") != _EEF_POSITION_FRAME:
        raise ValueError(f"Resume checkpoint eef_position_frame must be {_EEF_POSITION_FRAME!r}")
    try:
        saved_rotation = tuple(tuple(float(value) for value in row) for row in resume["R_world_from_action"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Resume checkpoint R_world_from_action is missing or malformed") from error
    if saved_rotation != _R_WORLD_FROM_ACTION:
        raise ValueError(f"Resume checkpoint R_world_from_action must be {_R_WORLD_FROM_ACTION!r}")
    if resume.get("residual_position_frame") != _RESIDUAL_POSITION_FRAME:
        raise ValueError(f"Resume checkpoint residual_position_frame must be {_RESIDUAL_POSITION_FRAME!r}")
    if resume.get("hand_target_semantics") != _HAND_TARGET_SEMANTICS:
        raise ValueError(f"Resume checkpoint hand_target_semantics must be {_HAND_TARGET_SEMANTICS!r}")
    _validate_hand_residual_scale_contract(resume.get("hand_residual_scale"))


def _validate_hand_residual_scale_contract(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("Resume checkpoint hand_residual_scale contract is missing")
    if metadata.get("mode") not in {"uniform", "thumb_override"}:
        raise ValueError("Resume checkpoint hand_residual_scale mode is invalid")
    if metadata.get("thumb_hand_indices") != list(_THUMB_HAND_INDICES):
        raise ValueError("Resume checkpoint thumb hand indices do not match contract v9")
    if metadata.get("thumb_latent_indices") != list(_THUMB_LATENT_INDICES):
        raise ValueError("Resume checkpoint thumb latent indices do not match contract v9")
    if metadata.get("thumb_joint_names") != list(_THUMB_JOINT_NAMES):
        raise ValueError("Resume checkpoint thumb joint names do not match contract v9")
    try:
        effective = tuple(float(value) for value in metadata["effective_scale_normalized"])
        default = float(metadata["default_scale_normalized"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Resume checkpoint hand_residual_scale values are malformed") from error
    if len(effective) != 10 or not all(math.isfinite(value) and value > 0.0 for value in (*effective, default)):
        raise ValueError("Resume checkpoint effective hand residual scales must be ten finite positive values")
    if metadata["mode"] == "uniform" and any(value != default for value in effective):
        raise ValueError("Resume checkpoint uniform hand residual scale vector is inconsistent")
    if metadata["mode"] == "thumb_override" and any(
        effective[index] != default for index in range(10) if index not in _THUMB_HAND_INDICES
    ):
        raise ValueError("Resume checkpoint non-thumb residual scales do not match the configured default")


def _validate_frozen_dp_training_contract(dp_config: Any) -> None:
    if dp_config.pred_horizon < _BASE_ACTION_HORIZON:
        raise ValueError(
            f"Frozen DP pred_horizon must be >= {_BASE_ACTION_HORIZON} for contract v9, got {dp_config.pred_horizon}"
        )
    if getattr(dp_config, "obs_horizon", 2) < 2:
        raise ValueError("Frozen DP obs_horizon must be >= 2 for the contract v9 live state delta")


def _validate_resume_train_args(saved_args: dict[str, Any], args: argparse.Namespace) -> None:
    for name in _RESUME_TRAIN_ARG_NAMES:
        if saved_args.get(name) != getattr(args, name):
            raise ValueError(f"Resume checkpoint training contract differs for {name}")


def _reward_v13_signals(info: dict[str, Any]) -> dict[str, Any]:
    """Return normalized GPU guidance signals shared by the critic and diagnostics."""
    import torch

    components = info["reward_components"]
    reaching = components["reaching"].float()
    pregrasp = components["opposed_pregrasp"].float()
    r0 = (reaching.clamp(0.0, 1.0) + 0.35 * pregrasp.clamp(0.0, 1.0)).clamp(0.0, 1.35)
    cN = info["non_thumb_anchor_contact_fraction_this_control_step"].float().clamp(0.0, 1.0)
    cT = info["thumb_anchor_contact_fraction_this_control_step"].float().clamp(0.0, 1.0)
    GN = torch.minimum(
        info["non_thumb_missing_thumb_geometry_progress_this_control_step"].float().clamp(0.0, 1.0),
        cN,
    )
    GT = torch.minimum(
        info["thumb_missing_non_thumb_geometry_progress_this_control_step"].float().clamp(0.0, 1.0),
        cT,
    )
    expected_shape = r0.shape
    for name, value in (("cN", cN), ("cT", cT), ("GN", GN), ("GT", GT)):
        if value.shape != expected_shape:
            raise ValueError(f"reward v13 {name} shape {tuple(value.shape)} does not match {tuple(expected_shape)}")

    finger_surface_gap = info["finger_surface_gap"].float()
    if finger_surface_gap.shape != (*expected_shape, len(_FINGER_NAMES)):
        raise ValueError(
            "finger_surface_gap must have shape "
            f"{(*expected_shape, len(_FINGER_NAMES))}, got {tuple(finger_surface_gap.shape)}"
        )
    thumb_gap = finger_surface_gap[..., 0].clamp_min(0.0)
    thumb_proximity = 1.0 - torch.tanh(thumb_gap / _PREGRASP_DISTANCE_SCALE_M)
    opposition_progress = info["non_thumb_guidance_opposition_progress_this_control_step"].float()
    z_progress = info["non_thumb_guidance_z_progress_this_control_step"].float()
    if opposition_progress.shape != expected_shape or z_progress.shape != expected_shape:
        raise ValueError("reward v13 guidance progress must match the reward batch shape")
    has_non_thumb_anchor = cN > 0.0
    guidance_opposition = torch.where(
        has_non_thumb_anchor,
        opposition_progress.clamp(0.0, 1.0) / cN.clamp_min(1.0e-6),
        torch.zeros_like(cN),
    ).clamp(0.0, 1.0)
    guidance_z = torch.where(
        has_non_thumb_anchor,
        z_progress.clamp(0.0, 1.0) / cN.clamp_min(1.0e-6),
        torch.zeros_like(cN),
    ).clamp(0.0, 1.0)
    unilateral_guidance_gain = components["unilateral_guidance_gain"].float()
    unilateral_contact_reward = components["unilateral_contact_reward"].float()
    if unilateral_guidance_gain.shape != expected_shape or unilateral_contact_reward.shape != expected_shape:
        raise ValueError("reward v13 unilateral components must match the reward batch shape")
    return {
        "reward_v13_r0": r0,
        "reward_v13_cN": cN,
        "reward_v13_cT": cT,
        "reward_v13_GN": GN,
        "reward_v13_GT": GT,
        "reward_v13_thumb_gap_m": thumb_gap,
        "reward_v13_thumb_proximity": thumb_proximity,
        "reward_v13_guidance_opposition": guidance_opposition,
        "reward_v13_guidance_z": guidance_z,
        "reward_v13_unilateral_guidance_gain": unilateral_guidance_gain,
        "reward_v13_unilateral_contact_reward": unilateral_contact_reward,
    }


def _privileged_task_state(info: dict[str, Any], config: GrootNewtonEnvConfig) -> Any:
    import torch
    import torch.nn.functional as functional

    phase = info["task_phase"].long().clamp(0, _TASK_PHASE_COUNT - 1)
    phase_one_hot = functional.one_hot(phase, num_classes=_TASK_PHASE_COUNT).float()
    flags = torch.stack(
        tuple(
            info[name].float()
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
        ),
        dim=-1,
    )
    continuous = torch.stack(
        (
            info["touching_finger_count"].float().div(5.0),
            info["xy_displacement"].float().div(config.bottle_min_xy_displacement).clamp(0.0, 5.0),
            info["current_lift_height"].float().div(config.bottle_lift_height).clamp(0.0, 1.0),
            info["orientation_error"].float().div(config.final_orientation_threshold_rad).clamp(0.0, 5.0),
            info["episode"]["length"].float().div(max(config.max_episode_steps, 1)).clamp(0.0, 1.0),
        ),
        dim=-1,
    )
    guidance = _reward_v13_signals(info)
    reward_guidance = torch.stack(
        tuple(
            guidance[name]
            for name in (
                "reward_v13_r0",
                "reward_v13_cN",
                "reward_v13_cT",
                "reward_v13_GN",
                "reward_v13_GT",
            )
        ),
        dim=-1,
    )
    output = torch.cat((phase_one_hot, flags, continuous, reward_guidance), dim=-1)
    if output.shape[-1] != _PRIVILEGED_STATE_DIM:
        raise RuntimeError(f"Expected privileged state width {_PRIVILEGED_STATE_DIM}, got {output.shape[-1]}")
    return output


@dataclass(frozen=True)
class _PreparedPolicyStep:
    """Exact cached DP row and actor input used for one control step."""

    policy_input: Any
    base_action: Any
    normalized_current_state: Any
    normalized_state_delta: Any
    finger_root_load: Any
    row_index: Any
    replanned: Any


class _PerLaneActionChunkCache:
    """GPU cache that advances DP action rows independently for every lane."""

    def __init__(self, num_lanes: int, action_dim: int, device: Any):
        import torch

        if num_lanes < 1 or action_dim < 1:
            raise ValueError("chunk cache lane and action dimensions must be positive")
        self.num_lanes = int(num_lanes)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self._all_indices = torch.arange(self.num_lanes, dtype=torch.long, device=self.device)
        self.chunk = torch.zeros(
            (self.num_lanes, _BASE_ACTION_HORIZON, self.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.condition: Any | None = None
        self.row = torch.zeros(self.num_lanes, dtype=torch.long, device=self.device)
        self.valid = torch.zeros(self.num_lanes, dtype=torch.bool, device=self.device)
        self.fresh = torch.zeros(self.num_lanes, dtype=torch.bool, device=self.device)
        self.plan_count = torch.zeros(self.num_lanes, dtype=torch.int64, device=self.device)

    def _lane_indices(self, lane_indices: Any | None = None) -> Any:
        import torch

        if lane_indices is None:
            return self._all_indices
        indices = lane_indices.to(device=self.device, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError(f"lane_indices must be one-dimensional, got {tuple(indices.shape)}")
        if bool(((indices < 0) | (indices >= self.num_lanes)).any()):
            raise IndexError(f"lane_indices must be in [0, {self.num_lanes})")
        if indices.numel() != torch.unique(indices).numel():
            raise ValueError("lane_indices cannot contain duplicates")
        return indices

    def refill(self, lane_indices: Any, predicted_chunks: Any, plan_condition: Any) -> None:
        import torch

        indices = self._lane_indices(lane_indices)
        expected_batch = indices.numel()
        if predicted_chunks.ndim != 3:
            raise ValueError(f"predicted_chunks must be [batch,horizon,action], got {tuple(predicted_chunks.shape)}")
        if predicted_chunks.shape[0] != expected_batch:
            raise ValueError(
                f"predicted chunk batch {predicted_chunks.shape[0]} does not match lane count {expected_batch}"
            )
        if predicted_chunks.shape[1] < _BASE_ACTION_HORIZON:
            raise ValueError(
                f"Frozen DP must return pred_horizon >= {_BASE_ACTION_HORIZON}, got {predicted_chunks.shape[1]}"
            )
        if predicted_chunks.shape[2] != self.action_dim:
            raise ValueError(f"Frozen DP action dimension must be {self.action_dim}, got {predicted_chunks.shape[2]}")
        if plan_condition.ndim != 2 or plan_condition.shape[0] != expected_batch:
            raise ValueError(
                f"plan_condition must be [batch,condition], got {tuple(plan_condition.shape)} for {expected_batch} lanes"
            )
        if self.condition is not None and plan_condition.shape[1] != self.condition.shape[1]:
            raise ValueError(
                f"plan condition width changed from {self.condition.shape[1]} to {plan_condition.shape[1]}"
            )

        chunk = predicted_chunks[:, :_BASE_ACTION_HORIZON].detach().to(device=self.device, dtype=self.chunk.dtype)
        condition = plan_condition.detach().to(device=self.device, dtype=torch.float32)
        if self.condition is None:
            self.condition = torch.zeros((self.num_lanes, condition.shape[1]), dtype=torch.float32, device=self.device)
        self.chunk[indices] = chunk
        self.condition[indices] = condition
        self.row[indices] = 0
        self.valid[indices] = True
        self.fresh[indices] = True
        self.plan_count[indices] += 1

    def peek(
        self,
        lane_indices: Any | None = None,
        *,
        eligible: Any | None = None,
        fallback_action: Any | None = None,
        validate: bool = True,
    ):
        import torch

        indices = self._lane_indices(lane_indices)
        batch_size = indices.numel()
        if eligible is None:
            eligible_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        else:
            eligible_mask = eligible.to(device=self.device, dtype=torch.bool)
            if eligible_mask.shape != (batch_size,):
                raise ValueError(f"eligible must have shape ({batch_size},), got {tuple(eligible_mask.shape)}")
        selected_valid = self.valid[indices]
        if validate and bool((eligible_mask & ~selected_valid).any()):
            raise RuntimeError("cannot select an eligible lane from an invalid DP chunk cache")
        if self.condition is None:
            raise RuntimeError("cannot select from an empty DP chunk cache")

        rows = self.row[indices]
        base_action = self.chunk[indices, rows.clamp(0, _BASE_ACTION_HORIZON - 1)]
        condition = self.condition[indices]
        if fallback_action is not None:
            if fallback_action.shape != (batch_size, self.action_dim):
                raise ValueError(
                    f"fallback_action must have shape ({batch_size}, {self.action_dim}), "
                    f"got {tuple(fallback_action.shape)}"
                )
            base_action = torch.where(eligible_mask[:, None], base_action, fallback_action.float())
        elif bool((~eligible_mask).any()):
            raise ValueError("fallback_action is required when any lane is ineligible")
        condition = torch.where(eligible_mask[:, None], condition, torch.zeros_like(condition))
        row_index = torch.where(eligible_mask, rows, torch.full_like(rows, -1))
        return condition, base_action, row_index

    def advance(self, executed_mask: Any, *, validate: bool = True) -> None:
        import torch

        executed = executed_mask.to(device=self.device, dtype=torch.bool)
        if executed.shape != (self.num_lanes,):
            raise ValueError(f"executed_mask must have shape ({self.num_lanes},), got {tuple(executed.shape)}")
        if validate and bool((executed & ~self.valid).any()):
            raise RuntimeError("cannot advance an invalid DP chunk cache lane")
        self.fresh[executed] = False
        next_row = self.row + executed.long()
        exhausted = executed & (next_row >= _BASE_ACTION_HORIZON)
        self.row.copy_(torch.where(executed, next_row, self.row))
        self.row[exhausted] = 0
        self.valid[exhausted] = False

    def remaining_rows(self, mask: Any) -> Any:
        import torch

        selected = mask.to(device=self.device, dtype=torch.bool)
        if selected.shape != (self.num_lanes,):
            raise ValueError(f"mask must have shape ({self.num_lanes},), got {tuple(selected.shape)}")
        return torch.where(selected & self.valid, _BASE_ACTION_HORIZON - self.row, torch.zeros_like(self.row))

    def invalidate(self, mask: Any | None = None) -> None:
        import torch

        if mask is None:
            selected = torch.ones(self.num_lanes, dtype=torch.bool, device=self.device)
        else:
            selected = mask.to(device=self.device, dtype=torch.bool)
            if selected.shape != (self.num_lanes,):
                raise ValueError(f"mask must have shape ({self.num_lanes},), got {tuple(selected.shape)}")
        self.valid[selected] = False
        self.fresh[selected] = False
        self.row[selected] = 0

    def clone_lanes(self, lane_indices: Any | None = None) -> _PerLaneActionChunkCache:
        indices = self._lane_indices(lane_indices)
        clone = _PerLaneActionChunkCache(indices.numel(), self.action_dim, self.device)
        clone.chunk.copy_(self.chunk[indices])
        clone.row.copy_(self.row[indices])
        clone.valid.copy_(self.valid[indices])
        clone.fresh.copy_(self.fresh[indices])
        clone.plan_count.copy_(self.plan_count[indices])
        if self.condition is not None:
            clone.condition = self.condition[indices].clone()
        return clone


def _hold_action_from_observation(observation: dict[str, Any]) -> Any:
    import torch

    state = observation[_STATE_KEY]
    if state.ndim != 3 or state.shape[-1] != 26:
        raise ValueError(f"Expected {_STATE_KEY} [batch,horizon,26], got {tuple(state.shape)}")
    current = state[:, -1].float()
    return torch.cat((current[:, 7:16], current[:, 16:26]), dim=-1)


def _normalize_current_state(state: Any, state_min: Any, state_max: Any) -> Any:
    import torch

    minimum = torch.as_tensor(state_min, dtype=state.dtype, device=state.device)
    maximum = torch.as_tensor(state_max, dtype=state.dtype, device=state.device)
    if state.shape[-1:] != (26,) or minimum.shape != (26,) or maximum.shape != (26,):
        raise ValueError(
            f"current state and bounds must end in dimension 26, got {state.shape}, {minimum.shape}, {maximum.shape}"
        )
    span = torch.clamp(maximum - minimum, min=1.0e-6)
    return torch.clamp(2.0 * (state - minimum) / span - 1.0, -1.0, 1.0)


def _live_finger_root_load(observation: dict[str, Any], batch_size: int) -> Any:
    load = observation.get(_FINGER_ROOT_LOAD_KEY)
    if load is None:
        raise ValueError(f"Contract v5 requires live {_FINGER_ROOT_LOAD_KEY!r}")
    if load.ndim == 3:
        load = load[:, -1]
    if load.ndim != 2 or load.shape != (batch_size, _FINGER_ROOT_LOAD_DIM):
        raise ValueError(f"Expected {_FINGER_ROOT_LOAD_KEY} [batch,5] or [batch,history,5], got {tuple(load.shape)}")
    return load.detach().float().clamp(0.0, 1.0)


def _prepare_policy_step(
    observation: dict[str, Any],
    frozen_dp: Any,
    scheduler: Any,
    cache: _PerLaneActionChunkCache,
    action_min: Any,
    action_max: Any,
    state_min: Any | None = None,
    state_max: Any | None = None,
    *,
    inference_steps: int,
    use_bfloat16: bool,
    generator: Any | None = None,
    initial_noise: Any | None = None,
    eligible: Any | None = None,
) -> _PreparedPolicyStep:
    import torch

    indices = cache._lane_indices()
    batch_size = indices.numel()
    if eligible is None:
        eligible_mask = torch.ones(batch_size, dtype=torch.bool, device=cache.device)
    else:
        eligible_mask = eligible.to(device=cache.device, dtype=torch.bool)
        if eligible_mask.shape != (batch_size,):
            raise ValueError(f"eligible must have shape ({batch_size},), got {tuple(eligible_mask.shape)}")
    needs_replan = eligible_mask & ~cache.valid[indices]
    if bool(needs_replan.any()):
        local_replan = torch.where(needs_replan)[0]
        global_replan = indices[local_replan]
        plan_observation = _select_tree(observation, local_replan)
        plan_noise = None if initial_noise is None else initial_noise[local_replan]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            plan_condition = frozen_dp.encode_observation(plan_observation)
            predicted_chunks = frozen_dp.predict_action_from_condition(
                plan_condition,
                scheduler,
                inference_steps=inference_steps,
                generator=generator,
                initial_noise=plan_noise,
            )
        cache.refill(global_replan, predicted_chunks, plan_condition)

    hold_action = _hold_action_from_observation(observation)
    plan_condition, base_action, row_index = cache.peek(
        None,
        eligible=eligible_mask,
        fallback_action=hold_action,
        validate=False,
    )
    base_action = base_action.detach().float()
    normalized_base = normalize_physical_action(base_action, action_min, action_max)
    state_history = observation[_STATE_KEY]
    if state_history.ndim != 3 or state_history.shape[0] != batch_size or state_history.shape[-1] != 26:
        raise ValueError(f"Expected {_STATE_KEY} [batch,history,26], got {tuple(state_history.shape)}")
    if state_history.shape[1] < 2:
        raise ValueError("Contract v5 requires at least two state-history frames for the live state delta")
    previous_state = state_history[:, -2].float()
    current_state = state_history[:, -1].float()
    if state_min is None:
        state_min = getattr(frozen_dp, "state_min", None)
    if state_max is None:
        state_max = getattr(frozen_dp, "state_max", None)
    if state_min is None or state_max is None:
        raise ValueError("DP state_min/state_max are required for the contract v9 actor input")
    normalized_previous_state = _normalize_current_state(previous_state, state_min, state_max)
    normalized_current_state = _normalize_current_state(current_state, state_min, state_max)
    normalized_state_delta = normalized_current_state - normalized_previous_state
    finger_root_load = _live_finger_root_load(observation, batch_size)
    row_one_hot = torch.nn.functional.one_hot(
        row_index.clamp(min=0),
        num_classes=_BASE_ACTION_HORIZON,
    ).float()
    row_one_hot *= (row_index >= 0).unsqueeze(-1)
    return _PreparedPolicyStep(
        policy_input=torch.cat(
            (
                plan_condition,
                normalized_base,
                normalized_current_state,
                row_one_hot,
                normalized_state_delta,
                finger_root_load,
            ),
            dim=-1,
        ),
        base_action=base_action,
        normalized_current_state=normalized_current_state,
        normalized_state_delta=normalized_state_delta,
        finger_root_load=finger_root_load,
        row_index=row_index,
        replanned=cache.fresh[indices] & eligible_mask,
    )


def _allocate_rollout(
    rollout_steps: int,
    num_envs: int,
    policy_input_dim: int,
    residual_dim: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    shape = (rollout_steps, num_envs)
    return {
        "policy_input": torch.empty((*shape, policy_input_dim), dtype=torch.float32, device=device),
        "privileged": torch.empty((*shape, _PRIVILEGED_STATE_DIM), dtype=torch.float32, device=device),
        "raw_latent": torch.empty((*shape, residual_dim), dtype=torch.float32, device=device),
        "log_prob": torch.empty(shape, dtype=torch.float32, device=device),
        "value": torch.empty(shape, dtype=torch.float32, device=device),
        "reward": torch.empty(shape, dtype=torch.float32, device=device),
        "terminated": torch.empty(shape, dtype=torch.bool, device=device),
        "truncated": torch.empty(shape, dtype=torch.bool, device=device),
        "timeout_value": torch.zeros(shape, dtype=torch.float32, device=device),
        "episode_done": torch.empty(shape, dtype=torch.bool, device=device),
        "episode_success": torch.empty(shape, dtype=torch.bool, device=device),
        "episode_fail": torch.empty(shape, dtype=torch.bool, device=device),
        "episode_return": torch.zeros(shape, dtype=torch.float32, device=device),
        "episode_length": torch.zeros(shape, dtype=torch.float32, device=device),
        "episode_max_lift_height_m": torch.zeros(shape, dtype=torch.float32, device=device),
        "episode_physical_max_lift_height_m": torch.zeros(shape, dtype=torch.float32, device=device),
        "current_lift_height_m": torch.zeros(shape, dtype=torch.float32, device=device),
        "physical_max_lift_height_m": torch.zeros(shape, dtype=torch.float32, device=device),
        "contacted_carry_max_lift_height_m": torch.zeros(shape, dtype=torch.float32, device=device),
        "opposed_pregrasp_score": torch.zeros(shape, dtype=torch.float32, device=device),
        **{name: torch.zeros(shape, dtype=torch.float32, device=device) for name in _REWARD_V13_SIGNAL_NAMES},
        "finger_root_load": torch.zeros((*shape, _FINGER_ROOT_LOAD_DIM), dtype=torch.float32, device=device),
        "actual_eef_delta_z_m": torch.zeros(shape, dtype=torch.float32, device=device),
        "pre_action_contact": torch.empty(shape, dtype=torch.bool, device=device),
        "pre_action_thumb_contact": torch.empty(shape, dtype=torch.bool, device=device),
        "pre_action_non_thumb_contact": torch.empty(shape, dtype=torch.bool, device=device),
        "pre_action_grasp": torch.empty(shape, dtype=torch.bool, device=device),
        "pre_action_carrying": torch.empty(shape, dtype=torch.bool, device=device),
        "finger_contact_any_frame": torch.empty((*shape, len(_FINGER_NAMES)), dtype=torch.bool, device=device),
        "opposed_grasp_any_frame": torch.empty(shape, dtype=torch.bool, device=device),
        "opposed_grasp_max_consecutive_physics_frames": torch.empty(shape, dtype=torch.int64, device=device),
        **{
            f"partial_contact_stage_{name}": torch.empty(shape, dtype=torch.bool, device=device)
            for name in _PARTIAL_CONTACT_STAGE_NAMES
        },
        "task_phase": torch.empty(shape, dtype=torch.int64, device=device),
        **{name: torch.empty(shape, dtype=torch.float32, device=device) for name in _ACTION_DIAGNOSTIC_NAMES},
        "base_action_row": torch.empty(shape, dtype=torch.int64, device=device),
        "base_action_replanned": torch.empty(shape, dtype=torch.bool, device=device),
        "base_action_discarded_rows": torch.zeros(shape, dtype=torch.int64, device=device),
        **{f"event_{name}_rise": torch.empty(shape, dtype=torch.bool, device=device) for name in _EVENT_INFO_KEYS},
    }


def _opposed_streak_metrics_from_histogram(histogram: Any, *, step_count: int) -> dict[str, float]:
    import torch

    if histogram.ndim != 1 or histogram.numel() < 2:
        raise ValueError("opposed-grasp streak histogram must contain at least bins 0 and 1")
    if step_count < 1 or int(histogram.sum()) != step_count:
        raise ValueError(f"opposed-grasp streak histogram must contain exactly {step_count} samples")
    frames_per_action = histogram.numel() - 1
    frame_values = torch.arange(histogram.numel(), dtype=torch.float32, device=histogram.device)
    cumulative = torch.cumsum(histogram, dim=0)

    def nearest_rank(quantile: float) -> float:
        rank = max(1, math.ceil(quantile * step_count))
        return float(torch.argmax((cumulative >= rank).to(torch.int32)))

    metrics = {
        "opposed_grasp_max_consecutive_physics_frames_mean": float((frame_values * histogram).sum() / step_count),
        "opposed_grasp_max_consecutive_physics_frames_p50": nearest_rank(0.50),
        "opposed_grasp_max_consecutive_physics_frames_p95": nearest_rank(0.95),
        "opposed_grasp_max_consecutive_physics_frames_max": float(torch.nonzero(histogram, as_tuple=False)[-1, 0]),
    }
    for frames in range(frames_per_action + 1):
        metrics[f"opposed_grasp_max_consecutive_physics_frames_eq_{frames}_step_fraction"] = (
            float(histogram[frames]) / step_count
        )
    return metrics


def _opposed_streak_metrics(
    max_consecutive: Any,
    *,
    step_count: int,
    frames_per_action: int,
) -> dict[str, float]:
    import torch

    if frames_per_action < 1:
        raise ValueError("frames_per_action must be positive")
    flattened = max_consecutive.long().reshape(-1)
    if flattened.numel() != step_count:
        raise ValueError(f"expected {step_count} opposed-grasp streak samples, got {flattened.numel()}")
    if bool(((flattened < 0) | (flattened > frames_per_action)).any()):
        raise ValueError(f"opposed-grasp streak must be in [0,{frames_per_action}]")
    histogram = torch.bincount(flattened, minlength=frames_per_action + 1)
    return _opposed_streak_metrics_from_histogram(histogram, step_count=step_count)


def _rollout_diagnostic_metrics(
    rollout: dict[str, Any],
    *,
    frames_per_action: int = 6,
) -> dict[str, float]:
    import torch

    task_phase = rollout["task_phase"]
    step_count = max(task_phase.numel(), 1)
    episode_done = rollout["episode_done"]
    episode_contacted_max_lifts = rollout["episode_max_lift_height_m"][episode_done]
    episode_physical_max_lifts = rollout["episode_physical_max_lift_height_m"][episode_done]
    metrics = {
        f"phase_{name}_fraction": float((task_phase == index).sum()) / step_count
        for index, name in enumerate(_PHASE_NAMES)
    }
    for name in _EVENT_INFO_KEYS:
        count = float(rollout[f"event_{name}_rise"].sum())
        metrics[f"event_{name}_rise_count"] = count
        metrics[f"event_{name}_rise_per_1000_steps"] = 1000.0 * count / step_count
    for row in range(_BASE_ACTION_HORIZON):
        metrics[f"base_action_row_{row}_fraction"] = float((rollout["base_action_row"] == row).sum()) / step_count
    replan_count = float(rollout["base_action_replanned"].sum())
    metrics["base_action_replan_lane_count"] = replan_count
    metrics["base_action_replan_lanes_per_1000_steps"] = 1000.0 * replan_count / step_count
    metrics["base_action_discarded_row_count"] = float(rollout["base_action_discarded_rows"].sum())
    for name in _ACTION_DIAGNOSTIC_NAMES:
        metrics[name] = float(rollout[name].mean())
    contact = rollout["pre_action_contact"].bool()
    thumb_contact = rollout["pre_action_thumb_contact"].bool()
    non_thumb_contact = rollout["pre_action_non_thumb_contact"].bool()
    grasp = rollout["pre_action_grasp"].bool()
    carrying = rollout["pre_action_carrying"].bool()
    finger_root_load = rollout["finger_root_load"]
    second_root_load = finger_root_load.topk(k=2, dim=-1).values[..., 1]
    metrics["contact_step_fraction"] = float(contact.sum()) / step_count
    metrics["thumb_contact_step_fraction"] = float(thumb_contact.sum()) / step_count
    metrics["non_thumb_contact_step_fraction"] = float(non_thumb_contact.sum()) / step_count
    metrics["opposed_grasp_step_fraction"] = float(grasp.sum()) / step_count
    metrics["thumb_only_contact_step_fraction"] = float((thumb_contact & ~non_thumb_contact).sum()) / step_count
    thumb_contact_count = int(thumb_contact.sum())
    metrics["thumb_to_opposed_grasp_step_conversion"] = float(grasp.sum()) / max(thumb_contact_count, 1)
    metrics["carrying_step_fraction"] = float(carrying.sum()) / step_count
    finger_contact_any_frame = rollout["finger_contact_any_frame"].bool()
    if finger_contact_any_frame.shape != (*task_phase.shape, len(_FINGER_NAMES)):
        raise ValueError(
            "finger_contact_any_frame must have shape "
            f"{(*task_phase.shape, len(_FINGER_NAMES))}, got {tuple(finger_contact_any_frame.shape)}"
        )
    thumb_contact_any_frame = finger_contact_any_frame[..., 0]
    non_thumb_contact_any_frame = finger_contact_any_frame[..., 1:].any(dim=-1)
    opposed_grasp_any_frame = rollout["opposed_grasp_any_frame"].bool()
    metrics["thumb_contact_any_frame_step_fraction"] = float(thumb_contact_any_frame.sum()) / step_count
    metrics["non_thumb_contact_any_frame_step_fraction"] = float(non_thumb_contact_any_frame.sum()) / step_count
    metrics["opposed_grasp_any_frame_step_fraction"] = float(opposed_grasp_any_frame.sum()) / step_count
    for name in _PARTIAL_CONTACT_STAGE_NAMES:
        value = rollout[f"partial_contact_stage_{name}"].bool()
        if value.shape != task_phase.shape:
            raise ValueError(f"partial_contact_stage_{name} must have shape {tuple(task_phase.shape)}")
        metrics[f"partial_contact_stage_{name}_step_fraction"] = float(value.sum()) / step_count
    for finger_index, finger_name in enumerate(_FINGER_NAMES):
        metrics[f"finger_{finger_name}_contact_any_frame_step_fraction"] = (
            float(finger_contact_any_frame[..., finger_index].sum()) / step_count
        )
    metrics.update(
        _opposed_streak_metrics(
            rollout["opposed_grasp_max_consecutive_physics_frames"],
            step_count=step_count,
            frames_per_action=frames_per_action,
        )
    )
    pregrasp_score = rollout["opposed_pregrasp_score"]
    metrics["opposed_pregrasp_score_mean"] = float(pregrasp_score.mean())
    metrics["opposed_pregrasp_score_p95"] = float(torch.quantile(pregrasp_score, 0.95))
    metrics["opposed_pregrasp_score_ge_0_1_step_fraction"] = float((pregrasp_score >= 0.1).float().mean())
    metrics["opposed_pregrasp_score_ge_0_5_step_fraction"] = float((pregrasp_score >= 0.5).float().mean())
    for name in _REWARD_V13_SIGNAL_NAMES:
        value = rollout[name]
        if value.shape != task_phase.shape:
            raise ValueError(f"{name} must have shape {tuple(task_phase.shape)}")
        metrics[f"{name}_mean"] = float(value.mean())
    for name in ("cN", "cT", "GN", "GT"):
        value = rollout[f"reward_v13_{name}"]
        metrics[f"reward_v13_{name}_positive_step_fraction"] = float((value > 0.0).float().mean())
    non_thumb_anchor = rollout["reward_v13_cN"] > 0.0
    non_thumb_anchor_count = int(non_thumb_anchor.sum())
    conditioned = {
        "thumb_gap_m": rollout["reward_v13_thumb_gap_m"],
        "thumb_proximity": rollout["reward_v13_thumb_proximity"],
        "guidance_opposition": rollout["reward_v13_guidance_opposition"],
        "guidance_z": rollout["reward_v13_guidance_z"],
    }
    for name, value in conditioned.items():
        selected = value[non_thumb_anchor]
        metrics[f"reward_v13_non_thumb_conditioned_{name}_mean"] = (
            float(selected.mean()) if non_thumb_anchor_count else 0.0
        )
    selected_gap = rollout["reward_v13_thumb_gap_m"][non_thumb_anchor]
    metrics["reward_v13_non_thumb_conditioned_thumb_gap_m_min"] = (
        float(selected_gap.min()) if non_thumb_anchor_count else 0.0
    )
    metrics["reward_v13_non_thumb_conditioned_thumb_gap_m_p50"] = (
        float(torch.quantile(selected_gap, 0.5)) if non_thumb_anchor_count else 0.0
    )
    contact_count = int(contact.sum())
    metrics["contact_to_grasp_step_conversion"] = float((contact & grasp).sum()) / max(contact_count, 1)
    contact_rises = float(rollout["event_contact_rise"].sum())
    grasp_rises = float(rollout["event_grasp_rise"].sum())
    metrics["contact_to_grasp_rise_conversion"] = grasp_rises / max(contact_rises, 1.0)
    for finger_index, finger_name in enumerate(_FINGER_NAMES):
        load = finger_root_load[..., finger_index]
        metrics[f"finger_root_load_{finger_name}_mean"] = float(load.mean())
        metrics[f"finger_root_load_{finger_name}_p95"] = float(torch.quantile(load, 0.95))
        metrics[f"finger_root_load_{finger_name}_saturation_fraction"] = float((load >= 0.95).float().mean())
    for condition_name, condition in (
        ("no_contact", ~contact),
        ("contact", contact),
        ("grasp", grasp),
    ):
        condition_count = int(condition.sum())
        metrics[f"finger_root_second_load_{condition_name}_mean"] = (
            float(second_root_load[condition].mean()) if condition_count else 0.0
        )
    no_load = second_root_load < _NO_LOAD_THRESHOLD
    two_finger_load = second_root_load >= _TWO_FINGER_LOAD_THRESHOLD
    metrics["vertical_residual_signed_m_no_load"] = (
        float(rollout["vertical_residual_signed_m"][no_load].mean()) if bool(no_load.any()) else 0.0
    )
    metrics["vertical_residual_signed_m_two_finger_load"] = (
        float(rollout["vertical_residual_signed_m"][two_finger_load].mean()) if bool(two_finger_load.any()) else 0.0
    )
    for condition_name, condition in (("contact", contact), ("carrying", carrying)):
        condition_count = int(condition.sum())
        for name in _PHASE_CONDITIONED_Z_NAMES:
            values = rollout[name]
            metrics[f"{condition_name}_{name}"] = float(values[condition].mean()) if condition_count else 0.0
    metrics["actual_eef_delta_z_m"] = float(rollout["actual_eef_delta_z_m"].mean())
    contacted_mean = float(episode_contacted_max_lifts.mean()) if episode_contacted_max_lifts.numel() else 0.0
    contacted_max = float(episode_contacted_max_lifts.max()) if episode_contacted_max_lifts.numel() else 0.0
    physical_mean = float(episode_physical_max_lifts.mean()) if episode_physical_max_lifts.numel() else 0.0
    physical_max = float(episode_physical_max_lifts.max()) if episode_physical_max_lifts.numel() else 0.0
    metrics["mean_episode_max_lift_height_m"] = contacted_mean
    metrics["max_episode_max_lift_height_m"] = contacted_max
    metrics["mean_episode_contacted_carry_max_lift_height_m"] = contacted_mean
    metrics["max_episode_contacted_carry_max_lift_height_m"] = contacted_max
    metrics["mean_episode_physical_max_lift_height_m"] = physical_mean
    metrics["max_episode_physical_max_lift_height_m"] = physical_max
    metrics["mean_current_lift_height_m"] = float(rollout["current_lift_height_m"].mean())
    metrics["max_rollout_physical_lift_height_m"] = float(rollout["physical_max_lift_height_m"].max())
    metrics["max_rollout_contacted_carry_lift_height_m"] = float(rollout["contacted_carry_max_lift_height_m"].max())
    for source_name, key in (
        ("physical_max_lift", "physical_max_lift_height_m"),
        ("contacted_carry_max_lift", "contacted_carry_max_lift_height_m"),
    ):
        lift_height = rollout[key]
        for threshold_name, threshold_m in _LIFT_DIAGNOSTIC_THRESHOLDS_M:
            reached = lift_height >= threshold_m
            metrics[f"rollout_{source_name}_ge_{threshold_name}_ever"] = float(reached.any())
            metrics[f"rollout_{source_name}_ge_{threshold_name}_step_fraction"] = float(reached.float().mean())
            metrics[f"rollout_{source_name}_ge_{threshold_name}_lane_ever_fraction"] = float(
                reached.any(dim=0).float().mean()
            )
    return metrics


def _ppo_update(
    actor_critic: Any,
    optimizer: Any,
    rollout: dict[str, Any],
    advantages: Any,
    returns: Any,
    args: argparse.Namespace,
    generator: Any,
) -> dict[str, float]:
    import torch

    batch_size = args.num_envs * args.rollout_steps
    flat = {
        name: value.reshape(batch_size, *value.shape[2:])
        for name, value in rollout.items()
        if name in {"policy_input", "privileged", "raw_latent", "log_prob", "value"}
    }
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)
    flat_advantages = (flat_advantages - flat_advantages.mean()) / flat_advantages.std(unbiased=False).clamp_min(1.0e-8)

    metric_totals = torch.zeros(3, dtype=torch.float32, device=flat_advantages.device)
    zero = torch.zeros((), dtype=torch.float32, device=flat_advantages.device)
    sample_count = 0
    stop_early = False
    optimizer_steps = 0
    max_update_step_kl = 0.0
    actor_grad_norm_sum = zero.clone()
    critic_grad_norm_sum = zero.clone()
    actor_grad_norm_max = zero.clone()
    critic_grad_norm_max = zero.clone()
    actor_grad_clip_count = zero.clone()
    critic_grad_clip_count = zero.clone()
    actor_parameters = [*actor_critic.actor.parameters(), actor_critic.log_std]
    critic_parameters = list(actor_critic.critic.parameters())
    for _epoch in range(args.update_epochs):
        permutation = torch.randperm(batch_size, device=flat_advantages.device, generator=generator)
        for start in range(0, batch_size, args.minibatch_size):
            indices = permutation[start : start + args.minibatch_size]
            new_log_prob, entropy, new_value = actor_critic.evaluate_actions(
                flat["policy_input"][indices],
                flat["privileged"][indices],
                flat["raw_latent"][indices],
            )
            log_ratio = new_log_prob - flat["log_prob"][indices]
            ratio = log_ratio.exp()
            with torch.no_grad():
                pre_step_kl = ((ratio - 1.0) - log_ratio).mean()
                pre_step_kl_value = float(pre_step_kl)
            max_update_step_kl = max(max_update_step_kl, pre_step_kl_value)
            if args.target_kl > 0.0 and pre_step_kl_value > args.target_kl:
                stop_early = True
                break

            mb_advantage = flat_advantages[indices]
            policy_loss = torch.maximum(
                -mb_advantage * ratio,
                -mb_advantage * ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef),
            ).mean()

            new_value = new_value.reshape(-1)
            if args.clip_vloss:
                unclipped = (new_value - flat_returns[indices]).square()
                clipped_value = flat["value"][indices] + (new_value - flat["value"][indices]).clamp(
                    -args.clip_coef, args.clip_coef
                )
                clipped = (clipped_value - flat_returns[indices]).square()
                value_loss = 0.5 * torch.maximum(unclipped, clipped).mean()
            else:
                value_loss = 0.5 * (new_value - flat_returns[indices]).square().mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm).detach()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm).detach()
            optimizer.step()

            with torch.no_grad():
                post_log_prob, _, _ = actor_critic.evaluate_actions(
                    flat["policy_input"][indices],
                    flat["privileged"][indices],
                    flat["raw_latent"][indices],
                )
                post_log_ratio = post_log_prob - flat["log_prob"][indices]
                post_ratio = post_log_ratio.exp()
                post_step_kl = ((post_ratio - 1.0) - post_log_ratio).mean()
                post_step_kl_value = float(post_step_kl)
            count = int(indices.numel())
            metric_totals[0] += policy_loss.detach() * count
            metric_totals[1] += value_loss.detach() * count
            metric_totals[2] += entropy_mean.detach() * count
            sample_count += count
            optimizer_steps += 1
            actor_grad_norm_sum += actor_grad_norm
            critic_grad_norm_sum += critic_grad_norm
            actor_grad_norm_max = torch.maximum(actor_grad_norm_max, actor_grad_norm)
            critic_grad_norm_max = torch.maximum(critic_grad_norm_max, critic_grad_norm)
            actor_grad_clip_count += (actor_grad_norm > args.max_grad_norm).float()
            critic_grad_clip_count += (critic_grad_norm > args.max_grad_norm).float()
            max_update_step_kl = max(max_update_step_kl, post_step_kl_value)
            if args.target_kl > 0.0 and post_step_kl_value > args.target_kl:
                stop_early = True
                break
        if stop_early:
            break

    final_kl_sum = zero.clone()
    final_clip_sum = zero.clone()
    final_max_minibatch_kl = zero.clone()
    with torch.no_grad():
        for start in range(0, batch_size, args.minibatch_size):
            indices = slice(start, min(start + args.minibatch_size, batch_size))
            final_log_prob, _, _ = actor_critic.evaluate_actions(
                flat["policy_input"][indices],
                flat["privileged"][indices],
                flat["raw_latent"][indices],
            )
            final_log_ratio = final_log_prob - flat["log_prob"][indices]
            final_ratio = final_log_ratio.exp()
            final_kl = (final_ratio - 1.0) - final_log_ratio
            final_kl_sum += final_kl.sum()
            final_clip_sum += ((final_ratio - 1.0).abs() > args.clip_coef).float().sum()
            final_max_minibatch_kl = torch.maximum(final_max_minibatch_kl, final_kl.mean())

    sample_denominator = max(sample_count, 1)
    step_denominator = max(optimizer_steps, 1)
    host_metrics = (
        torch.stack(
            (
                metric_totals[0] / sample_denominator,
                metric_totals[1] / sample_denominator,
                metric_totals[2] / sample_denominator,
                final_kl_sum / batch_size,
                final_max_minibatch_kl,
                final_clip_sum / batch_size,
                actor_grad_norm_sum / step_denominator,
                actor_grad_norm_max,
                actor_grad_clip_count / step_denominator,
                critic_grad_norm_sum / step_denominator,
                critic_grad_norm_max,
                critic_grad_clip_count / step_denominator,
            )
        )
        .cpu()
        .tolist()
    )
    return {
        "policy_loss": host_metrics[0],
        "value_loss": host_metrics[1],
        "entropy": host_metrics[2],
        "approx_kl": host_metrics[3],
        "max_minibatch_kl": host_metrics[4],
        "max_update_step_kl": max_update_step_kl,
        "clip_fraction": host_metrics[5],
        "kl_early_stop": float(stop_early),
        "optimizer_steps": float(optimizer_steps),
        "actor_grad_norm_mean": host_metrics[6],
        "actor_grad_norm_max": host_metrics[7],
        "actor_grad_clip_fraction": host_metrics[8],
        "critic_grad_norm_mean": host_metrics[9],
        "critic_grad_norm_max": host_metrics[10],
        "critic_grad_clip_fraction": host_metrics[11],
    }


def _checkpoint_payload(
    actor_critic: Any,
    optimizer: Any,
    *,
    update: int,
    global_step: int,
    best_eval_metrics: dict[str, float] | None,
    best_return_metrics: dict[str, float] | None,
    policy_config: Any,
    dp_sha256: str,
    env_config: GrootNewtonEnvConfig,
    finger_root_load_metadata: dict[str, Any],
    hand_target_metadata: dict[str, Any],
    args: argparse.Namespace,
    generators: dict[str, Any],
) -> dict[str, Any]:
    import torch

    return {
        "format": GROOT_RESIDUAL_PPO_CHECKPOINT_FORMAT,
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
        "hand_residual_scale": _hand_residual_scale_contract(args),
        "update": int(update),
        "global_step": int(global_step),
        "best_eval_success": None if best_eval_metrics is None else float(best_eval_metrics["success_rate"]),
        "best_eval_metrics": None if best_eval_metrics is None else dict(best_eval_metrics),
        "best_return_metrics": None if best_return_metrics is None else dict(best_return_metrics),
        "policy_config": asdict(policy_config),
        "frozen_dp_sha256": dp_sha256,
        "env_config": asdict(env_config),
        "finger_root_load": dict(finger_root_load_metadata),
        "hand_target": dict(hand_target_metadata),
        "train_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "actor_critic": actor_critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device=args.device),
        "generator_states": {name: generator.get_state() for name, generator in generators.items()},
    }


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _evaluate(
    env: GrootDiffusionPolicyEnv,
    frozen_dp: Any,
    scheduler: Any,
    actor_critic: Any,
    action_min: Any,
    action_max: Any,
    state_min: Any,
    state_max: Any,
    hand_residual_scale: Any,
    args: argparse.Namespace,
    *,
    episodes: int,
) -> tuple[dict[str, float], Any, dict[str, Any]]:
    import torch

    eval_generator = torch.Generator(device=args.device)
    eval_generator.manual_seed(args.seed + 10_003)
    initial_noise = torch.randn(
        (args.num_envs, frozen_dp.config.pred_horizon, frozen_dp.config.action_dim),
        dtype=torch.float32,
        device=args.device,
        generator=eval_generator,
    )
    quota = _EvaluationQuota(episodes, args.num_envs, args.device)
    success_count = 0
    fail_count = 0
    return_sum = 0.0
    length_sum = 0.0
    episode_contacted_max_lift_sum = 0.0
    episode_contacted_max_lift_max = 0.0
    episode_physical_max_lift_sum = 0.0
    episode_physical_max_lift_max = 0.0
    episode_contacted_lift_threshold_counts = {name: 0 for name, _ in _LIFT_DIAGNOSTIC_THRESHOLDS_M}
    episode_physical_lift_threshold_counts = {name: 0 for name, _ in _LIFT_DIAGNOSTIC_THRESHOLDS_M}
    phase_counts = torch.zeros(_TASK_PHASE_COUNT, dtype=torch.int64, device=args.device)
    base_row_counts = torch.zeros(_BASE_ACTION_HORIZON, dtype=torch.int64, device=args.device)
    base_replan_count = torch.zeros((), dtype=torch.int64, device=args.device)
    base_discarded_rows = torch.zeros((), dtype=torch.int64, device=args.device)
    active_step_count = torch.zeros((), dtype=torch.int64, device=args.device)
    pregrasp_score_sum = torch.zeros((), dtype=torch.float32, device=args.device)
    pregrasp_score_ge_tenth_count = torch.zeros((), dtype=torch.int64, device=args.device)
    pregrasp_score_ge_half_count = torch.zeros((), dtype=torch.int64, device=args.device)
    reward_v13_signal_sums = {
        name: torch.zeros((), dtype=torch.float32, device=args.device) for name in _REWARD_V13_SIGNAL_NAMES
    }
    reward_v13_positive_step_counts = {
        name: torch.zeros((), dtype=torch.int64, device=args.device) for name in ("cN", "cT", "GN", "GT")
    }
    reward_v13_positive_episode_counts = dict.fromkeys(reward_v13_positive_step_counts, 0)
    reward_v13_non_thumb_conditioned_count = torch.zeros((), dtype=torch.int64, device=args.device)
    reward_v13_non_thumb_conditioned_sums = {
        name: torch.zeros((), dtype=torch.float32, device=args.device)
        for name in ("thumb_gap_m", "thumb_proximity", "guidance_opposition", "guidance_z")
    }
    reward_v13_non_thumb_conditioned_thumb_gap_min = torch.full(
        (), float("inf"), dtype=torch.float32, device=args.device
    )
    event_episode_counts = dict.fromkeys(_EVENT_INFO_KEYS, 0)
    topology_step_counts = {
        name: torch.zeros((), dtype=torch.int64, device=args.device) for name in _CONTACT_TOPOLOGY_NAMES
    }
    topology_episode_counts = dict.fromkeys(_CONTACT_TOPOLOGY_NAMES, 0)
    frames_per_action = int(env.unwrapped.frames_per_action)
    any_frame_finger_step_counts = torch.zeros(len(_FINGER_NAMES), dtype=torch.int64, device=args.device)
    any_frame_topology_step_counts = {
        name: torch.zeros((), dtype=torch.int64, device=args.device)
        for name in ("thumb_contact", "non_thumb_contact", "opposed_grasp")
    }
    any_frame_finger_episode_counts = dict.fromkeys(_FINGER_NAMES, 0)
    any_frame_topology_episode_counts = dict.fromkeys(any_frame_topology_step_counts, 0)
    partial_contact_stage_step_counts = {
        name: torch.zeros((), dtype=torch.int64, device=args.device) for name in _PARTIAL_CONTACT_STAGE_NAMES
    }
    partial_contact_stage_episode_counts = dict.fromkeys(_PARTIAL_CONTACT_STAGE_NAMES, 0)
    opposed_streak_histogram = torch.zeros(frames_per_action + 1, dtype=torch.int64, device=args.device)
    action_diagnostic_sums = {
        name: torch.zeros((), dtype=torch.float32, device=args.device) for name in _ACTION_DIAGNOSTIC_NAMES
    }
    finger_root_load_sum = torch.zeros(_FINGER_ROOT_LOAD_DIM, dtype=torch.float32, device=args.device)
    finger_root_load_saturation_count = torch.zeros(_FINGER_ROOT_LOAD_DIM, dtype=torch.int64, device=args.device)
    second_load_sums = {
        name: torch.zeros((), dtype=torch.float32, device=args.device) for name in ("no_contact", "contact", "grasp")
    }
    second_load_counts = {name: torch.zeros((), dtype=torch.int64, device=args.device) for name in second_load_sums}
    load_conditioned_vertical_sums = {
        name: torch.zeros((), dtype=torch.float32, device=args.device) for name in ("no_load", "two_finger_load")
    }
    load_conditioned_vertical_counts = {
        name: torch.zeros((), dtype=torch.int64, device=args.device) for name in load_conditioned_vertical_sums
    }
    actual_eef_delta_z_sum = torch.zeros((), dtype=torch.float32, device=args.device)
    conditioned_step_counts = {
        name: torch.zeros((), dtype=torch.int64, device=args.device) for name in ("contact", "carrying")
    }
    conditioned_z_sums = {
        condition_name: {
            name: torch.zeros((), dtype=torch.float32, device=args.device) for name in _PHASE_CONDITIONED_Z_NAMES
        }
        for condition_name in conditioned_step_counts
    }
    contact_and_grasp_step_count = torch.zeros((), dtype=torch.int64, device=args.device)
    cache = _PerLaneActionChunkCache(args.num_envs, frozen_dp.config.action_dim, args.device)
    was_training = actor_critic.training
    actor_critic.eval()
    try:
        while not quota.complete:
            observation, current_info = env.reset()
            cache.invalidate()
            quota.start_wave()
            wave_event_ever = {
                name: torch.zeros(args.num_envs, dtype=torch.bool, device=args.device) for name in _EVENT_INFO_KEYS
            }
            wave_topology_ever = {
                name: torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
                for name in _CONTACT_TOPOLOGY_NAMES
            }
            wave_any_frame_finger_ever = torch.zeros(
                (args.num_envs, len(_FINGER_NAMES)), dtype=torch.bool, device=args.device
            )
            wave_any_frame_topology_ever = {
                name: torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
                for name in any_frame_topology_step_counts
            }
            wave_partial_contact_stage_ever = {
                name: torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
                for name in _PARTIAL_CONTACT_STAGE_NAMES
            }
            wave_reward_v13_positive_ever = {
                name: torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
                for name in reward_v13_positive_step_counts
            }
            for _wave_step in range(args.max_episode_steps):
                active_before_step = quota.active.clone()
                pre_action_contact, pre_action_grasp, pre_action_carrying = _pre_action_task_flags(current_info)
                pre_action_thumb, pre_action_non_thumb, _ = _finger_contact_topology(current_info)
                pre_action_contact = pre_action_contact & active_before_step
                pre_action_grasp = pre_action_grasp & active_before_step
                pre_action_carrying = pre_action_carrying & active_before_step
                pre_action_thumb = pre_action_thumb & active_before_step
                pre_action_non_thumb = pre_action_non_thumb & active_before_step
                current_eef_position = observation[_STATE_KEY][:, -1, 7:10].float()
                current_hand_position = observation[_STATE_KEY][:, -1, 16:26].float()
                prepared = _prepare_policy_step(
                    observation,
                    frozen_dp,
                    scheduler,
                    cache,
                    action_min,
                    action_max,
                    state_min,
                    state_max,
                    inference_steps=args.inference_steps,
                    use_bfloat16=args.bfloat16,
                    generator=eval_generator,
                    initial_noise=initial_noise,
                    eligible=active_before_step,
                )
                privileged = _privileged_task_state(current_info, env.unwrapped.config)
                with torch.no_grad():
                    raw_latent, _, _, _ = actor_critic.act(
                        prepared.policy_input,
                        privileged,
                        deterministic=True,
                    )
                    residual_action = compose_residual_action(
                        prepared.base_action,
                        raw_latent,
                        action_min,
                        action_max,
                        position_scale_m=args.position_residual_scale_m,
                        vertical_position_scale_m=args.vertical_residual_scale_m,
                        rotation_scale_rad=math.radians(args.rotation_residual_scale_deg),
                        hand_scale_normalized=hand_residual_scale,
                        world_from_action_rotation=_R_WORLD_FROM_ACTION,
                    )
                    action = torch.where(active_before_step[:, None], residual_action, prepared.base_action)
                    action_diagnostics = _action_diagnostics(
                        prepared.base_action,
                        raw_latent,
                        action_min,
                        action_max,
                        current_eef_position,
                        current_hand_position,
                        position_scale_m=args.position_residual_scale_m,
                        vertical_position_scale_m=args.vertical_residual_scale_m,
                        hand_scale_normalized=hand_residual_scale,
                        hand_max_joint_step_rad=env.unwrapped.config.hand_max_joint_step_rad,
                        world_from_action_rotation=_R_WORLD_FROM_ACTION,
                    )
                next_observation, _, terminated, truncated, step_info = env.step(action)
                done = terminated | truncated
                cache.advance(active_before_step, validate=False)
                eef_delta_action = next_observation[_STATE_KEY][:, -1, 7:10].float() - current_eef_position
                actual_eef_delta_z = _action_position_to_world(eef_delta_action)[..., 2]
                post_action_thumb, post_action_non_thumb, post_action_grasp = _finger_contact_topology(step_info)
                (
                    control_step_finger_contact,
                    control_step_thumb_contact,
                    control_step_non_thumb_contact,
                    control_step_opposed_grasp,
                    control_step_opposed_streak,
                ) = _control_step_contact_topology(step_info)

                pre_action_topology = {
                    "thumb_contact": pre_action_thumb,
                    "non_thumb_contact": pre_action_non_thumb,
                    "opposed_grasp": pre_action_grasp,
                    "thumb_only_contact": pre_action_thumb & ~pre_action_non_thumb,
                }
                post_action_topology = {
                    "thumb_contact": post_action_thumb,
                    "non_thumb_contact": post_action_non_thumb,
                    "opposed_grasp": post_action_grasp,
                    "thumb_only_contact": post_action_thumb & ~post_action_non_thumb,
                }
                for name in _CONTACT_TOPOLOGY_NAMES:
                    topology_step_counts[name] += pre_action_topology[name].sum()
                    wave_topology_ever[name] |= (
                        pre_action_topology[name] | post_action_topology[name]
                    ) & active_before_step

                active_finger_contact = control_step_finger_contact & active_before_step[:, None]
                any_frame_finger_step_counts += active_finger_contact.sum(dim=0)
                any_frame_topology = {
                    "thumb_contact": control_step_thumb_contact & active_before_step,
                    "non_thumb_contact": control_step_non_thumb_contact & active_before_step,
                    "opposed_grasp": control_step_opposed_grasp & active_before_step,
                }
                wave_any_frame_finger_ever |= active_finger_contact
                for name, value in any_frame_topology.items():
                    any_frame_topology_step_counts[name] += value.sum()
                    wave_any_frame_topology_ever[name] |= value
                for name, value in _partial_contact_reward_stages(step_info).items():
                    active_value = value & active_before_step
                    partial_contact_stage_step_counts[name] += active_value.sum()
                    wave_partial_contact_stage_ever[name] |= active_value
                opposed_streak_histogram += torch.bincount(
                    control_step_opposed_streak[active_before_step],
                    minlength=frames_per_action + 1,
                )

                phase = step_info["task_phase"].long().clamp(0, _TASK_PHASE_COUNT - 1)
                for phase_index in range(_TASK_PHASE_COUNT):
                    phase_counts[phase_index] += ((phase == phase_index) & active_before_step).sum()
                active_step_count += active_before_step.sum()
                pregrasp_score = step_info["opposed_pregrasp_score"].float()
                pregrasp_score_sum += (pregrasp_score * active_before_step.float()).sum()
                pregrasp_score_ge_tenth_count += ((pregrasp_score >= 0.1) & active_before_step).sum()
                pregrasp_score_ge_half_count += ((pregrasp_score >= 0.5) & active_before_step).sum()
                reward_v13_signals = _reward_v13_signals(step_info)
                active_float = active_before_step.float()
                for name, value in reward_v13_signals.items():
                    reward_v13_signal_sums[name] += (value * active_float).sum()
                for name in reward_v13_positive_step_counts:
                    positive = reward_v13_signals[f"reward_v13_{name}"] > 0.0
                    reward_v13_positive_step_counts[name] += (positive & active_before_step).sum()
                    wave_reward_v13_positive_ever[name] |= positive & active_before_step
                non_thumb_anchor = (reward_v13_signals["reward_v13_cN"] > 0.0) & active_before_step
                non_thumb_anchor_float = non_thumb_anchor.float()
                reward_v13_non_thumb_conditioned_count += non_thumb_anchor.sum()
                for name in reward_v13_non_thumb_conditioned_sums:
                    reward_v13_non_thumb_conditioned_sums[name] += (
                        reward_v13_signals[f"reward_v13_{name}"] * non_thumb_anchor_float
                    ).sum()
                masked_thumb_gap = torch.where(
                    non_thumb_anchor,
                    reward_v13_signals["reward_v13_thumb_gap_m"],
                    torch.full_like(reward_v13_signals["reward_v13_thumb_gap_m"], float("inf")),
                )
                reward_v13_non_thumb_conditioned_thumb_gap_min = torch.minimum(
                    reward_v13_non_thumb_conditioned_thumb_gap_min,
                    masked_thumb_gap.min(),
                )
                for row in range(_BASE_ACTION_HORIZON):
                    base_row_counts[row] += ((prepared.row_index == row) & active_before_step).sum()
                base_replan_count += (prepared.replanned & active_before_step).sum()
                for name, value in action_diagnostics.items():
                    action_diagnostic_sums[name] += (value * active_before_step.float()).sum()
                finger_root_load_sum += (prepared.finger_root_load * active_before_step[:, None].float()).sum(dim=0)
                finger_root_load_saturation_count += (
                    (prepared.finger_root_load >= 0.95) & active_before_step[:, None]
                ).sum(dim=0)
                second_root_load = prepared.finger_root_load.topk(k=2, dim=-1).values[:, 1]
                load_conditions = {
                    "no_contact": ~pre_action_contact & active_before_step,
                    "contact": pre_action_contact,
                    "grasp": pre_action_grasp,
                }
                for condition_name, condition in load_conditions.items():
                    second_load_sums[condition_name] += (second_root_load * condition.float()).sum()
                    second_load_counts[condition_name] += condition.sum()
                vertical_residual = action_diagnostics["vertical_residual_signed_m"]
                for condition_name, condition in (
                    ("no_load", (second_root_load < _NO_LOAD_THRESHOLD) & active_before_step),
                    ("two_finger_load", (second_root_load >= _TWO_FINGER_LOAD_THRESHOLD) & active_before_step),
                ):
                    load_conditioned_vertical_sums[condition_name] += (vertical_residual * condition.float()).sum()
                    load_conditioned_vertical_counts[condition_name] += condition.sum()
                actual_eef_delta_z_sum += (actual_eef_delta_z * active_before_step.float()).sum()
                contact_and_grasp_step_count += (pre_action_contact & pre_action_grasp).sum()
                conditioned_masks = {"contact": pre_action_contact, "carrying": pre_action_carrying}
                conditioned_values = {**action_diagnostics, "actual_eef_delta_z_m": actual_eef_delta_z}
                for condition_name, condition in conditioned_masks.items():
                    conditioned_step_counts[condition_name] += condition.sum()
                    for name in _PHASE_CONDITIONED_Z_NAMES:
                        conditioned_z_sums[condition_name][name] += (conditioned_values[name] * condition.float()).sum()

                current_events = _event_flags(step_info)
                for name, value in current_events.items():
                    wave_event_ever[name] |= value & active_before_step

                accepted = quota.accept(done)
                if bool(accepted.any()):
                    episode = step_info["episode"]
                    success_count += int((episode["success_once"] & accepted).sum())
                    fail_count += int((episode["fail_at_end"] & accepted).sum())
                    return_sum += float(episode["return"][accepted].sum())
                    length_sum += float(episode["length"][accepted].float().sum())
                    accepted_contacted_max_lift = step_info["max_contacted_carry_lift_height"][accepted].float()
                    accepted_physical_max_lift = step_info["physical_max_lift_height"][accepted].float()
                    episode_contacted_max_lift_sum += float(accepted_contacted_max_lift.sum())
                    episode_contacted_max_lift_max = max(
                        episode_contacted_max_lift_max,
                        float(accepted_contacted_max_lift.max()),
                    )
                    episode_physical_max_lift_sum += float(accepted_physical_max_lift.sum())
                    episode_physical_max_lift_max = max(
                        episode_physical_max_lift_max,
                        float(accepted_physical_max_lift.max()),
                    )
                    for threshold_name, threshold_m in _LIFT_DIAGNOSTIC_THRESHOLDS_M:
                        episode_contacted_lift_threshold_counts[threshold_name] += int(
                            (accepted_contacted_max_lift >= threshold_m).sum()
                        )
                        episode_physical_lift_threshold_counts[threshold_name] += int(
                            (accepted_physical_max_lift >= threshold_m).sum()
                        )
                    for name, value in wave_event_ever.items():
                        event_episode_counts[name] += int((value & accepted).sum())
                    for name, value in wave_topology_ever.items():
                        topology_episode_counts[name] += int((value & accepted).sum())
                    for finger_index, finger_name in enumerate(_FINGER_NAMES):
                        any_frame_finger_episode_counts[finger_name] += int(
                            (wave_any_frame_finger_ever[:, finger_index] & accepted).sum()
                        )
                    for name, value in wave_any_frame_topology_ever.items():
                        any_frame_topology_episode_counts[name] += int((value & accepted).sum())
                    for name, value in wave_partial_contact_stage_ever.items():
                        partial_contact_stage_episode_counts[name] += int((value & accepted).sum())
                    for name, value in wave_reward_v13_positive_ever.items():
                        reward_v13_positive_episode_counts[name] += int((value & accepted).sum())
                base_discarded_rows += cache.remaining_rows(done).sum()
                cache.invalidate(done)
                if quota.wave_complete:
                    break
                if bool(done.any()):
                    observation, current_info = env.reset(world_mask=done)
                else:
                    observation, current_info = next_observation, step_info
            else:
                raise RuntimeError(
                    f"Evaluation wave {quota.wave} did not finish within {args.max_episode_steps} control steps"
                )
    finally:
        actor_critic.train(was_training)

    episode_count = quota.completed
    if episode_count != episodes:
        raise RuntimeError(f"Evaluation collected {episode_count} episodes, expected exactly {episodes}")
    count = max(episode_count, 1)
    step_count = max(int(active_step_count), 1)
    contact_step_count = int(conditioned_step_counts["contact"])
    reward_v13_conditioned_count = int(reward_v13_non_thumb_conditioned_count)
    opposed_streak_metrics = _opposed_streak_metrics_from_histogram(
        opposed_streak_histogram,
        step_count=step_count,
    )
    metrics = {
        "episodes": float(episode_count),
        "success_count": float(success_count),
        "fail_count": float(fail_count),
        "success_rate": success_count / count,
        "fail_rate": fail_count / count,
        "mean_return": return_sum / count,
        "mean_length": length_sum / count,
        "mean_episode_max_lift_height_m": episode_contacted_max_lift_sum / count,
        "max_episode_max_lift_height_m": episode_contacted_max_lift_max,
        "mean_episode_contacted_carry_max_lift_height_m": episode_contacted_max_lift_sum / count,
        "max_episode_contacted_carry_max_lift_height_m": episode_contacted_max_lift_max,
        "mean_episode_physical_max_lift_height_m": episode_physical_max_lift_sum / count,
        "max_episode_physical_max_lift_height_m": episode_physical_max_lift_max,
        **{
            f"episode_contacted_carry_max_lift_ge_{threshold}_ever_rate": value / count
            for threshold, value in episode_contacted_lift_threshold_counts.items()
        },
        **{
            f"episode_physical_max_lift_ge_{threshold}_ever_rate": value / count
            for threshold, value in episode_physical_lift_threshold_counts.items()
        },
        **{
            f"phase_{name}_fraction": float(phase_counts[index]) / step_count for index, name in enumerate(_PHASE_NAMES)
        },
        **{f"event_{name}_ever_rate": value / count for name, value in event_episode_counts.items()},
        **{f"{name}_episode_ever_rate": value / count for name, value in topology_episode_counts.items()},
        **{
            f"finger_{finger_name}_contact_any_frame_episode_ever_rate": value / count
            for finger_name, value in any_frame_finger_episode_counts.items()
        },
        **{
            f"{name}_any_frame_episode_ever_rate": value / count
            for name, value in any_frame_topology_episode_counts.items()
        },
        **{
            f"partial_contact_stage_{name}_episode_ever_rate": value / count
            for name, value in partial_contact_stage_episode_counts.items()
        },
        **{
            f"base_action_row_{row}_fraction": float(base_row_counts[row]) / step_count
            for row in range(_BASE_ACTION_HORIZON)
        },
        "base_action_replan_lane_count": float(base_replan_count),
        "base_action_replan_lanes_per_1000_steps": 1000.0 * float(base_replan_count) / step_count,
        "base_action_discarded_row_count": float(base_discarded_rows),
        "opposed_pregrasp_score_mean": float(pregrasp_score_sum) / step_count,
        "opposed_pregrasp_score_ge_0_1_step_fraction": float(pregrasp_score_ge_tenth_count) / step_count,
        "opposed_pregrasp_score_ge_0_5_step_fraction": float(pregrasp_score_ge_half_count) / step_count,
        **{name + "_mean": float(value) / step_count for name, value in reward_v13_signal_sums.items()},
        **{
            f"reward_v13_{name}_positive_step_fraction": int(value) / step_count
            for name, value in reward_v13_positive_step_counts.items()
        },
        **{
            f"reward_v13_{name}_positive_episode_ever_rate": value / count
            for name, value in reward_v13_positive_episode_counts.items()
        },
        **{
            f"reward_v13_non_thumb_conditioned_{name}_mean": float(value) / max(reward_v13_conditioned_count, 1)
            for name, value in reward_v13_non_thumb_conditioned_sums.items()
        },
        "reward_v13_non_thumb_conditioned_thumb_gap_m_min": (
            float(reward_v13_non_thumb_conditioned_thumb_gap_min) if reward_v13_conditioned_count else 0.0
        ),
        "contact_step_fraction": contact_step_count / step_count,
        **{f"{name}_step_fraction": int(value) / step_count for name, value in topology_step_counts.items()},
        **{
            f"finger_{finger_name}_contact_any_frame_step_fraction": int(any_frame_finger_step_counts[finger_index])
            / step_count
            for finger_index, finger_name in enumerate(_FINGER_NAMES)
        },
        **{
            f"{name}_any_frame_step_fraction": int(value) / step_count
            for name, value in any_frame_topology_step_counts.items()
        },
        **{
            f"partial_contact_stage_{name}_step_fraction": int(value) / step_count
            for name, value in partial_contact_stage_step_counts.items()
        },
        **opposed_streak_metrics,
        "thumb_to_opposed_grasp_step_conversion": int(topology_step_counts["opposed_grasp"])
        / max(int(topology_step_counts["thumb_contact"]), 1),
        "carrying_step_fraction": int(conditioned_step_counts["carrying"]) / step_count,
        "contact_to_grasp_step_conversion": float(contact_and_grasp_step_count) / max(contact_step_count, 1),
        "contact_to_grasp_episode_conversion": event_episode_counts["grasp"] / max(event_episode_counts["contact"], 1),
        "actual_eef_delta_z_m": float(actual_eef_delta_z_sum) / step_count,
        **{name: float(value) / step_count for name, value in action_diagnostic_sums.items()},
        **{
            f"finger_root_load_{finger_name}_mean": float(finger_root_load_sum[finger_index]) / step_count
            for finger_index, finger_name in enumerate(_FINGER_NAMES)
        },
        **{
            f"finger_root_load_{finger_name}_saturation_fraction": float(
                finger_root_load_saturation_count[finger_index]
            )
            / step_count
            for finger_index, finger_name in enumerate(_FINGER_NAMES)
        },
        **{
            f"finger_root_second_load_{condition_name}_mean": float(second_load_sums[condition_name])
            / max(int(second_load_counts[condition_name]), 1)
            for condition_name in second_load_sums
        },
        **{
            f"vertical_residual_signed_m_{condition_name}": float(load_conditioned_vertical_sums[condition_name])
            / max(int(load_conditioned_vertical_counts[condition_name]), 1)
            for condition_name in load_conditioned_vertical_sums
        },
        **{
            f"{condition_name}_{name}": float(conditioned_z_sums[condition_name][name])
            / max(int(conditioned_step_counts[condition_name]), 1)
            for condition_name in conditioned_step_counts
            for name in _PHASE_CONDITIONED_Z_NAMES
        },
    }
    observation, current_info = env.reset()
    return metrics, observation, current_info


def main() -> None:
    args = create_parser().parse_args()
    _validate_args(args)
    import torch

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Residual PPO training requires a CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    hand_residual_scale_contract = _hand_residual_scale_contract(args)
    hand_residual_scale = torch.tensor(
        hand_residual_scale_contract["effective_scale_normalized"],
        dtype=torch.float32,
        device=device,
    )
    env_config = GrootNewtonEnvConfig(
        num_envs=args.num_envs,
        device=args.device,
        max_episode_steps=args.max_episode_steps,
        obs_mode="policy",
        control_mode="pd_eef_pose_abs",
        reward_mode="normalized_dense",
        terminate_on_success=False,
        terminate_on_fail=True,
        capture_graph=args.capture_graph,
        camera_textures=args.camera_textures,
        load_scene_visuals=args.scene_visuals,
        hydroelastic_contacts=args.hydroelastic,
        request_finger_root_load=True,
        triangle_pairs_per_env=args.triangle_pairs_per_env,
    )
    frozen_dp, dp_config, scheduler, stats, dp_sha256 = _load_frozen_dp(args.checkpoint, device)
    _validate_frozen_dp_training_contract(dp_config)
    action_min = torch.as_tensor(stats["action_min"], dtype=torch.float32, device=device)
    action_max = torch.as_tensor(stats["action_max"], dtype=torch.float32, device=device)
    state_min = torch.as_tensor(stats["state_min"], dtype=torch.float32, device=device)
    state_max = torch.as_tensor(stats["state_max"], dtype=torch.float32, device=device)
    condition_dim = dp_config.obs_horizon * (2 * dp_config.camera_feature_dim + dp_config.state_feature_dim)
    policy_config = GrootResidualActorCriticConfig(
        condition_dim=condition_dim,
        base_action_dim=dp_config.action_dim,
        current_state_dim=dp_config.state_dim,
        base_action_row_dim=_BASE_ACTION_HORIZON,
        state_delta_dim=dp_config.state_dim,
        finger_root_load_dim=_FINGER_ROOT_LOAD_DIM,
        privileged_dim=_PRIVILEGED_STATE_DIM,
        residual_dim=16,
        hidden_dim=args.hidden_dim,
        initial_log_std=args.initial_log_std,
    )
    actor_critic = GrootResidualActorCritic(policy_config).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=args.learning_rate, eps=1.0e-5)

    generators = {
        "dp": torch.Generator(device=device).manual_seed(args.seed + 1),
        "residual": torch.Generator(device=device).manual_seed(args.seed + 2),
        "shuffle": torch.Generator(device=device).manual_seed(args.seed + 3),
        "bootstrap_dp": torch.Generator(device=device).manual_seed(args.seed + 4),
    }
    start_update = 0
    global_step = 0
    best_eval_metrics: dict[str, float] | None = None
    best_return_metrics: dict[str, float] | None = None
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("format") != GROOT_RESIDUAL_PPO_CHECKPOINT_FORMAT:
            raise ValueError("Resume checkpoint is not a Groot residual PPO checkpoint")
        _validate_resume_training_contract(resume)
        if resume.get("frozen_dp_sha256") != dp_sha256:
            raise ValueError("Resume checkpoint was trained with a different frozen DP checkpoint")
        if resume.get("policy_config") != asdict(policy_config):
            raise ValueError("Resume checkpoint residual actor-critic configuration does not match")
        saved_env_config = dict(resume.get("env_config", {}))
        current_env_config = asdict(env_config)
        for ignored in ("num_envs", "device"):
            saved_env_config.pop(ignored, None)
            current_env_config.pop(ignored, None)
        if saved_env_config != current_env_config:
            raise ValueError("Resume checkpoint environment and staged-reward configuration does not match")
        saved_args = resume.get("train_args", {})
        _validate_resume_train_args(saved_args, args)
        if resume.get("hand_residual_scale") != hand_residual_scale_contract:
            raise ValueError("Resume checkpoint effective hand residual scale does not match the current CLI")
        actor_critic.load_state_dict(resume["actor_critic"])
        optimizer.load_state_dict(resume["optimizer"])
        start_update = int(resume["update"])
        global_step = int(resume["global_step"])
        expected_global_step = start_update * args.num_envs * args.rollout_steps
        if global_step != expected_global_step:
            raise ValueError(f"Resume checkpoint update/global_step are inconsistent: {start_update} vs {global_step}")
        saved_best_eval = resume.get("best_eval_metrics")
        saved_best_return = resume.get("best_return_metrics")
        best_eval_metrics = None if saved_best_eval is None else dict(saved_best_eval)
        best_return_metrics = None if saved_best_return is None else dict(saved_best_return)
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        torch.cuda.set_rng_state(resume["cuda_rng_state"].cpu(), device=device)
        saved_generator_states = resume.get("generator_states", {})
        if set(saved_generator_states) != set(generators):
            raise ValueError(
                f"Resume checkpoint generator states {sorted(saved_generator_states)} do not match {sorted(generators)}"
            )
        for name, state in saved_generator_states.items():
            generators[name].set_state(state.cpu())

    batch_size = args.num_envs * args.rollout_steps
    total_updates = args.total_timesteps // batch_size
    if total_updates < 1:
        raise ValueError(f"total_timesteps must be at least one rollout batch ({batch_size})")
    if start_update >= total_updates:
        raise ValueError(
            f"Resume checkpoint already reached update {start_update}; increase --total-timesteps beyond "
            f"{start_update * batch_size}"
        )
    base_env = GrootNewtonEnv(env_config)
    if args.resume is not None and resume.get("finger_root_load") != base_env.finger_root_load_metadata:
        raise ValueError("Resume checkpoint finger-root load calibration does not match the current environment")
    if args.resume is not None and resume.get("hand_target") != base_env.hand_target_metadata:
        raise ValueError("Resume checkpoint hand-target metadata does not match the current environment")
    env = GrootDiffusionPolicyEnv(base_env, obs_horizon=dp_config.obs_horizon, action_horizon=1)
    training_cache = _PerLaneActionChunkCache(args.num_envs, dp_config.action_dim, device)
    training_lanes = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    policy_input_dim = policy_config.policy_input_dim
    rollout = _allocate_rollout(
        args.rollout_steps,
        args.num_envs,
        policy_input_dim,
        policy_config.residual_dim,
        device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    if args.resume is None and metrics_path.exists():
        raise FileExistsError(f"Refusing to append a new run to existing metrics: {metrics_path}")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "policy_config": asdict(policy_config),
                "env_config": asdict(env_config),
                "finger_root_load": base_env.finger_root_load_metadata,
                "hand_target": base_env.hand_target_metadata,
                "frozen_dp_sha256": dp_sha256,
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
                "hand_residual_scale": hand_residual_scale_contract,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    observation, current_info = env.reset()
    previous_event_flags = {name: value.clone() for name, value in _event_flags(current_info).items()}
    torch.cuda.reset_peak_memory_stats(device)
    training_start = time.perf_counter()
    initial_global_step = global_step

    try:
        if args.eval_episodes > 0 and start_update == 0:
            baseline, observation, current_info = _evaluate(
                env,
                frozen_dp,
                scheduler,
                actor_critic,
                action_min,
                action_max,
                state_min,
                state_max,
                hand_residual_scale,
                args,
                episodes=args.eval_episodes,
            )
            baseline_line = json.dumps(
                {"type": "baseline_eval", "is_new_best": True, "is_new_best_return": True, **baseline},
                sort_keys=True,
            )
            print(baseline_line, flush=True)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(baseline_line + "\n")
            best_eval_metrics = dict(baseline)
            best_return_metrics = dict(baseline)
            baseline_payload = _checkpoint_payload(
                actor_critic,
                optimizer,
                update=0,
                global_step=0,
                best_eval_metrics=best_eval_metrics,
                best_return_metrics=best_return_metrics,
                policy_config=policy_config,
                dp_sha256=dp_sha256,
                env_config=env_config,
                finger_root_load_metadata=base_env.finger_root_load_metadata,
                hand_target_metadata=base_env.hand_target_metadata,
                args=args,
                generators=generators,
            )
            _save_checkpoint(baseline_payload, args.output_dir / "checkpoint_00000000.pt")
            _save_checkpoint(baseline_payload, args.output_dir / "best.pt")
            _save_checkpoint(baseline_payload, args.output_dir / "best_return.pt")

        for update in range(start_update + 1, total_updates + 1):
            if args.anneal_lr:
                fraction = 1.0 - (update - 1.0) / total_updates
                optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
            evaluation_due = (
                args.eval_every_updates > 0
                and args.eval_episodes > 0
                and (update % args.eval_every_updates == 0 or update == total_updates)
            )
            rollout["timeout_value"].zero_()
            rollout["base_action_discarded_rows"].zero_()
            rollout_start = time.perf_counter()
            for step in range(args.rollout_steps):
                pre_action_contact, pre_action_grasp, pre_action_carrying = _pre_action_task_flags(current_info)
                pre_action_thumb, pre_action_non_thumb, _ = _finger_contact_topology(current_info)
                current_eef_position = observation[_STATE_KEY][:, -1, 7:10].float()
                current_hand_position = observation[_STATE_KEY][:, -1, 16:26].float()
                prepared = _prepare_policy_step(
                    observation,
                    frozen_dp,
                    scheduler,
                    training_cache,
                    action_min,
                    action_max,
                    state_min,
                    state_max,
                    inference_steps=args.inference_steps,
                    use_bfloat16=args.bfloat16,
                    generator=generators["dp"],
                )
                privileged = _privileged_task_state(current_info, env_config)
                with torch.no_grad():
                    raw_latent, log_prob, _, value = actor_critic.act(
                        prepared.policy_input,
                        privileged,
                        generator=generators["residual"],
                    )
                    action = compose_residual_action(
                        prepared.base_action,
                        raw_latent,
                        action_min,
                        action_max,
                        position_scale_m=args.position_residual_scale_m,
                        vertical_position_scale_m=args.vertical_residual_scale_m,
                        rotation_scale_rad=math.radians(args.rotation_residual_scale_deg),
                        hand_scale_normalized=hand_residual_scale,
                        world_from_action_rotation=_R_WORLD_FROM_ACTION,
                    )
                    action_diagnostics = _action_diagnostics(
                        prepared.base_action,
                        raw_latent,
                        action_min,
                        action_max,
                        current_eef_position,
                        current_hand_position,
                        position_scale_m=args.position_residual_scale_m,
                        vertical_position_scale_m=args.vertical_residual_scale_m,
                        hand_scale_normalized=hand_residual_scale,
                        hand_max_joint_step_rad=env_config.hand_max_joint_step_rad,
                        world_from_action_rotation=_R_WORLD_FROM_ACTION,
                    )
                rollout["policy_input"][step].copy_(prepared.policy_input)
                rollout["privileged"][step].copy_(privileged)
                rollout["raw_latent"][step].copy_(raw_latent)
                rollout["log_prob"][step].copy_(log_prob)
                rollout["value"][step].copy_(value)
                rollout["finger_root_load"][step].copy_(prepared.finger_root_load)
                rollout["base_action_row"][step].copy_(prepared.row_index)
                rollout["base_action_replanned"][step].copy_(prepared.replanned)
                for name, diagnostic in action_diagnostics.items():
                    rollout[name][step].copy_(diagnostic)

                next_observation, reward, terminated, truncated, step_info = env.step(action)
                done = terminated | truncated
                training_cache.advance(training_lanes, validate=False)
                current_event_flags = _event_flags(step_info)
                (
                    control_step_finger_contact,
                    _,
                    _,
                    control_step_opposed_grasp,
                    control_step_opposed_streak,
                ) = _control_step_contact_topology(step_info)
                eef_delta_action = next_observation[_STATE_KEY][:, -1, 7:10].float() - current_eef_position
                rollout["actual_eef_delta_z_m"][step].copy_(_action_position_to_world(eef_delta_action)[..., 2])
                rollout["pre_action_contact"][step].copy_(pre_action_contact)
                rollout["pre_action_thumb_contact"][step].copy_(pre_action_thumb)
                rollout["pre_action_non_thumb_contact"][step].copy_(pre_action_non_thumb)
                rollout["pre_action_grasp"][step].copy_(pre_action_grasp)
                rollout["pre_action_carrying"][step].copy_(pre_action_carrying)
                rollout["finger_contact_any_frame"][step].copy_(control_step_finger_contact)
                rollout["opposed_grasp_any_frame"][step].copy_(control_step_opposed_grasp)
                rollout["opposed_grasp_max_consecutive_physics_frames"][step].copy_(control_step_opposed_streak)
                for name, value in _partial_contact_reward_stages(step_info).items():
                    rollout[f"partial_contact_stage_{name}"][step].copy_(value)
                rollout["reward"][step].copy_(reward)
                rollout["terminated"][step].copy_(terminated)
                rollout["truncated"][step].copy_(truncated)
                rollout["episode_done"][step].copy_(done)
                rollout["episode_success"][step].copy_(step_info["episode"]["success_once"] & done)
                rollout["episode_fail"][step].copy_(step_info["episode"]["fail_at_end"] & done)
                rollout["episode_return"][step].copy_(
                    torch.where(done, step_info["episode"]["return"], torch.zeros_like(reward))
                )
                rollout["episode_length"][step].copy_(
                    torch.where(done, step_info["episode"]["length"].float(), torch.zeros_like(reward))
                )
                rollout["episode_max_lift_height_m"][step].copy_(
                    torch.where(
                        done,
                        step_info["max_contacted_carry_lift_height"].float(),
                        torch.zeros_like(reward),
                    )
                )
                rollout["episode_physical_max_lift_height_m"][step].copy_(
                    torch.where(done, step_info["physical_max_lift_height"].float(), torch.zeros_like(reward))
                )
                rollout["current_lift_height_m"][step].copy_(step_info["current_lift_height"].float())
                rollout["physical_max_lift_height_m"][step].copy_(step_info["physical_max_lift_height"].float())
                rollout["contacted_carry_max_lift_height_m"][step].copy_(
                    step_info["max_contacted_carry_lift_height"].float()
                )
                rollout["opposed_pregrasp_score"][step].copy_(step_info["opposed_pregrasp_score"].float())
                for name, value in _reward_v13_signals(step_info).items():
                    rollout[name][step].copy_(value)
                rollout["task_phase"][step].copy_(step_info["task_phase"].long())
                for name, current_flag in current_event_flags.items():
                    rollout[f"event_{name}_rise"][step].copy_(current_flag & ~previous_event_flags[name])

                timeout_mask = truncated & ~terminated
                if args.bootstrap_time_limit and bool(timeout_mask.any()):
                    timeout_indices = torch.where(timeout_mask)[0]
                    terminal_observation = _select_tree(next_observation, timeout_indices)
                    terminal_cache = training_cache.clone_lanes(timeout_indices)
                    terminal_prepared = _prepare_policy_step(
                        terminal_observation,
                        frozen_dp,
                        scheduler,
                        terminal_cache,
                        action_min,
                        action_max,
                        state_min,
                        state_max,
                        inference_steps=args.inference_steps,
                        use_bfloat16=args.bfloat16,
                        generator=generators["bootstrap_dp"],
                    )
                    terminal_privileged = _privileged_task_state(step_info, env_config)[timeout_indices]
                    with torch.no_grad():
                        terminal_value = actor_critic.get_value(terminal_prepared.policy_input, terminal_privileged)
                    rollout["timeout_value"][step, timeout_indices] = terminal_value

                discarded_rows = training_cache.remaining_rows(done)
                rollout["base_action_discarded_rows"][step].copy_(discarded_rows)
                training_cache.invalidate(done)
                if bool(done.any()):
                    observation, current_info = env.reset(world_mask=done)
                else:
                    observation, current_info = next_observation, step_info
                previous_event_flags = {name: value.clone() for name, value in _event_flags(current_info).items()}
                global_step += args.num_envs

            value_only_plan = evaluation_due or update == total_updates
            value_cache = training_cache.clone_lanes() if value_only_plan else training_cache
            last_prepared = _prepare_policy_step(
                observation,
                frozen_dp,
                scheduler,
                value_cache,
                action_min,
                action_max,
                state_min,
                state_max,
                inference_steps=args.inference_steps,
                use_bfloat16=args.bfloat16,
                generator=generators["bootstrap_dp"] if value_only_plan else generators["dp"],
            )
            last_privileged = _privileged_task_state(current_info, env_config)
            with torch.no_grad():
                last_value = actor_critic.get_value(last_prepared.policy_input, last_privileged)
            advantages, returns = compute_gae(
                rollout["reward"],
                rollout["value"],
                rollout["terminated"],
                rollout["truncated"],
                rollout["timeout_value"],
                last_value,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                bootstrap_time_limit=args.bootstrap_time_limit,
            )
            rollout_seconds = time.perf_counter() - rollout_start
            update_metrics = _ppo_update(
                actor_critic,
                optimizer,
                rollout,
                advantages,
                returns,
                args,
                generators["shuffle"],
            )

            done_count = int(rollout["episode_done"].sum())
            success_count = int(rollout["episode_success"].sum())
            fail_count = int(rollout["episode_fail"].sum())
            episode_return_sum = float(rollout["episode_return"].sum())
            episode_length_sum = float(rollout["episode_length"].sum())
            elapsed = time.perf_counter() - training_start
            raw_latent = rollout["raw_latent"]
            position_scales = raw_latent.new_tensor(
                (
                    args.position_residual_scale_m,
                    args.position_residual_scale_m,
                    args.vertical_residual_scale_m,
                )
            )
            mean_position_residual = float(
                torch.linalg.vector_norm(position_scales * torch.tanh(raw_latent[..., :3]), dim=-1).mean()
            )
            mean_rotation_residual = float(
                args.rotation_residual_scale_deg
                * torch.tanh(torch.linalg.vector_norm(raw_latent[..., 3:6], dim=-1)).mean()
            )
            mean_hand_residual = float((hand_residual_scale * torch.tanh(raw_latent[..., 6:16]).abs()).mean())
            value_variance = torch.var(returns.reshape(-1), unbiased=False)
            explained_variance = float(
                1.0
                - torch.var(returns.reshape(-1) - rollout["value"].reshape(-1), unbiased=False)
                / value_variance.clamp_min(1.0e-8)
            )
            metrics = {
                "type": "train",
                "update": update,
                "global_step": global_step,
                "sps": (global_step - initial_global_step) / max(elapsed, 1.0e-6),
                "rollout_sps": batch_size / max(rollout_seconds, 1.0e-6),
                "mean_step_reward": float(rollout["reward"].mean()),
                "episodes": done_count,
                "success_rate": success_count / max(done_count, 1),
                "fail_rate": fail_count / max(done_count, 1),
                "mean_episode_return": episode_return_sum / max(done_count, 1),
                "mean_episode_length": episode_length_sum / max(done_count, 1),
                "explained_variance": explained_variance,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mean_position_residual_m": mean_position_residual,
                "mean_rotation_residual_deg": mean_rotation_residual,
                "mean_hand_residual_normalized": mean_hand_residual,
                "mean_policy_std": float(actor_critic.log_std.clamp(-5.0, 0.0).exp().mean()),
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                **_rollout_diagnostic_metrics(rollout, frames_per_action=base_env.frames_per_action),
                **update_metrics,
            }
            line = json.dumps(metrics, sort_keys=True)
            print(line, flush=True)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

            eval_metrics: dict[str, float] | None = None
            if evaluation_due:
                eval_metrics, observation, current_info = _evaluate(
                    env,
                    frozen_dp,
                    scheduler,
                    actor_critic,
                    action_min,
                    action_max,
                    state_min,
                    state_max,
                    hand_residual_scale,
                    args,
                    episodes=args.eval_episodes,
                )
                training_cache.invalidate()
                previous_event_flags = {name: value.clone() for name, value in _event_flags(current_info).items()}
            is_new_best = eval_metrics is not None and _is_better_eval(eval_metrics, best_eval_metrics)
            is_new_best_return = eval_metrics is not None and _is_better_return(eval_metrics, best_return_metrics)
            if eval_metrics is not None:
                eval_line = json.dumps(
                    {
                        "type": "eval",
                        "update": update,
                        "global_step": global_step,
                        "is_new_best": is_new_best,
                        "is_new_best_return": is_new_best_return,
                        **eval_metrics,
                    },
                    sort_keys=True,
                )
                print(eval_line, flush=True)
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(eval_line + "\n")

            should_save = update % args.save_every_updates == 0 or update == total_updates
            payload: dict[str, Any] | None = None
            if should_save or is_new_best or is_new_best_return:
                if is_new_best and eval_metrics is not None:
                    best_eval_metrics = dict(eval_metrics)
                if is_new_best_return and eval_metrics is not None:
                    best_return_metrics = dict(eval_metrics)
                payload = _checkpoint_payload(
                    actor_critic,
                    optimizer,
                    update=update,
                    global_step=global_step,
                    best_eval_metrics=best_eval_metrics,
                    best_return_metrics=best_return_metrics,
                    policy_config=policy_config,
                    dp_sha256=dp_sha256,
                    env_config=env_config,
                    finger_root_load_metadata=base_env.finger_root_load_metadata,
                    hand_target_metadata=base_env.hand_target_metadata,
                    args=args,
                    generators=generators,
                )
            if should_save and payload is not None:
                _save_checkpoint(payload, args.output_dir / f"checkpoint_{global_step:012d}.pt")
            if is_new_best and payload is not None:
                _save_checkpoint(payload, args.output_dir / "best.pt")
            if is_new_best_return and payload is not None:
                _save_checkpoint(payload, args.output_dir / "best_return.pt")
    finally:
        env.close()


if __name__ == "__main__":
    main()
