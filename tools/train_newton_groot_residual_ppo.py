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
from dataclasses import asdict
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
)

_PRIVILEGED_STATE_DIM = 20
_TASK_PHASE_COUNT = 5


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
    parser.add_argument("--rotation-residual-scale-deg", type=float, default=5.0)
    parser.add_argument("--hand-residual-scale-normalized", type=float, default=0.1)
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
    if (
        min(
            args.position_residual_scale_m,
            args.rotation_residual_scale_deg,
            args.hand_residual_scale_normalized,
        )
        <= 0.0
    ):
        raise ValueError("residual action scales must be positive")
    batch_size = args.num_envs * args.rollout_steps
    if args.minibatch_size > batch_size:
        raise ValueError(f"minibatch_size={args.minibatch_size} exceeds rollout batch size {batch_size}")


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
            info["final_z_error"].float().div(config.final_z_threshold).clamp(0.0, 5.0),
            info["orientation_error"].float().div(config.final_orientation_threshold_rad).clamp(0.0, 5.0),
            info["episode"]["length"].float().div(max(config.max_episode_steps, 1)).clamp(0.0, 1.0),
        ),
        dim=-1,
    )
    output = torch.cat((phase_one_hot, flags, continuous), dim=-1)
    if output.shape[-1] != _PRIVILEGED_STATE_DIM:
        raise RuntimeError(f"Expected privileged state width {_PRIVILEGED_STATE_DIM}, got {output.shape[-1]}")
    return output


def _encode_policy_input(
    observation: dict[str, Any],
    frozen_dp: Any,
    scheduler: Any,
    action_min: Any,
    action_max: Any,
    *,
    inference_steps: int,
    use_bfloat16: bool,
    generator: Any | None = None,
    initial_noise: Any | None = None,
) -> tuple[Any, Any]:
    import torch

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        condition = frozen_dp.encode_observation(observation)
        base_chunk = frozen_dp.predict_action_from_condition(
            condition,
            scheduler,
            inference_steps=inference_steps,
            generator=generator,
            initial_noise=initial_noise,
        )
    condition = condition.detach().float()
    base_action = base_chunk[:, 0].detach().float()
    normalized_base = normalize_physical_action(base_action, action_min, action_max)
    return torch.cat((condition, normalized_base), dim=-1), base_action


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
    }


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

    totals = dict.fromkeys(("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"), 0.0)
    sample_count = 0
    stop_early = False
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
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
            count = int(indices.numel())
            totals["policy_loss"] += float(policy_loss.detach()) * count
            totals["value_loss"] += float(value_loss.detach()) * count
            totals["entropy"] += float(entropy_mean.detach()) * count
            totals["approx_kl"] += float(approx_kl) * count
            totals["clip_fraction"] += float(clip_fraction) * count
            sample_count += count
            if args.target_kl > 0.0 and float(approx_kl) > args.target_kl:
                stop_early = True
                break
        if stop_early:
            break
    return {name: value / max(sample_count, 1) for name, value in totals.items()}


def _checkpoint_payload(
    actor_critic: Any,
    optimizer: Any,
    *,
    update: int,
    global_step: int,
    best_eval_success: float,
    policy_config: Any,
    dp_sha256: str,
    env_config: GrootNewtonEnvConfig,
    args: argparse.Namespace,
    generators: dict[str, Any],
) -> dict[str, Any]:
    import torch

    return {
        "format": GROOT_RESIDUAL_PPO_CHECKPOINT_FORMAT,
        "update": int(update),
        "global_step": int(global_step),
        "best_eval_success": float(best_eval_success),
        "policy_config": asdict(policy_config),
        "frozen_dp_sha256": dp_sha256,
        "env_config": asdict(env_config),
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
    args: argparse.Namespace,
    *,
    episodes: int,
) -> tuple[dict[str, float], Any, dict[str, Any]]:
    import torch

    observation, current_info = env.reset()
    eval_generator = torch.Generator(device=args.device)
    eval_generator.manual_seed(args.seed + 10_003)
    initial_noise = torch.randn(
        (args.num_envs, frozen_dp.config.pred_horizon, frozen_dp.config.action_dim),
        dtype=torch.float32,
        device=args.device,
        generator=eval_generator,
    )
    episode_count = 0
    success_count = 0
    fail_count = 0
    return_sum = 0.0
    length_sum = 0.0
    actor_critic.eval()
    while episode_count < episodes:
        policy_input, base_action = _encode_policy_input(
            observation,
            frozen_dp,
            scheduler,
            action_min,
            action_max,
            inference_steps=args.inference_steps,
            use_bfloat16=args.bfloat16,
            generator=eval_generator,
            initial_noise=initial_noise,
        )
        privileged = _privileged_task_state(current_info, env.unwrapped.config)
        with torch.no_grad():
            raw_latent, _, _, _ = actor_critic.act(
                policy_input,
                privileged,
                deterministic=True,
            )
            action = compose_residual_action(
                base_action,
                raw_latent,
                action_min,
                action_max,
                position_scale_m=args.position_residual_scale_m,
                rotation_scale_rad=math.radians(args.rotation_residual_scale_deg),
                hand_scale_normalized=args.hand_residual_scale_normalized,
            )
        next_observation, _, terminated, truncated, step_info = env.step(action)
        done = terminated | truncated
        if bool(done.any()):
            episode = step_info["episode"]
            episode_count += int(done.sum())
            success_count += int((episode["success_once"] & done).sum())
            fail_count += int((episode["fail_at_end"] & done).sum())
            return_sum += float(episode["return"][done].sum())
            length_sum += float(episode["length"][done].float().sum())
            observation, current_info = env.reset(world_mask=done)
        else:
            observation, current_info = next_observation, step_info
    actor_critic.train()
    count = max(episode_count, 1)
    metrics = {
        "episodes": float(episode_count),
        "success_rate": success_count / count,
        "fail_rate": fail_count / count,
        "mean_return": return_sum / count,
        "mean_length": length_sum / count,
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
    )
    frozen_dp, dp_config, scheduler, stats, dp_sha256 = _load_frozen_dp(args.checkpoint, device)
    action_min = torch.as_tensor(stats["action_min"], dtype=torch.float32, device=device)
    action_max = torch.as_tensor(stats["action_max"], dtype=torch.float32, device=device)
    condition_dim = dp_config.obs_horizon * (2 * dp_config.camera_feature_dim + dp_config.state_feature_dim)
    policy_config = GrootResidualActorCriticConfig(
        condition_dim=condition_dim,
        base_action_dim=dp_config.action_dim,
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
    }
    start_update = 0
    global_step = 0
    best_eval_success = -1.0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("format") != GROOT_RESIDUAL_PPO_CHECKPOINT_FORMAT:
            raise ValueError("Resume checkpoint is not a Groot residual PPO checkpoint")
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
        for name in (
            "num_envs",
            "rollout_steps",
            "seed",
            "inference_steps",
            "position_residual_scale_m",
            "rotation_residual_scale_deg",
            "hand_residual_scale_normalized",
            "gamma",
            "gae_lambda",
            "bootstrap_time_limit",
            "bfloat16",
        ):
            if saved_args.get(name) != getattr(args, name):
                raise ValueError(f"Resume checkpoint training contract differs for {name}")
        actor_critic.load_state_dict(resume["actor_critic"])
        optimizer.load_state_dict(resume["optimizer"])
        start_update = int(resume["update"])
        global_step = int(resume["global_step"])
        expected_global_step = start_update * args.num_envs * args.rollout_steps
        if global_step != expected_global_step:
            raise ValueError(f"Resume checkpoint update/global_step are inconsistent: {start_update} vs {global_step}")
        best_eval_success = float(resume.get("best_eval_success", best_eval_success))
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        torch.cuda.set_rng_state(resume["cuda_rng_state"].cpu(), device=device)
        for name, state in resume["generator_states"].items():
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
    env = GrootDiffusionPolicyEnv(base_env, obs_horizon=dp_config.obs_horizon, action_horizon=1)
    policy_input_dim = condition_dim + dp_config.action_dim
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
                "frozen_dp_sha256": dp_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    observation, current_info = env.reset()
    cached_policy_input: Any | None = None
    cached_base_action: Any | None = None
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
                args,
                episodes=args.eval_episodes,
            )
            baseline_line = json.dumps({"type": "baseline_eval", **baseline}, sort_keys=True)
            print(baseline_line, flush=True)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(baseline_line + "\n")
            best_eval_success = baseline["success_rate"]
            baseline_payload = _checkpoint_payload(
                actor_critic,
                optimizer,
                update=0,
                global_step=0,
                best_eval_success=best_eval_success,
                policy_config=policy_config,
                dp_sha256=dp_sha256,
                env_config=env_config,
                args=args,
                generators=generators,
            )
            _save_checkpoint(baseline_payload, args.output_dir / "checkpoint_00000000.pt")
            _save_checkpoint(baseline_payload, args.output_dir / "best.pt")

        for update in range(start_update + 1, total_updates + 1):
            if args.anneal_lr:
                fraction = 1.0 - (update - 1.0) / total_updates
                optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
            rollout["timeout_value"].zero_()
            rollout_start = time.perf_counter()
            for step in range(args.rollout_steps):
                if cached_policy_input is None or cached_base_action is None:
                    policy_input, base_action = _encode_policy_input(
                        observation,
                        frozen_dp,
                        scheduler,
                        action_min,
                        action_max,
                        inference_steps=args.inference_steps,
                        use_bfloat16=args.bfloat16,
                        generator=generators["dp"],
                    )
                else:
                    policy_input, base_action = cached_policy_input, cached_base_action
                    cached_policy_input = None
                    cached_base_action = None
                privileged = _privileged_task_state(current_info, env_config)
                with torch.no_grad():
                    raw_latent, log_prob, _, value = actor_critic.act(
                        policy_input,
                        privileged,
                        generator=generators["residual"],
                    )
                    action = compose_residual_action(
                        base_action,
                        raw_latent,
                        action_min,
                        action_max,
                        position_scale_m=args.position_residual_scale_m,
                        rotation_scale_rad=math.radians(args.rotation_residual_scale_deg),
                        hand_scale_normalized=args.hand_residual_scale_normalized,
                    )
                rollout["policy_input"][step].copy_(policy_input)
                rollout["privileged"][step].copy_(privileged)
                rollout["raw_latent"][step].copy_(raw_latent)
                rollout["log_prob"][step].copy_(log_prob)
                rollout["value"][step].copy_(value)

                next_observation, reward, terminated, truncated, step_info = env.step(action)
                done = terminated | truncated
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

                timeout_mask = truncated & ~terminated
                if args.bootstrap_time_limit and bool(timeout_mask.any()):
                    timeout_indices = torch.where(timeout_mask)[0]
                    terminal_observation = _select_tree(next_observation, timeout_indices)
                    terminal_input, _ = _encode_policy_input(
                        terminal_observation,
                        frozen_dp,
                        scheduler,
                        action_min,
                        action_max,
                        inference_steps=args.inference_steps,
                        use_bfloat16=args.bfloat16,
                        generator=generators["dp"],
                    )
                    terminal_privileged = _privileged_task_state(step_info, env_config)[timeout_indices]
                    with torch.no_grad():
                        terminal_value = actor_critic.get_value(terminal_input, terminal_privileged)
                    rollout["timeout_value"][step, timeout_indices] = terminal_value

                if bool(done.any()):
                    observation, current_info = env.reset(world_mask=done)
                else:
                    observation, current_info = next_observation, step_info
                global_step += args.num_envs

            cached_policy_input, cached_base_action = _encode_policy_input(
                observation,
                frozen_dp,
                scheduler,
                action_min,
                action_max,
                inference_steps=args.inference_steps,
                use_bfloat16=args.bfloat16,
                generator=generators["dp"],
            )
            last_privileged = _privileged_task_state(current_info, env_config)
            with torch.no_grad():
                last_value = actor_critic.get_value(cached_policy_input, last_privileged)
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
            mean_position_residual = float(
                torch.linalg.vector_norm(
                    args.position_residual_scale_m * torch.tanh(raw_latent[..., :3]), dim=-1
                ).mean()
            )
            mean_rotation_residual = float(
                args.rotation_residual_scale_deg
                * torch.tanh(torch.linalg.vector_norm(raw_latent[..., 3:6], dim=-1)).mean()
            )
            mean_hand_residual = float(
                args.hand_residual_scale_normalized * torch.tanh(raw_latent[..., 6:16]).abs().mean()
            )
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
                **update_metrics,
            }
            line = json.dumps(metrics, sort_keys=True)
            print(line, flush=True)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

            eval_metrics: dict[str, float] | None = None
            if (
                args.eval_every_updates > 0
                and args.eval_episodes > 0
                and (update % args.eval_every_updates == 0 or update == total_updates)
            ):
                eval_metrics, observation, current_info = _evaluate(
                    env,
                    frozen_dp,
                    scheduler,
                    actor_critic,
                    action_min,
                    action_max,
                    args,
                    episodes=args.eval_episodes,
                )
                cached_policy_input = None
                cached_base_action = None
                eval_line = json.dumps({"type": "eval", "update": update, "global_step": global_step, **eval_metrics})
                print(eval_line, flush=True)
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(eval_line + "\n")

            should_save = update % args.save_every_updates == 0 or update == total_updates
            is_new_best = eval_metrics is not None and eval_metrics["success_rate"] > best_eval_success
            payload: dict[str, Any] | None = None
            if should_save or is_new_best:
                if is_new_best and eval_metrics is not None:
                    best_eval_success = eval_metrics["success_rate"]
                payload = _checkpoint_payload(
                    actor_critic,
                    optimizer,
                    update=update,
                    global_step=global_step,
                    best_eval_success=best_eval_success,
                    policy_config=policy_config,
                    dp_sha256=dp_sha256,
                    env_config=env_config,
                    args=args,
                    generators=generators,
                )
            if should_save and payload is not None:
                _save_checkpoint(payload, args.output_dir / f"checkpoint_{global_step:012d}.pt")
            if is_new_best and payload is not None:
                _save_checkpoint(payload, args.output_dir / "best.pt")
    finally:
        env.close()


if __name__ == "__main__":
    main()
