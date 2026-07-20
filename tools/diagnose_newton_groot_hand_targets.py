#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare online DP/residual hand targets with a successful dataset replay."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from teleop_stack.envs.groot_newton_env import HAND_JOINT_NAMES

_THUMB_HAND_INDICES = (0, 1, 9)
_THUMB_JOINT_NAMES = tuple(HAND_JOINT_NAMES[index] for index in _THUMB_HAND_INDICES)
_HAND_ACTION_SLICE = slice(9, 19)
_SUMMARY_SCHEMA_VERSION = 1


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Frozen Diffusion Policy checkpoint")
    parser.add_argument("residual_checkpoint", type=Path, help="Residual PPO checkpoint to diagnose")
    parser.add_argument(
        "reference_dataset",
        type=Path,
        help="LeRobot dataset root or pre-extracted .npy corrected physical actions",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hand_target_diagnostic"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--inference-steps", type=int, default=6)
    parser.add_argument(
        "--triangle-pairs-per-env",
        type=int,
        default=196608,
        help="Diagnostic collision broad-phase capacity per environment.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capture-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-gates", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.num_envs, args.max_steps, args.inference_steps, args.triangle_pairs_per_env) < 1:
        raise ValueError("num-envs, max-steps, inference-steps, and triangle-pairs-per-env must be positive")
    if args.episode_index < 0:
        raise ValueError("episode-index cannot be negative")
    if not str(args.device).startswith("cuda"):
        raise ValueError("The hand-target diagnostic requires a CUDA device")


def _normalize_action_unclipped(action: Any, minimum: Any, maximum: Any) -> Any:
    import torch

    span = torch.clamp(maximum - minimum, min=1.0e-6)
    return 2.0 * (action - minimum) / span - 1.0


def _nearest_reference_rows(base_action: Any, reference_action: Any, minimum: Any, maximum: Any) -> Any:
    """Match successful targets while excluding the three thumb coordinates."""
    import torch

    if base_action.ndim != 2 or base_action.shape[1:] != (19,):
        raise ValueError(f"base_action must be [sample,19], got {tuple(base_action.shape)}")
    if reference_action.ndim != 2 or reference_action.shape[1:] != (19,) or reference_action.shape[0] < 1:
        raise ValueError(f"reference_action must be non-empty [sample,19], got {tuple(reference_action.shape)}")
    normalized_base = _normalize_action_unclipped(base_action, minimum, maximum).clamp(-1.0, 1.0)
    normalized_reference = _normalize_action_unclipped(reference_action, minimum, maximum).clamp(-1.0, 1.0)
    thumb_action_indices = {9 + index for index in _THUMB_HAND_INDICES}
    match_indices = [index for index in range(19) if index not in thumb_action_indices]
    distance = torch.cdist(normalized_base[:, match_indices], normalized_reference[:, match_indices])
    return distance.argmin(dim=1)


def _quantiles(value: Any) -> dict[str, float]:
    import torch

    return {
        "mean": float(value.mean()),
        "p50": float(torch.quantile(value, 0.50)),
        "p95": float(torch.quantile(value, 0.95)),
        "max": float(value.max()),
    }


def _target_comparison(
    base_action: Any,
    candidate_action: Any,
    executed_hand: Any,
    reference_action: Any,
    minimum: Any,
    maximum: Any,
    *,
    hand_scale_normalized: Any,
) -> dict[str, Any]:
    """Summarize residual reachability and remaining error for paired targets."""
    import torch

    sample_count = int(base_action.shape[0])
    expected = (sample_count, 19)
    if base_action.shape != expected or candidate_action.shape != expected or reference_action.shape != expected:
        raise ValueError("base, candidate, and reference actions must share shape [sample,19]")
    if executed_hand.shape != (sample_count, 10):
        raise ValueError("executed_hand must have shape [sample,10]")
    hand_minimum = minimum[_HAND_ACTION_SLICE]
    hand_maximum = maximum[_HAND_ACTION_SLICE]
    hand_span = torch.clamp(hand_maximum - hand_minimum, min=1.0e-6)
    hand_scale = torch.as_tensor(hand_scale_normalized, dtype=hand_span.dtype, device=hand_span.device)
    if hand_scale.ndim == 0:
        hand_scale = hand_scale.expand(10)
    if hand_scale.shape != (10,):
        raise ValueError(f"hand_scale_normalized must be scalar or shape (10,), got {tuple(hand_scale.shape)}")

    def normalize_hand(value: Any) -> Any:
        return _normalize_action_unclipped(value, hand_minimum, hand_maximum).clamp(-1.0, 1.0)

    base_hand = base_action[:, _HAND_ACTION_SLICE]
    candidate_hand = candidate_action[:, _HAND_ACTION_SLICE]
    reference_hand = reference_action[:, _HAND_ACTION_SLICE]
    normalized_base = normalize_hand(base_hand)
    normalized_candidate = normalize_hand(candidate_hand)
    normalized_executed = normalize_hand(executed_hand)
    normalized_reference = normalize_hand(reference_hand)
    required = normalized_reference - normalized_base
    candidate_delta = normalized_candidate - normalized_base
    candidate_remaining = normalized_reference - normalized_candidate
    executed_remaining = normalized_reference - normalized_executed
    capacity_physical = 0.5 * hand_span * hand_scale
    coverage = required.abs() <= hand_scale + 1.0e-6

    per_joint: dict[str, Any] = {}
    for joint_index, joint_name in enumerate(HAND_JOINT_NAMES):
        required_joint = required[:, joint_index]
        candidate_delta_joint = candidate_delta[:, joint_index]
        per_joint[joint_name] = {
            "hand_index": joint_index,
            "is_thumb": joint_index in _THUMB_HAND_INDICES,
            "physical_capacity_rad": float(capacity_physical[joint_index]),
            "physical_capacity_deg": math.degrees(float(capacity_physical[joint_index])),
            "required_normalized_signed_mean": float(required_joint.mean()),
            "required_normalized_signed_p05": float(torch.quantile(required_joint, 0.05)),
            "required_normalized_signed_p50": float(torch.quantile(required_joint, 0.50)),
            "required_normalized_signed_p95": float(torch.quantile(required_joint, 0.95)),
            "required_normalized_abs": _quantiles(required_joint.abs()),
            "required_physical_signed_mean_rad": float((0.5 * hand_span[joint_index] * required_joint).mean()),
            "coverage_fraction_at_configured_scale": float(coverage[:, joint_index].float().mean()),
            "candidate_delta_normalized_signed_mean": float(candidate_delta_joint.mean()),
            "candidate_remaining_normalized_abs": _quantiles(candidate_remaining[:, joint_index].abs()),
            "executed_remaining_normalized_abs": _quantiles(executed_remaining[:, joint_index].abs()),
            "candidate_execution_gap_rad": _quantiles(
                (candidate_hand[:, joint_index] - executed_hand[:, joint_index]).abs()
            ),
            "residual_direction_alignment_fraction": float(
                ((required_joint * candidate_delta_joint) > 0.0).float().mean()
            ),
        }

    thumb_rank = sorted(
        (
            {
                "joint_name": name,
                "hand_index": per_joint[name]["hand_index"],
                "required_normalized_abs_p50": per_joint[name]["required_normalized_abs"]["p50"],
                "required_normalized_abs_p95": per_joint[name]["required_normalized_abs"]["p95"],
                "coverage_fraction": per_joint[name]["coverage_fraction_at_configured_scale"],
                "required_normalized_signed_mean": per_joint[name]["required_normalized_signed_mean"],
                "required_normalized_signed_p50": per_joint[name]["required_normalized_signed_p50"],
                "candidate_delta_normalized_signed_mean": per_joint[name]["candidate_delta_normalized_signed_mean"],
            }
            for name in _THUMB_JOINT_NAMES
        ),
        key=lambda item: (
            item["required_normalized_abs_p50"] - float(hand_scale[item["hand_index"]]),
            item["required_normalized_abs_p95"],
        ),
        reverse=True,
    )
    return {
        "sample_count": sample_count,
        "hand_scale_normalized": [float(value) for value in hand_scale],
        "hand_mae_normalized": {
            "base_to_reference": float((normalized_base - normalized_reference).abs().mean()),
            "candidate_to_reference": float((normalized_candidate - normalized_reference).abs().mean()),
            "executed_to_reference": float((normalized_executed - normalized_reference).abs().mean()),
        },
        "thumb_mae_normalized": {
            "base_to_reference": float(
                (normalized_base[:, _THUMB_HAND_INDICES] - normalized_reference[:, _THUMB_HAND_INDICES]).abs().mean()
            ),
            "candidate_to_reference": float(
                (normalized_candidate[:, _THUMB_HAND_INDICES] - normalized_reference[:, _THUMB_HAND_INDICES])
                .abs()
                .mean()
            ),
            "executed_to_reference": float(
                (normalized_executed[:, _THUMB_HAND_INDICES] - normalized_reference[:, _THUMB_HAND_INDICES])
                .abs()
                .mean()
            ),
        },
        "all_thumb_coordinates_covered_fraction": float(coverage[:, _THUMB_HAND_INDICES].all(dim=1).float().mean()),
        "per_joint": per_joint,
        "thumb_priority": thumb_rank,
    }


def _executed_hand_target(env: Any) -> Any:
    import warp as wp  # noqa: PLC0415

    return wp.to_torch(env.control.joint_target_q)[env._hand_q_indices_torch].detach().float().clone()


def _load_residual_actor(checkpoint_path: Path, frozen_dp_sha256: str, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    from teleop_stack.policies import (  # noqa: PLC0415
        GROOT_RESIDUAL_PPO_CHECKPOINT_FORMAT,
        GrootResidualActorCritic,
        GrootResidualActorCriticConfig,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("format") != GROOT_RESIDUAL_PPO_CHECKPOINT_FORMAT:
        raise ValueError("residual_checkpoint is not a Groot residual PPO checkpoint")
    if checkpoint.get("frozen_dp_sha256") != frozen_dp_sha256:
        raise ValueError("Residual checkpoint was trained with a different frozen DP checkpoint")
    actor = GrootResidualActorCritic(GrootResidualActorCriticConfig(**checkpoint["policy_config"])).to(device)
    actor.load_state_dict(checkpoint["actor_critic"])
    actor.eval().requires_grad_(False)
    return actor, checkpoint


def _reference_replay(env: Any, actions: Any) -> dict[str, Any]:
    import torch

    from tools.train_newton_groot_residual_ppo import _finger_contact_topology  # noqa: PLC0415

    env.reset()
    executed = []
    thumb_steps = []
    non_thumb_steps = []
    opposed_steps = []
    confirmed_steps = []
    with torch.inference_mode():
        for step in range(actions.shape[0]):
            action = actions[step].expand(env.num_envs, -1)
            _, _, _, _, info = env.step(action)
            thumb, non_thumb, opposed = _finger_contact_topology(info)
            executed.append(_executed_hand_target(env.unwrapped).mean(dim=0))
            thumb_steps.append(thumb.any())
            non_thumb_steps.append(non_thumb.any())
            opposed_steps.append(opposed.any())
            confirmed_steps.append(info["grasp_confirmed"].bool().any())
    return {
        "action": actions,
        "executed_hand": torch.stack(executed),
        "thumb": torch.stack(thumb_steps),
        "non_thumb": torch.stack(non_thumb_steps),
        "opposed": torch.stack(opposed_steps),
        "confirmed": torch.stack(confirmed_steps),
    }


def _policy_rollout(
    env: Any,
    frozen_dp: Any,
    scheduler: Any,
    actor: Any,
    action_min: Any,
    action_max: Any,
    state_min: Any,
    state_max: Any,
    *,
    max_steps: int,
    inference_steps: int,
    use_bfloat16: bool,
    seed: int,
    action_scales: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from teleop_stack.policies import compose_residual_action  # noqa: PLC0415
    from tools.train_newton_groot_residual_ppo import (  # noqa: PLC0415
        _BASE_ACTION_HORIZON,
        _R_WORLD_FROM_ACTION,
        _finger_contact_topology,
        _PerLaneActionChunkCache,
        _prepare_policy_step,
        _privileged_task_state,
    )

    observation, current_info = env.reset()
    cache = _PerLaneActionChunkCache(env.num_envs, frozen_dp.config.action_dim, action_min.device)
    generator = torch.Generator(device=action_min.device).manual_seed(seed + 1)
    training_lanes = torch.ones(env.num_envs, dtype=torch.bool, device=action_min.device)
    hand_residual_scale = torch.as_tensor(
        action_scales["effective_hand_residual_scale_normalized"],
        dtype=action_min.dtype,
        device=action_min.device,
    )
    samples: dict[str, list[Any]] = {
        name: []
        for name in (
            "base",
            "candidate",
            "current_hand",
            "executed_hand",
            "raw_latent",
            "active",
            "thumb",
            "non_thumb",
            "opposed",
            "row",
            "step",
        )
    }
    with torch.inference_mode():
        for step in range(max_steps):
            prepared = _prepare_policy_step(
                observation,
                frozen_dp,
                scheduler,
                cache,
                action_min,
                action_max,
                state_min,
                state_max,
                inference_steps=inference_steps,
                use_bfloat16=use_bfloat16,
                generator=generator,
            )
            privileged = _privileged_task_state(current_info, env.unwrapped.config)
            raw_latent, _, _, _ = actor.act(prepared.policy_input, privileged, deterministic=True)
            current_hand = observation["observation.state"][:, -1, 16:26].detach().float().clone()
            candidate = compose_residual_action(
                prepared.base_action,
                raw_latent,
                action_min,
                action_max,
                position_scale_m=action_scales["position_residual_scale_m"],
                vertical_position_scale_m=action_scales["vertical_residual_scale_m"],
                rotation_scale_rad=math.radians(action_scales["rotation_residual_scale_deg"]),
                hand_scale_normalized=hand_residual_scale,
                world_from_action_rotation=_R_WORLD_FROM_ACTION,
            )
            next_observation, _, terminated, truncated, info = env.step(candidate)
            thumb, non_thumb, opposed = _finger_contact_topology(info)
            samples["base"].append(prepared.base_action.clone())
            samples["candidate"].append(candidate.clone())
            samples["current_hand"].append(current_hand)
            samples["executed_hand"].append(_executed_hand_target(env.unwrapped))
            samples["raw_latent"].append(raw_latent.clone())
            samples["active"].append(training_lanes)
            samples["thumb"].append(thumb.clone())
            samples["non_thumb"].append(non_thumb.clone())
            samples["opposed"].append(opposed.clone())
            samples["row"].append(prepared.row_index.clone())
            samples["step"].append(torch.full_like(prepared.row_index, step))
            cache.advance(training_lanes, validate=False)
            done = terminated | truncated
            cache.invalidate(done)
            if bool(done.any()):
                observation, current_info = env.reset(world_mask=done)
            else:
                observation, current_info = next_observation, info
    if not samples["base"]:
        raise RuntimeError("Policy rollout produced no samples")
    output = {name: torch.stack(value) for name, value in samples.items()}
    if output["row"].min() < 0 or output["row"].max() >= _BASE_ACTION_HORIZON:
        raise RuntimeError("Eligible rollout samples contain an invalid cached DP row")
    return output


def _summarize_condition(
    rollout: dict[str, Any],
    mask: Any,
    reference_action: Any,
    action_min: Any,
    action_max: Any,
    *,
    hand_scale_normalized: Any,
    hand_max_joint_step_rad: float,
) -> dict[str, Any]:
    import torch

    selected = mask & rollout["active"]
    sample_count = int(selected.sum())
    if sample_count == 0:
        return {"sample_count": 0}
    base = rollout["base"][selected]
    candidate = rollout["candidate"][selected]
    current_hand = rollout["current_hand"][selected]
    executed = rollout["executed_hand"][selected]
    match = _nearest_reference_rows(base, reference_action, action_min, action_max)
    matched_reference = reference_action[match]
    normalized_base = _normalize_action_unclipped(base, action_min, action_max).clamp(-1.0, 1.0)
    normalized_reference = _normalize_action_unclipped(matched_reference, action_min, action_max).clamp(-1.0, 1.0)
    thumb_action_indices = {9 + index for index in _THUMB_HAND_INDICES}
    match_indices = [index for index in range(19) if index not in thumb_action_indices]
    match_rms = (
        (normalized_base[:, match_indices] - normalized_reference[:, match_indices]).square().mean(dim=-1).sqrt()
    )
    summary = _target_comparison(
        base,
        candidate,
        executed,
        matched_reference,
        action_min,
        action_max,
        hand_scale_normalized=hand_scale_normalized,
    )
    summary["matched_reference_row"] = {
        "min": int(match.min()),
        "max": int(match.max()),
        "unique_count": int(match.unique().numel()),
    }
    summary["matched_reference_rms_normalized_excluding_thumb"] = _quantiles(match_rms)
    raw_hand = torch.tanh(rollout["raw_latent"][selected][:, 6:16])
    summary["learned_hand_residual_tanh_mean"] = {
        name: float(raw_hand[:, index].mean()) for index, name in enumerate(HAND_JOINT_NAMES)
    }
    for index, name in enumerate(HAND_JOINT_NAMES):
        raw_joint = raw_hand[:, index]
        requested_step = (candidate[:, 9 + index] - current_hand[:, index]).abs()
        summary["per_joint"][name]["learned_residual_tanh"] = {
            "signed_mean": float(raw_joint.mean()),
            "signed_p50": float(torch.quantile(raw_joint, 0.50)),
            "abs_p50": float(torch.quantile(raw_joint.abs(), 0.50)),
            "abs_p95": float(torch.quantile(raw_joint.abs(), 0.95)),
            "saturation_fraction": float((raw_joint.abs() >= 0.95).float().mean()),
        }
        summary["per_joint"][name]["dynamic_rate_limit_fraction"] = float(
            (requested_step > hand_max_joint_step_rad + 1.0e-6).float().mean()
        )
        summary["per_joint"][name]["requested_step_rad"] = _quantiles(requested_step)
    return summary


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from teleop_stack.envs import (  # noqa: PLC0415
        GrootDiffusionPolicyEnv,
        GrootNewtonEnv,
        GrootNewtonEnvConfig,
    )
    from tools.probe_newton_l10_finger_root_load_replay import _load_episode_actions  # noqa: PLC0415
    from tools.train_newton_groot_residual_ppo import _file_sha256, _load_frozen_dp  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("The hand-target diagnostic requires CUDA")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    frozen_dp, dp_config, scheduler, stats, dp_sha256 = _load_frozen_dp(args.checkpoint, device)
    actor, residual_checkpoint = _load_residual_actor(args.residual_checkpoint, dp_sha256, device)
    train_args = residual_checkpoint.get("train_args", {})
    action_scales = {
        name: float(train_args[name])
        for name in (
            "position_residual_scale_m",
            "vertical_residual_scale_m",
            "rotation_residual_scale_deg",
            "hand_residual_scale_normalized",
        )
    }
    hand_scale_contract = residual_checkpoint.get("hand_residual_scale", {})
    effective_hand_scale = hand_scale_contract.get("effective_scale_normalized")
    if effective_hand_scale is None:
        effective_hand_scale = [action_scales["hand_residual_scale_normalized"]] * 10
    if len(effective_hand_scale) != 10:
        raise ValueError("Residual checkpoint effective hand residual scale must have length 10")
    effective_hand_scale = torch.as_tensor(effective_hand_scale, dtype=torch.float32, device=device)
    action_scales["effective_hand_residual_scale_normalized"] = [float(value) for value in effective_hand_scale]
    env_payload = residual_checkpoint.get("env_config")
    if not isinstance(env_payload, dict):
        raise ValueError("Residual checkpoint does not contain env_config")
    env_config = replace(
        GrootNewtonEnvConfig(**env_payload),
        num_envs=args.num_envs,
        device=args.device,
        capture_graph=args.capture_graph,
        triangle_pairs_per_env=args.triangle_pairs_per_env,
    )
    action_min = torch.as_tensor(stats["action_min"], dtype=torch.float32, device=device)
    action_max = torch.as_tensor(stats["action_max"], dtype=torch.float32, device=device)
    state_min = torch.as_tensor(stats["state_min"], dtype=torch.float32, device=device)
    state_max = torch.as_tensor(stats["state_max"], dtype=torch.float32, device=device)
    reference_actions_cpu = _load_episode_actions(args.reference_dataset, args.episode_index)
    reference_actions = reference_actions_cpu.to(device=device)
    base_env = GrootNewtonEnv(env_config)
    env = GrootDiffusionPolicyEnv(base_env, obs_horizon=dp_config.obs_horizon, action_horizon=1)
    try:
        reference = _reference_replay(env, reference_actions)
        strict_reference_mask = reference["opposed"]
        if not bool(strict_reference_mask.any()):
            raise RuntimeError("Reference dataset replay never reached a live opposed grasp")
        strict_reference_actions = reference["action"][strict_reference_mask]
        rollout = _policy_rollout(
            env,
            frozen_dp,
            scheduler,
            actor,
            action_min,
            action_max,
            state_min,
            state_max,
            max_steps=args.max_steps,
            inference_steps=args.inference_steps,
            use_bfloat16=args.bfloat16,
            seed=args.seed,
            action_scales=action_scales,
        )
        active = rollout["active"]
        conditions = {
            "all_active": active,
            "any_contact": rollout["thumb"] | rollout["non_thumb"],
            "thumb_contact": rollout["thumb"],
            "non_thumb_contact": rollout["non_thumb"],
            "non_thumb_without_thumb": rollout["non_thumb"] & ~rollout["thumb"],
            "thumb_without_non_thumb": rollout["thumb"] & ~rollout["non_thumb"],
            "opposed_grasp": rollout["opposed"],
        }
        condition_summary = {
            name: _summarize_condition(
                rollout,
                mask,
                strict_reference_actions,
                action_min,
                action_max,
                hand_scale_normalized=effective_hand_scale,
                hand_max_joint_step_rad=env_config.hand_max_joint_step_rad,
            )
            for name, mask in conditions.items()
        }
        recommendation_source = next(
            name
            for name in ("non_thumb_without_thumb", "non_thumb_contact", "any_contact", "all_active")
            if condition_summary[name]["sample_count"] > 0
        )
        priority = condition_summary[recommendation_source]["thumb_priority"]
        recommended = priority[0]
        configured_scale = float(effective_hand_scale[recommended["hand_index"]])
        scale_covers_p50 = recommended["required_normalized_abs_p50"] <= configured_scale + 1.0e-6
        summary = {
            "schema_version": _SUMMARY_SCHEMA_VERSION,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _file_sha256(args.checkpoint),
            "residual_checkpoint": str(args.residual_checkpoint.resolve()),
            "residual_update": int(residual_checkpoint.get("update", -1)),
            "reference_dataset": str(args.reference_dataset.resolve()),
            "episode_index": args.episode_index,
            "num_envs": args.num_envs,
            "triangle_pairs_per_env": args.triangle_pairs_per_env,
            "policy_steps": int(rollout["base"].shape[0]),
            "action_scales": action_scales,
            "hand_joint_names": HAND_JOINT_NAMES,
            "thumb_hand_indices": _THUMB_HAND_INDICES,
            "reference_replay": {
                "action_steps": int(reference_actions.shape[0]),
                "thumb_contact_step_count": int(reference["thumb"].sum()),
                "non_thumb_contact_step_count": int(reference["non_thumb"].sum()),
                "live_opposed_step_count": int(reference["opposed"].sum()),
                "confirmed_grasp_step_count": int(reference["confirmed"].sum()),
                "strict_reference_action_count": int(strict_reference_actions.shape[0]),
                "command_execution_gap_rad": {
                    name: _quantiles((reference["action"][:, 9 + index] - reference["executed_hand"][:, index]).abs())
                    for index, name in enumerate(HAND_JOINT_NAMES)
                },
            },
            "policy_contact_counts": {
                name: int((mask & active).sum()) for name, mask in conditions.items() if name != "all_active"
            },
            "condition_comparison": condition_summary,
            "diagnosis": {
                "matching_method": (
                    "nearest live-opposed reference action using normalized EEF, rotation, and non-thumb targets"
                ),
                "source_condition": recommendation_source,
                "highest_priority_thumb_joint": recommended,
                "configured_scale_covers_priority_joint_p50": scale_covers_p50,
                "configured_priority_joint_scale_normalized": configured_scale,
                "configured_scale_covers_all_thumb_coordinates_fraction": condition_summary[recommendation_source][
                    "all_thumb_coordinates_covered_fraction"
                ],
            },
        }
        finite = all(
            torch.isfinite(value).all()
            for name, value in rollout.items()
            if name in {"base", "candidate", "current_hand", "executed_hand", "raw_latent"}
        )
        summary["passed"] = bool(finite and reference["opposed"].any())
        return summary
    finally:
        env.close()


def main() -> int:
    args = create_parser().parse_args()
    _validate_args(args)
    summary = _run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    diagnosis = summary["diagnosis"]
    priority = diagnosis["highest_priority_thumb_joint"]
    print(
        "hand-target diagnostic: "
        f"update={summary['residual_update']} source={diagnosis['source_condition']} "
        f"priority={priority['joint_name']} required_p50={priority['required_normalized_abs_p50']:.6f} "
        f"scale={diagnosis['configured_priority_joint_scale_normalized']:.6f} "
        f"covered={diagnosis['configured_scale_covers_priority_joint_p50']} passed={summary['passed']}"
    )
    print(f"summary: {output_path}")
    return 0 if summary["passed"] or not args.strict_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
