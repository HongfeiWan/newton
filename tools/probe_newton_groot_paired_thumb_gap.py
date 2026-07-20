#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnose signed thumb residuals against paired frozen-DP baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

_THUMB_HAND_INDICES = (0, 1, 9)
_THUMB_ACTION_INDICES = tuple(9 + index for index in _THUMB_HAND_INDICES)
_THUMB_JOINT_NAMES = ("thumb_cmc_pitch", "thumb_cmc_yaw", "thumb_cmc_roll")
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_REFERENCE_THUMB_RESIDUAL = (0.1425, 0.3715, 0.4159)
_DEFAULT_THUMB_RESIDUALS = (_REFERENCE_THUMB_RESIDUAL, tuple(-value for value in _REFERENCE_THUMB_RESIDUAL))
_CHUNK_HORIZON = 8
_SCHEMA_VERSION = "newton.groot_paired_thumb_gap_probe.v1"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Frozen Diffusion Policy checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/groot_paired_thumb_gap_probe"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-pairs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--inference-steps", type=int, default=6)
    parser.add_argument(
        "--thumb-residual-normalized",
        dest="thumb_residuals_normalized",
        type=float,
        nargs=3,
        action="append",
        default=None,
        metavar=("PITCH", "YAW", "ROLL"),
        help=(
            "Normalized pitch/yaw/roll residual for one treatment group; repeat the option to compare groups. "
            "The default compares the positive reference prior and its exact opposite."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--triangle-pairs-per-env", type=int, default=196_608)
    parser.add_argument("--capture-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-textures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hydroelastic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-gates", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resolve_thumb_residuals(args: argparse.Namespace) -> tuple[tuple[float, float, float], ...]:
    raw = args.thumb_residuals_normalized
    if raw is None:
        return _DEFAULT_THUMB_RESIDUALS
    return tuple(tuple(float(value) for value in residual) for residual in raw)


def _validate_args(args: argparse.Namespace) -> None:
    positive = (args.num_pairs, args.max_steps, args.inference_steps, args.triangle_pairs_per_env)
    if min(positive) < 1:
        raise ValueError("num-pairs, max-steps, inference-steps, and triangle-pairs-per-env must be positive")
    if not str(args.device).startswith("cuda"):
        raise ValueError("The paired thumb-gap probe requires a CUDA device")
    residuals = _resolve_thumb_residuals(args)
    if len(residuals) < 2:
        raise ValueError("At least two signed thumb residual groups are required")
    if len(residuals) > args.num_pairs:
        raise ValueError("The number of thumb residual groups cannot exceed num-pairs")
    if len(set(residuals)) != len(residuals):
        raise ValueError("Thumb residual groups must be distinct")
    for residual in residuals:
        if len(residual) != len(_THUMB_HAND_INDICES):
            raise ValueError("Each thumb residual must contain pitch, yaw, and roll")
        if not all(math.isfinite(value) and abs(value) <= 2.0 for value in residual):
            raise ValueError("Thumb residuals must be finite and stay within [-2, 2]")
        if not any(value != 0.0 for value in residual):
            raise ValueError("The zero residual is implicit and cannot be a treatment group")


def _assign_pair_variants(
    residuals: tuple[tuple[float, float, float], ...],
    num_pairs: int,
) -> tuple[list[int], list[tuple[float, float, float]]]:
    if not residuals or num_pairs < len(residuals):
        raise ValueError("Every residual variant needs at least one pair")
    variant_indices = [pair % len(residuals) for pair in range(num_pairs)]
    return variant_indices, [residuals[index] for index in variant_indices]


def _build_paired_actions(
    pair_base_action: Any,
    pair_residuals: Any,
    action_min: Any,
    action_max: Any,
) -> tuple[Any, Any, Any]:
    """Duplicate each zero action and add its treatment's normalized residual."""
    import torch

    if pair_base_action.ndim != 2 or pair_base_action.shape[1:] != (19,):
        raise ValueError(f"pair_base_action must have shape [pair,19], got {tuple(pair_base_action.shape)}")
    if pair_residuals.shape != (pair_base_action.shape[0], 3):
        raise ValueError(
            f"pair_residuals must have shape ({pair_base_action.shape[0]},3), got {tuple(pair_residuals.shape)}"
        )
    if action_min.shape != (19,) or action_max.shape != (19,):
        raise ValueError("action_min/action_max must have shape [19]")

    paired_base = pair_base_action.repeat_interleave(2, dim=0)
    action = paired_base.clone()
    span = torch.clamp(action_max - action_min, min=1.0e-6)
    normalized_base = 2.0 * (pair_base_action - action_min) / span - 1.0
    candidate_thumb = normalized_base[:, _THUMB_ACTION_INDICES] + pair_residuals
    clamped_thumb = candidate_thumb.clamp(-1.0, 1.0)
    thumb_indices = list(_THUMB_ACTION_INDICES)
    treatment_thumb = 0.5 * (clamped_thumb + 1.0) * span[thumb_indices] + action_min[thumb_indices]
    action[1::2, _THUMB_ACTION_INDICES] = treatment_thumb

    absolute_clamp = torch.zeros((2 * pair_base_action.shape[0], 3), dtype=torch.bool, device=action.device)
    absolute_clamp[0::2] = (normalized_base[:, _THUMB_ACTION_INDICES] < -1.0) | (
        normalized_base[:, _THUMB_ACTION_INDICES] > 1.0
    )
    absolute_clamp[1::2] = (candidate_thumb < -1.0) | (candidate_thumb > 1.0)
    return action, paired_base, absolute_clamp


def _expected_row_counts(max_steps: int, num_pairs: int) -> list[int]:
    return [((max_steps + _CHUNK_HORIZON - 1 - row) // _CHUNK_HORIZON) * num_pairs for row in range(_CHUNK_HORIZON)]


def _distribution(values: Any) -> dict[str, float]:
    import torch

    flattened = values.float().reshape(-1)
    if flattened.numel() == 0 or not bool(torch.isfinite(flattened).all()):
        raise ValueError("Distribution samples must be non-empty and finite")
    return {
        "mean": float(flattened.mean()),
        "mean_abs": float(flattened.abs().mean()),
        "min": float(flattened.min()),
        "p50": float(torch.quantile(flattened, 0.50)),
        "p95": float(torch.quantile(flattened, 0.95)),
        "max": float(flattened.max()),
    }


def _joint_distributions(values: Any) -> dict[str, dict[str, float]]:
    if values.shape[-1:] != (3,):
        raise ValueError(f"Expected three thumb coordinates, got {tuple(values.shape)}")
    return {name: _distribution(values[..., index]) for index, name in enumerate(_THUMB_JOINT_NAMES)}


def _lane_summary(samples: dict[str, Any], lane: int) -> dict[str, Any]:
    import torch

    requested = samples["requested_thumb_action"][:, lane]
    requested_from_pre = samples["requested_minus_pre_state"][:, lane]
    executed = samples["post_minus_pre_state"][:, lane]
    tracking_error = samples["post_minus_requested"][:, lane]
    dynamic = samples["dynamic_rate_limit"][:, lane]
    absolute = samples["absolute_clamp"][:, lane]
    contact = samples["finger_contact_any_frame"][:, lane]
    opposed = samples["opposed_any_frame"][:, lane]
    streak = samples["opposed_streak"][:, lane]
    score = samples["pregrasp_score"][:, lane]
    return {
        "requested_thumb_action_rad": _joint_distributions(requested),
        "requested_minus_pre_state_rad": _joint_distributions(requested_from_pre),
        "executed_post_minus_pre_state_rad": _joint_distributions(executed),
        "post_state_minus_requested_rad": _joint_distributions(tracking_error),
        "dynamic_rate_limit": {
            name: {"count": int(dynamic[:, index].sum()), "fraction": float(dynamic[:, index].float().mean())}
            for index, name in enumerate(_THUMB_JOINT_NAMES)
        },
        "absolute_clamp": {
            name: {"count": int(absolute[:, index].sum()), "fraction": float(absolute[:, index].float().mean())}
            for index, name in enumerate(_THUMB_JOINT_NAMES)
        },
        "thumb_surface_gap_m": _distribution(samples["thumb_surface_gap"][:, lane]),
        "best_non_thumb_surface_gap_m": _distribution(samples["best_non_thumb_gap"][:, lane]),
        "opposed_pregrasp_score": _distribution(score),
        "finger_contact_any_frame": {
            name: {"steps": int(contact[:, index].sum()), "ever": bool(contact[:, index].any())}
            for index, name in enumerate(_FINGER_NAMES)
        },
        "opposed": {
            "any_frame_steps": int(opposed.sum()),
            "ever": bool(opposed.any()),
            "max_consecutive_physics_frames": int(streak.max()),
            "mean_consecutive_physics_frames": float(streak.float().mean()),
        },
        "finite": bool(
            torch.isfinite(requested).all()
            and torch.isfinite(executed).all()
            and torch.isfinite(samples["thumb_surface_gap"][:, lane]).all()
            and torch.isfinite(samples["best_non_thumb_gap"][:, lane]).all()
            and torch.isfinite(score).all()
        ),
    }


def _paired_metric_samples(samples: dict[str, Any]) -> dict[str, Any]:
    """Return time/pair treatment-minus-zero tensors for diagnostic ranking."""
    return {
        "requested_thumb_action_rad": samples["requested_thumb_action"][:, 1::2]
        - samples["requested_thumb_action"][:, 0::2],
        "post_thumb_state_rad": samples["post_thumb_state"][:, 1::2] - samples["post_thumb_state"][:, 0::2],
        "executed_step_delta_rad": samples["post_minus_pre_state"][:, 1::2] - samples["post_minus_pre_state"][:, 0::2],
        "dynamic_rate_limit_fraction": samples["dynamic_rate_limit"][:, 1::2].float()
        - samples["dynamic_rate_limit"][:, 0::2].float(),
        "zero_dynamic_rate_limit": samples["dynamic_rate_limit"][:, 0::2].float(),
        "treatment_dynamic_rate_limit": samples["dynamic_rate_limit"][:, 1::2].float(),
        "absolute_clamp_fraction": samples["absolute_clamp"][:, 1::2].float()
        - samples["absolute_clamp"][:, 0::2].float(),
        "zero_absolute_clamp": samples["absolute_clamp"][:, 0::2].float(),
        "treatment_absolute_clamp": samples["absolute_clamp"][:, 1::2].float(),
        "thumb_surface_gap_m": samples["thumb_surface_gap"][:, 1::2] - samples["thumb_surface_gap"][:, 0::2],
        "best_non_thumb_surface_gap_m": samples["best_non_thumb_gap"][:, 1::2] - samples["best_non_thumb_gap"][:, 0::2],
        "opposed_pregrasp_score": samples["pregrasp_score"][:, 1::2] - samples["pregrasp_score"][:, 0::2],
        "finger_contact_any_frame_fraction": samples["finger_contact_any_frame"][:, 1::2].float()
        - samples["finger_contact_any_frame"][:, 0::2].float(),
        "opposed_any_frame_fraction": samples["opposed_any_frame"][:, 1::2].float()
        - samples["opposed_any_frame"][:, 0::2].float(),
        "opposed_streak_frames": samples["opposed_streak"][:, 1::2].float()
        - samples["opposed_streak"][:, 0::2].float(),
    }


def _summarize_variant_deltas(
    paired: dict[str, Any],
    pair_variant_indices: Any,
    variant_count: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in range(variant_count):
        selected = pair_variant_indices == variant
        if not bool(selected.any()):
            raise ValueError(f"Residual variant {variant} has no paired samples")
        gap = paired["thumb_surface_gap_m"][:, selected]
        score = paired["opposed_pregrasp_score"][:, selected]
        per_pair_gap_mean = gap.mean(dim=0)
        per_pair_score_mean = score.mean(dim=0)
        output.append(
            {
                "variant_index": variant,
                "pair_count": int(selected.sum()),
                "treatment_minus_zero": {
                    "requested_thumb_action_rad": _joint_distributions(
                        paired["requested_thumb_action_rad"][:, selected]
                    ),
                    "post_thumb_state_rad": _joint_distributions(paired["post_thumb_state_rad"][:, selected]),
                    "executed_step_delta_rad": _joint_distributions(paired["executed_step_delta_rad"][:, selected]),
                    "dynamic_rate_limit_fraction": _joint_distributions(
                        paired["dynamic_rate_limit_fraction"][:, selected]
                    ),
                    "zero_dynamic_rate_limit": _joint_distributions(paired["zero_dynamic_rate_limit"][:, selected]),
                    "treatment_dynamic_rate_limit": _joint_distributions(
                        paired["treatment_dynamic_rate_limit"][:, selected]
                    ),
                    "absolute_clamp_fraction": _joint_distributions(paired["absolute_clamp_fraction"][:, selected]),
                    "zero_absolute_clamp": _joint_distributions(paired["zero_absolute_clamp"][:, selected]),
                    "treatment_absolute_clamp": _joint_distributions(paired["treatment_absolute_clamp"][:, selected]),
                    "thumb_surface_gap_m": _distribution(gap),
                    "best_non_thumb_surface_gap_m": _distribution(paired["best_non_thumb_surface_gap_m"][:, selected]),
                    "opposed_pregrasp_score": _distribution(score),
                    "finger_contact_any_frame_fraction": {
                        name: _distribution(paired["finger_contact_any_frame_fraction"][:, selected, index])
                        for index, name in enumerate(_FINGER_NAMES)
                    },
                    "opposed_any_frame_fraction": _distribution(paired["opposed_any_frame_fraction"][:, selected]),
                    "opposed_streak_frames": _distribution(paired["opposed_streak_frames"][:, selected]),
                },
                "pair_outcomes": {
                    "thumb_gap_reduced": int((per_pair_gap_mean < 0.0).sum()),
                    "thumb_gap_tied": int((per_pair_gap_mean == 0.0).sum()),
                    "thumb_gap_increased": int((per_pair_gap_mean > 0.0).sum()),
                    "pregrasp_score_improved": int((per_pair_score_mean > 0.0).sum()),
                    "pregrasp_score_tied": int((per_pair_score_mean == 0.0).sum()),
                    "pregrasp_score_worsened": int((per_pair_score_mean < 0.0).sum()),
                },
            }
        )
    return output


def _summarize_pair_delta(paired: dict[str, Any], pair: int) -> dict[str, Any]:
    """Summarize one treatment-minus-zero pair across its control steps."""
    return {
        "requested_thumb_action_rad": _joint_distributions(paired["requested_thumb_action_rad"][:, pair]),
        "post_thumb_state_rad": _joint_distributions(paired["post_thumb_state_rad"][:, pair]),
        "executed_step_delta_rad": _joint_distributions(paired["executed_step_delta_rad"][:, pair]),
        "thumb_surface_gap_m": _distribution(paired["thumb_surface_gap_m"][:, pair]),
        "best_non_thumb_surface_gap_m": _distribution(paired["best_non_thumb_surface_gap_m"][:, pair]),
        "opposed_pregrasp_score": _distribution(paired["opposed_pregrasp_score"][:, pair]),
        "opposed_any_frame_fraction": _distribution(paired["opposed_any_frame_fraction"][:, pair]),
        "opposed_streak_frames": _distribution(paired["opposed_streak_frames"][:, pair]),
    }


def _diagnose_variants(variant_deltas: list[dict[str, Any]]) -> dict[str, Any]:
    effects = []
    for item in variant_deltas:
        delta = item["treatment_minus_zero"]
        state_delta = sum(delta["post_thumb_state_rad"][name]["mean_abs"] for name in _THUMB_JOINT_NAMES)
        requested_delta = sum(delta["requested_thumb_action_rad"][name]["mean_abs"] for name in _THUMB_JOINT_NAMES)
        rate_fraction = sum(delta["treatment_dynamic_rate_limit"][name]["mean"] for name in _THUMB_JOINT_NAMES) / len(
            _THUMB_JOINT_NAMES
        )
        clamp_fraction = sum(delta["treatment_absolute_clamp"][name]["mean"] for name in _THUMB_JOINT_NAMES) / len(
            _THUMB_JOINT_NAMES
        )
        gap_delta = delta["thumb_surface_gap_m"]["mean"]
        if clamp_fraction >= 0.5:
            classification = "absolute_bound_clamp_dominant"
        elif rate_fraction >= 0.5 and state_delta < 0.25 * max(requested_delta, 1.0e-9):
            classification = "dynamic_rate_limit_or_tracking_dominant"
        elif state_delta < 0.05 * max(requested_delta, 1.0e-9):
            classification = "requested_residual_barely_changes_realized_thumb_state"
        elif gap_delta < 0.0:
            classification = "residual_sign_reduces_thumb_gap"
        else:
            classification = "residual_sign_does_not_reduce_thumb_gap"
        effects.append(
            {
                "variant_index": item["variant_index"],
                "thumb_residual_normalized": item.get("thumb_residual_normalized"),
                "classification": classification,
                "mean_thumb_gap_delta_m": gap_delta,
                "mean_pregrasp_score_delta": delta["opposed_pregrasp_score"]["mean"],
                "requested_thumb_delta_l1_rad": requested_delta,
                "realized_thumb_state_delta_l1_rad": state_delta,
                "treatment_dynamic_rate_limit_fraction": rate_fraction,
                "treatment_absolute_clamp_fraction": clamp_fraction,
            }
        )
    best = min(effects, key=lambda item: (item["mean_thumb_gap_delta_m"], -item["mean_pregrasp_score_delta"]))
    return {
        "best_thumb_gap_variant_index": best["variant_index"],
        "best_thumb_gap_residual_normalized": best["thumb_residual_normalized"],
        "selection_rule": "minimum paired mean thumb-gap delta, then maximum paired mean pregrasp-score delta",
        "variant_effects": effects,
    }


def _evaluate_gates(
    *,
    num_pairs: int,
    num_envs: int,
    variant_pair_counts: list[int],
    control_steps: int,
    expected_steps: int,
    finite_counts: dict[str, int],
    expected_lane_samples: int,
    paired_deltas_finite: bool,
    initial_pair_state_max_abs_delta: float,
    shared_pair_base_max_abs_delta: float,
    row_counts: list[int],
    expected_row_counts: list[int],
    triangle_buffer_available: bool,
    rigid_overflow_frames: int,
    rigid_overflow_excess: int,
    triangle_overflow_frames: int,
    triangle_overflow_excess: int,
) -> dict[str, Any]:
    gates: dict[str, Any] = {
        "paired_lane_count": {
            "passed": num_pairs > 0 and num_envs == 2 * num_pairs,
            "num_pairs": num_pairs,
            "num_envs": num_envs,
        },
        "all_variants_have_pairs": {
            "passed": bool(variant_pair_counts) and all(count > 0 for count in variant_pair_counts),
            "pair_counts": variant_pair_counts,
        },
        "exact_control_steps": {
            "passed": control_steps == expected_steps,
            "expected": expected_steps,
            "actual": control_steps,
        },
        "finite_rollout_samples": {
            "passed": bool(finite_counts) and all(count == expected_lane_samples for count in finite_counts.values()),
            "expected_per_signal": expected_lane_samples,
            **finite_counts,
        },
        "finite_paired_deltas": {"passed": paired_deltas_finite},
        "same_initial_state_within_pair": {
            "passed": initial_pair_state_max_abs_delta <= 1.0e-6,
            "max_abs_delta": initial_pair_state_max_abs_delta,
        },
        "shared_base_within_pair": {
            "passed": shared_pair_base_max_abs_delta == 0.0,
            "max_abs_delta": shared_pair_base_max_abs_delta,
        },
        "chunk_rows_0_through_7": {
            "passed": row_counts == expected_row_counts,
            "actual": row_counts,
            "expected": expected_row_counts,
        },
        "triangle_buffer_available": {"passed": triangle_buffer_available},
        "rigid_contact_buffer_clean": {
            "passed": rigid_overflow_frames == 0 and rigid_overflow_excess == 0,
            "overflow_frames": rigid_overflow_frames,
            "overflow_excess": rigid_overflow_excess,
        },
        "triangle_pair_buffer_clean": {
            "passed": triangle_overflow_frames == 0 and triangle_overflow_excess == 0,
            "overflow_frames": triangle_overflow_frames,
            "overflow_excess": triangle_overflow_excess,
        },
    }
    gates["passed"] = all(bool(gate["passed"]) for gate in gates.values())
    return gates


def _accumulate_buffer(
    accumulator: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    prefix: str,
    diagnostic_prefix: str,
) -> None:
    import torch

    frame_max = diagnostics[f"{diagnostic_prefix}_frame_max"].to(torch.int64).reshape(())
    overflow_frames = diagnostics[f"{diagnostic_prefix}_overflow_frame_count"].to(torch.int64).reshape(())
    overflow_excess = diagnostics[f"{diagnostic_prefix}_overflow_excess_count"].to(torch.int64).reshape(())
    accumulator[f"{prefix}_max"] = torch.maximum(accumulator[f"{prefix}_max"], frame_max)
    accumulator[f"{prefix}_overflow_steps"] += (overflow_frames > 0).to(torch.int64)
    accumulator[f"{prefix}_overflow_frames"] += overflow_frames
    accumulator[f"{prefix}_overflow_excess"] += overflow_excess


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from teleop_stack.envs import GrootDiffusionPolicyEnv, GrootNewtonEnv, GrootNewtonEnvConfig  # noqa: PLC0415
    from tools.compare_newton_groot_dp_rollouts import _load_frozen_dp  # noqa: PLC0415
    from tools.train_newton_groot_residual_ppo import (  # noqa: PLC0415
        _BASE_ACTION_HORIZON,
        _PerLaneActionChunkCache,
        _prepare_policy_step,
        _validate_frozen_dp_training_contract,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("The paired thumb-gap probe requires CUDA")
    residuals = _resolve_thumb_residuals(args)
    pair_variant_indices_list, pair_residuals_list = _assign_pair_variants(residuals, args.num_pairs)
    num_envs = 2 * args.num_pairs
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    frozen_dp, dp_config, scheduler, stats, checkpoint_sha256 = _load_frozen_dp(args.checkpoint, device)
    _validate_frozen_dp_training_contract(dp_config)
    action_min = torch.as_tensor(stats["action_min"], dtype=torch.float32, device=device)
    action_max = torch.as_tensor(stats["action_max"], dtype=torch.float32, device=device)
    state_min = torch.as_tensor(stats["state_min"], dtype=torch.float32, device=device)
    state_max = torch.as_tensor(stats["state_max"], dtype=torch.float32, device=device)
    pair_residuals = torch.as_tensor(pair_residuals_list, dtype=torch.float32, device=device)
    pair_variant_indices = torch.as_tensor(pair_variant_indices_list, dtype=torch.long, device=device)
    zero_lane_indices = 2 * torch.arange(args.num_pairs, dtype=torch.long, device=device)
    executed_pairs = torch.ones(args.num_pairs, dtype=torch.bool, device=device)

    env_config = GrootNewtonEnvConfig(
        num_envs=num_envs,
        device=args.device,
        max_episode_steps=0,
        obs_mode="policy",
        control_mode="pd_eef_pose_abs",
        reward_mode="normalized_dense",
        terminate_on_success=False,
        terminate_on_fail=False,
        capture_graph=args.capture_graph,
        camera_textures=args.camera_textures,
        load_scene_visuals=args.scene_visuals,
        hydroelastic_contacts=args.hydroelastic,
        request_finger_root_load=True,
        triangle_pairs_per_env=args.triangle_pairs_per_env,
    )
    base_env = GrootNewtonEnv(env_config)
    env = GrootDiffusionPolicyEnv(base_env, obs_horizon=dp_config.obs_horizon, action_horizon=1)
    cache = _PerLaneActionChunkCache(args.num_pairs, dp_config.action_dim, device)
    samples = {
        "requested_thumb_action": torch.empty((args.max_steps, num_envs, 3), device=device),
        "requested_minus_pre_state": torch.empty((args.max_steps, num_envs, 3), device=device),
        "post_thumb_state": torch.empty((args.max_steps, num_envs, 3), device=device),
        "post_minus_pre_state": torch.empty((args.max_steps, num_envs, 3), device=device),
        "post_minus_requested": torch.empty((args.max_steps, num_envs, 3), device=device),
        "dynamic_rate_limit": torch.empty((args.max_steps, num_envs, 3), dtype=torch.bool, device=device),
        "absolute_clamp": torch.empty((args.max_steps, num_envs, 3), dtype=torch.bool, device=device),
        "thumb_surface_gap": torch.empty((args.max_steps, num_envs), device=device),
        "best_non_thumb_gap": torch.empty((args.max_steps, num_envs), device=device),
        "pregrasp_score": torch.empty((args.max_steps, num_envs), device=device),
        "finger_contact_any_frame": torch.empty(
            (args.max_steps, num_envs, len(_FINGER_NAMES)), dtype=torch.bool, device=device
        ),
        "opposed_any_frame": torch.empty((args.max_steps, num_envs), dtype=torch.bool, device=device),
        "opposed_streak": torch.empty((args.max_steps, num_envs), dtype=torch.int32, device=device),
    }
    row_counts = torch.zeros(_BASE_ACTION_HORIZON, dtype=torch.int64, device=device)
    finite_action_count = torch.zeros((), dtype=torch.int64, device=device)
    finite_state_count = torch.zeros((), dtype=torch.int64, device=device)
    shared_pair_base_max_abs_delta = torch.zeros((), dtype=torch.float32, device=device)
    buffer = {
        name: torch.zeros((), dtype=torch.int64, device=device)
        for name in (
            "rigid_max",
            "rigid_overflow_steps",
            "rigid_overflow_frames",
            "rigid_overflow_excess",
            "triangle_max",
            "triangle_overflow_steps",
            "triangle_overflow_frames",
            "triangle_overflow_excess",
        )
    }
    control_steps = 0
    try:
        observation, reset_info = env.reset(seed=args.seed)
        initial_state = observation["observation.state"][:, -1].float()
        initial_pair_state_max_abs_delta = (initial_state[1::2] - initial_state[0::2]).abs().max()
        reset_diagnostics = reset_info["control_step_diagnostics"]
        rigid_capacity = int(reset_diagnostics["rigid_contact_capacity"])
        triangle_capacity = int(reset_diagnostics["triangle_pair_capacity"])
        triangle_available = bool(reset_diagnostics["triangle_pair_buffer_available"])
        with torch.inference_mode():
            for step in range(args.max_steps):
                zero_observation = {key: value[zero_lane_indices] for key, value in observation.items()}
                prepared = _prepare_policy_step(
                    zero_observation,
                    frozen_dp,
                    scheduler,
                    cache,
                    action_min,
                    action_max,
                    state_min,
                    state_max,
                    inference_steps=args.inference_steps,
                    use_bfloat16=args.bfloat16,
                    generator=generator,
                )
                action, paired_base, absolute_clamp = _build_paired_actions(
                    prepared.base_action,
                    pair_residuals,
                    action_min,
                    action_max,
                )
                finite_action_count += torch.isfinite(action).all(dim=-1).sum(dtype=torch.int64)
                shared_pair_base_max_abs_delta = torch.maximum(
                    shared_pair_base_max_abs_delta,
                    (paired_base[1::2] - paired_base[0::2]).abs().max(),
                )
                for row in range(_BASE_ACTION_HORIZON):
                    row_counts[row] += (prepared.row_index == row).sum(dtype=torch.int64)

                pre_hand = observation["observation.state"][:, -1, 16:26].float()
                pre_thumb = pre_hand[:, _THUMB_HAND_INDICES]
                requested_thumb = action[:, _THUMB_ACTION_INDICES]
                dynamic_rate_limit = (
                    requested_thumb - pre_thumb
                ).abs() > base_env.config.hand_max_joint_step_rad + 1.0e-6
                samples["requested_thumb_action"][step].copy_(requested_thumb)
                samples["requested_minus_pre_state"][step].copy_(requested_thumb - pre_thumb)
                samples["dynamic_rate_limit"][step].copy_(dynamic_rate_limit)
                samples["absolute_clamp"][step].copy_(absolute_clamp)

                observation, _, _, _, info = env.step(action)
                cache.advance(executed_pairs, validate=True)
                control_steps += 1
                post_hand = observation["observation.state"][:, -1, 16:26].float()
                finite_state_count += (
                    torch.isfinite(observation["observation.state"]).all(dim=(1, 2)).sum(dtype=torch.int64)
                )
                post_thumb = post_hand[:, _THUMB_HAND_INDICES]
                gap = info["finger_surface_gap"].float()
                samples["post_thumb_state"][step].copy_(post_thumb)
                samples["post_minus_pre_state"][step].copy_(post_thumb - pre_thumb)
                samples["post_minus_requested"][step].copy_(post_thumb - requested_thumb)
                samples["thumb_surface_gap"][step].copy_(gap[:, 0])
                samples["best_non_thumb_gap"][step].copy_(gap[:, 1:].amin(dim=-1))
                samples["pregrasp_score"][step].copy_(info["opposed_pregrasp_score"].float())
                samples["finger_contact_any_frame"][step].copy_(
                    info["finger_contact_any_frame_this_control_step"].bool()
                )
                samples["opposed_any_frame"][step].copy_(info["opposed_grasp_any_frame_this_control_step"].bool())
                samples["opposed_streak"][step].copy_(
                    info["opposed_grasp_max_consecutive_physics_frames_this_control_step"].to(torch.int32)
                )
                diagnostics = info["control_step_diagnostics"]
                _accumulate_buffer(buffer, diagnostics, prefix="rigid", diagnostic_prefix="rigid_contact")
                _accumulate_buffer(buffer, diagnostics, prefix="triangle", diagnostic_prefix="triangle_pair")

        paired = _paired_metric_samples(samples)
        variant_deltas = _summarize_variant_deltas(paired, pair_variant_indices, len(residuals))
        for item, residual in zip(variant_deltas, residuals, strict=True):
            item["thumb_residual_normalized"] = dict(zip(_THUMB_JOINT_NAMES, residual, strict=True))
        lane_summaries = [_lane_summary(samples, lane) for lane in range(num_envs)]
        finite_counts = {
            "actions": int(finite_action_count),
            "states": int(finite_state_count),
            "requested_thumb_actions": int(torch.isfinite(samples["requested_thumb_action"]).all(dim=-1).sum()),
            "executed_state_deltas": int(torch.isfinite(samples["post_minus_pre_state"]).all(dim=-1).sum()),
            "thumb_gaps": int(torch.isfinite(samples["thumb_surface_gap"]).sum()),
            "best_non_thumb_gaps": int(torch.isfinite(samples["best_non_thumb_gap"]).sum()),
            "pregrasp_scores": int(torch.isfinite(samples["pregrasp_score"]).sum()),
        }
        paired_deltas_finite = all(bool(torch.isfinite(value).all()) for value in paired.values())
        row_count_values = [int(value) for value in row_counts.cpu().tolist()]
        buffer_values = {name: int(value) for name, value in buffer.items()}
        initial_pair_delta_value = float(initial_pair_state_max_abs_delta)
        shared_base_delta_value = float(shared_pair_base_max_abs_delta)
    finally:
        env.close()

    pair_results = []
    for pair, variant_index in enumerate(pair_variant_indices_list):
        pair_results.append(
            {
                "pair": pair,
                "variant_index": variant_index,
                "zero": lane_summaries[2 * pair],
                "treatment": lane_summaries[2 * pair + 1],
                "treatment_minus_zero": _summarize_pair_delta(paired, pair),
            }
        )
    variant_pair_counts = [pair_variant_indices_list.count(index) for index in range(len(residuals))]
    expected_lane_samples = args.max_steps * num_envs
    expected_rows = _expected_row_counts(args.max_steps, args.num_pairs)
    gates = _evaluate_gates(
        num_pairs=args.num_pairs,
        num_envs=num_envs,
        variant_pair_counts=variant_pair_counts,
        control_steps=control_steps,
        expected_steps=args.max_steps,
        finite_counts=finite_counts,
        expected_lane_samples=expected_lane_samples,
        paired_deltas_finite=paired_deltas_finite,
        initial_pair_state_max_abs_delta=initial_pair_delta_value,
        shared_pair_base_max_abs_delta=shared_base_delta_value,
        row_counts=row_count_values,
        expected_row_counts=expected_rows,
        triangle_buffer_available=triangle_available,
        rigid_overflow_frames=buffer_values["rigid_overflow_frames"],
        rigid_overflow_excess=buffer_values["rigid_overflow_excess"],
        triangle_overflow_frames=buffer_values["triangle_overflow_frames"],
        triangle_overflow_excess=buffer_values["triangle_overflow_excess"],
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "config": {
            "device": args.device,
            "num_pairs": args.num_pairs,
            "num_envs": num_envs,
            "max_steps": args.max_steps,
            "inference_steps": args.inference_steps,
            "seed": args.seed,
            "chunk_horizon": _CHUNK_HORIZON,
            "pair_lane_order": ["zero", "treatment"],
            "replan_observation_lane": "zero",
            "shared_frozen_dp_chunk_within_pair": True,
            "thumb_residuals_normalized": [
                dict(zip(_THUMB_JOINT_NAMES, residual, strict=True)) for residual in residuals
            ],
            "pair_variant_indices": pair_variant_indices_list,
            "hand_max_joint_step_rad": base_env.config.hand_max_joint_step_rad,
            "triangle_pairs_per_env": args.triangle_pairs_per_env,
        },
        "control_steps": control_steps,
        "base_action_row_counts": {str(index): count for index, count in enumerate(row_count_values)},
        "collision_buffers": {
            "rigid_contacts": {
                "capacity": rigid_capacity,
                "peak": buffer_values["rigid_max"],
                "overflow_steps": buffer_values["rigid_overflow_steps"],
                "overflow_frames": buffer_values["rigid_overflow_frames"],
                "overflow_excess": buffer_values["rigid_overflow_excess"],
            },
            "triangle_pairs": {
                "available": triangle_available,
                "capacity": triangle_capacity,
                "peak": buffer_values["triangle_max"],
                "overflow_steps": buffer_values["triangle_overflow_steps"],
                "overflow_frames": buffer_values["triangle_overflow_frames"],
                "overflow_excess": buffer_values["triangle_overflow_excess"],
            },
        },
        "variant_paired_deltas": variant_deltas,
        "diagnosis": _diagnose_variants(variant_deltas),
        "pairs": pair_results,
        "gates": gates,
        "passed": bool(gates["passed"]),
    }


def main() -> int:
    args = create_parser().parse_args()
    _validate_args(args)
    summary = _run_probe(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    diagnosis = summary["diagnosis"]
    print(
        f"paired thumb-gap probe: best_variant={diagnosis['best_thumb_gap_variant_index']} passed={summary['passed']}"
    )
    print(f"summary: {output_path}")
    return 0 if summary["passed"] or not args.strict_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
