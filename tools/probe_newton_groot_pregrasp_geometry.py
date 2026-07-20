#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Probe frozen-DP pregrasp geometry signals in a headless Newton rollout."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_CHUNK_HORIZON = 8
_PREGRASP_DISTANCE_SCALE_M = 0.08
_ZERO_EPSILON = 1.0e-7
_USABLE_PROXIMITY_EPSILON = 1.0e-4
_SCHEMA_VERSION = "newton.groot_pregrasp_geometry_probe.v1"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Frozen Diffusion Policy checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/groot_pregrasp_geometry_probe"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--inference-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--triangle-pairs-per-env", type=int, default=196_608)
    parser.add_argument("--capture-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-textures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hydroelastic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-gates", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.num_envs, args.max_steps, args.inference_steps, args.triangle_pairs_per_env) < 1:
        raise ValueError("num-envs, max-steps, inference-steps, and triangle-pairs-per-env must be positive")
    if not str(args.device).startswith("cuda"):
        raise ValueError("The pregrasp geometry probe requires a CUDA device")


def _distribution(value: Any, *, include_min: bool = False, include_max: bool = False) -> dict[str, float]:
    import torch

    flattened = value.float().reshape(-1)
    if flattened.numel() == 0:
        raise ValueError("Cannot summarize an empty tensor")
    if not bool(torch.isfinite(flattened).all()):
        raise ValueError("Cannot summarize non-finite values")
    summary = {
        "mean": float(flattened.mean()),
        "p50": float(torch.quantile(flattened, 0.50)),
        "p95": float(torch.quantile(flattened, 0.95)),
    }
    if include_min:
        summary["min"] = float(flattened.min())
    if include_max:
        summary["max"] = float(flattened.max())
    return summary


def _summarize_geometry_samples(
    finger_surface_gap: Any,
    opposed_pregrasp_score: Any,
    *,
    distance_scale_m: float = _PREGRASP_DISTANCE_SCALE_M,
    zero_epsilon: float = _ZERO_EPSILON,
    usable_proximity_epsilon: float = _USABLE_PROXIMITY_EPSILON,
) -> dict[str, Any]:
    """Summarize gap, proximity, and zero-score attribution tensors."""
    import torch

    if finger_surface_gap.ndim < 2 or finger_surface_gap.shape[-1] != len(_FINGER_NAMES):
        raise ValueError(
            f"finger_surface_gap must end in {len(_FINGER_NAMES)} fingers, got {tuple(finger_surface_gap.shape)}"
        )
    expected_score_shape = finger_surface_gap.shape[:-1]
    if opposed_pregrasp_score.shape != expected_score_shape:
        raise ValueError(
            "opposed_pregrasp_score must match the gap sample dimensions, "
            f"got {tuple(opposed_pregrasp_score.shape)} versus {tuple(expected_score_shape)}"
        )
    if not math.isfinite(distance_scale_m) or distance_scale_m <= 0.0:
        raise ValueError("distance_scale_m must be finite and positive")
    if min(zero_epsilon, usable_proximity_epsilon) < 0.0:
        raise ValueError("zero and usable-proximity epsilons cannot be negative")

    gap = finger_surface_gap.float()
    score = opposed_pregrasp_score.float()
    if not bool(torch.isfinite(gap).all() and torch.isfinite(score).all()):
        raise ValueError("Pregrasp geometry samples must be finite")
    proximity = 1.0 - torch.tanh(gap / distance_scale_m)
    thumb_gap = gap[..., 0]
    best_non_thumb_gap = gap[..., 1:].amin(dim=-1)
    thumb_proximity = proximity[..., 0]
    best_non_thumb_proximity = proximity[..., 1:].amax(dim=-1)
    bilateral_proximity_cap = torch.minimum(thumb_proximity, best_non_thumb_proximity)
    bilateral_worst_gap = torch.maximum(thumb_gap, best_non_thumb_gap)
    score_zero = score <= zero_epsilon
    distance_saturated = bilateral_proximity_cap <= usable_proximity_epsilon
    downstream_gate_zero = score_zero & ~distance_saturated
    sample_count = score.numel()

    return {
        "sample_count": sample_count,
        "distance_scale_m": distance_scale_m,
        "per_finger_gap_m": {
            name: _distribution(gap[..., index], include_min=True) for index, name in enumerate(_FINGER_NAMES)
        },
        "bilateral_gap_m": {
            "thumb": _distribution(thumb_gap, include_min=True),
            "best_non_thumb": _distribution(best_non_thumb_gap, include_min=True),
            "worse_side": _distribution(bilateral_worst_gap, include_min=True),
        },
        "bilateral_proximity_cap": _distribution(bilateral_proximity_cap, include_min=True, include_max=True),
        "opposed_pregrasp_score": {
            **_distribution(score, include_max=True),
            "nonzero_fraction": float((score > zero_epsilon).float().mean()),
            "zero_fraction": float(score_zero.float().mean()),
        },
        "zero_score_attribution": {
            "zero_epsilon": zero_epsilon,
            "usable_proximity_epsilon": usable_proximity_epsilon,
            "distance_saturated_fraction": float(distance_saturated.float().mean()),
            "distance_saturated_and_score_zero_fraction": float((distance_saturated & score_zero).float().mean()),
            "usable_bilateral_proximity_but_score_zero_fraction": float(downstream_gate_zero.float().mean()),
            "usable_bilateral_proximity_fraction": float((~distance_saturated).float().mean()),
        },
    }


def _diagnose_zero_score(summary: dict[str, Any]) -> dict[str, Any]:
    score = summary["opposed_pregrasp_score"]
    attribution = summary["zero_score_attribution"]
    nonzero = float(score["nonzero_fraction"])
    distance_zero = float(attribution["distance_saturated_and_score_zero_fraction"])
    downstream_zero = float(attribution["usable_bilateral_proximity_but_score_zero_fraction"])
    if nonzero > 0.0:
        cause = "score_not_always_zero"
    elif downstream_zero > distance_zero:
        cause = "downstream_pair_gate_likely"
    elif distance_zero > 0.0:
        cause = "distance_proximity_saturation_likely"
    else:
        cause = "indeterminate"
    return {
        "classification": cause,
        "interpretation": (
            "A zero score with usable bilateral proximity excludes tanh distance saturation; "
            "the remaining multiplicative zero is most likely radial opposition (or an extreme z-pair gate)."
        ),
        "score_nonzero_fraction": nonzero,
        "distance_saturated_zero_fraction": distance_zero,
        "downstream_pair_gate_zero_fraction": downstream_zero,
    }


def _evaluate_gates(
    *,
    num_envs: int,
    control_steps: int,
    expected_control_steps: int,
    finite_action_samples: int,
    finite_state_samples: int,
    finite_gap_samples: int,
    finite_score_samples: int,
    nonnegative_gap_samples: int,
    bounded_score_samples: int,
    expected_lane_samples: int,
    cache_rows_in_range: bool,
    triangle_buffer_available: bool,
    rigid_overflow_frames: int,
    rigid_overflow_excess: int,
    triangle_overflow_frames: int,
    triangle_overflow_excess: int,
) -> dict[str, Any]:
    counts = {
        "finite_actions": finite_action_samples,
        "finite_states": finite_state_samples,
        "finite_gaps": finite_gap_samples,
        "finite_scores": finite_score_samples,
        "nonnegative_gaps": nonnegative_gap_samples,
        "bounded_scores": bounded_score_samples,
    }
    gates: dict[str, Any] = {
        "positive_lane_count": {"passed": num_envs > 0, "actual": num_envs},
        "exact_control_steps": {
            "passed": control_steps == expected_control_steps,
            "expected": expected_control_steps,
            "actual": control_steps,
        },
        "sample_integrity": {
            "passed": all(count == expected_lane_samples for count in counts.values()),
            "expected_per_signal": expected_lane_samples,
            **counts,
        },
        "cache_rows_in_range": {"passed": cache_rows_in_range},
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
        raise RuntimeError("The pregrasp geometry probe requires CUDA")
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
    env_config = GrootNewtonEnvConfig(
        num_envs=args.num_envs,
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
    cache = _PerLaneActionChunkCache(args.num_envs, dp_config.action_dim, device)
    executed = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    gap_samples = torch.empty(
        (args.max_steps, args.num_envs, len(_FINGER_NAMES)),
        dtype=torch.float32,
        device=device,
    )
    score_samples = torch.empty((args.max_steps, args.num_envs), dtype=torch.float32, device=device)
    row_counts = torch.zeros(_BASE_ACTION_HORIZON, dtype=torch.int64, device=device)
    counters = {
        name: torch.zeros((), dtype=torch.int64, device=device)
        for name in (
            "finite_actions",
            "finite_states",
            "finite_gaps",
            "finite_scores",
            "nonnegative_gaps",
            "bounded_scores",
        )
    }
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
        reset_diagnostics = reset_info["control_step_diagnostics"]
        rigid_capacity = int(reset_diagnostics["rigid_contact_capacity"])
        triangle_capacity = int(reset_diagnostics["triangle_pair_capacity"])
        triangle_available = bool(reset_diagnostics["triangle_pair_buffer_available"])
        with torch.inference_mode():
            for step in range(args.max_steps):
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
                    generator=generator,
                )
                action = prepared.base_action
                next_observation, _, _, _, info = env.step(action)
                cache.advance(executed, validate=True)
                control_steps += 1

                gap = info["finger_surface_gap"].float()
                score = info["opposed_pregrasp_score"].float()
                gap_samples[step].copy_(gap)
                score_samples[step].copy_(score)
                row_counts += torch.bincount(prepared.row_index, minlength=_BASE_ACTION_HORIZON)
                counters["finite_actions"] += torch.isfinite(action).all(dim=-1).sum(dtype=torch.int64)
                counters["finite_states"] += (
                    torch.isfinite(next_observation["observation.state"]).all(dim=(1, 2)).sum(dtype=torch.int64)
                )
                counters["finite_gaps"] += torch.isfinite(gap).all(dim=-1).sum(dtype=torch.int64)
                counters["finite_scores"] += torch.isfinite(score).sum(dtype=torch.int64)
                counters["nonnegative_gaps"] += (gap >= 0.0).all(dim=-1).sum(dtype=torch.int64)
                counters["bounded_scores"] += ((score >= 0.0) & (score <= 1.0)).sum(dtype=torch.int64)
                diagnostics = info["control_step_diagnostics"]
                _accumulate_buffer(buffer, diagnostics, prefix="rigid", diagnostic_prefix="rigid_contact")
                _accumulate_buffer(buffer, diagnostics, prefix="triangle", diagnostic_prefix="triangle_pair")
                observation = next_observation

        geometry = _summarize_geometry_samples(gap_samples, score_samples)
        diagnosis = _diagnose_zero_score(geometry)
        packed = torch.stack((*counters.values(), *buffer.values(), *row_counts)).to(torch.float64).cpu().tolist()
    finally:
        env.close()

    counter_names = tuple(counters)
    buffer_names = tuple(buffer)
    counter_values = {name: int(packed[index]) for index, name in enumerate(counter_names)}
    buffer_start = len(counter_names)
    buffer_values = {name: int(packed[buffer_start + index]) for index, name in enumerate(buffer_names)}
    row_start = buffer_start + len(buffer_names)
    row_values = [int(value) for value in packed[row_start:]]
    expected_lane_samples = args.num_envs * args.max_steps
    gates = _evaluate_gates(
        num_envs=args.num_envs,
        control_steps=control_steps,
        expected_control_steps=args.max_steps,
        finite_action_samples=counter_values["finite_actions"],
        finite_state_samples=counter_values["finite_states"],
        finite_gap_samples=counter_values["finite_gaps"],
        finite_score_samples=counter_values["finite_scores"],
        nonnegative_gap_samples=counter_values["nonnegative_gaps"],
        bounded_score_samples=counter_values["bounded_scores"],
        expected_lane_samples=expected_lane_samples,
        cache_rows_in_range=sum(row_values) == expected_lane_samples,
        triangle_buffer_available=triangle_available,
        rigid_overflow_frames=buffer_values["rigid_overflow_frames"],
        rigid_overflow_excess=buffer_values["rigid_overflow_excess"],
        triangle_overflow_frames=buffer_values["triangle_overflow_frames"],
        triangle_overflow_excess=buffer_values["triangle_overflow_excess"],
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "config": {
            "device": args.device,
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "inference_steps": args.inference_steps,
            "seed": args.seed,
            "chunk_horizon": _CHUNK_HORIZON,
            "bfloat16": args.bfloat16,
            "capture_graph": args.capture_graph,
            "triangle_pairs_per_env": args.triangle_pairs_per_env,
        },
        "control_steps": control_steps,
        "lane_samples": expected_lane_samples,
        "base_action_row_counts": {str(index): value for index, value in enumerate(row_values)},
        "geometry": geometry,
        "diagnosis": diagnosis,
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
        "pregrasp geometry probe: "
        f"samples={summary['lane_samples']} "
        f"score_nonzero={diagnosis['score_nonzero_fraction']:.6f} "
        f"classification={diagnosis['classification']} passed={summary['passed']}"
    )
    print(f"summary: {output_path}")
    return 0 if summary["passed"] or not args.strict_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
