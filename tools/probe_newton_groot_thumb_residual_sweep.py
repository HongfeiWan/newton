#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sweep fixed normalized thumb offsets around a frozen Groot DP baseline."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

_THUMB_HAND_INDICES = (0, 1, 9)
_THUMB_JOINT_NAMES = ("thumb_cmc_pitch", "thumb_cmc_yaw", "thumb_cmc_roll")
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_CHUNK_HORIZON = 8
_SCHEMA_VERSION = "newton.groot_thumb_residual_sweep.v1"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Frozen Diffusion Policy checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/groot_thumb_residual_sweep"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--inference-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pitch-offsets", type=float, nargs=3, default=(0.0, 0.1, 0.2))
    parser.add_argument("--yaw-offsets", type=float, nargs=3, default=(0.0, 0.4, 0.8))
    parser.add_argument("--roll-offsets", type=float, nargs=3, default=(0.0, 0.4, 0.8))
    parser.add_argument("--triangle-pairs-per-env", type=int, default=131_072)
    parser.add_argument("--capture-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-textures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hydroelastic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-gates", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.max_steps, args.inference_steps, args.triangle_pairs_per_env) < 1:
        raise ValueError("max-steps, inference-steps, and triangle-pairs-per-env must be positive")
    if not str(args.device).startswith("cuda"):
        raise ValueError("The frozen-DP thumb sweep requires a CUDA device")
    for name in ("pitch_offsets", "yaw_offsets", "roll_offsets"):
        values = tuple(float(value) for value in getattr(args, name))
        if len(values) != 3 or len(set(values)) != 3:
            raise ValueError(f"{name} must contain exactly three distinct values")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} must contain only finite offsets")
        if max(abs(value) for value in values) > 2.0:
            raise ValueError(f"{name} offsets must stay within [-2, 2]")


def _build_variants(
    pitch_offsets: tuple[float, ...] | list[float],
    yaw_offsets: tuple[float, ...] | list[float],
    roll_offsets: tuple[float, ...] | list[float],
) -> list[tuple[float, float, float]]:
    variants = list(itertools.product(pitch_offsets, yaw_offsets, roll_offsets))
    if len(variants) != 27 or len(set(variants)) != 27:
        raise ValueError("The thumb residual sweep requires exactly 27 unique variants")
    return [(float(pitch), float(yaw), float(roll)) for pitch, yaw, roll in variants]


def _apply_thumb_offsets(
    base_action: Any,
    variant_offsets: Any,
    action_min: Any,
    action_max: Any,
) -> tuple[Any, Any]:
    """Add pitch/yaw/roll offsets in the DP normalized action coordinates."""
    import torch

    if base_action.ndim != 2 or base_action.shape[1:] != (19,):
        raise ValueError(f"base_action must have shape [lane,19], got {tuple(base_action.shape)}")
    if variant_offsets.shape != (base_action.shape[0], 3):
        raise ValueError(
            f"variant_offsets must have shape ({base_action.shape[0]},3), got {tuple(variant_offsets.shape)}"
        )
    if action_min.shape != (19,) or action_max.shape != (19,):
        raise ValueError("action_min/action_max must have shape [19]")
    span = torch.clamp(action_max - action_min, min=1.0e-6)
    normalized = torch.clamp(2.0 * (base_action - action_min) / span - 1.0, -1.0, 1.0)
    candidate = normalized.clone()
    thumb_action_indices = tuple(9 + index for index in _THUMB_HAND_INDICES)
    unclamped_thumb = normalized[:, thumb_action_indices] + variant_offsets
    candidate[:, thumb_action_indices] = unclamped_thumb.clamp(-1.0, 1.0)
    physical = 0.5 * (candidate + 1.0) * span + action_min
    clamped = unclamped_thumb > 1.0
    return physical, clamped


def _evaluate_gates(
    *,
    num_variants: int,
    control_steps: int,
    expected_steps: int,
    finite_action_count: int,
    finite_state_count: int,
    expected_action_count: int,
    finite_results: bool,
    initial_state_max_abs_lane_delta: float,
    triangle_buffer_available: bool,
    rigid_overflow_frames: int,
    rigid_overflow_excess: int,
    triangle_overflow_frames: int,
    triangle_overflow_excess: int,
) -> dict[str, Any]:
    gates = {
        "exact_27_variants": {"passed": num_variants == 27, "actual": num_variants},
        "exact_control_steps": {
            "passed": control_steps == expected_steps,
            "expected": expected_steps,
            "actual": control_steps,
        },
        "finite_actions": {
            "passed": finite_action_count == expected_action_count,
            "finite": finite_action_count,
            "expected": expected_action_count,
        },
        "finite_states": {
            "passed": finite_state_count == expected_action_count,
            "finite": finite_state_count,
            "expected": expected_action_count,
        },
        "finite_results": {"passed": finite_results},
        "same_initial_robot_state": {
            "passed": initial_state_max_abs_lane_delta <= 1.0e-6,
            "max_abs_lane_delta": initial_state_max_abs_lane_delta,
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


def _run_sweep(args: argparse.Namespace) -> dict[str, Any]:
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
        raise RuntimeError("The frozen-DP thumb sweep requires CUDA")
    variants = _build_variants(args.pitch_offsets, args.yaw_offsets, args.roll_offsets)
    num_envs = len(variants)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    frozen_dp, dp_config, scheduler, stats, checkpoint_sha256 = _load_frozen_dp(args.checkpoint, device)
    _validate_frozen_dp_training_contract(dp_config)
    action_min = torch.as_tensor(stats["action_min"], dtype=torch.float32, device=device)
    action_max = torch.as_tensor(stats["action_max"], dtype=torch.float32, device=device)
    thumb_normalized_unit_rad = tuple(
        0.5 * (float(stats["action_max"][9 + index]) - float(stats["action_min"][9 + index]))
        for index in _THUMB_HAND_INDICES
    )
    state_min = torch.as_tensor(stats["state_min"], dtype=torch.float32, device=device)
    state_max = torch.as_tensor(stats["state_max"], dtype=torch.float32, device=device)
    variant_tensor = torch.as_tensor(variants, dtype=torch.float32, device=device)
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
    cache = _PerLaneActionChunkCache(1, dp_config.action_dim, device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    baseline_lane = torch.zeros(1, dtype=torch.long, device=device)
    baseline_executed = torch.ones(1, dtype=torch.bool, device=device)

    finger_steps = torch.zeros((num_envs, 5), dtype=torch.int64, device=device)
    thumb_steps = torch.zeros(num_envs, dtype=torch.int64, device=device)
    non_thumb_steps = torch.zeros_like(thumb_steps)
    opposed_steps = torch.zeros_like(thumb_steps)
    confirmed_steps = torch.zeros_like(thumb_steps)
    carrying_steps = torch.zeros_like(thumb_steps)
    any_contact_steps = torch.zeros_like(thumb_steps)
    ever = torch.zeros((num_envs, 5), dtype=torch.bool, device=device)
    rate_limit_counts = torch.zeros((num_envs, 3), dtype=torch.int64, device=device)
    normalized_clamp_counts = torch.zeros_like(rate_limit_counts)
    finite_action_count = torch.zeros((), dtype=torch.int64, device=device)
    finite_state_count = torch.zeros((), dtype=torch.int64, device=device)
    row_counts = torch.zeros(_BASE_ACTION_HORIZON, dtype=torch.int64, device=device)
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
    max_current_lift = torch.zeros(num_envs, dtype=torch.float32, device=device)
    max_contacted_lift = torch.zeros_like(max_current_lift)
    policy_calls = 0
    control_steps = 0
    try:
        observation, reset_info = env.reset(seed=args.seed)
        initial_state = observation["observation.state"][:, -1].float()
        initial_state_max_abs_lane_delta = (initial_state - initial_state[:1]).abs().max()
        reset_diagnostics = reset_info["control_step_diagnostics"]
        rigid_capacity = int(reset_diagnostics["rigid_contact_capacity"])
        triangle_capacity = int(reset_diagnostics["triangle_pair_capacity"])
        triangle_available = bool(reset_diagnostics["triangle_pair_buffer_available"])
        with torch.inference_mode():
            for control_step in range(args.max_steps):
                initial_noise = None
                if control_step % _CHUNK_HORIZON == 0:
                    shared_noise = torch.randn(
                        (1, dp_config.pred_horizon, dp_config.action_dim),
                        dtype=torch.float32,
                        device=device,
                        generator=generator,
                    )
                    initial_noise = shared_noise
                    policy_calls += 1
                baseline_observation = {key: value[baseline_lane] for key, value in observation.items()}
                prepared = _prepare_policy_step(
                    baseline_observation,
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
                    initial_noise=initial_noise,
                )
                action, normalized_clamped = _apply_thumb_offsets(
                    prepared.base_action.expand(num_envs, -1),
                    variant_tensor,
                    action_min,
                    action_max,
                )
                current_hand = observation["observation.state"][:, -1, 16:26].float()
                thumb_action_indices = tuple(9 + index for index in _THUMB_HAND_INDICES)
                requested_thumb_step = (action[:, thumb_action_indices] - current_hand[:, _THUMB_HAND_INDICES]).abs()
                rate_limited = requested_thumb_step > env_config.hand_max_joint_step_rad + 1.0e-6
                rate_limit_counts += rate_limited.to(torch.int64)
                normalized_clamp_counts += normalized_clamped.to(torch.int64)
                finite_action_count += torch.isfinite(action).all(dim=-1).sum(dtype=torch.int64)
                for row in range(_BASE_ACTION_HORIZON):
                    row_counts[row] += (prepared.row_index == row).sum(dtype=torch.int64) * num_envs

                observation, _, _, _, info = env.step(action)
                control_steps += 1
                cache.advance(baseline_executed, validate=False)
                finger = info["finger_contact_counts"] > 0
                thumb = finger[:, 0]
                non_thumb = finger[:, 1:].any(dim=-1)
                opposed = info["is_grasped"].bool()
                confirmed = info["grasp_confirmed"].bool()
                carrying = info["task_phase"].long() == 1
                any_contact = info["had_hand_contact_this_control_step"].bool()
                finger_steps += finger.to(torch.int64)
                thumb_steps += thumb.to(torch.int64)
                non_thumb_steps += non_thumb.to(torch.int64)
                opposed_steps += opposed.to(torch.int64)
                confirmed_steps += confirmed.to(torch.int64)
                carrying_steps += carrying.to(torch.int64)
                any_contact_steps += any_contact.to(torch.int64)
                ever[:, 0] |= thumb
                ever[:, 1] |= non_thumb
                ever[:, 2] |= opposed
                ever[:, 3] |= confirmed
                ever[:, 4] |= carrying
                max_current_lift = torch.maximum(max_current_lift, info["current_lift_height"].float())
                max_contacted_lift = torch.maximum(
                    max_contacted_lift,
                    info["max_contacted_carry_lift_height"].float(),
                )
                diagnostics = info["control_step_diagnostics"]
                _accumulate_buffer(buffer, diagnostics, prefix="rigid", diagnostic_prefix="rigid_contact")
                _accumulate_buffer(buffer, diagnostics, prefix="triangle", diagnostic_prefix="triangle_pair")
                finite_state = (
                    torch.isfinite(observation["observation.state"]).all(dim=(1, 2))
                    & torch.isfinite(info["current_lift_height"])
                    & torch.isfinite(info["max_contacted_carry_lift_height"])
                )
                finite_state_count += finite_state.sum(dtype=torch.int64)

        task = base_env.evaluate()
        packed = torch.cat(
            (
                finger_steps.float(),
                thumb_steps[:, None].float(),
                non_thumb_steps[:, None].float(),
                opposed_steps[:, None].float(),
                confirmed_steps[:, None].float(),
                carrying_steps[:, None].float(),
                any_contact_steps[:, None].float(),
                ever.float(),
                rate_limit_counts.float(),
                normalized_clamp_counts.float(),
                max_current_lift[:, None],
                max_contacted_lift[:, None],
                task["physical_max_lift_height"].float()[:, None],
            ),
            dim=1,
        ).cpu()
        packed_global = (
            torch.stack(
                (
                    finite_action_count,
                    finite_state_count,
                    initial_state_max_abs_lane_delta,
                    *buffer.values(),
                    *row_counts,
                )
            )
            .to(torch.float64)
            .cpu()
        )
    finally:
        env.close()

    results = []
    for lane, (pitch, yaw, roll) in enumerate(variants):
        row = packed[lane]
        finger_contact_steps = {name: int(row[index]) for index, name in enumerate(_FINGER_NAMES)}
        results.append(
            {
                "lane": lane,
                "thumb_offset_normalized": {"pitch": pitch, "yaw": yaw, "roll": roll},
                "thumb_offset_unclamped_rad": {
                    name: offset * unit
                    for name, offset, unit in zip(
                        _THUMB_JOINT_NAMES,
                        (pitch, yaw, roll),
                        thumb_normalized_unit_rad,
                        strict=True,
                    )
                },
                "finger_contact_steps": finger_contact_steps,
                "thumb_contact_steps": int(row[5]),
                "non_thumb_contact_steps": int(row[6]),
                "live_opposed_grasp_steps": int(row[7]),
                "confirmed_grasp_steps": int(row[8]),
                "carrying_steps": int(row[9]),
                "any_contact_steps": int(row[10]),
                "ever": {
                    "thumb_contact": bool(row[11]),
                    "non_thumb_contact": bool(row[12]),
                    "live_opposed_grasp": bool(row[13]),
                    "confirmed_grasp": bool(row[14]),
                    "carrying": bool(row[15]),
                },
                "dynamic_rate_limit_steps": {
                    name: int(row[16 + index]) for index, name in enumerate(_THUMB_JOINT_NAMES)
                },
                "normalized_action_clamp_steps": {
                    name: int(row[19 + index]) for index, name in enumerate(_THUMB_JOINT_NAMES)
                },
                "max_current_lift_height_m": float(row[22]),
                "max_contacted_carry_lift_height_m": float(row[23]),
                "physical_max_lift_height_m": float(row[24]),
            }
        )

    ranking = sorted(
        range(num_envs),
        key=lambda lane: (
            results[lane]["confirmed_grasp_steps"],
            results[lane]["live_opposed_grasp_steps"],
            results[lane]["carrying_steps"],
            results[lane]["max_contacted_carry_lift_height_m"],
            results[lane]["thumb_contact_steps"],
        ),
        reverse=True,
    )
    global_values = packed_global.tolist()
    finite_count = int(global_values[0])
    buffer_names = tuple(buffer)
    finite_state_count_value = int(global_values[1])
    initial_state_delta = float(global_values[2])
    buffer_values = {name: int(global_values[3 + index]) for index, name in enumerate(buffer_names)}
    row_values_start = 3 + len(buffer_names)
    row_count_values = [int(value) for value in global_values[row_values_start:]]
    expected_action_count = args.max_steps * num_envs
    finite_results = bool(torch.isfinite(packed).all())
    gates = _evaluate_gates(
        num_variants=num_envs,
        control_steps=control_steps,
        expected_steps=args.max_steps,
        finite_action_count=finite_count,
        finite_state_count=finite_state_count_value,
        expected_action_count=expected_action_count,
        finite_results=finite_results,
        initial_state_max_abs_lane_delta=initial_state_delta,
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
            "max_steps": args.max_steps,
            "inference_steps": args.inference_steps,
            "seed": args.seed,
            "num_envs": num_envs,
            "chunk_horizon": _CHUNK_HORIZON,
            "shared_frozen_dp_base_lane": 0,
            "shared_dp_initial_noise_per_replan": True,
            "pitch_offsets": list(args.pitch_offsets),
            "yaw_offsets": list(args.yaw_offsets),
            "roll_offsets": list(args.roll_offsets),
            "hand_max_joint_step_rad": env_config.hand_max_joint_step_rad,
            "thumb_normalized_unit_rad": dict(zip(_THUMB_JOINT_NAMES, thumb_normalized_unit_rad, strict=True)),
            "triangle_pairs_per_env": args.triangle_pairs_per_env,
        },
        "policy_calls": policy_calls,
        "control_steps": control_steps,
        "initial_state_max_abs_lane_delta": initial_state_delta,
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
        "variants_with_thumb_contact": sum(item["ever"]["thumb_contact"] for item in results),
        "variants_with_live_opposed_grasp": sum(item["ever"]["live_opposed_grasp"] for item in results),
        "variants_with_confirmed_grasp": sum(item["ever"]["confirmed_grasp"] for item in results),
        "variants_with_carrying": sum(item["ever"]["carrying"] for item in results),
        "best_lane": ranking[0],
        "ranking": ranking,
        "results": results,
        "gates": gates,
        "passed": bool(gates["passed"]),
    }


def main() -> int:
    args = create_parser().parse_args()
    _validate_args(args)
    summary = _run_sweep(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    best = summary["results"][summary["best_lane"]]
    print(
        "thumb residual sweep: "
        f"thumb={summary['variants_with_thumb_contact']}/27 "
        f"opposed={summary['variants_with_live_opposed_grasp']}/27 "
        f"confirmed={summary['variants_with_confirmed_grasp']}/27 "
        f"best={best['thumb_offset_normalized']} passed={summary['passed']}"
    )
    print(f"summary: {output_path}")
    return 0 if summary["passed"] or not args.strict_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
