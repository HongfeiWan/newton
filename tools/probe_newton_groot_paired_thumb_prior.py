#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare a fixed positive thumb prior against a paired frozen-DP baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

_THUMB_HAND_INDICES = (0, 1, 9)
_THUMB_JOINT_NAMES = ("thumb_cmc_pitch", "thumb_cmc_yaw", "thumb_cmc_roll")
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_VARIANT_NAMES = ("zero", "positive_prior")
_CHUNK_HORIZON = 8
_DEFAULT_THUMB_OFFSETS = (0.1425, 0.3715, 0.4159)
_SCHEMA_VERSION = "newton.groot_paired_thumb_prior_probe.v1"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Frozen Diffusion Policy checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/groot_paired_thumb_prior_probe"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-pairs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--inference-steps", type=int, default=6)
    parser.add_argument(
        "--thumb-offsets-normalized",
        type=float,
        nargs=3,
        default=_DEFAULT_THUMB_OFFSETS,
        metavar=("PITCH", "YAW", "ROLL"),
        help="Positive normalized pitch/yaw/roll offsets applied only to the prior lane in each pair",
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


def _validate_args(args: argparse.Namespace) -> None:
    positive = (args.num_pairs, args.max_steps, args.inference_steps, args.triangle_pairs_per_env)
    if min(positive) < 1:
        raise ValueError("num-pairs, max-steps, inference-steps, and triangle-pairs-per-env must be positive")
    if not str(args.device).startswith("cuda"):
        raise ValueError("The paired thumb-prior probe requires a CUDA device")
    offsets = tuple(float(value) for value in args.thumb_offsets_normalized)
    if len(offsets) != len(_THUMB_HAND_INDICES):
        raise ValueError("thumb-offsets-normalized must contain pitch, yaw, and roll")
    if not all(math.isfinite(value) and 0.0 < value <= 2.0 for value in offsets):
        raise ValueError("thumb-offsets-normalized must be finite and lie in (0, 2]")


def _build_paired_actions(
    pair_base_action: Any,
    positive_offsets: Any,
    action_min: Any,
    action_max: Any,
) -> tuple[Any, Any, Any]:
    """Duplicate each pair base and add normalized offsets to its odd lane."""
    import torch

    if pair_base_action.ndim != 2 or pair_base_action.shape[1:] != (19,):
        raise ValueError(f"pair_base_action must have shape [pair,19], got {tuple(pair_base_action.shape)}")
    if positive_offsets.shape != (3,):
        raise ValueError(f"positive_offsets must have shape [3], got {tuple(positive_offsets.shape)}")
    if action_min.shape != (19,) or action_max.shape != (19,):
        raise ValueError("action_min/action_max must have shape [19]")

    paired_base = pair_base_action.repeat_interleave(2, dim=0)
    action = paired_base.clone()
    span = torch.clamp(action_max - action_min, min=1.0e-6)
    prior_base = pair_base_action
    normalized_prior = torch.clamp(2.0 * (prior_base - action_min) / span - 1.0, -1.0, 1.0)
    thumb_action_indices = tuple(9 + index for index in _THUMB_HAND_INDICES)
    unclamped_thumb = normalized_prior[:, thumb_action_indices] + positive_offsets
    normalized_prior = normalized_prior.clone()
    normalized_prior[:, thumb_action_indices] = unclamped_thumb.clamp(-1.0, 1.0)
    prior_action = 0.5 * (normalized_prior + 1.0) * span + action_min
    action[1::2] = prior_action
    normalized_clamped = unclamped_thumb > 1.0
    return action, paired_base, normalized_clamped


def _evaluate_gates(
    *,
    num_pairs: int,
    num_envs: int,
    control_steps: int,
    expected_steps: int,
    finite_action_count: int,
    finite_state_count: int,
    expected_lane_samples: int,
    finite_results: bool,
    initial_state_max_abs_lane_delta: float,
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
        "exact_control_steps": {
            "passed": control_steps == expected_steps,
            "expected": expected_steps,
            "actual": control_steps,
        },
        "finite_actions": {
            "passed": finite_action_count == expected_lane_samples,
            "finite": finite_action_count,
            "expected": expected_lane_samples,
        },
        "finite_states": {
            "passed": finite_state_count == expected_lane_samples,
            "finite": finite_state_count,
            "expected": expected_lane_samples,
        },
        "finite_results": {"passed": finite_results},
        "fixed_initial_state": {
            "passed": initial_state_max_abs_lane_delta <= 1.0e-6,
            "max_abs_lane_delta": initial_state_max_abs_lane_delta,
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


def _expected_row_counts(max_steps: int, num_pairs: int) -> list[int]:
    return [((max_steps + _CHUNK_HORIZON - 1 - row) // _CHUNK_HORIZON) * num_pairs for row in range(8)]


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


def _variant_totals(results: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected = [pair[variant] for pair in results]
    return {
        "lanes": len(selected),
        "lanes_with_any_contact": sum(item["ever"]["any_contact"] for item in selected),
        "lanes_with_thumb_contact": sum(item["ever"]["thumb_contact"] for item in selected),
        "lanes_with_live_opposed": sum(item["ever"]["live_opposed"] for item in selected),
        "lanes_with_confirmed_grasp": sum(item["ever"]["confirmed_grasp"] for item in selected),
        "lanes_with_carrying": sum(item["ever"]["carrying"] for item in selected),
        "mean_any_contact_steps": sum(item["any_contact_steps"] for item in selected) / len(selected),
        "mean_physical_max_lift_height_m": sum(item["physical_max_lift_height_m"] for item in selected) / len(selected),
        "mean_gated_max_lift_height_m": sum(item["gated_max_lift_height_m"] for item in selected) / len(selected),
    }


def _paired_deltas(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "any_contact_steps",
        "live_opposed_steps",
        "confirmed_grasp_steps",
        "carrying_steps",
        "physical_max_lift_height_m",
        "gated_max_lift_height_m",
    )
    output: dict[str, Any] = {}
    for metric in metrics:
        deltas = [float(pair["positive_prior"][metric]) - float(pair["zero"][metric]) for pair in results]
        output[metric] = {
            "mean_prior_minus_zero": sum(deltas) / len(deltas),
            "prior_wins": sum(delta > 0.0 for delta in deltas),
            "ties": sum(delta == 0.0 for delta in deltas),
            "zero_wins": sum(delta < 0.0 for delta in deltas),
        }
    return output


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
        raise RuntimeError("The paired thumb-prior probe requires CUDA")
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
    offsets = torch.as_tensor(args.thumb_offsets_normalized, dtype=torch.float32, device=device)
    thumb_normalized_unit_rad = tuple(
        0.5 * (float(stats["action_max"][9 + index]) - float(stats["action_min"][9 + index]))
        for index in _THUMB_HAND_INDICES
    )
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
    executed_pairs = torch.ones(args.num_pairs, dtype=torch.bool, device=device)
    zero_lane_indices = 2 * torch.arange(args.num_pairs, dtype=torch.long, device=device)

    finger_steps = torch.zeros((num_envs, len(_FINGER_NAMES)), dtype=torch.int64, device=device)
    any_contact_steps = torch.zeros(num_envs, dtype=torch.int64, device=device)
    live_opposed_steps = torch.zeros_like(any_contact_steps)
    confirmed_steps = torch.zeros_like(any_contact_steps)
    carrying_steps = torch.zeros_like(any_contact_steps)
    ever = torch.zeros((num_envs, 5), dtype=torch.bool, device=device)
    max_current_lift = torch.zeros(num_envs, dtype=torch.float32, device=device)
    max_gated_lift = torch.zeros_like(max_current_lift)
    normalized_clamp_counts = torch.zeros((args.num_pairs, 3), dtype=torch.int64, device=device)
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
        initial_state_max_abs_lane_delta = (initial_state - initial_state[:1]).abs().max()
        reset_diagnostics = reset_info["control_step_diagnostics"]
        rigid_capacity = int(reset_diagnostics["rigid_contact_capacity"])
        triangle_capacity = int(reset_diagnostics["triangle_pair_capacity"])
        triangle_available = bool(reset_diagnostics["triangle_pair_buffer_available"])
        with torch.inference_mode():
            for _ in range(args.max_steps):
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
                action, paired_base, clamped = _build_paired_actions(
                    prepared.base_action,
                    offsets,
                    action_min,
                    action_max,
                )
                pair_base_delta = (paired_base[0::2] - paired_base[1::2]).abs().max()
                shared_pair_base_max_abs_delta = torch.maximum(shared_pair_base_max_abs_delta, pair_base_delta)
                normalized_clamp_counts += clamped.to(torch.int64)
                finite_action_count += torch.isfinite(action).all(dim=-1).sum(dtype=torch.int64)
                for row in range(_BASE_ACTION_HORIZON):
                    row_counts[row] += (prepared.row_index == row).sum(dtype=torch.int64)

                observation, _, _, _, info = env.step(action)
                cache.advance(executed_pairs, validate=True)
                control_steps += 1
                finger = info["finger_contact_any_frame_this_control_step"].bool()
                any_contact = info["had_hand_contact_this_control_step"].bool()
                live_opposed = info["is_grasped"].bool()
                confirmed = info["grasp_confirmed"].bool()
                carrying = info["task_phase"].long() == 1
                finger_steps += finger.to(torch.int64)
                any_contact_steps += any_contact.to(torch.int64)
                live_opposed_steps += live_opposed.to(torch.int64)
                confirmed_steps += confirmed.to(torch.int64)
                carrying_steps += carrying.to(torch.int64)
                ever[:, 0] |= any_contact
                ever[:, 1] |= finger[:, 0]
                ever[:, 2] |= live_opposed
                ever[:, 3] |= confirmed
                ever[:, 4] |= carrying
                max_current_lift = torch.maximum(max_current_lift, info["current_lift_height"].float())
                max_gated_lift = torch.maximum(max_gated_lift, info["max_contacted_carry_lift_height"].float())
                diagnostics = info["control_step_diagnostics"]
                _accumulate_buffer(buffer, diagnostics, prefix="rigid", diagnostic_prefix="rigid_contact")
                _accumulate_buffer(buffer, diagnostics, prefix="triangle", diagnostic_prefix="triangle_pair")
                finite_state = (
                    torch.isfinite(observation["observation.state"]).all(dim=(1, 2))
                    & torch.isfinite(info["current_lift_height"])
                    & torch.isfinite(info["max_contacted_carry_lift_height"])
                )
                finite_state_count += finite_state.sum(dtype=torch.int64)

        evaluation = base_env.evaluate()
        packed = torch.cat(
            (
                finger_steps.float(),
                any_contact_steps[:, None].float(),
                live_opposed_steps[:, None].float(),
                confirmed_steps[:, None].float(),
                carrying_steps[:, None].float(),
                ever.float(),
                max_current_lift[:, None],
                max_gated_lift[:, None],
                evaluation["physical_max_lift_height"].float()[:, None],
            ),
            dim=1,
        ).cpu()
        packed_global = (
            torch.stack(
                (
                    finite_action_count,
                    finite_state_count,
                    initial_state_max_abs_lane_delta,
                    shared_pair_base_max_abs_delta,
                    cache.plan_count.sum(),
                    *buffer.values(),
                    *row_counts,
                )
            )
            .to(torch.float64)
            .cpu()
        )
        packed_clamps = normalized_clamp_counts.cpu()
    finally:
        env.close()

    lane_results: list[dict[str, Any]] = []
    for lane in range(num_envs):
        row = packed[lane]
        lane_results.append(
            {
                "lane": lane,
                "variant": _VARIANT_NAMES[lane % 2],
                "finger_contact_steps": {name: int(row[index]) for index, name in enumerate(_FINGER_NAMES)},
                "any_contact_steps": int(row[5]),
                "live_opposed_steps": int(row[6]),
                "confirmed_grasp_steps": int(row[7]),
                "carrying_steps": int(row[8]),
                "ever": {
                    "any_contact": bool(row[9]),
                    "thumb_contact": bool(row[10]),
                    "live_opposed": bool(row[11]),
                    "confirmed_grasp": bool(row[12]),
                    "carrying": bool(row[13]),
                },
                "max_current_lift_height_m": float(row[14]),
                "gated_max_lift_height_m": float(row[15]),
                "physical_max_lift_height_m": float(row[16]),
            }
        )
    results = []
    for pair in range(args.num_pairs):
        prior = lane_results[2 * pair + 1]
        prior["normalized_action_clamp_steps"] = {
            name: int(packed_clamps[pair, index]) for index, name in enumerate(_THUMB_JOINT_NAMES)
        }
        results.append({"pair": pair, "zero": lane_results[2 * pair], "positive_prior": prior})

    global_values = packed_global.tolist()
    buffer_names = tuple(buffer)
    buffer_values = {name: int(global_values[5 + index]) for index, name in enumerate(buffer_names)}
    row_start = 5 + len(buffer_names)
    row_count_values = [int(value) for value in global_values[row_start:]]
    expected_rows = _expected_row_counts(args.max_steps, args.num_pairs)
    expected_lane_samples = args.max_steps * num_envs
    gates = _evaluate_gates(
        num_pairs=args.num_pairs,
        num_envs=num_envs,
        control_steps=control_steps,
        expected_steps=args.max_steps,
        finite_action_count=int(global_values[0]),
        finite_state_count=int(global_values[1]),
        expected_lane_samples=expected_lane_samples,
        finite_results=bool(torch.isfinite(packed).all()),
        initial_state_max_abs_lane_delta=float(global_values[2]),
        shared_pair_base_max_abs_delta=float(global_values[3]),
        row_counts=row_count_values,
        expected_row_counts=expected_rows,
        triangle_buffer_available=triangle_available,
        rigid_overflow_frames=buffer_values["rigid_overflow_frames"],
        rigid_overflow_excess=buffer_values["rigid_overflow_excess"],
        triangle_overflow_frames=buffer_values["triangle_overflow_frames"],
        triangle_overflow_excess=buffer_values["triangle_overflow_excess"],
    )
    offsets_tuple = tuple(float(value) for value in args.thumb_offsets_normalized)
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
            "pair_lane_order": list(_VARIANT_NAMES),
            "replan_observation_lane": "zero",
            "shared_frozen_dp_chunk_within_pair": True,
            "thumb_offsets_normalized": dict(zip(_THUMB_JOINT_NAMES, offsets_tuple, strict=True)),
            "thumb_offsets_unclamped_rad": dict(
                zip(
                    _THUMB_JOINT_NAMES,
                    (offset * unit for offset, unit in zip(offsets_tuple, thumb_normalized_unit_rad, strict=True)),
                    strict=True,
                )
            ),
            "triangle_pairs_per_env": args.triangle_pairs_per_env,
        },
        "control_steps": control_steps,
        "policy_lane_plan_count": int(global_values[4]),
        "initial_state_max_abs_lane_delta": float(global_values[2]),
        "shared_pair_base_max_abs_delta": float(global_values[3]),
        "base_action_row_counts_per_pair_lane": {str(index): count for index, count in enumerate(row_count_values)},
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
        "variant_totals": {variant: _variant_totals(results, variant) for variant in _VARIANT_NAMES},
        "paired_deltas": _paired_deltas(results),
        "results": results,
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
    totals = summary["variant_totals"]
    print(
        "paired thumb prior: "
        f"zero_thumb={totals['zero']['lanes_with_thumb_contact']}/{args.num_pairs} "
        f"prior_thumb={totals['positive_prior']['lanes_with_thumb_contact']}/{args.num_pairs} "
        f"zero_opposed={totals['zero']['lanes_with_live_opposed']}/{args.num_pairs} "
        f"prior_opposed={totals['positive_prior']['lanes_with_live_opposed']}/{args.num_pairs} "
        f"passed={summary['passed']}"
    )
    print(f"summary: {output_path}")
    return 0 if summary["passed"] or not args.strict_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
