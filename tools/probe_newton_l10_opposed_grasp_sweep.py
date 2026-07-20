#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sweep EEF offsets and non-thumb closure over a recorded L10 trajectory."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from tools.probe_newton_l10_finger_root_load_replay import _load_episode_actions

_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_NON_THUMB_PITCH_ACTION_SLICE = slice(11, 15)
_NON_THUMB_PITCH_MAX = 1.3607


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", type=Path, help="Corrected physical actions stored as a [time,19] NPY file")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/opposed_grasp_sweep"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=0, help="Zero replays the complete trajectory")
    parser.add_argument("--position-offsets", type=float, nargs="+", default=(-0.015, 0.0, 0.015))
    parser.add_argument("--non-thumb-closures", type=float, nargs="+", default=(0.0, 0.1, 0.2))
    parser.add_argument("--triangle-pairs-per-env", type=int, default=131_072)
    parser.add_argument("--capture-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hydroelastic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-gates", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from teleop_stack.envs import GrootNewtonEnv, GrootNewtonEnvConfig  # noqa: PLC0415

    if args.max_steps < 0 or args.triangle_pairs_per_env < 1:
        raise ValueError("max-steps cannot be negative and triangle-pairs-per-env must be positive")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("The opposed-grasp sweep requires CUDA")

    variants = list(itertools.product(args.position_offsets, args.position_offsets, args.non_thumb_closures))
    if not variants:
        raise ValueError("At least one sweep variant is required")
    actions_cpu = _load_episode_actions(args.actions, episode_index=0)
    action_count = actions_cpu.shape[0] if args.max_steps == 0 else min(args.max_steps, actions_cpu.shape[0])
    actions = actions_cpu[:action_count].to(device=args.device)
    variant_tensor = torch.as_tensor(variants, dtype=torch.float32, device=args.device)
    num_envs = len(variants)

    env = GrootNewtonEnv(
        GrootNewtonEnvConfig(
            num_envs=num_envs,
            device=args.device,
            max_episode_steps=max(action_count + 1, 300),
            obs_mode="policy",
            control_mode="pd_eef_pose_abs",
            reward_mode="normalized_dense",
            terminate_on_success=False,
            terminate_on_fail=False,
            capture_graph=args.capture_graph,
            render_images=False,
            camera_textures=False,
            load_scene_visuals=True,
            hydroelastic_contacts=args.hydroelastic,
            request_finger_root_load=True,
            triangle_pairs_per_env=args.triangle_pairs_per_env,
        )
    )
    finger_steps = torch.zeros((num_envs, 5), dtype=torch.int64, device=args.device)
    any_contact_steps = torch.zeros(num_envs, dtype=torch.int64, device=args.device)
    live_opposed_steps = torch.zeros_like(any_contact_steps)
    confirmed_steps = torch.zeros_like(any_contact_steps)
    carrying_steps = torch.zeros_like(any_contact_steps)
    max_current_lift = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
    triangle_pair_frame_max = torch.zeros((), dtype=torch.int64, device=args.device)
    triangle_pair_overflow_frames = torch.zeros((), dtype=torch.int64, device=args.device)
    triangle_pair_capacity = 0
    try:
        env.reset(seed=0)
        with torch.inference_mode():
            for step in range(action_count):
                action = actions[step].expand(num_envs, -1).clone()
                # The recorded pose is in the Nero/CAN base frame. CAN y/z span the
                # two world-horizontal axes under the training contract-v6 transform.
                action[:, 1] += variant_tensor[:, 0]
                action[:, 2] += variant_tensor[:, 1]
                action[:, _NON_THUMB_PITCH_ACTION_SLICE] = torch.clamp(
                    action[:, _NON_THUMB_PITCH_ACTION_SLICE] + variant_tensor[:, 2:3],
                    min=0.0,
                    max=_NON_THUMB_PITCH_MAX,
                )
                _, _, _, _, info = env.step(action)
                finger = info["finger_contact_counts"] > 0
                finger_steps += finger
                any_contact_steps += info["had_hand_contact_this_control_step"].to(torch.int64)
                live_opposed_steps += info["is_grasped"].to(torch.int64)
                confirmed_steps += info["grasp_confirmed"].to(torch.int64)
                carrying_steps += (info["task_phase"] == 1).to(torch.int64)
                max_current_lift = torch.maximum(max_current_lift, info["current_lift_height"].float())
                diagnostics = info["control_step_diagnostics"]
                triangle_pair_frame_max = torch.maximum(
                    triangle_pair_frame_max,
                    diagnostics["triangle_pair_frame_max"].to(torch.int64).reshape(()),
                )
                triangle_pair_overflow_frames += (
                    diagnostics["triangle_pair_overflow_frame_count"].to(torch.int64).reshape(())
                )
                triangle_pair_capacity = int(diagnostics["triangle_pair_capacity"])

        task = env.evaluate()
        packed = torch.cat(
            (
                finger_steps.float(),
                any_contact_steps[:, None].float(),
                live_opposed_steps[:, None].float(),
                confirmed_steps[:, None].float(),
                carrying_steps[:, None].float(),
                max_current_lift[:, None],
                task["physical_max_lift_height"].float()[:, None],
                task["max_contacted_carry_lift_height"].float()[:, None],
                triangle_pair_frame_max.expand(num_envs, 1).float(),
                triangle_pair_overflow_frames.expand(num_envs, 1).float(),
            ),
            dim=1,
        ).cpu()
    finally:
        env.close()

    results: list[dict[str, Any]] = []
    for lane, (can_y, can_z, closure) in enumerate(variants):
        row = packed[lane]
        finger_counts = {name: int(row[index].item()) for index, name in enumerate(_FINGER_NAMES)}
        results.append(
            {
                "lane": lane,
                "can_y_offset_m": can_y,
                "can_z_offset_m": can_z,
                "non_thumb_pitch_closure_rad": closure,
                "finger_contact_steps": finger_counts,
                "non_thumb_contact_steps": sum(finger_counts[name] for name in _FINGER_NAMES[1:]),
                "any_contact_control_steps": int(row[5].item()),
                "live_opposed_grasp_steps": int(row[6].item()),
                "confirmed_grasp_steps": int(row[7].item()),
                "carrying_steps": int(row[8].item()),
                "max_current_lift_height_m": float(row[9].item()),
                "physical_max_lift_height_m": float(row[10].item()),
                "contacted_carry_max_lift_height_m": float(row[11].item()),
            }
        )
    results.sort(
        key=lambda item: (
            item["confirmed_grasp_steps"],
            item["live_opposed_grasp_steps"],
            item["non_thumb_contact_steps"],
            item["physical_max_lift_height_m"],
        ),
        reverse=True,
    )
    triangle_pair_overflow_frames = int(packed[0, 13].item())
    return {
        "schema_version": "newton.l10_opposed_grasp_sweep.v1",
        "actions": str(args.actions.resolve()),
        "action_steps": action_count,
        "num_envs": num_envs,
        "triangle_pairs_per_env": args.triangle_pairs_per_env,
        "triangle_pair_capacity": triangle_pair_capacity,
        "triangle_pair_frame_max": int(packed[0, 12].item()),
        "triangle_pair_overflow_frames": triangle_pair_overflow_frames,
        "passed": triangle_pair_overflow_frames == 0,
        "variants_with_non_thumb_contact": sum(item["non_thumb_contact_steps"] > 0 for item in results),
        "variants_with_live_opposed_grasp": sum(item["live_opposed_grasp_steps"] > 0 for item in results),
        "variants_with_confirmed_grasp": sum(item["confirmed_grasp_steps"] > 0 for item in results),
        "results": results,
    }


def main() -> int:
    args = create_parser().parse_args()
    summary = _run_sweep(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    best = summary["results"][0]
    print(
        "opposed-grasp sweep: "
        f"variants={summary['num_envs']} non_thumb={summary['variants_with_non_thumb_contact']} "
        f"live={summary['variants_with_live_opposed_grasp']} confirmed={summary['variants_with_confirmed_grasp']} "
        f"triangle_overflow_frames={summary['triangle_pair_overflow_frames']}"
    )
    print(
        "best: "
        f"can_y={best['can_y_offset_m']:+.3f} can_z={best['can_z_offset_m']:+.3f} "
        f"closure={best['non_thumb_pitch_closure_rad']:+.3f} "
        f"live={best['live_opposed_grasp_steps']} confirmed={best['confirmed_grasp_steps']} "
        f"fingers={best['finger_contact_steps']}"
    )
    print(f"summary: {output_path}")
    return 0 if summary["passed"] or not args.strict_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
