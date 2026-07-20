#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Probe L10 actuator loads while replaying a successful dataset trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.probe_newton_l10_finger_root_load import _LOAD_KEY, _summarize_samples


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="LeRobot dataset root, or a pre-extracted .npy file containing corrected physical actions",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/finger_root_load_replay_probe"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=0, help="Zero replays the complete episode")
    parser.add_argument("--capture-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hydroelastic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-gates", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _load_episode_actions(dataset: Path, episode_index: int) -> Any:
    import numpy as np  # noqa: PLC0415
    import torch

    if dataset.suffix == ".npy":
        path = dataset
        if not path.is_file():
            raise FileNotFoundError(path)
        actions = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    else:
        import pyarrow.parquet as parquet  # noqa: PLC0415

        path = dataset / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        table = parquet.read_table(path, columns=["action"])
        actions = np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 19 or actions.shape[0] < 1:
        raise ValueError(f"Expected non-empty [time,19] actions in {path}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError(f"Non-finite action in {path}")
    return torch.from_numpy(actions)


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from teleop_stack.envs import GrootNewtonEnv, GrootNewtonEnvConfig  # noqa: PLC0415

    if args.num_envs < 1 or args.episode_index < 0 or args.max_steps < 0:
        raise ValueError("num_envs must be positive and episode-index/max-steps cannot be negative")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Finger-root replay probe requires CUDA")
    actions_cpu = _load_episode_actions(args.dataset, args.episode_index)
    action_count = actions_cpu.shape[0] if args.max_steps == 0 else min(args.max_steps, actions_cpu.shape[0])
    actions = actions_cpu[:action_count].to(device=args.device)
    env = GrootNewtonEnv(
        GrootNewtonEnvConfig(
            num_envs=args.num_envs,
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
        )
    )
    load_samples: list[Any] = []
    contact_samples: list[Any] = []
    grasp_samples: list[Any] = []
    try:
        observation, _ = env.reset()
        reset_zero = (observation[_LOAD_KEY] == 0.0).all()
        with torch.inference_mode():
            for step in range(action_count):
                action = actions[step].expand(args.num_envs, -1)
                observation, _, _, _, info = env.step(action)
                load_samples.append(observation[_LOAD_KEY].float().clone())
                contact_samples.append(info["had_hand_contact_this_control_step"].bool().clone())
                grasp_samples.append(info["grasp_confirmed"].bool().clone())
        loads = torch.cat(load_samples, dim=0)
        contact = torch.cat(contact_samples, dim=0)
        grasp = torch.cat(grasp_samples, dim=0)
        summary = _summarize_samples(
            loads,
            contact,
            grasp,
            reset_zero=reset_zero,
            episodes_completed=torch.as_tensor(args.num_envs, device=args.device),
            episodes_expected=args.num_envs,
        )
        task = env.evaluate()
        packed_task = torch.stack(
            (
                task["has_hand_contact"].float().sum(),
                task["grasp_confirmed"].float().sum(),
                task["physical_max_lift_height"].float().mean(),
                task["physical_max_lift_height"].float().max(),
                task["max_contacted_carry_lift_height"].float().mean(),
                task["max_contacted_carry_lift_height"].float().max(),
            )
        ).cpu()
        summary["replay"] = {
            "dataset": str(args.dataset.resolve()),
            "episode_index": args.episode_index,
            "action_steps": action_count,
            "lanes_with_contact_at_end": int(packed_task[0]),
            "lanes_with_grasp_at_end": int(packed_task[1]),
            "mean_physical_max_lift_height_m": float(packed_task[2]),
            "max_physical_max_lift_height_m": float(packed_task[3]),
            "mean_contacted_carry_max_lift_height_m": float(packed_task[4]),
            "max_contacted_carry_max_lift_height_m": float(packed_task[5]),
        }
        summary["finger_root_load"] = env.finger_root_load_metadata
        return summary
    finally:
        env.close()


def main() -> int:
    args = create_parser().parse_args()
    summary = _run_probe(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        f"finger-root-load replay: steps={summary['replay']['action_steps']} "
        f"physical_lift={summary['replay']['max_physical_max_lift_height_m']:.6f}m "
        f"grasp_samples={summary['condition_sample_count']['grasp']} "
        f"margin={summary['second_largest_load']['grasp_minus_free']} "
        f"auc={summary['second_largest_load']['grasp_auc']} passed={summary['passed']}"
    )
    print(f"summary: {output_path}")
    return 0 if summary["passed"] or not args.strict_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
