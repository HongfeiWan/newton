#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Probe whether L10 root-actuator load separates free motion from grasping."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_LOAD_KEY = "observation.finger_root_load"
_STATE_KEY = "observation.state"
_CHUNK_HORIZON = 8


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Frozen Diffusion Policy checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/finger_root_load_probe"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--inference-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-textures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hydroelastic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-gates", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _binary_auc(scores: Any, labels: Any, *, bins: int = 1001) -> Any:
    """Return tie-aware histogram AUC for scores constrained to ``[0, 1]``."""
    import torch

    scores = scores.float().clamp(0.0, 1.0)
    labels = labels.bool()
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.numel() == 0 or negatives.numel() == 0:
        return torch.full((), float("nan"), dtype=torch.float64, device=scores.device)
    positive_histogram = torch.histc(positives, bins=bins, min=0.0, max=1.0).double()
    negative_histogram = torch.histc(negatives, bins=bins, min=0.0, max=1.0).double()
    negatives_below = torch.cumsum(negative_histogram, dim=0) - negative_histogram
    concordant = (positive_histogram * (negatives_below + 0.5 * negative_histogram)).sum()
    return concordant / float(positives.numel() * negatives.numel())


def _masked_column_mean(values: Any, mask: Any) -> Any:
    import torch

    weights = mask.double()[:, None]
    count = weights.sum()
    mean = (values.double() * weights).sum(dim=0) / torch.clamp(count, min=1.0)
    return torch.where(count > 0, mean, torch.full_like(mean, float("nan")))


def _masked_scalar_mean(values: Any, mask: Any) -> Any:
    import torch

    weights = mask.double()
    count = weights.sum()
    mean = (values.double() * weights).sum() / torch.clamp(count, min=1.0)
    return torch.where(count > 0, mean, torch.full_like(mean, float("nan")))


def _summarize_samples(
    loads: Any,
    contact: Any,
    grasp: Any,
    *,
    reset_zero: Any,
    episodes_completed: Any,
    episodes_expected: int,
) -> dict[str, Any]:
    """Compute probe metrics on-device and transfer one packed vector to the host."""
    import torch

    loads = loads.float()
    contact = contact.bool()
    grasp = grasp.bool()
    free = ~contact
    second_load = loads.topk(k=2, dim=-1).values[:, 1]
    overall_mean = loads.double().mean(dim=0)
    overall_p95 = torch.quantile(loads.float(), 0.95, dim=0).double()
    overall_saturation = (loads >= 0.95).double().mean(dim=0)
    free_mean = _masked_column_mean(loads, free)
    contact_mean = _masked_column_mean(loads, contact)
    grasp_mean = _masked_column_mean(loads, grasp)
    free_saturation = _masked_scalar_mean((loads >= 0.95).float(), free[:, None].expand_as(loads))
    second_free = _masked_scalar_mean(second_load, free)
    second_contact = _masked_scalar_mean(second_load, contact)
    second_grasp = _masked_scalar_mean(second_load, grasp)
    auc = _binary_auc(second_load, grasp)
    finite_loads = torch.isfinite(loads).all()
    packed = torch.cat(
        (
            torch.stack(
                (
                    reset_zero.double(),
                    finite_loads.double(),
                    episodes_completed.double(),
                    torch.as_tensor(float(loads.shape[0]), dtype=torch.float64, device=loads.device),
                    free.sum().double(),
                    contact.sum().double(),
                    grasp.sum().double(),
                    free_saturation,
                    second_free,
                    second_contact,
                    second_grasp,
                    second_grasp - second_free,
                    auc,
                )
            ),
            overall_mean,
            overall_p95,
            overall_saturation,
            free_mean,
            contact_mean,
            grasp_mean,
        )
    )
    values = packed.detach().cpu().tolist()

    def scalar(index: int) -> float | None:
        value = float(values[index])
        return value if math.isfinite(value) else None

    cursor = 13

    def finger_values() -> dict[str, float | None]:
        nonlocal cursor
        result = {name: scalar(cursor + index) for index, name in enumerate(_FINGER_NAMES)}
        cursor += len(_FINGER_NAMES)
        return result

    summary = {
        "episodes_expected": episodes_expected,
        "episodes_completed": int(values[2]),
        "sample_count": int(values[3]),
        "condition_sample_count": {"free": int(values[4]), "contact": int(values[5]), "grasp": int(values[6])},
        "reset_zero": bool(values[0]),
        "finite_loads": bool(values[1]),
        "per_finger": {
            "mean": finger_values(),
            "p95": finger_values(),
            "saturation_rate": finger_values(),
            "free_mean": finger_values(),
            "contact_mean": finger_values(),
            "grasp_mean": finger_values(),
        },
        "free_saturation_rate": scalar(7),
        "second_largest_load": {
            "free_mean": scalar(8),
            "contact_mean": scalar(9),
            "grasp_mean": scalar(10),
            "grasp_minus_free": scalar(11),
            "grasp_auc": scalar(12),
        },
    }
    exact_episodes = summary["episodes_completed"] == episodes_expected
    free_saturation_passed = summary["free_saturation_rate"] is not None and summary["free_saturation_rate"] < 0.10
    separation = summary["second_largest_load"]["grasp_minus_free"]
    separation_passed = separation is not None and separation > 0.10
    grasp_auc = summary["second_largest_load"]["grasp_auc"]
    auc_passed = grasp_auc is not None and grasp_auc > 0.75
    gates = {
        "reset_zero": summary["reset_zero"],
        "finite_loads": summary["finite_loads"],
        "exact_episode_count": exact_episodes,
        "free_saturation_below_10pct": free_saturation_passed,
        "grasp_second_load_margin_above_0_1": separation_passed,
        "grasp_auc_above_0_75": auc_passed,
    }
    gates["passed"] = all(gates.values())
    summary["gates"] = gates
    summary["passed"] = gates["passed"]
    return summary


def _mask_inactive_action(action: Any, state: Any, active: Any) -> Any:
    import torch

    hold = torch.cat((state[:, 7:16], state[:, 16:26]), dim=-1).to(dtype=action.dtype)
    return torch.where(active[:, None], action, hold)


def _load_frozen_dp(checkpoint: Path, device: Any) -> tuple[Any, Any, Any, dict[str, Any], str]:
    from tools.train_newton_groot_residual_ppo import _load_frozen_dp as load_frozen_dp  # noqa: PLC0415

    return load_frozen_dp(checkpoint, device)


def _run_probe(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from teleop_stack.envs import (  # noqa: PLC0415
        GrootDiffusionPolicyEnv,
        GrootNewtonEnv,
        GrootNewtonEnvConfig,
    )

    if args.num_envs < 1 or args.episodes < 1 or args.episodes > args.num_envs:
        raise ValueError("episodes must be in [1, num_envs]")
    if args.max_episode_steps < 1 or args.inference_steps < 1:
        raise ValueError("max_episode_steps and inference_steps must be positive")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Finger-root load probe requires CUDA")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    frozen_dp, dp_config, scheduler, _stats, checkpoint_sha256 = _load_frozen_dp(args.checkpoint, device)
    if dp_config.pred_horizon < _CHUNK_HORIZON:
        raise ValueError(f"chunk8 probe requires pred_horizon >= {_CHUNK_HORIZON}")
    config = GrootNewtonEnvConfig(
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
    )
    base_env = GrootNewtonEnv(config)
    env = GrootDiffusionPolicyEnv(base_env, obs_horizon=dp_config.obs_horizon, action_horizon=1)
    load_samples: list[Any] = []
    contact_samples: list[Any] = []
    grasp_samples: list[Any] = []
    active_samples: list[Any] = []
    try:
        observation, _ = env.reset(seed=args.seed)
        reset_zero = (observation[_LOAD_KEY][:, -1] == 0.0).all()
        active = torch.arange(args.num_envs, device=device) < args.episodes
        completed = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        cached_chunk = None
        with torch.inference_mode():
            for control_step in range(args.max_episode_steps):
                row = control_step % _CHUNK_HORIZON
                if row == 0:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bfloat16):
                        cached_chunk = frozen_dp.predict_action(
                            observation,
                            scheduler,
                            inference_steps=args.inference_steps,
                            generator=generator,
                        )
                if cached_chunk is None:
                    raise RuntimeError("DP action chunk was not initialized")
                state = observation[_STATE_KEY][:, -1].float()
                action = _mask_inactive_action(cached_chunk[:, row].float(), state, active)
                next_observation, _, terminated, truncated, info = env.step(action)
                load = next_observation[_LOAD_KEY][:, -1].float()
                load_samples.append(load.clone())
                contact_samples.append(info["had_hand_contact_this_control_step"].bool().clone())
                grasp_samples.append(info["grasp_confirmed"].bool().clone())
                active_samples.append(active.clone())
                first_terminal = active & (terminated.bool() | truncated.bool())
                completed.logical_or_(first_terminal)
                active.logical_and_(~first_terminal)
                observation = next_observation
        sample_mask = torch.cat(active_samples, dim=0)
        loads = torch.cat(load_samples, dim=0)[sample_mask]
        contact = torch.cat(contact_samples, dim=0)[sample_mask]
        grasp = torch.cat(grasp_samples, dim=0)[sample_mask]
        summary = _summarize_samples(
            loads,
            contact,
            grasp,
            reset_zero=reset_zero,
            episodes_completed=completed.sum(),
            episodes_expected=args.episodes,
        )
        metadata = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "config": {
                "num_envs": args.num_envs,
                "episodes": args.episodes,
                "max_episode_steps": args.max_episode_steps,
                "inference_steps": args.inference_steps,
                "seed": args.seed,
                "execution": "cached_chunk_rows_0_through_7",
            },
            "finger_root_load": base_env.finger_root_load_metadata,
        }
        return summary, metadata
    finally:
        env.close()


def main() -> int:
    args = create_parser().parse_args()
    summary, metadata = _run_probe(args)
    output = {**metadata, "results": summary, "passed": summary["passed"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        f"finger-root-load probe: episodes={summary['episodes_completed']}/{summary['episodes_expected']} "
        f"free_sat={summary['free_saturation_rate']} "
        f"margin={summary['second_largest_load']['grasp_minus_free']} "
        f"auc={summary['second_largest_load']['grasp_auc']} passed={summary['passed']}"
    )
    print(f"summary: {output_path}")
    return 0 if summary["passed"] or not args.strict_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
