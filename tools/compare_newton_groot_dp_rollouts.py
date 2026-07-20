#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare frozen Groot DP online action-row execution modes in Newton."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MODES = ("index0", "index1", "chunk8")
_CHUNK_HORIZON = 8
_SUMMARY_SCHEMA_VERSION = 1
_STATE_KEY = "observation.state"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Frozen Diffusion Policy checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dp_index_compare"))
    parser.add_argument("--modes", nargs="+", choices=_MODES, default=list(_MODES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--inference-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-textures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hydroelastic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--strict-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit with status 2 unless all three modes and launch gates pass.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs < 1 or args.max_episode_steps < 1 or args.inference_steps < 1:
        raise ValueError("num_envs, max_episode_steps, and inference_steps must be positive")
    if len(args.modes) != len(set(args.modes)):
        raise ValueError("--modes cannot contain duplicates")
    if not str(args.device).startswith("cuda"):
        raise ValueError("Frozen DP rollout comparison requires a CUDA device")


def _validate_modes(modes: list[str] | tuple[str, ...], pred_horizon: int) -> None:
    unknown = set(modes).difference(_MODES)
    if unknown:
        raise ValueError(f"Unsupported modes: {sorted(unknown)}")
    required_horizon = _CHUNK_HORIZON if "chunk8" in modes else 2 if "index1" in modes else 1
    if pred_horizon < required_horizon:
        raise ValueError(
            f"Requested modes require pred_horizon >= {required_horizon}, got checkpoint pred_horizon={pred_horizon}"
        )


def _action_row(mode: str, control_step: int) -> int:
    if mode == "index0":
        return 0
    if mode == "index1":
        return 1
    if mode == "chunk8":
        return control_step % _CHUNK_HORIZON
    raise ValueError(f"Unsupported mode {mode!r}")


def _needs_policy_prediction(mode: str, control_step: int) -> bool:
    if mode not in _MODES:
        raise ValueError(f"Unsupported mode {mode!r}")
    return mode != "chunk8" or control_step % _CHUNK_HORIZON == 0


def _advance_first_terminal(active: Any, terminated: Any, truncated: Any) -> tuple[Any, Any]:
    """Return the newly completed lanes and the next active-lane mask."""
    first_terminal = active & (terminated | truncated)
    return first_terminal, active & ~first_terminal


def _done_reset_mask(terminated: Any, truncated: Any) -> Any:
    """Reset every terminal lane, including lanes whose episode was already counted."""
    return terminated | truncated


def _mask_inactive_action(action: Any, state: Any, active: Any) -> Any:
    """Replace inactive-lane DP actions with same-frame absolute hold targets."""
    import torch

    hold_action = torch.cat((state[:, 7:16], state[:, 16:26]), dim=-1).to(dtype=action.dtype)
    return torch.where(active.unsqueeze(-1), action, hold_action)


def _masked_sum(value: Any, mask: Any) -> Any:
    """Sum active values without propagating non-finite inactive values."""
    import torch

    return torch.where(mask, value, torch.zeros((), dtype=value.dtype, device=value.device)).sum(dtype=torch.float64)


def _accumulate_collision_buffer(
    accumulators: dict[str, Any],
    *,
    prefix: str,
    frame_max: Any,
    overflow_frame_count: Any,
    overflow_excess_count: Any,
) -> None:
    import torch

    frame_max_int64 = frame_max.to(dtype=torch.int64)
    overflow_frames_int64 = overflow_frame_count.to(dtype=torch.int64)
    accumulators[f"{prefix}_overflow_step_count"].add_((overflow_frames_int64 > 0).to(dtype=torch.int64))
    accumulators[f"{prefix}_overflow_frame_count"].add_(overflow_frames_int64)
    accumulators[f"{prefix}_overflow_excess_count"].add_(overflow_excess_count.to(dtype=torch.int64))
    torch.maximum(
        accumulators[f"{prefix}_max_observed_count"],
        frame_max_int64,
        out=accumulators[f"{prefix}_max_observed_count"],
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_dp(checkpoint_path: Path, device: Any) -> tuple[Any, Any, Any, dict[str, Any], str]:
    # Keep checkpoint validation identical to residual PPO without making this
    # standalone tool depend on how it selects a base-action row.
    try:
        from tools.train_newton_groot_residual_ppo import _load_frozen_dp as load_frozen_dp  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name not in {"tools", "tools.train_newton_groot_residual_ppo"}:
            raise
        from train_newton_groot_residual_ppo import _load_frozen_dp as load_frozen_dp  # noqa: PLC0415

    return load_frozen_dp(checkpoint_path, device)


def _mode_result_from_gpu(
    accumulators: dict[str, Any],
    *,
    mode: str,
    num_envs: int,
    max_episode_steps: int,
    policy_calls: int,
    env_step_calls: int,
    elapsed_s: float,
    rigid_contact_capacity: int,
    triangle_pair_capacity: int,
    triangle_pair_buffer_available: bool,
) -> dict[str, Any]:
    import torch

    scalar_names = (
        "episodes_completed",
        "terminated_count",
        "truncated_count",
        "success_episode_count",
        "fail_episode_count",
        "contact_episode_count",
        "grasp_episode_count",
        "grasp_confirmed_episode_count",
        "transport_episode_count",
        "lift_episode_count",
        "released_episode_count",
        "return_sum",
        "length_sum",
        "active_control_steps",
        "xyz_command_displacement_sum_m",
        "finite_action_sample_count",
        "action_sample_count",
        "rigid_contact_overflow_step_count",
        "rigid_contact_overflow_frame_count",
        "rigid_contact_overflow_excess_count",
        "rigid_contact_max_observed_count",
        "triangle_pair_overflow_step_count",
        "triangle_pair_overflow_frame_count",
        "triangle_pair_overflow_excess_count",
        "triangle_pair_max_observed_count",
    )
    packed = torch.stack(tuple(accumulators[name].to(dtype=torch.float64) for name in scalar_names))
    values = packed.detach().cpu().tolist()
    value = dict(zip(scalar_names, values, strict=True))
    episodes = int(value["episodes_completed"])
    action_samples = int(value["action_sample_count"])
    finite_action_samples = int(value["finite_action_sample_count"])
    mean_return = value["return_sum"] / max(episodes, 1)
    mean_length = value["length_sum"] / max(episodes, 1)
    mean_xyz_displacement = value["xyz_command_displacement_sum_m"] / max(action_samples, 1)
    result = {
        "mode": mode,
        "episodes_expected": num_envs,
        "episodes_completed": episodes,
        "terminated_count": int(value["terminated_count"]),
        "truncated_count": int(value["truncated_count"]),
        "success_episode_count": int(value["success_episode_count"]),
        "fail_episode_count": int(value["fail_episode_count"]),
        "contact_episode_count": int(value["contact_episode_count"]),
        "grasp_episode_count": int(value["grasp_episode_count"]),
        "grasp_confirmed_episode_count": int(value["grasp_confirmed_episode_count"]),
        "transport_episode_count": int(value["transport_episode_count"]),
        "lift_episode_count": int(value["lift_episode_count"]),
        "released_episode_count": int(value["released_episode_count"]),
        "success_rate": int(value["success_episode_count"]) / max(episodes, 1),
        "fail_rate": int(value["fail_episode_count"]) / max(episodes, 1),
        "mean_return": mean_return,
        "mean_episode_length": mean_length,
        "mean_xyz_command_displacement_m": mean_xyz_displacement,
        "action_sample_count": action_samples,
        "finite_action_sample_count": finite_action_samples,
        "active_control_steps": int(value["active_control_steps"]),
        "max_episode_steps": max_episode_steps,
        "env_step_calls": env_step_calls,
        "policy_calls": policy_calls,
        "elapsed_s": elapsed_s,
        "active_control_steps_per_s": int(value["active_control_steps"]) / max(elapsed_s, 1.0e-12),
        "collision_buffers": {
            "rigid_contacts": {
                "available": True,
                "capacity": rigid_contact_capacity,
                "max_observed_count": int(value["rigid_contact_max_observed_count"]),
                "overflow_step_count": int(value["rigid_contact_overflow_step_count"]),
                "overflow_frame_count": int(value["rigid_contact_overflow_frame_count"]),
                "overflow_excess_count": int(value["rigid_contact_overflow_excess_count"]),
            },
            "triangle_pairs": {
                "available": triangle_pair_buffer_available,
                "capacity": triangle_pair_capacity,
                "max_observed_count": int(value["triangle_pair_max_observed_count"]),
                "overflow_step_count": int(value["triangle_pair_overflow_step_count"]),
                "overflow_frame_count": int(value["triangle_pair_overflow_frame_count"]),
                "overflow_excess_count": int(value["triangle_pair_overflow_excess_count"]),
            },
        },
    }
    finite_names = (
        "success_rate",
        "fail_rate",
        "mean_return",
        "mean_episode_length",
        "mean_xyz_command_displacement_m",
        "elapsed_s",
        "active_control_steps_per_s",
    )
    finite_values = all(math.isfinite(float(result[name])) for name in finite_names)
    result["finite"] = finite_values and finite_action_samples == action_samples
    for name in finite_names:
        if not math.isfinite(float(result[name])):
            result[name] = None
    return result


def _evaluate_gates(
    results: dict[str, dict[str, Any]],
    *,
    expected_episodes: int,
    runtime_errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    runtime_errors = runtime_errors or {}
    required_modes_present = all(mode in results for mode in _MODES)
    exact_episode_count = required_modes_present and all(
        result["episodes_completed"] == expected_episodes for result in results.values()
    )
    finite_results = required_modes_present and all(bool(result["finite"]) for result in results.values())
    collision_buffer_status = {
        mode: {
            name: {
                "available": bool(buffer.get("available", False)),
                "capacity": buffer.get("capacity"),
                "max_observed_count": buffer.get("max_observed_count"),
                "overflow_step_count": buffer["overflow_step_count"],
                "overflow_frame_count": buffer.get("overflow_frame_count"),
                "overflow_excess_count": buffer["overflow_excess_count"],
            }
            for name, buffer in result.get("collision_buffers", {}).items()
        }
        for mode, result in results.items()
    }
    collision_buffers_clean = required_modes_present and all(
        set(result.get("collision_buffers", {})) == {"rigid_contacts", "triangle_pairs"}
        and all(
            bool(buffer.get("available", False))
            and int(buffer.get("overflow_step_count", -1)) == 0
            and int(buffer.get("overflow_excess_count", -1)) == 0
            for buffer in result["collision_buffers"].values()
        )
        for result in results.values()
    )

    index0_motion = results.get("index0", {}).get("mean_xyz_command_displacement_m")
    index1_motion = results.get("index1", {}).get("mean_xyz_command_displacement_m")
    motion_threshold = max(2.0 * float(index0_motion), 0.0005) if index0_motion is not None else None
    index1_motion_passed = (
        index1_motion is not None and motion_threshold is not None and float(index1_motion) > motion_threshold
    )

    index1_contacts = results.get("index1", {}).get("contact_episode_count")
    index1_contact_passed = index1_contacts is not None and int(index1_contacts) > 0

    index1 = results.get("index1", {})
    chunk8 = results.get("chunk8", {})
    chunk8_grasp_without_index1 = (
        int(chunk8.get("grasp_episode_count", 0)) > 0 and int(index1.get("grasp_episode_count", 0)) == 0
    ) or (
        int(chunk8.get("grasp_confirmed_episode_count", 0)) > 0
        and int(index1.get("grasp_confirmed_episode_count", 0)) == 0
    )
    chunk8_lift_without_index1 = (
        int(chunk8.get("lift_episode_count", 0)) > 0 and int(index1.get("lift_episode_count", 0)) == 0
    )
    receding_chunk_not_required = required_modes_present and not (
        chunk8_grasp_without_index1 or chunk8_lift_without_index1
    )

    gates = {
        "runtime_clean": {"passed": not runtime_errors, "errors": runtime_errors},
        "required_modes_present": {"passed": required_modes_present, "required": list(_MODES)},
        "exact_episode_count": {
            "passed": exact_episode_count,
            "expected_per_mode": expected_episodes,
            "actual_per_mode": {mode: result["episodes_completed"] for mode, result in results.items()},
        },
        "finite_results": {"passed": finite_results},
        "collision_buffers_clean": {
            "passed": collision_buffers_clean,
            "per_mode": collision_buffer_status,
        },
        "index1_motion": {
            "passed": index1_motion_passed,
            "index0_mean_m": index0_motion,
            "index1_mean_m": index1_motion,
            "strict_threshold_m": motion_threshold,
        },
        "index1_contact": {
            "passed": index1_contact_passed,
            "contact_episode_count": index1_contacts,
        },
        "receding_chunk_not_required": {
            "passed": receding_chunk_not_required,
            "chunk8_grasp_without_index1": chunk8_grasp_without_index1,
            "chunk8_lift_without_index1": chunk8_lift_without_index1,
        },
    }
    gates["passed"] = all(bool(gate["passed"]) for gate in gates.values())
    return gates


def _build_summary(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    config: dict[str, Any],
    results: dict[str, dict[str, Any]],
    runtime_errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    gates = _evaluate_gates(
        results,
        expected_episodes=int(config["num_envs"]),
        runtime_errors=runtime_errors,
    )
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "config": config,
        "results": results,
        "gates": gates,
        "passed": bool(gates["passed"]),
    }


def _write_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _run_mode(
    *,
    mode: str,
    frozen_dp: Any,
    scheduler: Any,
    dp_config: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    from teleop_stack.envs import GrootDiffusionPolicyEnv, GrootNewtonEnv, GrootNewtonEnvConfig  # noqa: PLC0415

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 1)
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
    base_env = GrootNewtonEnv(env_config)
    env = GrootDiffusionPolicyEnv(base_env, obs_horizon=dp_config.obs_horizon, action_horizon=1)
    device = torch.device(args.device)

    def bool_zeros() -> Any:
        return torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    def float_zero() -> Any:
        return torch.zeros((), dtype=torch.float64, device=device)

    def int_zero() -> Any:
        return torch.zeros((), dtype=torch.int64, device=device)

    active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    ever = {
        "contact": bool_zeros(),
        "grasp": bool_zeros(),
        "grasp_confirmed": bool_zeros(),
        "transport": bool_zeros(),
        "lift": bool_zeros(),
        "released": bool_zeros(),
        "success": bool_zeros(),
        "fail": bool_zeros(),
    }
    accumulators = {
        "episodes_completed": int_zero(),
        "terminated_count": int_zero(),
        "truncated_count": int_zero(),
        "success_episode_count": int_zero(),
        "fail_episode_count": int_zero(),
        "contact_episode_count": int_zero(),
        "grasp_episode_count": int_zero(),
        "grasp_confirmed_episode_count": int_zero(),
        "transport_episode_count": int_zero(),
        "lift_episode_count": int_zero(),
        "released_episode_count": int_zero(),
        "return_sum": float_zero(),
        "length_sum": float_zero(),
        "active_control_steps": int_zero(),
        "xyz_command_displacement_sum_m": float_zero(),
        "finite_action_sample_count": int_zero(),
        "action_sample_count": int_zero(),
        "rigid_contact_overflow_step_count": int_zero(),
        "rigid_contact_overflow_frame_count": int_zero(),
        "rigid_contact_overflow_excess_count": int_zero(),
        "rigid_contact_max_observed_count": int_zero(),
        "triangle_pair_overflow_step_count": int_zero(),
        "triangle_pair_overflow_frame_count": int_zero(),
        "triangle_pair_overflow_excess_count": int_zero(),
        "triangle_pair_max_observed_count": int_zero(),
    }
    policy_calls = 0
    env_step_calls = 0
    cached_chunk = None
    try:
        observation, reset_info = env.reset(seed=args.seed)
        reset_diagnostics = reset_info["control_step_diagnostics"]
        rigid_contact_capacity = int(reset_diagnostics["rigid_contact_capacity"])
        triangle_pair_capacity = int(reset_diagnostics["triangle_pair_capacity"])
        triangle_pair_buffer_available = bool(reset_diagnostics["triangle_pair_buffer_available"])
        torch.cuda.synchronize(device=device)
        start = time.perf_counter()
        with torch.inference_mode():
            for control_step in range(args.max_episode_steps):
                if _needs_policy_prediction(mode, control_step):
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=args.bfloat16,
                    ):
                        cached_chunk = frozen_dp.predict_action(
                            observation,
                            scheduler,
                            inference_steps=args.inference_steps,
                            generator=generator,
                        )
                    policy_calls += 1
                if cached_chunk is None:
                    raise RuntimeError("DP action chunk was not initialized")
                state = observation[_STATE_KEY][:, -1].float()
                selected_action = cached_chunk[:, _action_row(mode, control_step)].float()
                action = _mask_inactive_action(selected_action, state, active)
                state_xyz = state[:, 7:10]
                displacement = torch.linalg.vector_norm(action[:, :3] - state_xyz, dim=-1)
                active_count = active.sum(dtype=torch.int64)
                accumulators["active_control_steps"].add_(active_count)
                accumulators["action_sample_count"].add_(active_count)
                accumulators["finite_action_sample_count"].add_(
                    (torch.isfinite(action).all(dim=-1) & active).sum(dtype=torch.int64)
                )
                accumulators["xyz_command_displacement_sum_m"].add_(_masked_sum(displacement, active))

                next_observation, _, terminated, truncated, info = env.step(action)
                env_step_calls += 1
                diagnostics = info["control_step_diagnostics"]
                _accumulate_collision_buffer(
                    accumulators,
                    prefix="rigid_contact",
                    frame_max=diagnostics["rigid_contact_frame_max"][0],
                    overflow_frame_count=diagnostics["rigid_contact_overflow_frame_count"][0],
                    overflow_excess_count=diagnostics["rigid_contact_overflow_excess_count"][0],
                )
                _accumulate_collision_buffer(
                    accumulators,
                    prefix="triangle_pair",
                    frame_max=diagnostics["triangle_pair_frame_max"][0],
                    overflow_frame_count=diagnostics["triangle_pair_overflow_frame_count"][0],
                    overflow_excess_count=diagnostics["triangle_pair_overflow_excess_count"][0],
                )
                for name, info_name in (
                    ("contact", "had_hand_contact_this_control_step"),
                    ("grasp", "is_grasped"),
                    ("grasp_confirmed", "grasp_confirmed"),
                    ("transport", "transport_started"),
                    ("lift", "is_lifted"),
                    ("released", "released"),
                    ("success", "success"),
                    ("fail", "fail"),
                ):
                    ever[name].logical_or_(info[info_name].bool() & active)

                done = _done_reset_mask(terminated.bool(), truncated.bool())
                first_terminal, active = _advance_first_terminal(active, terminated.bool(), truncated.bool())
                accumulators["episodes_completed"].add_(first_terminal.sum(dtype=torch.int64))
                accumulators["terminated_count"].add_((first_terminal & terminated).sum(dtype=torch.int64))
                accumulators["truncated_count"].add_((first_terminal & truncated).sum(dtype=torch.int64))
                for name in (
                    "contact",
                    "grasp",
                    "grasp_confirmed",
                    "transport",
                    "lift",
                    "released",
                    "success",
                    "fail",
                ):
                    accumulator_name = f"{name}_episode_count"
                    accumulators[accumulator_name].add_((ever[name] & first_terminal).sum(dtype=torch.int64))
                episode = info["episode"]
                accumulators["return_sum"].add_(_masked_sum(episode["return"], first_terminal))
                accumulators["length_sum"].add_(_masked_sum(episode["length"].to(dtype=torch.float32), first_terminal))
                done_any, active_any = torch.stack((done.any(), active.any())).cpu().tolist()
                if done_any:
                    observation, _ = env.reset(world_mask=done)
                else:
                    observation = next_observation
                if not active_any:
                    break
        torch.cuda.synchronize(device=device)
        elapsed_s = time.perf_counter() - start
        return _mode_result_from_gpu(
            accumulators,
            mode=mode,
            num_envs=args.num_envs,
            max_episode_steps=args.max_episode_steps,
            policy_calls=policy_calls,
            env_step_calls=env_step_calls,
            elapsed_s=elapsed_s,
            rigid_contact_capacity=rigid_contact_capacity,
            triangle_pair_capacity=triangle_pair_capacity,
            triangle_pair_buffer_available=triangle_pair_buffer_available,
        )
    finally:
        env.close()


def main() -> int:
    args = create_parser().parse_args()
    _validate_args(args)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Frozen DP rollout comparison requires CUDA")
    device = torch.device(args.device)
    frozen_dp, dp_config, scheduler, _stats, checkpoint_sha256 = _load_frozen_dp(args.checkpoint, device)
    _validate_modes(args.modes, dp_config.pred_horizon)

    results: dict[str, dict[str, Any]] = {}
    runtime_errors: dict[str, str] = {}
    for mode in args.modes:
        try:
            result = _run_mode(
                mode=mode,
                frozen_dp=frozen_dp,
                scheduler=scheduler,
                dp_config=dp_config,
                args=args,
            )
            results[mode] = result
            print(
                f"{mode}: episodes={result['episodes_completed']}/{result['episodes_expected']} "
                f"xyz_command={result['mean_xyz_command_displacement_m']:.6f}m "
                f"contact={result['contact_episode_count']} grasp={result['grasp_confirmed_episode_count']} "
                f"lift={result['lift_episode_count']} success={result['success_episode_count']}"
            )
        except Exception as exc:
            runtime_errors[mode] = f"{type(exc).__name__}: {exc}"
            break

    config = {
        "modes": list(args.modes),
        "device": args.device,
        "num_envs": args.num_envs,
        "max_episode_steps": args.max_episode_steps,
        "inference_steps": args.inference_steps,
        "seed": args.seed,
        "camera_textures": args.camera_textures,
        "scene_visuals": args.scene_visuals,
        "capture_graph": args.capture_graph,
        "hydroelastic": args.hydroelastic,
        "bfloat16": args.bfloat16,
        "pred_horizon": dp_config.pred_horizon,
        "chunk_horizon": _CHUNK_HORIZON,
    }
    summary = _build_summary(
        checkpoint=args.checkpoint,
        checkpoint_sha256=checkpoint_sha256 or _file_sha256(args.checkpoint),
        config=config,
        results=results,
        runtime_errors=runtime_errors,
    )
    summary_path = _write_summary(summary, args.output_dir)
    print(f"Wrote {summary_path}; strict gates passed={summary['passed']}")
    if runtime_errors:
        return 1
    if args.strict_gates and not summary["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
