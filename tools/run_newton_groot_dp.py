#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a trained dual-camera Diffusion Policy in the batched Newton task."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from teleop_stack.envs import GrootDiffusionPolicyEnv, GrootNewtonEnv, GrootNewtonEnvConfig
from teleop_stack.policies import GROOT_DP_CHECKPOINT_FORMAT, GrootDiffusionPolicy, GrootDiffusionPolicyConfig


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--policy-steps", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--no-camera-textures", action="store_true")
    parser.add_argument("--no-hydroelastic", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: PLC0415

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != GROOT_DP_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Checkpoint format {checkpoint.get('format')!r} is incompatible with the canonical row-first "
            f"state-target environment; expected {GROOT_DP_CHECKPOINT_FORMAT!r}. Retrain from the repaired dataset."
        )
    config = GrootDiffusionPolicyConfig(**checkpoint["config"])
    if config.state_dim != 26 or config.action_dim != 19:
        raise ValueError(
            f"Expected checkpoint state/action dimensions 26/19, got {config.state_dim}/{config.action_dim}"
        )
    if not 1 <= args.action_horizon <= config.pred_horizon:
        raise ValueError(f"action_horizon must be between 1 and checkpoint pred_horizon={config.pred_horizon}")
    state = checkpoint["model"]
    stats = checkpoint.get("train_dataset_stats")
    if not isinstance(stats, dict):
        raise ValueError("Checkpoint does not contain train_dataset_stats")
    canonical = json.dumps(stats, sort_keys=True, separators=(",", ":")).encode("utf-8")
    stats_sha256 = hashlib.sha256(canonical).hexdigest()
    if checkpoint.get("train_dataset_stats_sha256") != stats_sha256:
        raise ValueError("Checkpoint train statistics SHA-256 does not match its payload")
    for name, width in (("state_min", 26), ("state_max", 26), ("action_min", 19), ("action_max", 19)):
        values = np.asarray(stats.get(name), dtype=np.float32)
        if values.shape != (width,) or not np.isfinite(values).all():
            raise ValueError(f"Checkpoint {name} must be finite with shape ({width},)")
        model_values = state.get(name)
        if model_values is None or not np.array_equal(np.asarray(model_values.cpu()), values):
            raise ValueError(f"Checkpoint {name} payload does not match the model normalization buffer")
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
    model.eval().to(args.device)
    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion_train_steps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    base_env = GrootNewtonEnv(
        GrootNewtonEnvConfig(
            num_envs=args.num_envs,
            device=args.device,
            obs_mode="policy",
            control_mode="pd_eef_pose_abs",
            camera_textures=not args.no_camera_textures,
            hydroelastic_contacts=not args.no_hydroelastic,
        )
    )
    env = GrootDiffusionPolicyEnv(
        base_env,
        obs_horizon=config.obs_horizon,
        action_horizon=args.action_horizon,
    )
    try:
        observation, _ = env.reset()
        torch.cuda.synchronize(device=args.device)
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.policy_steps):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    action = model.predict_action(observation, scheduler, inference_steps=args.inference_steps)
                observation, _, terminated, truncated, _ = env.step(action[:, : args.action_horizon])
                done = terminated | truncated
                observation, _ = env.reset(world_mask=done)
        torch.cuda.synchronize(device=args.device)
        elapsed = time.perf_counter() - start
        print(
            f"Completed {args.policy_steps} DP calls across {args.num_envs} worlds in {elapsed:.3f}s "
            f"({args.policy_steps * args.num_envs / max(elapsed, 1.0e-6):.2f} policy-worlds/s)."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
