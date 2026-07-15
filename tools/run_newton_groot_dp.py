#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a trained dual-camera Diffusion Policy in the batched Newton task."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from teleop_stack.envs import GrootDiffusionPolicyEnv, GrootNewtonEnv, GrootNewtonEnvConfig
from teleop_stack.policies import GrootDiffusionPolicy, GrootDiffusionPolicyConfig


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
    config = GrootDiffusionPolicyConfig(**checkpoint["config"])
    state = checkpoint["model"]
    model = GrootDiffusionPolicy(
        state_min=state.get("state_min", np.zeros(config.state_dim, dtype=np.float32)),
        state_max=state.get("state_max", np.ones(config.state_dim, dtype=np.float32)),
        action_min=state.get("action_min", np.zeros(config.action_dim, dtype=np.float32)),
        action_max=state.get("action_max", np.ones(config.action_dim, dtype=np.float32)),
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
