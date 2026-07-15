#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train the dual-camera Nero + L10 Diffusion Policy from LeRobot data."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from teleop_stack.datasets import GrootLeRobotWindowDataset
from teleop_stack.policies import GrootDiffusionPolicy, GrootDiffusionPolicyConfig


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("local_data/groot/smooth"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/dp/groot_l10_pick"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--compile", action="store_true")
    return parser


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device=device, non_blocking=True) for key, value in batch.items()}


def _checkpoint_payload(model: Any, optimizer: Any, step: int, config: Any) -> dict[str, Any]:
    return {
        "format": "teleop_stack.groot_l10_diffusion_policy.v1",
        "step": int(step),
        "config": asdict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def main() -> None:
    args = create_parser().parse_args()
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: PLC0415

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("This trainer is GPU-first and requires a CUDA device")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = GrootLeRobotWindowDataset(
        args.dataset,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    config = GrootDiffusionPolicyConfig(obs_horizon=args.obs_horizon, pred_horizon=args.pred_horizon)
    model = GrootDiffusionPolicy(
        state_min=dataset.stats.state_min,
        state_max=dataset.stats.state_max,
        action_min=dataset.stats.action_min,
        action_max=dataset.stats.action_max,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion_train_steps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
    checkpoint_model = model
    if args.compile:
        model = torch.compile(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    running_loss = torch.zeros((), dtype=torch.float32, device=device)
    print_start = time.perf_counter()
    try:
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model.compute_loss(batch, scheduler)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss.add_(loss.detach().float())

            if step % args.print_every == 0:
                elapsed = time.perf_counter() - print_start
                samples_per_second = args.print_every * args.batch_size / max(elapsed, 1.0e-6)
                print(
                    f"step={step} loss={float(running_loss) / args.print_every:.6f} samples/s={samples_per_second:.2f}",
                    flush=True,
                )
                running_loss.zero_()
                print_start = time.perf_counter()
            if step % args.save_every == 0 or step == args.steps:
                checkpoint_path = args.output_dir / f"checkpoint_{step:08d}.pt"
                torch.save(_checkpoint_payload(checkpoint_model, optimizer, step, config), checkpoint_path)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
