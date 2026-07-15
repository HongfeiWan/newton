#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train the dual-camera Nero + L10 Diffusion Policy from LeRobot data."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from teleop_stack.datasets import GrootLeRobotWindowDataset, create_groot_lerobot_bc_split
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
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--validate-every", type=int, default=5_000)
    parser.add_argument("--validation-batches", type=int, default=0, help="Zero validates on the complete split")
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--compile", action="store_true")
    return parser


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device=device, non_blocking=True) for key, value in batch.items()}


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    step: int,
    config: Any,
    dataset_split: Any,
    best_validation_loss: float,
) -> dict[str, Any]:
    return {
        "format": "teleop_stack.groot_l10_diffusion_policy.v1",
        "step": int(step),
        "config": asdict(config),
        "dataset_split": asdict(dataset_split),
        "best_validation_loss": float(best_validation_loss),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def _validate(model: Any, loader: Any, scheduler: Any, device: Any, *, seed: int, max_batches: int) -> float:
    import torch

    model.eval()
    total_loss = 0.0
    total_samples = 0
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]), torch.no_grad():
        torch.manual_seed(seed)
        for batch_index, cpu_batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = _to_device(cpu_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model.compute_loss(batch, scheduler)
            batch_size = int(batch["action"].shape[0])
            total_loss += float(loss) * batch_size
            total_samples += batch_size
    model.train()
    if total_samples == 0:
        raise RuntimeError("Validation loader did not produce any samples")
    return total_loss / total_samples


def main() -> None:
    args = create_parser().parse_args()
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: PLC0415

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("This trainer is GPU-first and requires a CUDA device")
    if args.validate_every < 1 or args.validation_batches < 0:
        raise ValueError("validate_every must be positive and validation_batches must be non-negative")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset_split = create_groot_lerobot_bc_split(
        args.dataset,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    train_dataset = GrootLeRobotWindowDataset(
        args.dataset,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        episode_indices=dataset_split.train_episode_indices,
    )
    validation_dataset = GrootLeRobotWindowDataset(
        args.dataset,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        episode_indices=dataset_split.validation_episode_indices,
        stats=train_dataset.stats,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    config = GrootDiffusionPolicyConfig(obs_horizon=args.obs_horizon, pred_horizon=args.pred_horizon)
    model = GrootDiffusionPolicy(
        state_min=train_dataset.stats.state_min,
        state_max=train_dataset.stats.state_max,
        action_min=train_dataset.stats.action_min,
        action_max=train_dataset.stats.action_max,
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
    best_validation_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("dataset_split") != asdict(dataset_split):
            raise ValueError("Resume checkpoint dataset split does not match the current dataset metadata and options")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint.get("best_validation_loss", best_validation_loss))
    checkpoint_model = model
    if args.compile:
        model = torch.compile(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_split.json").write_text(
        json.dumps(asdict(dataset_split), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"dataset source={dataset_split.source_episode_count} episodes/{dataset_split.source_frame_count} frames "
        f"excluded_failed={len(dataset_split.excluded_unsuccessful_episode_indices)} "
        f"excluded_duplicates={len(dataset_split.excluded_duplicate_episode_indices)} "
        f"train={len(train_dataset.episodes)} episodes/{len(train_dataset)} frames "
        f"validation={len(validation_dataset.episodes)} episodes/{len(validation_dataset)} frames",
        flush=True,
    )
    iterator = iter(train_loader)
    running_loss = torch.zeros((), dtype=torch.float32, device=device)
    print_start = time.perf_counter()
    try:
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
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
            if step % args.validate_every == 0 or step == args.steps:
                validation_loss = _validate(
                    model,
                    validation_loader,
                    scheduler,
                    device,
                    seed=args.seed + 1,
                    max_batches=args.validation_batches,
                )
                print(f"step={step} validation_loss={validation_loss:.6f}", flush=True)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    torch.save(
                        _checkpoint_payload(
                            checkpoint_model,
                            optimizer,
                            step,
                            config,
                            dataset_split,
                            best_validation_loss,
                        ),
                        args.output_dir / "best.pt",
                    )
            if step % args.save_every == 0 or step == args.steps:
                checkpoint_path = args.output_dir / f"checkpoint_{step:08d}.pt"
                torch.save(
                    _checkpoint_payload(
                        checkpoint_model,
                        optimizer,
                        step,
                        config,
                        dataset_split,
                        best_validation_loss,
                    ),
                    checkpoint_path,
                )
    finally:
        train_dataset.close()
        validation_dataset.close()


if __name__ == "__main__":
    main()
