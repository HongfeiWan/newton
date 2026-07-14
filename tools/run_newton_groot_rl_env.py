#!/usr/bin/env python3
"""Run the headless, batched dual Nero + L10 reinforcement-learning environment."""

from __future__ import annotations

import argparse
import time

import warp as wp

from teleop_stack.envs import GrootNewtonEnv, GrootNewtonEnvConfig


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--control-hz", type=int, default=10)
    parser.add_argument("--simulation-hz", type=int, default=60)
    parser.add_argument("--substeps-per-frame", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--ego-width", type=int, default=320)
    parser.add_argument("--ego-height", type=int, default=180)
    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--rigid-contacts-per-env", type=int, default=1024)
    parser.add_argument("--mujoco-njmax", type=int, default=2048)
    parser.add_argument("--mujoco-nconmax", type=int, default=1024)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--no-camera-textures", action="store_true")
    parser.add_argument("--no-scene-visuals", action="store_true")
    parser.add_argument("--no-hydroelastic", action="store_true")
    parser.add_argument("--no-capture-graph", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    config = GrootNewtonEnvConfig(
        num_envs=args.num_envs,
        device=args.device,
        control_hz=args.control_hz,
        simulation_hz=args.simulation_hz,
        substeps_per_frame=args.substeps_per_frame,
        max_episode_steps=args.max_episode_steps,
        capture_graph=not args.no_capture_graph,
        render_images=not args.no_images,
        camera_textures=not args.no_camera_textures,
        load_scene_visuals=not args.no_scene_visuals,
        hydroelastic_contacts=not args.no_hydroelastic,
        ego_width=args.ego_width,
        ego_height=args.ego_height,
        wrist_width=args.wrist_width,
        wrist_height=args.wrist_height,
        rigid_contacts_per_env=args.rigid_contacts_per_env,
        mujoco_njmax=args.mujoco_njmax,
        mujoco_nconmax=args.mujoco_nconmax,
    )
    env = GrootNewtonEnv(config)
    try:
        observation, _ = env.reset()
        action = env.hold_action()
        wp.synchronize_device(env.device)
        start = time.perf_counter()
        for _ in range(max(0, args.steps)):
            observation, _, _, _, _ = env.step(action)
        wp.synchronize_device(env.device)
        elapsed = time.perf_counter() - start
        steps_per_second = float(args.steps) / elapsed if elapsed > 0.0 else 0.0
        shapes = {
            group: {name: tuple(value.shape) for name, value in values.items()} for group, values in observation.items()
        }
        print(
            f"Completed {args.steps} batched steps across {args.num_envs} worlds "
            f"in {elapsed:.3f}s ({steps_per_second:.2f} env.step calls/s)."
        )
        print(f"GPU observation shapes: {shapes}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
