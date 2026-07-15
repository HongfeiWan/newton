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
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument(
        "--obs-mode", choices=("state", "state_dict", "rgb", "state_dict+rgb", "policy"), default="state_dict+rgb"
    )
    parser.add_argument(
        "--control-mode",
        choices=("pd_eef_pose_abs", "pd_joint_pos", "pd_joint_delta_pos"),
        default="pd_eef_pose_abs",
    )
    parser.add_argument(
        "--reward-mode", choices=("none", "sparse", "dense", "normalized_dense"), default="normalized_dense"
    )
    parser.add_argument("--arm-action-delta", type=float, default=0.1)
    parser.add_argument("--hand-action-delta", type=float, default=0.1)
    parser.add_argument("--ik-iterations", type=int, default=4)
    parser.add_argument("--ik-damping-lambda", type=float, default=0.02)
    parser.add_argument("--ik-max-joint-step-rad", type=float, default=0.045)
    parser.add_argument("--hand-max-joint-step-rad", type=float, default=0.08)
    parser.add_argument("--bottle-lift-height", type=float, default=0.1)
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
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        arm_action_delta=args.arm_action_delta,
        hand_action_delta=args.hand_action_delta,
        ik_iterations=args.ik_iterations,
        ik_damping_lambda=args.ik_damping_lambda,
        ik_max_joint_step_rad=args.ik_max_joint_step_rad,
        hand_max_joint_step_rad=args.hand_max_joint_step_rad,
        bottle_lift_height=args.bottle_lift_height,
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

        def shapes(value):
            if isinstance(value, dict):
                return {key: shapes(child) for key, child in value.items()}
            return tuple(value.shape)

        print(
            f"Completed {args.steps} batched steps across {args.num_envs} worlds "
            f"in {elapsed:.3f}s ({steps_per_second:.2f} env.step calls/s)."
        )
        print(f"GPU observation shapes: {shapes(observation)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
