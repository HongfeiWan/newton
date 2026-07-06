#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop_stack.paths import resolve_isaac_teleop_root


def _default_output_path(isaac_teleop_root: Path) -> Path:
    return isaac_teleop_root / "examples" / "camera_streamer" / "build" / "newton_sim_screen_xr.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a minimal IsaacTeleop camera_streamer XR config for the Newton sim screen.",
    )
    parser.add_argument("--isaac-teleop-root", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="/dev/video44")
    parser.add_argument("--name", default="sim_screen")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--fps",
        type=int,
        default=0,
        help="Consumer-side FPS for camera_streamer. 0 keeps the V4L2 loopback producer-paced.",
    )
    parser.add_argument("--plane-distance", type=float, default=1.6)
    parser.add_argument("--plane-width", type=float, default=1.2)
    parser.add_argument("--plane-offset-x", type=float, default=0.0)
    parser.add_argument("--plane-offset-y", type=float, default=0.0)
    parser.add_argument("--lock-mode", choices=("lazy", "world", "head"), default="lazy")
    parser.add_argument("--look-away-angle", type=float, default=45.0)
    parser.add_argument("--reposition-distance", type=float, default=0.5)
    parser.add_argument("--reposition-delay", type=float, default=0.5)
    parser.add_argument("--transition-duration", type=float, default=0.3)
    parser.add_argument("--cuda-device", type=int, default=0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    isaac_root = resolve_isaac_teleop_root(args.isaac_teleop_root)
    output = args.output.resolve() if args.output else _default_output_path(isaac_root)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": "local",
        "cameras": {
            args.name: {
                "enabled": True,
                "type": "v4l2",
                "stereo": False,
                "device": args.device,
                "width": int(args.width),
                "height": int(args.height),
                "fps": int(args.fps),
            },
        },
        "display": {
            "mode": "xr",
            "cuda_device": int(args.cuda_device),
            "monitor": {
                "width": int(args.width),
                "height": int(args.height),
                "title": "Newton sim screen",
                "padding": 4,
                "stream_timeout": 2.0,
            },
            "xr": {
                "planes": {
                    args.name: {
                        "distance": float(args.plane_distance),
                        "width": float(args.plane_width),
                        "offset_x": float(args.plane_offset_x),
                        "offset_y": float(args.plane_offset_y),
                    },
                },
                "lock_mode": args.lock_mode,
                "look_away_angle": float(args.look_away_angle),
                "reposition_distance": float(args.reposition_distance),
                "reposition_delay": float(args.reposition_delay),
                "transition_duration": float(args.transition_duration),
            },
        },
    }

    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
