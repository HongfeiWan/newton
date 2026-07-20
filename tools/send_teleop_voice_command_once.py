from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop_stack.session.voice_controls import VoiceTeleopCommand


def _default_voice_udp_port() -> int:
    for name in ("TELEOP_QUEST_VOICE_UDP_PORT", "TELEOP_VOICE_UDP_PORT"):
        value = os.environ.get(name)
        if value:
            return int(value)
    return 9910


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send one host-side teleop voice command packet to the local session control socket.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="UDP host of the local teleop voice control receiver.")
    parser.add_argument(
        "--port",
        type=int,
        default=_default_voice_udp_port(),
        help="UDP port of the local teleop voice control receiver.",
    )
    parser.add_argument(
        "--command",
        required=True,
        choices=["engage", "clutch", "resume", "recenter", "stop", "disengage", "exit", "estop"],
        help="Voice control command to send.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Send the packet as a JSON object instead of a plain text command word.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    command: VoiceTeleopCommand = args.command

    if args.json:
        payload = json.dumps({"command": command}, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    else:
        payload = command.encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(payload, (args.host, args.port))
    finally:
        sock.close()

    print("send_teleop_voice_command_once: PASS")
    print(f"udp_target=udp://{args.host}:{args.port}")
    print(f"command={command}")
    print(f"encoding={'json' if args.json else 'text'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
