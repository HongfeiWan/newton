from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop_stack.session.quest_voice_bridge import (
    QuestVoiceCommandBridgeConfig,
    QuestVoiceCommandBridgeServer,
)


def _default_cloudxr_cert_dir() -> Path:
    return Path.home() / ".cloudxr" / "certs"


def _default_voice_udp_port() -> int:
    for name in ("TELEOP_QUEST_VOICE_UDP_PORT", "TELEOP_VOICE_UDP_PORT"):
        value = os.environ.get(name)
        if value:
            return int(value)
    return 9910


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive Quest browser microphone audio from the CloudXR 8443 reverse proxy, transcribe with Vosk, and forward teleop voice commands over UDP.",
    )
    parser.add_argument(
        "--asr-backend",
        choices=("vosk", "funasr"),
        default="vosk",
        help="Streaming ASR backend. vosk preserves the existing chain; funasr uses paraformer streaming plus FSMN-VAD.",
    )
    parser.add_argument("--model-path", default=None, help="Path to the local Vosk model directory.")
    parser.add_argument(
        "--bind-host", default="127.0.0.1", help="Host interface to bind for the local Quest voice WS server."
    )
    parser.add_argument("--bind-port", type=int, default=8766, help="Local WS port for the Quest voice uplink bridge.")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="PCM sample rate expected from the browser uplink."
    )
    parser.add_argument("--udp-host", default="127.0.0.1", help="Destination teleop voice UDP host.")
    parser.add_argument(
        "--udp-port", type=int, default=_default_voice_udp_port(), help="Destination teleop voice UDP port."
    )
    parser.add_argument("--cooldown-s", type=float, default=1.0, help="Minimum resend interval for the same command.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.98,
        help="Only forward recognized commands whose mean word confidence is at least this threshold.",
    )
    parser.add_argument(
        "--grammar-constrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Constrain Vosk decoding to command phrases. Default is off so non-command speech is not forced into a command.",
    )
    parser.add_argument(
        "--funasr-model",
        default="paraformer-zh-streaming",
        help="FunASR streaming ASR model name or local path for --asr-backend=funasr.",
    )
    parser.add_argument(
        "--funasr-vad-model",
        default="fsmn-vad",
        help="FunASR streaming VAD model name or local path for --asr-backend=funasr.",
    )
    parser.add_argument(
        "--funasr-device",
        default="cpu",
        help="FunASR device, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--enable-xr-status-fallback",
        action="store_true",
        help="Also publish simple XR status badges when no teleop session is consuming voice UDP commands.",
    )
    parser.add_argument(
        "--xr-status-path",
        default=None,
        help="Optional teleop_xr_status.json path for --enable-xr-status-fallback.",
    )
    parser.add_argument(
        "--ssl-cert",
        default=str(_default_cloudxr_cert_dir() / "server.crt"),
        help="TLS certificate path if you explicitly want this bridge to expose WSS directly instead of using the 8443 reverse proxy.",
    )
    parser.add_argument(
        "--ssl-key",
        default=str(_default_cloudxr_cert_dir() / "server.key"),
        help="TLS private key path if you explicitly want this bridge to expose WSS directly instead of using the 8443 reverse proxy.",
    )
    tls_group = parser.add_mutually_exclusive_group()
    tls_group.add_argument(
        "--tls",
        dest="tls",
        action="store_true",
        help="Enable direct WSS on this bridge. Only use this if you are not going through the CloudXR 8443 reverse proxy.",
    )
    tls_group.add_argument(
        "--no-tls",
        dest="tls",
        action="store_false",
        help="Serve plain local WS. This is the default and is recommended behind the CloudXR 8443 reverse proxy.",
    )
    parser.set_defaults(tls=False)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else None
    ssl_cert = None if not args.tls else Path(args.ssl_cert).expanduser().resolve()
    ssl_key = None if not args.tls else Path(args.ssl_key).expanduser().resolve()

    if args.tls:
        if not ssl_cert or not ssl_cert.is_file():
            raise SystemExit(f"SSL certificate not found: {ssl_cert}")
        if not ssl_key or not ssl_key.is_file():
            raise SystemExit(f"SSL key not found: {ssl_key}")

    if args.asr_backend == "vosk":
        if model_path is None:
            raise SystemExit("--model-path is required with --asr-backend=vosk.")
        if not model_path.is_dir():
            raise SystemExit(f"Vosk model path not found: {model_path}")

    server = QuestVoiceCommandBridgeServer(
        QuestVoiceCommandBridgeConfig(
            model_path=model_path,
            asr_backend=str(args.asr_backend),
            bind_host=args.bind_host,
            bind_port=int(args.bind_port),
            sample_rate_hz=int(args.sample_rate),
            udp_host=args.udp_host,
            udp_port=int(args.udp_port),
            cooldown_s=float(args.cooldown_s),
            min_confidence=float(args.min_confidence),
            ssl_cert_path=ssl_cert,
            ssl_key_path=ssl_key,
            enable_xr_status_fallback=bool(args.enable_xr_status_fallback),
            xr_status_path=args.xr_status_path,
            grammar_constrained=bool(args.grammar_constrained),
            funasr_model=str(args.funasr_model),
            funasr_vad_model=str(args.funasr_vad_model),
            funasr_device=str(args.funasr_device),
        )
    )
    asyncio.run(server.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
