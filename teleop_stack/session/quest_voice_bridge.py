from __future__ import annotations

import asyncio
import json
import shlex
import socket
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import websockets

from teleop_stack.session.voice_asr import (
    FunasrStreamingConfig,
    StreamingFunasrVoiceCommandRecognizer,
    StreamingVoskVoiceCommandRecognizer,
    VoskAsrConfig,
    confidence_meets_threshold,
)
from teleop_stack.session.xr_status import XrTeleopStatusPublisher


@dataclass(frozen=True)
class QuestVoiceCommandBridgeConfig:
    model_path: Path | None = None
    asr_backend: str = "vosk"
    bind_host: str = "127.0.0.1"
    bind_port: int = 8766
    sample_rate_hz: int = 16000
    udp_host: str = "127.0.0.1"
    udp_port: int = 9910
    cooldown_s: float = 1.0
    min_confidence: float = 0.98
    ssl_cert_path: Path | None = None
    ssl_key_path: Path | None = None
    enable_xr_status_fallback: bool = False
    xr_status_path: str | None = None
    grammar_constrained: bool = False
    funasr_model: str = "paraformer-zh-streaming"
    funasr_vad_model: str = "fsmn-vad"
    funasr_device: str = "cpu"


@dataclass
class _ConnectionState:
    recognizer: Any
    last_sent_at: dict[str, float] = field(default_factory=dict)


class QuestVoiceCommandBridgeServer:
    def __init__(self, config: QuestVoiceCommandBridgeConfig):
        self.config = config
        self._xr_status = (
            XrTeleopStatusPublisher(config.xr_status_path)
            if config.enable_xr_status_fallback
            else None
        )
        self._fallback_mode = "ready"

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        if self.config.ssl_cert_path is None or self.config.ssl_key_path is None:
            return None
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(
            certfile=str(self.config.ssl_cert_path),
            keyfile=str(self.config.ssl_key_path),
        )
        return context

    async def run_forever(self) -> None:
        import websockets.server

        ssl_context = self._build_ssl_context()
        scheme = "wss" if ssl_context is not None else "ws"
        print(
            "[quest-voice-bridge] starting "
            f"{scheme}://{self.config.bind_host}:{self.config.bind_port} "
            f"-> udp://{self.config.udp_host}:{self.config.udp_port}",
            flush=True,
        )
        print(
            "[quest-voice-bridge] commands: 开始/暂停/继续/重置/停止遥操/结束遥操/退出/急停/成功/失败/放弃/结束任务 "
            "(Quest mic -> same-origin WSS -> local WS bridge -> Vosk -> UDP)",
            flush=True,
        )
        print(
            f"[quest-voice-bridge] min_confidence={float(self.config.min_confidence):.3f}",
            flush=True,
        )
        print(
            f"[quest-voice-bridge] grammar_constrained={bool(self.config.grammar_constrained)}",
            flush=True,
        )
        print(
            f"[quest-voice-bridge] asr_backend={self.config.asr_backend}",
            flush=True,
        )
        if self.config.asr_backend == "funasr":
            print(
                "[quest-voice-bridge] funasr="
                f"model={self.config.funasr_model} "
                f"vad_model={self.config.funasr_vad_model} "
                f"device={self.config.funasr_device}",
                flush=True,
            )
        self._publish_status(lifecycle_event="session_started", force=True)

        async with websockets.serve(
            self._handle_connection,
            self.config.bind_host,
            self.config.bind_port,
            ssl=ssl_context,
            max_size=None,
        ):
            await asyncio.Future()

    async def _handle_connection(self, websocket) -> None:
        remote = getattr(websocket, "remote_address", None)
        print(f"[quest-voice-bridge] client_connected remote={remote}", flush=True)
        state = _ConnectionState(
            recognizer=self._build_recognizer()
        )

        try:
            try:
                async for message in websocket:
                    if isinstance(message, str):
                        self._handle_text_message(message)
                        continue

                    result = state.recognizer.accept_pcm16le(bytes(message))
                    if result is None or not result.transcript:
                        continue
                    self._handle_recognized_result(result=result, state=state)
            except websockets.exceptions.ConnectionClosed:
                pass
        finally:
            final_result = state.recognizer.finalize()
            if final_result is not None:
                self._handle_recognized_result(result=final_result, state=state)
            print(f"[quest-voice-bridge] client_disconnected remote={remote}", flush=True)

    def _build_recognizer(self):
        backend = str(self.config.asr_backend).strip().lower()
        if backend == "vosk":
            if self.config.model_path is None:
                raise RuntimeError("--model-path is required when --asr-backend=vosk.")
            return StreamingVoskVoiceCommandRecognizer(
                VoskAsrConfig(
                    model_path=self.config.model_path,
                    sample_rate_hz=self.config.sample_rate_hz,
                    grammar_constrained=bool(self.config.grammar_constrained),
                )
            )
        if backend == "funasr":
            return StreamingFunasrVoiceCommandRecognizer(
                FunasrStreamingConfig(
                    model=str(self.config.funasr_model),
                    vad_model=str(self.config.funasr_vad_model),
                    device=str(self.config.funasr_device),
                    sample_rate_hz=int(self.config.sample_rate_hz),
                )
            )
        raise RuntimeError(f"Unsupported ASR backend: {self.config.asr_backend}")

    def _handle_text_message(self, message: str) -> None:
        text = message.strip()
        if not text:
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            print(f"[quest-voice-bridge] control={shlex.quote(text)}", flush=True)
            return
        print(f"[quest-voice-bridge] control={payload}", flush=True)
        message_type = str(payload.get("type") or "").strip().lower()
        if message_type == "session_start":
            self._publish_status(lifecycle_event="session_started", force=True)

    def _handle_recognized_result(self, *, result, state: _ConnectionState) -> None:
        confidence_text = "n/a" if result.confidence is None else f"{result.confidence:.3f}"
        print(
            f"[quest-voice-bridge] transcript={shlex.quote(result.transcript)} "
            f"command={result.command or 'none'} conf={confidence_text}",
            flush=True,
        )
        if result.command is None:
            return
        if not confidence_meets_threshold(result.confidence, self.config.min_confidence):
            print(
                f"[quest-voice-bridge] filtered command={result.command} "
                f"reason=low_confidence threshold={float(self.config.min_confidence):.3f}",
                flush=True,
            )
            return

        now = time.monotonic()
        if now - state.last_sent_at.get(result.command, -1e9) < self.config.cooldown_s:
            return

        state.last_sent_at[result.command] = now
        self._send_udp_command(result.command)
        self._publish_status_for_command(result.command)
        print(f"[quest-voice-bridge] forwarded command={result.command}", flush=True)

    def _send_udp_command(self, command: str) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(command.encode("utf-8"), (self.config.udp_host, self.config.udp_port))
        finally:
            sock.close()

    def _publish_status_for_command(self, command: str) -> None:
        event_by_command = {
            "engage": ("engaged", "engaged"),
            "clutch": ("clutched", "entered_clutch"),
            "resume": ("engaged", "resumed_from_clutch"),
            "recenter": (self._fallback_mode, "recentered"),
            "stop": ("ready", "disengaged"),
            "disengage": ("ready", "disengaged"),
            "estop": ("fault", "estop"),
        }
        next_mode, last_event = event_by_command.get(command, (self._fallback_mode, ""))
        self._fallback_mode = next_mode
        self._publish_status(
            snapshot={
                "mode": next_mode,
                "last_event": last_event,
                "guard_events": (),
            },
            force=True,
        )

    def _publish_status(
        self,
        *,
        snapshot: dict | None = None,
        lifecycle_event: str | None = None,
        force: bool = False,
    ) -> None:
        if self._xr_status is None:
            return
        self._xr_status.publish(
            snapshot=snapshot or {"mode": self._fallback_mode, "last_event": "", "guard_events": ()},
            lifecycle_event=lifecycle_event,
            force=force,
        )
