from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from errno import EADDRINUSE
from typing import Literal

VoiceTeleopCommand = Literal[
    "engage",
    "clutch",
    "resume",
    "recenter",
    "stop",
    "disengage",
    "exit",
    "estop",
    "success",
    "failure",
    "abort",
]

_VALID_COMMANDS = frozenset(
    {
        "engage",
        "clutch",
        "resume",
        "recenter",
        "stop",
        "disengage",
        "exit",
        "estop",
        "success",
        "failure",
        "abort",
    }
)


def _normalize_command(raw_value: object) -> VoiceTeleopCommand | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if not text:
        return None
    alias_map = {
        "start": "engage",
        "start_follow": "engage",
        "开始": "engage",
        "启动": "engage",
        "开始遥操": "engage",
        "pause": "clutch",
        "hold": "clutch",
        "暂停": "clutch",
        "暂停遥操": "clutch",
        "continue": "resume",
        "继续": "resume",
        "继续遥操": "resume",
        "reset": "recenter",
        "recenter": "recenter",
        "重置": "recenter",
        "回正": "recenter",
        "resume": "resume",
        "stop": "stop",
        "停止": "stop",
        "停止遥操": "stop",
        "结束遥操": "stop",
        "disable": "disengage",
        "disable_teleop": "disengage",
        "exit": "exit",
        "park": "exit",
        "退出": "exit",
        "退出遥操": "exit",
        "收回": "exit",
        "回桌面": "exit",
        "emergency_stop": "estop",
        "e_stop": "estop",
        "estop": "estop",
        "急停": "estop",
        "succeed": "success",
        "succeeded": "success",
        "成功": "success",
        "任务成功": "success",
        "task_success": "success",
        "fail": "failure",
        "failed": "failure",
        "失败": "failure",
        "任务失败": "failure",
        "task_failure": "failure",
        "abort": "abort",
        "give_up": "abort",
        "放弃": "abort",
        "结束任务": "abort",
        "终止任务": "abort",
    }
    text = alias_map.get(text, text)
    if text not in _VALID_COMMANDS:
        return None
    return text  # type: ignore[return-value]


@dataclass(frozen=True)
class VoiceTeleopControlConfig:
    host: str = "127.0.0.1"
    port: int = 9910
    max_packet_size: int = 4096


@dataclass(frozen=True)
class VoiceTeleopControlEvents:
    engage_requested: bool = False
    clutch_requested: bool = False
    resume_requested: bool = False
    recenter_requested: bool = False
    stop_requested: bool = False
    exit_requested: bool = False
    estop_requested: bool = False
    commands_seen: tuple[VoiceTeleopCommand, ...] = ()


@dataclass
class VoiceCommandUdpReceiver:
    config: VoiceTeleopControlConfig
    connected: bool = field(default=False, init=False)
    _socket: socket.socket | None = field(default=None, init=False, repr=False)

    def connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.config.host, self.config.port))
        except OSError as exc:
            sock.close()
            if exc.errno == EADDRINUSE:
                raise RuntimeError(
                    "Voice control UDP listener is already running on "
                    f"udp://{self.config.host}:{self.config.port}. "
                    "A previous wrist/controller teleop session is probably still alive; "
                    "stop the old session or reuse it instead of starting a duplicate."
                ) from exc
            raise
        sock.setblocking(False)
        self._socket = sock
        self.connected = True
        print(f"[voice_control] listening on udp://{self.config.host}:{self.config.port}")

    def poll_commands(self) -> tuple[VoiceTeleopCommand, ...]:
        if not self.connected or self._socket is None:
            raise RuntimeError("Voice command receiver is not connected.")

        commands: list[VoiceTeleopCommand] = []
        while True:
            try:
                packet, _ = self._socket.recvfrom(self.config.max_packet_size)
            except BlockingIOError:
                break

            command = _parse_voice_packet(packet)
            if command is not None:
                print(f"[voice_control] command={command}")
                commands.append(command)
        return tuple(commands)

    def disconnect(self) -> None:
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self.connected = False


def _parse_voice_packet(packet: bytes) -> VoiceTeleopCommand | None:
    raw_text = packet.decode("utf-8", errors="ignore").strip()
    if not raw_text:
        return None
    if raw_text.startswith("{"):
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return _normalize_command(payload.get("command"))
    return _normalize_command(raw_text)


class VoiceTeleopControlPolicy:
    def __init__(self, config: VoiceTeleopControlConfig):
        self.config = config
        self._receiver = VoiceCommandUdpReceiver(config)

    def connect(self) -> None:
        self._receiver.connect()

    def disconnect(self) -> None:
        self._receiver.disconnect()

    def update(self) -> VoiceTeleopControlEvents:
        commands = self._receiver.poll_commands()
        return VoiceTeleopControlEvents(
            engage_requested="engage" in commands,
            clutch_requested="clutch" in commands,
            resume_requested="resume" in commands,
            recenter_requested="recenter" in commands,
            stop_requested=("stop" in commands or "disengage" in commands),
            exit_requested="exit" in commands,
            estop_requested="estop" in commands,
            commands_seen=commands,
        )
