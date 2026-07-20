from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np

from teleop_stack.models import ArmSide, GripperCommand, Pose7, SingleArmTeleopCommand
from teleop_stack.retargeting.linker_l10_dex_retargeter import retarget_openxr_hand_to_linker_l10_right
from teleop_stack.robots.base import RobotInterface


@dataclass(frozen=True)
class OverlayHandLogSessionConfig:
    trace_path: str
    arm_side: ArmSide = "right"
    hand_side: str = "right"
    use_teleop_orientation: bool = True
    loop_hz: float = 60.0
    print_every_n_frames: int = 30
    stale_after_s: float = 1.0
    teleop_trace_path: str | None = None


def _normalize_quaternion_xyzw(quaternion_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = sqrt(sum(float(value) * float(value) for value in quaternion_xyzw))
    if norm <= 1.0e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(value) / norm for value in quaternion_xyzw)  # type: ignore[return-value]


class OverlayHandLogSession:
    def __init__(self, config: OverlayHandLogSessionConfig, robot: RobotInterface):
        self.config = config
        self.robot = robot
        self.trace_path = Path(config.trace_path).expanduser()
        self._handle = None
        self._trace_handle = None
        self._latest_sample: dict[str, object] | None = None
        self._last_warn_time_s = 0.0
        self._frame_count = 0

    def __enter__(self) -> OverlayHandLogSession:
        self.robot.connect()
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.trace_path.open("a+", encoding="utf-8")
            self._handle.seek(0, os.SEEK_END)
            if self.config.teleop_trace_path:
                trace_path = Path(self.config.teleop_trace_path).expanduser()
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                self._trace_handle = trace_path.open("w", encoding="utf-8")
                print(f"[overlay-hand-log-session] writing trace to {trace_path}", flush=True)
            print(
                "[overlay-hand-log-session] using camera overlay hand log "
                f"path={self.trace_path} hand={self.config.hand_side} arm={self.config.arm_side}",
                flush=True,
            )
        except Exception:
            self.robot.stop()
            self.robot.disconnect()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._handle is not None:
                self._handle.close()
            if self._trace_handle is not None:
                self._trace_handle.close()
        finally:
            self._handle = None
            self._trace_handle = None
            self.robot.stop()
            self.robot.disconnect()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _accept_sample(self, sample: dict[str, object]) -> bool:
        if sample.get("event") != "frame":
            return False
        if self.config.hand_side != "auto" and str(sample.get("hand")) != self.config.hand_side:
            return False
        try:
            positions = np.asarray(sample.get("raw_hand_positions_xyz"), dtype=np.float32)
            valid = np.asarray(sample.get("joint_valid"), dtype=np.uint8)
        except Exception:
            return False
        return positions.shape == (26, 3) and valid.shape == (26,) and int(valid.sum()) >= 10

    def _read_latest_sample(self) -> dict[str, object] | None:
        if self._handle is None:
            return None
        while True:
            raw_line = self._handle.readline()
            if raw_line == "":
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(sample, dict) and self._accept_sample(sample):
                self._latest_sample = sample
        return self._latest_sample

    def _sample_age_s(self, sample: dict[str, object]) -> float | None:
        raw_monotonic = sample.get("monotonic_time_s")
        if isinstance(raw_monotonic, (int, float)):
            return max(0.0, time.monotonic() - float(raw_monotonic))
        raw_time = sample.get("time_s")
        if isinstance(raw_time, (int, float)):
            return max(0.0, time.time() - float(raw_time))
        return None

    @staticmethod
    def _hand_debug_from_sample(sample: dict[str, object]) -> dict[str, object]:
        names = sample.get("raw_hand_joint_names")
        positions = sample.get("raw_hand_positions_xyz")
        orientations = sample.get("raw_hand_orientations_xyzw")
        valid = sample.get("joint_valid")
        return {
            "joint_valid_count": int(sample.get("valid_joint_count", 0)),
            "joint_positions_xyz": positions if isinstance(positions, list) else [],
            "joint_quaternions_xyzw": orientations if isinstance(orientations, list) else [],
            "joint_valid": valid if isinstance(valid, list) else [],
            "joint_names": names if isinstance(names, list) else None,
            "source": "camera_overlay_hand_log",
            "hand": sample.get("hand"),
        }

    def _command_from_sample(self, sample: dict[str, object]) -> SingleArmTeleopCommand:
        positions = np.asarray(sample["raw_hand_positions_xyz"], dtype=np.float32)
        orientations = np.asarray(sample.get("raw_hand_orientations_xyzw"), dtype=np.float32)
        joint_valid = np.asarray(sample["joint_valid"], dtype=np.uint8)
        wrist_pos = tuple(float(value) for value in positions[1])
        if self.config.use_teleop_orientation and orientations.shape == (26, 4):
            wrist_quat = _normalize_quaternion_xyzw(tuple(float(value) for value in orientations[1]))
        else:
            wrist_quat = (0.0, 0.0, 0.0, 1.0)

        try:
            hand_target = retarget_openxr_hand_to_linker_l10_right(
                positions,
                joint_orientations_xyzw=orientations if orientations.shape == (26, 4) else None,
                joint_valid=joint_valid,
            )
        except Exception:
            hand_target = None

        self._frame_count += 1
        return SingleArmTeleopCommand(
            arm_side=self.config.arm_side,
            ee_target=Pose7(position_xyz=wrist_pos, quaternion_xyzw=wrist_quat),
            gripper=GripperCommand(normalized_position=0.0),
            source_name="camera_overlay_hand_log",
            timestamp_s=float(sample.get("monotonic_time_s", time.monotonic())),
            frame_id=self._frame_count,
            hand_target=hand_target,
        )

    def step(self) -> SingleArmTeleopCommand | None:
        sample = self._read_latest_sample()
        now = time.monotonic()
        if sample is None:
            if now - self._last_warn_time_s > 2.0:
                print(f"[overlay-hand-log-session] waiting for overlay hand samples: {self.trace_path}", flush=True)
                self._last_warn_time_s = now
            return None

        age_s = self._sample_age_s(sample)
        if age_s is not None and age_s > max(0.1, float(self.config.stale_after_s)):
            if now - self._last_warn_time_s > 2.0:
                print(
                    "[overlay-hand-log-session] overlay hand samples are stale "
                    f"age_s={age_s:.2f} path={self.trace_path}",
                    flush=True,
                )
                self._last_warn_time_s = now
            return None

        command = self._command_from_sample(sample)
        hand_debug = self._hand_debug_from_sample(sample)
        updater = getattr(self.robot, "update_hand_debug", None)
        if callable(updater):
            updater(hand_debug, timestamp_s=command.timestamp_s)
        self.robot.send_command(command)
        if self._trace_handle is not None:
            self._trace_handle.write(
                json.dumps(
                    {"command": command.as_dict(), "hand_debug": hand_debug},
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            )
            self._trace_handle.write("\n")
        if self._frame_count == 1 or self._frame_count % max(1, int(self.config.print_every_n_frames)) == 0:
            print(
                f"[overlay-hand-log-session] frame={self._frame_count} "
                f"hand={sample.get('hand')} valid={sample.get('valid_joint_count')} "
                f"hand_target={'yes' if command.hand_target is not None else 'no'}",
                flush=True,
            )
        return command

    def run(self, duration_s: float | None = None) -> None:
        if duration_s is not None and duration_s <= 0:
            duration_s = None
        started = time.monotonic()
        period_s = 0.0 if self.config.loop_hz <= 0 else 1.0 / float(self.config.loop_hz)
        while True:
            loop_started = time.monotonic()
            self.step()
            if duration_s is not None and time.monotonic() - started >= duration_s:
                print(f"[overlay-hand-log-session] duration reached: {duration_s:.1f}s", flush=True)
                break
            remaining = period_s - (time.monotonic() - loop_started)
            if remaining > 0:
                time.sleep(remaining)
