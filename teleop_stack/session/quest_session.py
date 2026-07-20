from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from teleop_palm_plane import apply_palm_plane_wrist_orientation_correction
from teleop_stack.devices.quest import QuestInputConfig, build_quest_input_bundle
from teleop_stack.models import ArmSide, Pose7, SingleArmTeleopCommand
from teleop_stack.retargeting.converters import (
    ArmPoseCommandMode,
    optional_hand_debug_snapshot,
    session_result_to_single_arm_command,
)
from teleop_stack.retargeting.pipelines import (
    PoseInputMode,
    SingleArmPipelineConfig,
    build_single_arm_pose_gripper_pipeline,
)
from teleop_stack.robots.base import RobotInterface


@dataclass(frozen=True)
class QuestRobotSessionConfig:
    app_name: str = "HarnessNeroQuestTeleop"
    arm_side: ArmSide = "right"
    pose_input_mode: PoseInputMode = "hand_abs"
    arm_pose_command_mode: ArmPoseCommandMode = "raw_wrist_position_full_orientation"
    fixed_arm_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    use_wrist_position_for_hand: bool = True
    use_wrist_rotation_for_hand: bool = True
    palm_plane_wrist_orientation_blend_alpha: float = 1.0
    loop_hz: float = 60.0
    print_every_n_frames: int = 30
    enable_head_tracker: bool = False
    enable_synthetic_hands_plugin: bool = True
    isaac_teleop_root: str | None = None
    startup_timeout_s: float = 30.0
    startup_retry_interval_s: float = 1.0
    teleop_trace_path: str | None = None


def _is_retryable_openxr_startup_error(exc: Exception) -> bool:
    message = str(exc)
    return "Failed to get OpenXR system: -35" in message or "Failed to get OpenXR system: -33" in message


def _correct_command_with_palm_plane(
    command: SingleArmTeleopCommand,
    hand_debug: dict[str, object] | None,
    *,
    blend_alpha: float,
) -> tuple[SingleArmTeleopCommand, dict[str, object] | None]:
    correction = apply_palm_plane_wrist_orientation_correction(
        command.ee_target.quaternion_xyzw,
        hand_debug,
        blend_alpha=blend_alpha,
    )
    if correction is None:
        return command, None
    corrected = SingleArmTeleopCommand(
        arm_side=command.arm_side,
        ee_target=Pose7(
            position_xyz=command.ee_target.position_xyz,
            quaternion_xyzw=correction.corrected_quaternion_xyzw,
        ),
        gripper=command.gripper,
        source_name=command.source_name,
        timestamp_s=command.timestamp_s,
        frame_id=command.frame_id,
        hand_target=command.hand_target,
    )
    debug = {
        "enabled": True,
        "applied": True,
        "blend_alpha": float(correction.blend_alpha),
        "raw_to_palm_error_rad": float(correction.raw_to_palm_error_rad),
        "raw_wrist_quaternion_xyzw": [float(v) for v in command.ee_target.quaternion_xyzw],
        "corrected_wrist_quaternion_xyzw": [float(v) for v in correction.corrected_quaternion_xyzw],
        **correction.palm_plane.as_dict(),
    }
    return corrected, debug


class QuestRobotSession:
    def __init__(self, config: QuestRobotSessionConfig, robot: RobotInterface):
        self.config = config
        self.robot = robot
        self._session = None
        self._trace_handle = None

    def _build_teleop_session(self):
        from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig

        enable_controllers = self.config.pose_input_mode == "controller_abs"
        input_bundle = build_quest_input_bundle(
            QuestInputConfig(
                enable_hands=True,
                enable_controllers=enable_controllers,
                enable_head=self.config.enable_head_tracker,
                enable_synthetic_hands_plugin=self.config.enable_synthetic_hands_plugin,
                isaac_teleop_root=self.config.isaac_teleop_root,
            )
        )
        pipeline = build_single_arm_pose_gripper_pipeline(
            input_bundle,
            SingleArmPipelineConfig(
                arm_side=self.config.arm_side,
                pose_input_mode=self.config.pose_input_mode,
                use_wrist_position_for_hand=self.config.use_wrist_position_for_hand,
                use_wrist_rotation_for_hand=self.config.use_wrist_rotation_for_hand,
            ),
        )
        return TeleopSession(
            TeleopSessionConfig(
                app_name=self.config.app_name,
                trackers=[],
                pipeline=pipeline,
                plugins=input_bundle.plugins,
            )
        )

    def __enter__(self) -> QuestRobotSession:
        self.robot.connect()
        try:
            if self.config.teleop_trace_path:
                trace_path = Path(self.config.teleop_trace_path).expanduser()
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                self._trace_handle = trace_path.open("w", encoding="utf-8")
                print(f"[quest-session] writing trace to {trace_path}")
            deadline = time.monotonic() + max(0.0, float(self.config.startup_timeout_s))
            attempt = 0
            while True:
                attempt += 1
                self._session = self._build_teleop_session()
                try:
                    self._session.__enter__()
                    print("[quest-session] started")
                    break
                except Exception as exc:
                    try:
                        self._session.__exit__(type(exc), exc, exc.__traceback__)
                    except Exception:
                        pass
                    self._session = None
                    if not _is_retryable_openxr_startup_error(exc):
                        raise
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "OpenXR runtime is available, but no active Quest/WebXR session was found before "
                            f"startup_timeout_s={self.config.startup_timeout_s:.1f}. "
                            "Keep `python -m isaacteleop.cloudxr --accept-eula` running, open the CloudXR/WebXR "
                            "client in the headset, accept the certificate if prompted, and enter the immersive VR session."
                        ) from exc
                    remaining = max(0.0, deadline - time.monotonic())
                    print(
                        "[quest-session] OpenXR runtime found but no active Quest session yet; "
                        f"retrying in {self.config.startup_retry_interval_s:.1f}s "
                        f"attempt={attempt} remaining_s={remaining:.1f}"
                    )
                    time.sleep(max(0.0, float(self.config.startup_retry_interval_s)))
        except Exception:
            self.robot.stop()
            self.robot.disconnect()
            if self._trace_handle is not None:
                self._trace_handle.close()
                self._trace_handle = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._session is not None:
                self._session.__exit__(exc_type, exc_value, traceback)
        finally:
            if self._trace_handle is not None:
                self._trace_handle.close()
                self._trace_handle = None
            self.robot.stop()
            self.robot.disconnect()

    @property
    def frame_count(self) -> int:
        if self._session is None:
            return 0
        return int(self._session.frame_count)

    def step(self) -> SingleArmTeleopCommand:
        if self._session is None:
            raise RuntimeError("QuestRobotSession must be entered before stepping")
        result = self._session.step()
        timestamp_s = float(self._session.get_elapsed_time())
        hand_debug = optional_hand_debug_snapshot(result)
        command = session_result_to_single_arm_command(
            result,
            arm_side=self.config.arm_side,
            timestamp_s=timestamp_s,
            frame_id=self.frame_count,
            pose_input_mode=self.config.pose_input_mode,
            arm_pose_command_mode=self.config.arm_pose_command_mode,
            fixed_arm_orientation_xyzw=self.config.fixed_arm_orientation_xyzw,
        )
        command, palm_debug = _correct_command_with_palm_plane(
            command,
            hand_debug,
            blend_alpha=self.config.palm_plane_wrist_orientation_blend_alpha,
        )
        if hand_debug is not None and palm_debug is not None:
            hand_debug["palm_plane_wrist_orientation"] = palm_debug
        updater = getattr(self.robot, "update_hand_debug", None)
        if callable(updater):
            updater(hand_debug, timestamp_s=timestamp_s)
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
        return command

    def run(self, duration_s: float | None = None) -> None:
        if duration_s is not None and duration_s <= 0:
            duration_s = None
        started = time.monotonic()
        period_s = 0.0 if self.config.loop_hz <= 0 else 1.0 / float(self.config.loop_hz)
        while True:
            loop_started = time.monotonic()
            command = self.step()
            if self.frame_count % max(1, int(self.config.print_every_n_frames)) == 0:
                pos = command.ee_target.position_xyz
                print(
                    f"[quest-session] frame={self.frame_count} t={command.timestamp_s:.2f}s "
                    f"raw_pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
                    f"gripper={command.gripper.normalized_position:.3f}"
                )
            if duration_s is not None and time.monotonic() - started >= duration_s:
                print(f"[quest-session] duration reached: {duration_s:.1f}s")
                break
            remaining = period_s - (time.monotonic() - loop_started)
            if remaining > 0:
                time.sleep(remaining)
