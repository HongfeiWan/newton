from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from teleop_stack.paths import resolve_cloudxr_env_path


def resolve_xr_status_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    env_value = os.environ.get("TELEOP_XR_STATUS_PATH")
    if env_value:
        return Path(env_value).expanduser().resolve()

    runtime_dir = resolve_cloudxr_env_path().parent
    return (runtime_dir / "teleop_xr_status.json").resolve()


@dataclass(frozen=True)
class _BadgeStyle:
    state: str
    label: str
    color: str


_BADGE_BY_MODE: dict[str, _BadgeStyle] = {
    "ready": _BadgeStyle(state="ready", label="READY", color="#7A808C"),
    "engaged": _BadgeStyle(state="running", label="RUN", color="#1FD16F"),
    "clutched": _BadgeStyle(state="paused", label="PAUSE", color="#FFB020"),
    "fault": _BadgeStyle(state="fault", label="FAULT", color="#FF5E5B"),
    "stopped": _BadgeStyle(state="stopped", label="STOP", color="#FF5E5B"),
    "error": _BadgeStyle(state="error", label="ERROR", color="#FF5E5B"),
}

_BADGE_BY_INPUT_TRACKING: dict[str, _BadgeStyle] = {
    "missing": _BadgeStyle(state="warning", label="NO HAND", color="#FFB020"),
}

_BADGE_BY_HAND_POSE_GATE: dict[str, _BadgeStyle] = {
    "reacquiring": _BadgeStyle(state="warning", label="RELOCK", color="#39A7FF"),
    "unstable": _BadgeStyle(state="warning", label="HOLD", color="#FFB020"),
}

_BADGE_BY_AUTHORITY: dict[str, _BadgeStyle] = {
    "policy_rollout": _BadgeStyle(state="running", label="POLICY", color="#1FD16F"),
    "policy_stopped_hold": _BadgeStyle(state="paused", label="PAUSE", color="#FFB020"),
    "human_intervention": _BadgeStyle(state="running", label="HUMAN", color="#39A7FF"),
    "human_recovery_hold": _BadgeStyle(state="paused", label="HOLD", color="#FFB020"),
    "resume_pending": _BadgeStyle(state="paused", label="RESUME", color="#39A7FF"),
    "safety_hold": _BadgeStyle(state="warning", label="HOLD", color="#FFB020"),
}

_ALIGNMENT_BADGE_ACTIVE = _BadgeStyle(state="warning", label="ALIGN", color="#39A7FF")

_TOAST_BY_EVENT: dict[str, tuple[str, str]] = {
    "engaged": ("RUN", "#1FD16F"),
    "entered_clutch": ("PAUSE", "#FFB020"),
    "resumed_from_clutch": ("RUN", "#1FD16F"),
    "recentered": ("RESET", "#39A7FF"),
    "disengaged": ("STOP", "#FF5E5B"),
    "session_started": ("READY", "#7A808C"),
    "session_stopped": ("STOP", "#FF5E5B"),
    "session_error": ("ERROR", "#FF5E5B"),
    "hand_tracking_lost": ("NO HAND", "#FFB020"),
    "hand_tracking_restored": ("HAND", "#1FD16F"),
    "hand_pose_reacquiring": ("RELOCK", "#39A7FF"),
    "hand_pose_unstable": ("HOLD", "#FFB020"),
    "hand_pose_stable": ("HAND", "#1FD16F"),
}


def _clean_ascii_label(raw_value: str | None, fallback: str) -> str:
    if not raw_value:
        return fallback
    cleaned = "".join(
        ch if ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch in {" ", "-", "<", ">", "^", "V"} else " "
        for ch in raw_value.upper()
    )
    collapsed = " ".join(cleaned.split())
    return collapsed or fallback


def _float_list(payload: Any, length: int) -> list[float] | None:
    if not isinstance(payload, list) or len(payload) < length:
        return None
    try:
        return [float(value) for value in payload[:length]]
    except (TypeError, ValueError):
        return None


def _flatten_hand_anatomical_frame(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    frame = snapshot.get("hand_anatomical_frame_overlay")
    if not isinstance(frame, Mapping):
        return {"hand_frame_visible": 0}
    origin = _float_list(frame.get("origin_xyz"), 3)
    axes = frame.get("axes")
    if not isinstance(axes, Mapping):
        return {"hand_frame_visible": 0}
    x_axis = _float_list(axes.get("x"), 3)
    y_axis = _float_list(axes.get("y"), 3)
    z_axis = _float_list(axes.get("z"), 3)
    if origin is None or x_axis is None or y_axis is None or z_axis is None:
        return {"hand_frame_visible": 0}
    return {
        "hand_frame_visible": 1,
        "hand_frame_origin_x": origin[0],
        "hand_frame_origin_y": origin[1],
        "hand_frame_origin_z": origin[2],
        "hand_frame_x_axis_x": x_axis[0],
        "hand_frame_x_axis_y": x_axis[1],
        "hand_frame_x_axis_z": x_axis[2],
        "hand_frame_y_axis_x": y_axis[0],
        "hand_frame_y_axis_y": y_axis[1],
        "hand_frame_y_axis_z": y_axis[2],
        "hand_frame_z_axis_x": z_axis[0],
        "hand_frame_z_axis_y": z_axis[1],
        "hand_frame_z_axis_z": z_axis[2],
    }


class XrTeleopStatusPublisher:
    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        heartbeat_s: float = 0.25,
        toast_ttl_s: float = 1.5,
    ) -> None:
        self.path = resolve_xr_status_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_s = max(0.05, float(heartbeat_s))
        self.toast_ttl_s = max(0.2, float(toast_ttl_s))
        self._last_payload = ""
        self._last_write_monotonic_s = 0.0
        self._last_mapper_event = ""
        self._last_guard_signature: tuple[str, ...] = ()
        self._last_input_tracking_state = ""
        self._last_hand_pose_gate_state = ""
        self._last_alignment_phase = ""
        self._toast_label = ""
        self._toast_color = "#000000"
        self._toast_until_s = 0.0
        self._write_error: str | None = None

    def publish(
        self,
        *,
        snapshot: Mapping[str, Any] | None = None,
        lifecycle_event: str | None = None,
        force: bool = False,
    ) -> None:
        now_monotonic = time.monotonic()
        now_wall = time.time()
        payload = self._build_payload(
            snapshot=snapshot or {},
            lifecycle_event=lifecycle_event,
            now_wall_s=now_wall,
        )
        serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if not force and serialized == self._last_payload and (now_monotonic - self._last_write_monotonic_s) < self.heartbeat_s:
            return
        try:
            self._write_atomic(serialized)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            if error_text != self._write_error:
                print(
                    "[xr_status] publish_nonfatal_write_error "
                    f"path={self.path} error={error_text}"
                )
                self._write_error = error_text
            return
        if self._write_error is not None:
            print(f"[xr_status] publish_recovered path={self.path}")
            self._write_error = None
        self._last_payload = serialized
        self._last_write_monotonic_s = now_monotonic

    def _build_payload(
        self,
        *,
        snapshot: Mapping[str, Any],
        lifecycle_event: str | None,
        now_wall_s: float,
    ) -> dict[str, Any]:
        mapper_mode = str(snapshot.get("mode") or "").strip().lower()
        last_event = str(snapshot.get("last_event") or "").strip().lower()
        input_tracking_state = str(snapshot.get("input_tracking_state") or "").strip().lower()
        hand_pose_gate_state = str(snapshot.get("hand_pose_gate_state") or "").strip().lower()
        mapper_control_profile = str(snapshot.get("mapper_control_profile") or "").strip().lower()
        authority_state = str(snapshot.get("authority_state") or "").strip().lower()
        controller_available = bool(snapshot.get("controller_available"))
        policy_hil_controller_ready = mapper_control_profile == "policy_hil" and controller_available
        if lifecycle_event == "session_stopped":
            badge = _BADGE_BY_MODE["stopped"]
        elif lifecycle_event == "session_error":
            badge = _BADGE_BY_MODE["error"]
        elif mapper_control_profile == "policy_hil" and authority_state in _BADGE_BY_AUTHORITY:
            badge = _BADGE_BY_AUTHORITY[authority_state]
        elif input_tracking_state in _BADGE_BY_INPUT_TRACKING and not policy_hil_controller_ready:
            badge = _BADGE_BY_INPUT_TRACKING[input_tracking_state]
        elif hand_pose_gate_state in _BADGE_BY_HAND_POSE_GATE and not policy_hil_controller_ready:
            badge = _BADGE_BY_HAND_POSE_GATE[hand_pose_gate_state]
        elif mapper_mode == "ready" and last_event == "disengaged":
            badge = _BADGE_BY_MODE["stopped"]
        else:
            badge = _BADGE_BY_MODE.get(mapper_mode, _BADGE_BY_MODE["ready"])

        arm_status = snapshot.get("arm_status")
        if isinstance(arm_status, Mapping):
            alignment_events = tuple(str(event) for event in arm_status.get("events") or ())
            alignment_event_set = set(alignment_events)
            alignment_gate_active = (
                "orientation_alignment_gate_active" in alignment_event_set
                and "orientation_alignment_gate_opened" not in alignment_event_set
                and "orientation_alignment_gate_timeout" not in alignment_event_set
            )
            if alignment_gate_active:
                badge = _ALIGNMENT_BADGE_ACTIVE

        if lifecycle_event:
            self._maybe_activate_toast(lifecycle_event, now_wall_s)
        elif last_event and last_event != self._last_mapper_event:
            self._last_mapper_event = last_event
            self._maybe_activate_toast(last_event, now_wall_s)

        guard_events = tuple(str(event) for event in snapshot.get("guard_events") or ())
        if guard_events != self._last_guard_signature:
            if guard_events:
                self._activate_toast(label="WARN", color="#FFB020", now_wall_s=now_wall_s)
            self._last_guard_signature = guard_events

        if input_tracking_state != self._last_input_tracking_state:
            if input_tracking_state == "missing" and not policy_hil_controller_ready:
                self._maybe_activate_toast("hand_tracking_lost", now_wall_s)
            elif self._last_input_tracking_state == "missing" and input_tracking_state == "tracked":
                self._maybe_activate_toast("hand_tracking_restored", now_wall_s)
            self._last_input_tracking_state = input_tracking_state

        if hand_pose_gate_state != self._last_hand_pose_gate_state:
            if hand_pose_gate_state == "reacquiring" and not policy_hil_controller_ready:
                self._maybe_activate_toast("hand_pose_reacquiring", now_wall_s)
            elif hand_pose_gate_state == "unstable" and not policy_hil_controller_ready:
                self._maybe_activate_toast("hand_pose_unstable", now_wall_s)
            elif self._last_hand_pose_gate_state in {"reacquiring", "unstable"} and hand_pose_gate_state == "stable":
                self._maybe_activate_toast("hand_pose_stable", now_wall_s)
            self._last_hand_pose_gate_state = hand_pose_gate_state

        if isinstance(arm_status, Mapping):
            self._update_alignment_toast(arm_status=arm_status, now_wall_s=now_wall_s)

        toast_label = ""
        toast_color = "#000000"
        if now_wall_s < self._toast_until_s:
            toast_label = self._toast_label
            toast_color = self._toast_color

        payload = {
            "version": 1,
            "badge_state": badge.state,
            "badge_label": badge.label,
            "badge_color": badge.color,
            "toast_label": toast_label,
            "toast_color": toast_color,
            "toast_until_s": round(self._toast_until_s, 3) if toast_label else 0.0,
            "updated_at_s": round(now_wall_s, 3),
        }
        payload.update(_flatten_hand_anatomical_frame(snapshot))
        return payload

    def _maybe_activate_toast(self, event_key: str, now_wall_s: float) -> None:
        mapped = _TOAST_BY_EVENT.get(event_key)
        if mapped is None:
            return
        self._activate_toast(label=mapped[0], color=mapped[1], now_wall_s=now_wall_s)

    def _update_alignment_toast(self, *, arm_status: Mapping[str, Any], now_wall_s: float) -> None:
        events = tuple(str(event) for event in arm_status.get("events") or ())
        event_set = set(events)
        hint_label = _clean_ascii_label(
            str(arm_status.get("orientation_alignment_hint_label") or ""),
            "ALIGN",
        )
        alignment = arm_status.get("orientation_alignment")
        if isinstance(alignment, Mapping):
            hint_label = _clean_ascii_label(str(alignment.get("hint_label") or hint_label), "ALIGN")
        if "orientation_alignment_gate_timeout" in event_set:
            phase = "timeout"
            label = "ALIGN TO"
            color = "#FFB020"
        elif "orientation_alignment_gate_aligned" in event_set:
            phase = "aligned"
            label = "ALIGNED"
            color = "#1FD16F"
        elif "orientation_alignment_gate_active" in event_set:
            phase = "active"
            label = hint_label if hint_label and hint_label != "OK" else "ALIGN"
            color = "#39A7FF"
        else:
            phase = ""
            label = ""
            color = "#000000"
        if phase == "active" and label:
            self._activate_toast(label=label, color=color, now_wall_s=now_wall_s)
        elif phase != self._last_alignment_phase:
            if phase:
                self._activate_toast(label=label, color=color, now_wall_s=now_wall_s)
        self._last_alignment_phase = phase

    def _activate_toast(self, *, label: str, color: str, now_wall_s: float) -> None:
        self._toast_label = _clean_ascii_label(label, "INFO")
        self._toast_color = color
        self._toast_until_s = now_wall_s + self.toast_ttl_s

    def _write_atomic(self, payload: str) -> None:
        tmp_path = self.path.with_name(f".{self.path.name}.tmp")
        last_error: Exception | None = None
        for _ in range(2):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(tmp_path, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.write("\n")
                os.replace(tmp_path, self.path)
                return
            except FileNotFoundError as exc:
                last_error = exc
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to write XR status payload to {self.path}")
