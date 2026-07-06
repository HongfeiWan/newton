from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

import newton
from teleop_stack.ik.differential_ik import PositionJacobian, SpatialJacobian
from teleop_stack.ik.full_pose_controller import (
    FullPoseDifferentialIkController,
    FullPoseDifferentialIkControllerConfig,
)
from teleop_stack.ik.so3 import (
    QuaternionXYZW,
    quat_align_hemisphere_xyzw,
    quat_inverse_xyzw,
    quat_log_rotvec_xyzw,
    quat_multiply_xyzw,
    quat_normalize_xyzw,
)
from teleop_stack.ik.types import RobotStateSnapshot, TaskSpaceTarget
from teleop_stack.models import ArmSide, NamedJointValues, Pose7, SingleArmTeleopCommand
from teleop_stack.robots.base import RobotInterface
from teleop_stack.robots.nero_runtime import NeroTeleopMappingConfig
from teleop_stack.teleop.openxr_genesis_adapter import (
    adapt_openxr_hand_frame_to_genesis_parent,
    adapt_openxr_hand_frame_to_genesis_wrist_frame,
)
from teleop_stack.teleop.orientation_tracker import (
    OrientationTargetTracker,
    OrientationTrackerResult,
    QuaternionWXYZ,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from teleop_stack.teleop.spatial_frames import (
    BeavrHandFrameSmoother,
    FrameAxes,
    HandAnatomicalFrame,
    hand_anatomical_frame_from_debug,
    hand_beavr_anatomical_frame_from_debug,
    matrix_from_axes,
    matrix_to_quat_xyzw,
    quat_xyzw_to_matrix,
)

if TYPE_CHECKING:
    from debug.import_dual_nero_linker_l10 import Example


@dataclass(frozen=True)
class NewtonRuntimeRobotConfig:
    arm_side: ArmSide = "right"
    drive_ik: bool = True
    relative_control: bool = True
    require_initial_anchor: bool = True
    eef_body_suffix_by_side: dict[ArmSide, str] = field(
        default_factory=lambda: {"left": "/left_revo2_flange", "right": "/right_revo2_flange"}
    )
    openxr_yaw_recenter: bool = True
    finite_difference_rad: float = 1.0e-4
    hand_max_joint_step_rad: float = 0.0
    hand_publish_kinematic_velocity: bool = False
    mapping: NeroTeleopMappingConfig = field(default_factory=NeroTeleopMappingConfig)
    ik_config_overrides: dict[str, object] = field(default_factory=dict)


class NewtonLinkKinematicsModel:
    def __init__(
        self,
        *,
        model: newton.Model,
        side: ArmSide,
        arm_joint_q_indices: tuple[int, ...],
        eef_body_suffix: str,
        finite_difference_rad: float,
    ) -> None:
        if len(arm_joint_q_indices) != 7:
            raise ValueError(f"Newton Nero kinematics expects seven arm joints, got {len(arm_joint_q_indices)}")
        self.model = model
        self.side = side
        self.arm_joint_q_indices = tuple(int(v) for v in arm_joint_q_indices)
        self.eef_body_index = _find_body_index(model, eef_body_suffix)
        self.finite_difference_rad = max(abs(float(finite_difference_rad)), 1.0e-6)
        self._kin_state = model.state()
        self._joint_q_host = model.joint_q.numpy().copy()
        self._joint_qd_zero = wp.zeros_like(model.joint_qd)
        self._last_jacobian_q: tuple[float, ...] | None = None
        self._last_spatial_jacobian: SpatialJacobian | None = None

    def sync_joint_q(self, joint_q: np.ndarray) -> None:
        self._joint_q_host = np.asarray(joint_q, dtype=np.float32).copy()
        self.clear_cache()

    def forward_pose(self, joint_positions_rad: tuple[float, ...]) -> Pose7:
        position, quaternion_xyzw = self._pose_after_set(joint_positions_rad)
        return Pose7(
            position_xyz=tuple(float(v) for v in position),
            quaternion_xyzw=tuple(float(v) for v in quaternion_xyzw),  # type: ignore[arg-type]
        )

    def position_jacobian(self, joint_positions_rad: tuple[float, ...]) -> PositionJacobian:
        spatial = self.spatial_jacobian(joint_positions_rad)
        return (
            tuple(float(v) for v in spatial[0]),
            tuple(float(v) for v in spatial[1]),
            tuple(float(v) for v in spatial[2]),
        )

    def spatial_jacobian(self, joint_positions_rad: tuple[float, ...]) -> SpatialJacobian:
        q = _require_seven_dof(joint_positions_rad)
        if self._last_jacobian_q is not None and _same_joint_tuple(q, self._last_jacobian_q):
            assert self._last_spatial_jacobian is not None
            return self._last_spatial_jacobian

        eps = self.finite_difference_rad
        columns: list[tuple[float, float, float, float, float, float]] = []
        for joint_index in range(7):
            q_plus = list(q)
            q_minus = list(q)
            q_plus[joint_index] += eps
            q_minus[joint_index] -= eps
            pos_plus, quat_plus = self._pose_after_set(tuple(q_plus))
            pos_minus, quat_minus = self._pose_after_set(tuple(q_minus))
            linear = (pos_plus - pos_minus) / (2.0 * eps)
            quat_plus = quat_align_hemisphere_xyzw(quat_plus, quat_minus)
            delta_quat = quat_multiply_xyzw(quat_plus, quat_inverse_xyzw(quat_minus))
            angular = np.asarray(quat_log_rotvec_xyzw(delta_quat), dtype=np.float64) / (2.0 * eps)
            columns.append(
                (
                    float(linear[0]),
                    float(linear[1]),
                    float(linear[2]),
                    float(angular[0]),
                    float(angular[1]),
                    float(angular[2]),
                )
            )

        self._pose_after_set(q)
        spatial: SpatialJacobian = tuple(
            tuple(float(columns[col][row]) for col in range(7))
            for row in range(6)
        )  # type: ignore[assignment]
        self._last_jacobian_q = q
        self._last_spatial_jacobian = spatial
        return spatial

    def clear_cache(self) -> None:
        self._last_jacobian_q = None
        self._last_spatial_jacobian = None

    def _pose_after_set(self, joint_positions_rad: tuple[float, ...]) -> tuple[np.ndarray, QuaternionXYZW]:
        q = _require_seven_dof(joint_positions_rad)
        joint_q = self._joint_q_host.copy()
        for target_index, value in zip(self.arm_joint_q_indices, q, strict=True):
            joint_q[int(target_index)] = float(value)
        newton.eval_fk(
            self.model,
            wp.array(joint_q, dtype=wp.float32, device=self.model.device),
            self._joint_qd_zero,
            self._kin_state,
        )
        body_q = self._kin_state.body_q.numpy()[self.eef_body_index]
        position = np.asarray(body_q[:3], dtype=np.float64)
        quaternion_xyzw = quat_normalize_xyzw(tuple(float(v) for v in body_q[3:7]))  # type: ignore[arg-type]
        return position, quaternion_xyzw


class NewtonRuntimeRobotInterface(RobotInterface):
    def __init__(self, example: "Example", config: NewtonRuntimeRobotConfig | None = None, *, print_every_n: int = 30):
        self.example = example
        self.config = config or NewtonRuntimeRobotConfig()
        self.print_every_n = max(1, int(print_every_n))
        self.command_count = 0
        self._connected = False
        self._human_anchor_xyz: tuple[float, float, float] | None = None
        self._target_anchor_xyz: tuple[float, float, float] | None = None
        self._joint_q_host: np.ndarray | None = None
        self._joint_qd_host: np.ndarray | None = None
        self._arm_joint_q_indices: tuple[int, ...] = ()
        self._arm_joint_qd_indices: tuple[int, ...] = ()
        self._joint_q_index_by_label: dict[str, int] = {}
        self._joint_qd_index_by_label: dict[str, int] = {}
        self._kinematics: NewtonLinkKinematicsModel | None = None
        self._ik: FullPoseDifferentialIkController | None = None
        self._last_hand_debug: dict[str, object] | None = None
        self._last_hand_debug_timestamp_s = 0.0
        self._target_anchor_quaternion_wxyz: QuaternionWXYZ | None = None
        self._orientation_tracker: OrientationTargetTracker | None = None
        self._orientation_debug: OrientationTrackerResult | None = None
        self._last_orientation_timestamp_s: float | None = None
        self._last_orientation_source_quaternion_xyzw: QuaternionXYZW | None = None
        self._last_orientation_source_debug: dict[str, object] | None = None
        self._orientation_anchor_source_actual: str | None = None
        self._beavr_hand_frame_smoother = BeavrHandFrameSmoother(moving_average_limit=5)
        self._openxr_yaw_correction_rad: float | None = None
        self._openxr_yaw_recenter_debug: dict[str, object] | None = None
        self._command_gate_enabled = True
        self._gate_mode = "engaged"
        self._gate_last_event = "direct_control"
        self._last_ik_result = None
        self._eef_body_suffix = ""

    def connect(self) -> None:
        model = self.example.model
        self._joint_q_host = self.example.state_0.joint_q.numpy().copy()
        self._joint_qd_host = self.example.state_0.joint_qd.numpy().copy()
        self._joint_q_index_by_label, self._joint_qd_index_by_label = _joint_scalar_index_maps(model)
        self._arm_joint_q_indices = _arm_joint_indices(self._joint_q_index_by_label, self.config.arm_side)
        self._arm_joint_qd_indices = _arm_joint_indices(self._joint_qd_index_by_label, self.config.arm_side)
        self._kinematics = NewtonLinkKinematicsModel(
            model=model,
            side=self.config.arm_side,
            arm_joint_q_indices=self._arm_joint_q_indices,
            eef_body_suffix=self.config.eef_body_suffix_by_side[self.config.arm_side],
            finite_difference_rad=self.config.finite_difference_rad,
        )
        self._eef_body_suffix = self.config.eef_body_suffix_by_side[self.config.arm_side]
        current_q = self._current_arm_q()
        ik_config = FullPoseDifferentialIkControllerConfig(
            seed_joint_positions_rad=current_q,
            neutral_joint_positions_rad=current_q,
            kinematics_model=self._kinematics,
            **self.config.ik_config_overrides,
        )
        self._ik = FullPoseDifferentialIkController(ik_config)
        self._ik.reset(self._robot_state(timestamp_s=0.0))
        self._orientation_tracker = (
            OrientationTargetTracker(self.config.mapping.orientation_tracker_config())
            if bool(self.config.mapping.use_teleop_orientation)
            else None
        )
        self._connected = True
        print(
            "[newton-quest-teleop] connected"
            f" side={self.config.arm_side}"
            f" drive_ik={'on' if self.config.drive_ik else 'hand-only'}"
            f" relative={'on' if self.config.relative_control else 'off'}"
            f" openxr_adapter={self.config.mapping.openxr_coordinate_adapter}"
            f" eef={self._eef_body_suffix}"
        )

    def send_command(self, command: SingleArmTeleopCommand) -> None:
        if not self._connected:
            raise RuntimeError("NewtonRuntimeRobotInterface is not connected")
        if command.arm_side != self.config.arm_side:
            return

        if not self._command_gate_enabled:
            self.command_count += 1
            if self.command_count == 1 or self.command_count % self.print_every_n == 0:
                print(
                    f"[newton-quest-teleop] frame={command.frame_id} mode={self._gate_mode} "
                    f"event={self._gate_last_event} holding_until_voice_start"
                )
            return

        self._ensure_anchor(command)
        if self.config.drive_ik:
            self._apply_arm_command(command)
        if command.hand_target is not None:
            self._apply_hand_target(command.hand_target)
        self._publish_joint_state()

        self.command_count += 1
        if self.command_count == 1 or self.command_count % self.print_every_n == 0:
            target = self._target_position(command.ee_target)
            eef = self._current_eef_pose().position_xyz
            ik_status = getattr(self._last_ik_result, "status", "none")
            ik_pos_error = getattr(self._last_ik_result, "target_position_error_m", None)
            ik_ori_error = getattr(self._last_ik_result, "target_orientation_error_rad", None)
            ik_error_text = ""
            if ik_pos_error is not None:
                ik_error_text += f" ik_pos_err={float(ik_pos_error):.4f}"
            if ik_ori_error is not None:
                ik_error_text += f" ik_ori_err={float(ik_ori_error):.4f}"
            print(
                f"[newton-quest-teleop] frame={command.frame_id} side={self.config.arm_side} "
                f"target=({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f}) "
                f"eef=({eef[0]:+.3f},{eef[1]:+.3f},{eef[2]:+.3f}) "
                f"ik_status={ik_status}{ik_error_text} "
                f"gripper={command.gripper.normalized_position:.3f}"
            )

    def update_hand_debug(self, hand_debug: dict[str, object] | None, *, timestamp_s: float) -> None:
        self._last_hand_debug = hand_debug
        self._last_hand_debug_timestamp_s = float(timestamp_s)

    def stop(self) -> None:
        if self._joint_q_host is not None:
            self._publish_joint_state()

    def disconnect(self) -> None:
        self._connected = False

    def set_command_gate(self, enabled: bool, *, mode: str, last_event: str) -> None:
        self._command_gate_enabled = bool(enabled)
        self._gate_mode = str(mode)
        self._gate_last_event = str(last_event)

    def reset_relative_anchor(self) -> None:
        self._human_anchor_xyz = None
        self._target_anchor_xyz = None
        self._target_anchor_quaternion_wxyz = None
        self._last_orientation_timestamp_s = None
        self._last_orientation_source_quaternion_xyzw = None
        self._last_orientation_source_debug = None
        self._orientation_anchor_source_actual = None
        self._orientation_debug = None
        self._openxr_yaw_correction_rad = None
        self._openxr_yaw_recenter_debug = None
        self._beavr_hand_frame_smoother.reset()
        if self._ik is not None:
            self._ik.reset(self._robot_state(timestamp_s=0.0))

    def recenter_teleop(self) -> None:
        self.reset_relative_anchor()
        if self._joint_q_host is not None and self._kinematics is not None:
            self._kinematics.sync_joint_q(self._joint_q_host)
        print(f"[newton-quest-teleop] recentered side={self.config.arm_side}")

    def reset_to_scene_state(self) -> None:
        if not self._connected:
            return
        self._joint_q_host = self.example.state_0.joint_q.numpy().copy()
        self._joint_qd_host = self.example.state_0.joint_qd.numpy().copy()
        self._last_ik_result = None
        self.command_count = 0
        if self._kinematics is not None:
            self._kinematics.sync_joint_q(self._joint_q_host)
        self.reset_relative_anchor()
        print(f"[newton-quest-teleop] reset_to_scene_state side={self.config.arm_side}")

    def xr_status_snapshot(self, *, mode: str, last_event: str) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "mode": mode,
            "last_event": last_event,
            "mapper_control_profile": "voice",
            "arm_side": self.config.arm_side,
            "command_count": self.command_count,
            "controller_available": True,
            "command_gate_enabled": self._command_gate_enabled,
        }
        if self._last_hand_debug is None:
            snapshot["input_tracking_state"] = "missing"
        else:
            snapshot["input_tracking_state"] = "tracking"
            snapshot["hand_debug_timestamp_s"] = self._last_hand_debug_timestamp_s
            snapshot.update(self._last_hand_debug)
        if self._last_orientation_source_debug is not None:
            snapshot["orientation_source_debug"] = self._last_orientation_source_debug
        if self._orientation_debug is not None:
            snapshot["orientation_debug"] = self._orientation_debug.as_dict()
        if self._last_ik_result is not None:
            snapshot["ik_status"] = getattr(self._last_ik_result, "status", None)
            snapshot["ik_target_position_error_m"] = getattr(self._last_ik_result, "target_position_error_m", None)
            snapshot["ik_target_orientation_error_rad"] = getattr(
                self._last_ik_result,
                "target_orientation_error_rad",
                None,
            )
            snapshot["ik_events"] = getattr(self._last_ik_result, "events", ())
        return snapshot

    def _ensure_anchor(self, command: SingleArmTeleopCommand) -> None:
        if self._human_anchor_xyz is not None and self._target_anchor_xyz is not None:
            return
        self._human_anchor_xyz = tuple(float(v) for v in command.ee_target.position_xyz)
        self._target_anchor_xyz = tuple(float(v) for v in self._current_eef_pose().position_xyz)
        self._target_anchor_quaternion_wxyz = xyzw_to_wxyz(self._current_eef_pose().quaternion_xyzw)
        if self.config.openxr_yaw_recenter and self._openxr_yaw_correction_rad is None:
            self._recenter_openxr_yaw_from_hand(command.ee_target, source="anchor")
        if self._orientation_tracker is not None:
            source_quat_xyzw, source_debug = self._orientation_source_quaternion_xyzw(command.ee_target)
            self._orientation_tracker.reset_anchor(source_quat_xyzw, self._target_anchor_quaternion_wxyz)
            self._orientation_debug = None
            self._last_orientation_timestamp_s = None
            self._orientation_anchor_source_actual = str(source_debug.get("actual", "unknown"))
        print(
            f"[newton-quest-teleop] anchor side={self.config.arm_side} "
            f"human=({self._human_anchor_xyz[0]:+.3f},{self._human_anchor_xyz[1]:+.3f},{self._human_anchor_xyz[2]:+.3f}) "
            f"target=({self._target_anchor_xyz[0]:+.3f},{self._target_anchor_xyz[1]:+.3f},{self._target_anchor_xyz[2]:+.3f}) "
            f"orientation={'on' if self._orientation_tracker is not None else 'off'} "
            f"orientation_source={self.config.mapping.orientation_source} "
            f"orientation_reference_mode={self.config.mapping.orientation_reference_mode}"
        )

    def _target_position(self, pose: Pose7) -> tuple[float, float, float]:
        if not self.config.relative_control:
            mapped = self.config.mapping.map_vector(pose.position_xyz)
            mapped = self._apply_openxr_yaw_correction_to_vector(mapped)
            return tuple(
                float(self.config.mapping.workspace_origin_xyz[i])
                + float(self.config.mapping.translation_scale_xyz[i]) * float(mapped[i])
                for i in range(3)
            )  # type: ignore[return-value]
        if self._human_anchor_xyz is None or self._target_anchor_xyz is None:
            raise RuntimeError("Relative teleop anchor is not initialized")
        delta = tuple(float(pose.position_xyz[i]) - float(self._human_anchor_xyz[i]) for i in range(3))
        mapped_delta = self.config.mapping.map_vector(delta)  # type: ignore[arg-type]
        mapped_delta = self._apply_openxr_yaw_correction_to_vector(mapped_delta)
        return tuple(
            float(self._target_anchor_xyz[i])
            + float(self.config.mapping.translation_scale_xyz[i]) * float(mapped_delta[i])
            for i in range(3)
        )  # type: ignore[return-value]

    def _target_quaternion_xyzw(self, pose: Pose7, *, timestamp_s: float) -> QuaternionXYZW:
        if self._orientation_tracker is not None:
            if self._target_anchor_quaternion_wxyz is None:
                return self._current_eef_pose().quaternion_xyzw
            source_quat, source_debug = self._orientation_source_quaternion_xyzw(pose)
            source_actual = str(source_debug.get("actual", "unknown"))
            if (
                self._orientation_anchor_source_actual is not None
                and source_actual in {"hand_anatomical_frame", "hand_beavr_anatomical_frame", "hand_genesis_wrist_frame"}
                and self._orientation_anchor_source_actual != source_actual
            ):
                self.reset_relative_anchor()
                return self._current_eef_pose().quaternion_xyzw
            if self._last_orientation_timestamp_s is None:
                dt_s = float(self.example.frame_dt)
            else:
                dt_s = max(0.0, float(timestamp_s) - float(self._last_orientation_timestamp_s))
                if dt_s <= 0.0:
                    dt_s = float(self.example.frame_dt)
            self._orientation_debug = self._orientation_tracker.update(source_quat, dt_s=dt_s)
            self._last_orientation_timestamp_s = float(timestamp_s)
            self._orientation_anchor_source_actual = source_actual
            return wxyz_to_xyzw(self._orientation_debug.cmd_target_quat_wxyz)
        if self.config.mapping.fixed_quaternion_wxyz is not None:
            return quat_normalize_xyzw(wxyz_to_xyzw(self.config.mapping.fixed_quaternion_wxyz))
        return self._current_eef_pose().quaternion_xyzw

    @staticmethod
    def _yaw_rotation_matrix(rad: float) -> np.ndarray:
        cos_v = math.cos(float(rad))
        sin_v = math.sin(float(rad))
        return np.asarray(
            (
                (cos_v, -sin_v, 0.0),
                (sin_v, cos_v, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], ...]:
        return tuple(tuple(float(value) for value in row) for row in np.asarray(matrix, dtype=np.float64))

    def _set_openxr_yaw_correction_from_genesis_forward(
        self,
        forward_xyz: tuple[float, float, float],
        *,
        source: str,
    ) -> bool:
        if self.config.mapping.openxr_coordinate_adapter != "openxr_genesis":
            self._openxr_yaw_correction_rad = None
            self._openxr_yaw_recenter_debug = {
                "enabled": False,
                "source": source,
                "reason": "openxr_coordinate_adapter_not_openxr_genesis",
                "openxr_coordinate_adapter": self.config.mapping.openxr_coordinate_adapter,
            }
            return False

        measured = np.asarray(forward_xyz, dtype=np.float64).reshape(3)
        measured_xy = np.asarray((measured[0], measured[1]), dtype=np.float64)
        norm_xy = float(np.linalg.norm(measured_xy))
        if norm_xy <= 1.0e-9:
            self._openxr_yaw_correction_rad = None
            self._openxr_yaw_recenter_debug = {
                "enabled": False,
                "source": source,
                "reason": "forward_axis_horizontal_norm_too_small",
                "measured_forward_xyz": [float(v) for v in measured],
            }
            return False

        measured_xy /= norm_xy
        target_xy = np.asarray((-1.0, 0.0), dtype=np.float64)
        cross_z = float(measured_xy[0] * target_xy[1] - measured_xy[1] * target_xy[0])
        dot = float(np.dot(measured_xy, target_xy))
        yaw_rad = math.atan2(cross_z, dot)
        self._openxr_yaw_correction_rad = float(yaw_rad)
        corrected = self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in measured))
        self._openxr_yaw_recenter_debug = {
            "enabled": True,
            "source": source,
            "yaw_correction_rad": float(yaw_rad),
            "yaw_correction_deg": float(math.degrees(yaw_rad)),
            "measured_forward_xyz": [float(v) for v in measured],
            "measured_forward_xy_normalized": [float(v) for v in measured_xy],
            "target_forward_xyz": [-1.0, 0.0, 0.0],
            "corrected_forward_xyz": [float(v) for v in corrected],
        }
        print(
            "[newton-quest-teleop] openxr_yaw_recenter "
            f"source={source} yaw_deg={math.degrees(yaw_rad):+.2f} "
            f"measured_forward=({measured[0]:+.3f},{measured[1]:+.3f},{measured[2]:+.3f})"
        )
        return True

    def _apply_openxr_yaw_correction_to_vector(
        self,
        vector_xyz: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if self._openxr_yaw_correction_rad is None:
            return tuple(float(v) for v in vector_xyz)  # type: ignore[return-value]
        corrected = self._yaw_rotation_matrix(self._openxr_yaw_correction_rad) @ np.asarray(
            vector_xyz,
            dtype=np.float64,
        )
        return (float(corrected[0]), float(corrected[1]), float(corrected[2]))

    def _apply_openxr_yaw_correction_to_quaternion(
        self,
        quaternion_xyzw: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if self._openxr_yaw_correction_rad is None:
            return tuple(float(v) for v in quaternion_xyzw)  # type: ignore[return-value]
        yaw_matrix = self._yaw_rotation_matrix(self._openxr_yaw_correction_rad)
        rotation = np.asarray(quat_xyzw_to_matrix(quaternion_xyzw), dtype=np.float64)
        return matrix_to_quat_xyzw(self._matrix_tuple(yaw_matrix @ rotation))  # type: ignore[return-value]

    def _apply_openxr_yaw_correction_to_frame(self, frame: HandAnatomicalFrame) -> HandAnatomicalFrame:
        if self._openxr_yaw_correction_rad is None:
            return frame
        axes = FrameAxes(
            x=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.axes.x)),
            y=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.axes.y)),
            z=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.axes.z)),
        )
        return HandAnatomicalFrame(
            origin_xyz=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.origin_xyz)),
            axes=axes,
            quaternion_xyzw=matrix_to_quat_xyzw(matrix_from_axes(axes)),
            handedness_det=float(frame.handedness_det),
            thumb_alignment=float(frame.thumb_alignment),
            legacy_palm_normal_alignment=frame.legacy_palm_normal_alignment,
            construction=f"{frame.construction}_openxr_yaw_recentered",
            raw_axes=frame.raw_axes,
            axis_adapter={
                **(frame.axis_adapter or {}),
                "session_yaw_recenter": "yaw-only Genesis +Z correction; operator front -> robot front",
            },
        )

    def _hand_orientation_frame(
        self,
        requested: str,
        *,
        apply_openxr_yaw_correction: bool = True,
    ) -> HandAnatomicalFrame | None:
        if not isinstance(self._last_hand_debug, dict):
            return None
        if requested == "hand_anatomical_frame":
            frame = hand_anatomical_frame_from_debug(self._last_hand_debug)
        elif requested in {"hand_beavr_anatomical_frame", "hand_genesis_wrist_frame"}:
            frame = self._beavr_hand_frame_smoother.update(self._last_hand_debug)
            if frame is None:
                frame = hand_beavr_anatomical_frame_from_debug(self._last_hand_debug)
        else:
            return None
        if frame is None:
            return None
        if requested == "hand_genesis_wrist_frame":
            adapted = adapt_openxr_hand_frame_to_genesis_wrist_frame(frame)
            return self._apply_openxr_yaw_correction_to_frame(adapted) if apply_openxr_yaw_correction else adapted
        if self.config.mapping.openxr_coordinate_adapter == "openxr_genesis":
            adapted = adapt_openxr_hand_frame_to_genesis_parent(frame)
            return self._apply_openxr_yaw_correction_to_frame(adapted) if apply_openxr_yaw_correction else adapted
        return frame

    def _recenter_openxr_yaw_from_hand(self, pose: Pose7, *, source: str) -> bool:
        frame = self._hand_orientation_frame("hand_genesis_wrist_frame", apply_openxr_yaw_correction=False)
        if frame is None:
            source_quat = self.config.mapping.adapt_openxr_quaternion(tuple(float(v) for v in pose.quaternion_xyzw))
            forward = quat_xyzw_to_matrix(source_quat)
            return self._set_openxr_yaw_correction_from_genesis_forward(
                (float(forward[0][2]), float(forward[1][2]), float(forward[2][2])),
                source=f"{source}:wrist_quat_fallback",
            )
        return self._set_openxr_yaw_correction_from_genesis_forward(
            tuple(float(v) for v in frame.axes.z),
            source=f"{source}:hand_genesis_wrist_frame_z",
        )

    def _orientation_source_quaternion_xyzw(self, pose: Pose7) -> tuple[QuaternionXYZW, dict[str, object]]:
        requested = str(self.config.mapping.orientation_source)
        wrist_quat = tuple(float(v) for v in pose.quaternion_xyzw)
        if requested == "wrist_quat":
            adapted = self.config.mapping.adapt_openxr_quaternion(wrist_quat)
            adapted = self._apply_openxr_yaw_correction_to_quaternion(adapted)
            adapted = quat_normalize_xyzw(adapted)
            debug: dict[str, object] = {
                "requested": requested,
                "actual": "wrist_quat",
                "fallback": False,
                "reason": None,
                "openxr_coordinate_adapter": self.config.mapping.openxr_coordinate_adapter,
                "openxr_yaw_recenter": self._openxr_yaw_recenter_debug,
            }
            self._last_orientation_source_quaternion_xyzw = adapted
            self._last_orientation_source_debug = debug
            return adapted, debug

        if requested not in {"hand_anatomical_frame", "hand_beavr_anatomical_frame", "hand_genesis_wrist_frame"}:
            raise ValueError(f"unsupported teleop orientation source: {requested!r}")

        frame = self._hand_orientation_frame(requested)
        if frame is not None:
            quat = quat_normalize_xyzw(tuple(float(v) for v in frame.quaternion_xyzw))
            debug = {
                "requested": requested,
                "actual": requested,
                "fallback": False,
                "reason": None,
                "openxr_coordinate_adapter": self.config.mapping.openxr_coordinate_adapter,
                "openxr_yaw_recenter": self._openxr_yaw_recenter_debug,
                requested: frame.as_dict(),
            }
            self._last_orientation_source_quaternion_xyzw = quat
            self._last_orientation_source_debug = debug
            return quat, debug

        if self._last_orientation_source_quaternion_xyzw is not None:
            debug = {
                "requested": requested,
                "actual": f"last_{requested}",
                "fallback": True,
                "reason": f"{requested}_unavailable",
                "openxr_coordinate_adapter": self.config.mapping.openxr_coordinate_adapter,
                "openxr_yaw_recenter": self._openxr_yaw_recenter_debug,
            }
            self._last_orientation_source_debug = debug
            return self._last_orientation_source_quaternion_xyzw, debug

        adapted = self.config.mapping.adapt_openxr_quaternion(wrist_quat)
        adapted = self._apply_openxr_yaw_correction_to_quaternion(adapted)
        adapted = quat_normalize_xyzw(adapted)
        debug = {
            "requested": requested,
            "actual": "wrist_quat",
            "fallback": True,
            "reason": f"{requested}_unavailable",
            "openxr_coordinate_adapter": self.config.mapping.openxr_coordinate_adapter,
            "openxr_yaw_recenter": self._openxr_yaw_recenter_debug,
        }
        self._last_orientation_source_quaternion_xyzw = adapted
        self._last_orientation_source_debug = debug
        return adapted, debug

    def _apply_arm_command(self, command: SingleArmTeleopCommand) -> None:
        assert self._ik is not None
        target_pose = Pose7(
            position_xyz=self._target_position(command.ee_target),
            quaternion_xyzw=self._target_quaternion_xyzw(command.ee_target, timestamp_s=command.timestamp_s),
        )
        target = TaskSpaceTarget(
            arm_side=self.config.arm_side,
            source_name=command.source_name,
            timestamp_s=command.timestamp_s,
            frame_id=command.frame_id,
            ee_target=target_pose,
            orientation_mode="teleop_full_pose",
            target_frame="world",
        )
        self._ik.set_target(target)
        result = self._ik.step(self._robot_state(timestamp_s=command.timestamp_s), self.example.frame_dt)
        self._last_ik_result = result
        assert self._joint_q_host is not None and self._joint_qd_host is not None
        for target_index, value in zip(self._arm_joint_q_indices, result.q_cmd, strict=True):
            self._joint_q_host[int(target_index)] = float(value)
        if result.dq_cmd is not None:
            for target_index, value in zip(self._arm_joint_qd_indices, result.dq_cmd, strict=True):
                self._joint_qd_host[int(target_index)] = float(value)
        if self._kinematics is not None:
            self._kinematics.sync_joint_q(self._joint_q_host)

    def _apply_hand_target(self, hand_target: NamedJointValues) -> None:
        joint_values = _expand_l10_mimic_joint_values(hand_target)
        assert self._joint_q_host is not None and self._joint_qd_host is not None
        max_step = max(0.0, float(self.config.hand_max_joint_step_rad))
        for joint_name, value in zip(joint_values.joint_names, joint_values.joint_positions, strict=True):
            label_suffix = _l10_joint_label_suffix(self.config.arm_side, str(joint_name))
            q_index = self._joint_q_index_by_label.get(label_suffix)
            if q_index is None:
                continue
            current_value = float(self._joint_q_host[int(q_index)])
            target_value = float(value)
            if max_step > 0.0:
                target_value = float(np.clip(target_value, current_value - max_step, current_value + max_step))
            self._joint_q_host[int(q_index)] = target_value
            qd_index = self._joint_qd_index_by_label.get(label_suffix)
            if qd_index is not None:
                if self.config.hand_publish_kinematic_velocity:
                    hand_qd = (target_value - current_value) / float(self.example.frame_dt)
                else:
                    hand_qd = 0.0
                self._joint_qd_host[int(qd_index)] = hand_qd

    def _publish_joint_state(self) -> None:
        assert self._joint_q_host is not None and self._joint_qd_host is not None
        model = self.example.model
        joint_q = wp.array(self._joint_q_host, dtype=wp.float32, device=model.device)
        joint_qd = wp.array(self._joint_qd_host, dtype=wp.float32, device=model.device)
        model.joint_q = joint_q
        model.joint_qd = joint_qd
        self.example.state_0.joint_q = joint_q
        self.example.state_0.joint_qd = joint_qd
        self.example.state_1.joint_q = wp.clone(joint_q)
        self.example.state_1.joint_qd = wp.clone(joint_qd)
        self.example.control.joint_target_q = wp.clone(joint_q)
        self.example.control.joint_target_qd = wp.clone(joint_qd)
        newton.eval_fk(
            model,
            joint_q,
            joint_qd,
            self.example.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )
        model.bvh_refit_shapes(self.example.state_0)

    def _robot_state(self, *, timestamp_s: float) -> RobotStateSnapshot:
        return RobotStateSnapshot(
            timestamp_s=float(timestamp_s),
            joint_positions_rad=self._current_arm_q(),
            joint_velocities_rad_s=self._current_arm_qd(),
            ee_pose=self._current_eef_pose(),
        )

    def _current_arm_q(self) -> tuple[float, ...]:
        if self._joint_q_host is None:
            joint_q = self.example.state_0.joint_q.numpy()
        else:
            joint_q = self._joint_q_host
        return tuple(float(joint_q[index]) for index in self._arm_joint_q_indices)

    def _current_arm_qd(self) -> tuple[float, ...]:
        if self._joint_qd_host is None:
            joint_qd = self.example.state_0.joint_qd.numpy()
        else:
            joint_qd = self._joint_qd_host
        return tuple(float(joint_qd[index]) for index in self._arm_joint_qd_indices)

    def _current_eef_pose(self) -> Pose7:
        assert self._kinematics is not None
        return self._kinematics.forward_pose(self._current_arm_q())


def _find_body_index(model: newton.Model, label_suffix: str) -> int:
    body_index = next((i for i, label in enumerate(model.body_label) if label.endswith(label_suffix)), None)
    if body_index is None:
        raise ValueError(f"Body ending with {label_suffix!r} not found")
    return int(body_index)


def _joint_scalar_index_maps(model: newton.Model) -> tuple[dict[str, int], dict[str, int]]:
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    q_map: dict[str, int] = {}
    qd_map: dict[str, int] = {}
    for joint_index, label in enumerate(model.joint_label):
        q0 = int(q_start[joint_index])
        q1 = int(q_start[joint_index + 1]) if joint_index + 1 < len(q_start) else int(model.joint_coord_count)
        qd0 = int(qd_start[joint_index])
        qd1 = int(qd_start[joint_index + 1]) if joint_index + 1 < len(qd_start) else int(model.joint_dof_count)
        if q1 - q0 == 1:
            q_map[_short_joint_label(label)] = q0
        if qd1 - qd0 == 1:
            qd_map[_short_joint_label(label)] = qd0
    return q_map, qd_map


def _short_joint_label(label: str) -> str:
    return str(label).rsplit("/", maxsplit=1)[-1]


def _arm_joint_indices(index_by_label: dict[str, int], side: ArmSide) -> tuple[int, ...]:
    indices = []
    for joint_index in range(1, 8):
        label = f"{side}_joint{joint_index}"
        if label not in index_by_label:
            raise ValueError(f"Missing Newton arm joint: {label}")
        indices.append(int(index_by_label[label]))
    return tuple(indices)


def _l10_joint_label_suffix(side: ArmSide, joint_name: str) -> str:
    if joint_name.startswith(f"{side}_l10_"):
        return joint_name
    if joint_name.startswith("l10_"):
        return f"{side}_{joint_name}"
    return f"{side}_l10_{joint_name}"


def _expand_l10_mimic_joint_values(joint_values: NamedJointValues) -> NamedJointValues:
    try:
        from teleop_stack.retargeting.hand_config import load_linker_l10_right_hand_spec

        spec = load_linker_l10_right_hand_spec()
        return spec.expand_mimic_joint_values(joint_values)
    except Exception:
        return joint_values


def _require_seven_dof(joint_positions_rad: tuple[float, ...]) -> tuple[float, ...]:
    if len(joint_positions_rad) != 7:
        raise ValueError(f"Newton Nero kinematics expects 7 joints, got {len(joint_positions_rad)}")
    values = tuple(float(v) for v in joint_positions_rad)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Newton Nero joint positions must be finite")
    return values


def _same_joint_tuple(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> bool:
    if len(lhs) != len(rhs):
        return False
    return all(abs(float(a) - float(b)) <= 1.0e-12 for a, b in zip(lhs, rhs, strict=True))
