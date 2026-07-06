from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

from teleop_stack.models import ArmSide, Pose7, SingleArmTeleopCommand
from teleop_stack.robots.base import RobotInterface
from teleop_stack.teleop.openxr_genesis_adapter import (
    OpenXrCoordinateAdapterName,
    map_openxr_quaternion_to_genesis_parent,
    map_openxr_vector_to_genesis,
    validate_adapter_name,
)
from teleop_stack.teleop.orientation_tracker import (
    OrientationTrackerConfig,
    QuaternionWXYZ,
    xyzw_to_wxyz,
)


AxisMap = tuple[str, str, str]


def _axis_value(values_xyz: tuple[float, float, float], token: str) -> float:
    token = token.strip().lower()
    sign = -1.0 if token.startswith("-") else 1.0
    axis = token[-1]
    index = {"x": 0, "y": 1, "z": 2}[axis]
    return sign * float(values_xyz[index])


def _map_vec3_axes(values_xyz: tuple[float, float, float], axis_map: AxisMap) -> tuple[float, float, float]:
    return tuple(_axis_value(values_xyz, token) for token in axis_map)  # type: ignore[return-value]


@dataclass(frozen=True)
class NeroTeleopMappingConfig:
    translation_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    workspace_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    input_axis_map: AxisMap = ("x", "y", "z")
    openxr_coordinate_adapter: OpenXrCoordinateAdapterName = "openxr_genesis"
    use_teleop_orientation: bool = True
    fixed_quaternion_wxyz: QuaternionWXYZ | None = None
    orientation_axis_map: AxisMap = ("x", "y", "z")
    orientation_max_speed_rad_s: float = 3.0
    orientation_tool_offset_wxyz: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)
    orientation_reference_mode: str = "calibrated_tool_local"
    orientation_source: str = "hand_genesis_wrist_frame"

    def __post_init__(self) -> None:
        validate_adapter_name(self.openxr_coordinate_adapter)

    def adapt_openxr_vector(self, values_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        if self.openxr_coordinate_adapter == "openxr_genesis":
            return map_openxr_vector_to_genesis(values_xyz)
        return tuple(float(value) for value in values_xyz)  # type: ignore[return-value]

    def adapt_openxr_quaternion(
        self,
        quaternion_xyzw: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if self.openxr_coordinate_adapter == "openxr_genesis":
            return map_openxr_quaternion_to_genesis_parent(quaternion_xyzw)
        return tuple(float(value) for value in quaternion_xyzw)  # type: ignore[return-value]

    def map_vector(self, values_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        return _map_vec3_axes(self.adapt_openxr_vector(values_xyz), self.input_axis_map)

    def map_position_absolute(self, pose: Pose7) -> tuple[float, float, float]:
        mapped = self.map_vector(pose.position_xyz)
        return tuple(
            float(self.workspace_origin_xyz[i]) + float(self.translation_scale_xyz[i]) * float(mapped[i])
            for i in range(3)
        )  # type: ignore[return-value]

    def map_delta(self, delta_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        mapped = self.map_vector(delta_xyz)
        return tuple(float(self.translation_scale_xyz[i]) * float(mapped[i]) for i in range(3))  # type: ignore[return-value]

    def map_quaternion(self, pose: Pose7) -> tuple[float, float, float, float] | None:
        if self.use_teleop_orientation:
            return xyzw_to_wxyz(pose.quaternion_xyzw)
        return self.fixed_quaternion_wxyz

    def orientation_tracker_config(self) -> OrientationTrackerConfig:
        return OrientationTrackerConfig(
            axis_map=self.orientation_axis_map,
            max_speed_rad_s=float(self.orientation_max_speed_rad_s),
            tool_offset_wxyz=self.orientation_tool_offset_wxyz,
            reference_mode=self.orientation_reference_mode,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class NeroRuntimeRobotConfig:
    arm_side: ArmSide = "right"
    backend: str = "gpu"
    show_viewer: bool = True
    linker_hand_side: ArmSide = "right"
    show_palm_plane_axes: bool = True
    drive_ik: bool = True
    relative_control: bool = True
    require_initial_anchor: bool = True
    runtime_config_overrides: dict[str, object] = field(default_factory=dict)
    mapping: NeroTeleopMappingConfig = field(default_factory=NeroTeleopMappingConfig)


class NeroRuntimeRobotInterface(RobotInterface):
    def __init__(self, config: NeroRuntimeRobotConfig | None = None, *, print_every_n: int = 30) -> None:
        self.config = config or NeroRuntimeRobotConfig()
        self.print_every_n = max(1, int(print_every_n))
        self.runtime = None
        self.command_count = 0
        self._human_anchor_xyz: tuple[float, float, float] | None = None
        self._target_anchor_xyz: tuple[float, float, float] | None = None
        self._latest_hand_debug: dict[str, object] | None = None

    def connect(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        assets_root = repo_root / "assets"
        nero_root = assets_root / "nero_twin"
        if str(assets_root) not in sys.path:
            sys.path.insert(0, str(assets_root))
        if str(nero_root) not in sys.path:
            sys.path.insert(0, str(nero_root))

        from assets.nero_arm_linker_l10_genesis_config import make_runtime_config
        from nero_dual_runtime import NeroDualArmRuntime

        runtime_config = make_runtime_config(
            backend=self.config.backend,
            show_viewer=self.config.show_viewer,
            linker_hand_side=self.config.linker_hand_side,
        )
        if self.config.show_palm_plane_axes:
            runtime_config = runtime_config.__class__(
                **{
                    **runtime_config.__dict__,
                    "show_palm_plane_axes": True,
                }
            )
        if self.config.runtime_config_overrides:
            runtime_config = runtime_config.__class__(
                **{
                    **runtime_config.__dict__,
                    **self.config.runtime_config_overrides,
                }
            )
        self.runtime = NeroDualArmRuntime(runtime_config)
        self.runtime.connect()
        print(
            f"[nero-vr] connected side={self.config.arm_side} backend={self.config.backend} "
            f"drive_ik={'on' if self.config.drive_ik else 'markers-only'} relative={'on' if self.config.relative_control else 'off'}"
        )

    def update_hand_debug(self, hand_debug: dict[str, object] | None, *, timestamp_s: float) -> None:
        self._latest_hand_debug = hand_debug
        if self.runtime is None or hand_debug is None:
            return
        palm = hand_debug.get("palm_plane_wrist_orientation")
        if isinstance(palm, dict):
            axes = {
                "across": palm.get("palm_across_xyz"),
                "forward": palm.get("palm_forward_xyz"),
                "normal": palm.get("palm_normal_xyz"),
            }
            self.runtime.update_palm_plane_axis_markers(self.config.arm_side, axes)

    def _ensure_anchor(self, command: SingleArmTeleopCommand) -> None:
        if self.runtime is None:
            raise RuntimeError("Nero runtime is not connected")
        if self._human_anchor_xyz is not None and self._target_anchor_xyz is not None:
            return
        side = self.config.arm_side
        self._human_anchor_xyz = tuple(float(v) for v in command.ee_target.position_xyz)
        self._target_anchor_xyz = tuple(float(v) for v in self.runtime.targets[side])
        print(
            f"[nero-vr] anchor side={side} "
            f"human=({self._human_anchor_xyz[0]:+.3f},{self._human_anchor_xyz[1]:+.3f},{self._human_anchor_xyz[2]:+.3f}) "
            f"target=({self._target_anchor_xyz[0]:+.3f},{self._target_anchor_xyz[1]:+.3f},{self._target_anchor_xyz[2]:+.3f})"
        )

    def _target_position(self, pose: Pose7) -> tuple[float, float, float]:
        if not self.config.relative_control:
            return self.config.mapping.map_position_absolute(pose)
        if self._human_anchor_xyz is None or self._target_anchor_xyz is None:
            raise RuntimeError("Relative teleop anchor is not initialized")
        delta = tuple(float(pose.position_xyz[i]) - float(self._human_anchor_xyz[i]) for i in range(3))
        mapped_delta = self.config.mapping.map_delta(delta)  # type: ignore[arg-type]
        return tuple(float(self._target_anchor_xyz[i]) + float(mapped_delta[i]) for i in range(3))  # type: ignore[return-value]

    def send_command(self, command: SingleArmTeleopCommand) -> None:
        if self.runtime is None:
            raise RuntimeError("Nero runtime is not connected")
        self._ensure_anchor(command)
        side = self.config.arm_side
        from nero_dual_runtime import NeroArmTarget

        target = NeroArmTarget(
            position_xyz=self._target_position(command.ee_target),
            quaternion_wxyz=self.config.mapping.map_quaternion(command.ee_target),
        )
        if self.config.drive_ik:
            status = self.runtime.step({side: target}, selected=side)
        else:
            status = self.runtime.step_targets_only({side: target}, selected=side)
        self.command_count += 1
        if self.command_count == 1 or self.command_count % self.print_every_n == 0:
            target_xyz = status.target_left_xyz if side == "left" else status.target_right_xyz
            print(
                f"[nero-vr] frame={command.frame_id} side={side} "
                f"target=({target_xyz[0]:+.3f},{target_xyz[1]:+.3f},{target_xyz[2]:+.3f}) "
                f"gripper={command.gripper.normalized_position:.3f}"
            )

    def stop(self) -> None:
        if self.runtime is not None:
            self.runtime.stop()

    def disconnect(self) -> None:
        if self.runtime is not None:
            self.runtime.disconnect()
            self.runtime = None
