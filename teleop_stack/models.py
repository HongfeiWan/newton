from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

ArmSide = Literal["left", "right"]


@dataclass(frozen=True)
class Pose7:
    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> Pose7:
        data = [float(v) for v in values]
        if len(data) != 7:
            raise ValueError(f"Expected 7 pose values, got {len(data)}")
        return cls((data[0], data[1], data[2]), (data[3], data[4], data[5], data[6]))

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "position_xyz": [float(v) for v in self.position_xyz],
            "quaternion_xyzw": [float(v) for v in self.quaternion_xyzw],
        }


@dataclass(frozen=True)
class GripperCommand:
    normalized_position: float

    def as_dict(self) -> dict[str, float]:
        return {"normalized_position": float(self.normalized_position)}


@dataclass(frozen=True)
class NamedJointValues:
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.joint_names) != len(self.joint_positions):
            raise ValueError("joint_names and joint_positions must have the same length")

    def as_dict(self) -> dict[str, list[object]]:
        return {
            "joint_names": [str(name) for name in self.joint_names],
            "joint_positions": [float(value) for value in self.joint_positions],
        }


@dataclass(frozen=True)
class SingleArmTeleopCommand:
    arm_side: ArmSide
    ee_target: Pose7
    gripper: GripperCommand
    source_name: str
    timestamp_s: float
    frame_id: int = 0
    hand_target: NamedJointValues | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "frame_id": int(self.frame_id),
            "timestamp_s": float(self.timestamp_s),
            "arm_side": self.arm_side,
            "source_name": self.source_name,
            "ee_target": self.ee_target.as_dict(),
            "gripper": self.gripper.as_dict(),
        }
        if self.hand_target is not None:
            payload["hand_target"] = self.hand_target.as_dict()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=True, separators=(",", ":"))
