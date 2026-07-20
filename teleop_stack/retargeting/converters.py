from __future__ import annotations

import logging
from math import sqrt
from typing import Any, Literal

import numpy as np

from teleop_stack.models import ArmSide, GripperCommand, NamedJointValues, Pose7, SingleArmTeleopCommand
from teleop_stack.retargeting.linker_l10_dex_retargeter import retarget_openxr_hand_to_linker_l10_right

logger = logging.getLogger(__name__)
_LINKER_HAND_SPEC_WARNING_EMITTED = False


ArmPoseCommandMode = Literal[
    "legacy_retargeted_ee",
    "raw_wrist_position_fixed_orientation",
    "raw_wrist_position_full_orientation",
]


def _first_tensor(result: dict[str, Any], key: str):
    import numpy as np

    if key not in result:
        raise KeyError(f"Teleop result does not contain key: {key}")
    return np.asarray(result[key][0], dtype=float)


def _tensor_value_to_numpy(value: Any, *, dtype: Any):
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    try:
        return np.from_dlpack(value).astype(dtype, copy=False)
    except Exception:
        return np.asarray(value, dtype=dtype)


def _optional_linker_hand_target(result: dict[str, Any], *, arm_side: ArmSide) -> NamedJointValues | None:
    global _LINKER_HAND_SPEC_WARNING_EMITTED

    if arm_side != "right":
        return None
    hand_group = result.get("raw_hand")
    if hand_group is None or getattr(hand_group, "is_none", False):
        return None

    try:
        from isaacteleop.retargeting_engine.tensor_types.indices import HandInputIndex
    except ModuleNotFoundError:
        return None

    try:
        joint_positions = _tensor_value_to_numpy(hand_group[HandInputIndex.JOINT_POSITIONS], dtype=np.float32)
        joint_valid = _tensor_value_to_numpy(hand_group[HandInputIndex.JOINT_VALID], dtype=np.uint8)
        try:
            joint_orientations = _tensor_value_to_numpy(hand_group[HandInputIndex.JOINT_ORIENTATIONS], dtype=np.float32)
        except (KeyError, IndexError, TypeError):
            joint_orientations = None
    except Exception:
        return None

    if joint_positions.shape != (26, 3) or joint_valid.shape != (26,):
        return None
    if int(joint_valid.sum()) < 10:
        return None
    if joint_orientations is not None and joint_orientations.shape != (26, 4):
        joint_orientations = None

    try:
        return retarget_openxr_hand_to_linker_l10_right(
            joint_positions,
            joint_orientations_xyzw=joint_orientations,
            joint_valid=joint_valid,
        )
    except FileNotFoundError as exc:
        if not _LINKER_HAND_SPEC_WARNING_EMITTED:
            logger.warning("Linker Hand retargeting assets are unavailable; continuing without hand_target. %s", exc)
            _LINKER_HAND_SPEC_WARNING_EMITTED = True
        return None
    except Exception as exc:
        logger.debug("Failed to retarget OpenXR hand to Linker L10 target: %s", exc)
        return None


def _normalize_quaternion_xyzw(quaternion_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = sqrt(sum(float(value) * float(value) for value in quaternion_xyzw))
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(v) / norm for v in quaternion_xyzw)  # type: ignore[return-value]


def _optional_raw_wrist_pose(
    result: dict[str, Any],
    *,
    fixed_orientation_xyzw: tuple[float, float, float, float],
    use_raw_wrist_orientation: bool = False,
) -> Pose7 | None:
    hand_group = result.get("raw_hand")
    if hand_group is None or getattr(hand_group, "is_none", False):
        return None

    try:
        from isaacteleop.retargeting_engine.tensor_types.indices import HandInputIndex, HandJointIndex
    except ModuleNotFoundError:
        return None

    try:
        joint_positions = _tensor_value_to_numpy(hand_group[HandInputIndex.JOINT_POSITIONS], dtype=np.float32)
        joint_valid = _tensor_value_to_numpy(hand_group[HandInputIndex.JOINT_VALID], dtype=np.uint8)
        joint_orientations = (
            _tensor_value_to_numpy(hand_group[HandInputIndex.JOINT_ORIENTATIONS], dtype=np.float32)
            if use_raw_wrist_orientation
            else None
        )
    except Exception:
        return None

    if joint_positions.shape != (26, 3) or joint_valid.shape != (26,):
        return None
    if use_raw_wrist_orientation and (joint_orientations is None or joint_orientations.shape != (26, 4)):
        return None

    wrist_index = int(HandJointIndex.WRIST)
    if wrist_index < 0 or wrist_index >= len(joint_valid) or not bool(joint_valid[wrist_index]):
        return None

    wrist_position = tuple(float(value) for value in joint_positions[wrist_index])
    wrist_orientation = (
        _normalize_quaternion_xyzw(tuple(float(value) for value in joint_orientations[wrist_index]))  # type: ignore[index,arg-type]
        if use_raw_wrist_orientation and joint_orientations is not None
        else _normalize_quaternion_xyzw(fixed_orientation_xyzw)
    )
    return Pose7(position_xyz=wrist_position, quaternion_xyzw=wrist_orientation)  # type: ignore[arg-type]


def optional_hand_debug_snapshot(result: dict[str, Any]) -> dict[str, object] | None:
    hand_group = result.get("raw_hand")
    if hand_group is None or getattr(hand_group, "is_none", False):
        return None

    try:
        from isaacteleop.retargeting_engine.tensor_types.indices import HandInputIndex, HandJointIndex
    except ModuleNotFoundError:
        return None

    try:
        joint_positions = _tensor_value_to_numpy(hand_group[HandInputIndex.JOINT_POSITIONS], dtype=np.float32)
        joint_orientations = _tensor_value_to_numpy(hand_group[HandInputIndex.JOINT_ORIENTATIONS], dtype=np.float32)
        joint_valid = _tensor_value_to_numpy(hand_group[HandInputIndex.JOINT_VALID], dtype=np.uint8)
    except Exception:
        return None

    if joint_positions.shape != (26, 3) or joint_orientations.shape != (26, 4) or joint_valid.shape != (26,):
        return None

    try:
        joint_names = [str(HandJointIndex(index).name).lower() for index in range(26)]
    except Exception:
        joint_names = None

    return {
        "joint_valid_count": int(joint_valid.sum()),
        "joint_positions_xyz": [[float(value) for value in row] for row in joint_positions],
        "joint_quaternions_xyzw": [[float(value) for value in row] for row in joint_orientations],
        "joint_valid": [bool(value) for value in joint_valid],
        "joint_names": joint_names,
    }


def result_has_valid_hand_tracking(result: dict[str, Any]) -> bool:
    hand_group = result.get("raw_hand")
    if hand_group is None or getattr(hand_group, "is_none", False):
        return False
    try:
        from isaacteleop.retargeting_engine.tensor_types.indices import HandInputIndex

        joint_valid = _tensor_value_to_numpy(hand_group[HandInputIndex.JOINT_VALID], dtype=np.uint8)
    except Exception:
        return False
    return joint_valid.shape == (26,) and int(joint_valid.sum()) >= 10


def session_result_to_single_arm_command(
    result: dict[str, Any],
    *,
    arm_side: ArmSide,
    timestamp_s: float,
    source_name: str = "quest",
    frame_id: int = 0,
    pose_input_mode: Literal["controller_abs", "hand_abs"] = "controller_abs",
    arm_pose_command_mode: ArmPoseCommandMode = "legacy_retargeted_ee",
    fixed_arm_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> SingleArmTeleopCommand:
    pose: Pose7 | None = None
    if pose_input_mode == "hand_abs" and arm_pose_command_mode in {
        "raw_wrist_position_fixed_orientation",
        "raw_wrist_position_full_orientation",
    }:
        pose = _optional_raw_wrist_pose(
            result,
            fixed_orientation_xyzw=fixed_arm_orientation_xyzw,
            use_raw_wrist_orientation=arm_pose_command_mode == "raw_wrist_position_full_orientation",
        )
    if pose is None:
        pose = Pose7.from_iterable(_first_tensor(result, "ee_pose"))
    gripper_value = float(_first_tensor(result, "gripper_command").reshape(-1)[0])
    return SingleArmTeleopCommand(
        arm_side=arm_side,
        ee_target=pose,
        gripper=GripperCommand(normalized_position=gripper_value),
        source_name=source_name,
        timestamp_s=timestamp_s,
        frame_id=frame_id,
        hand_target=_optional_linker_hand_target(result, arm_side=arm_side),
    )
