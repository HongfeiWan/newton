from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from teleop_stack.devices.quest import QuestInputBundle
from teleop_stack.models import ArmSide


PoseInputMode = Literal["controller_abs", "hand_abs"]


@dataclass(frozen=True)
class SingleArmPipelineConfig:
    arm_side: ArmSide = "right"
    pose_input_mode: PoseInputMode = "controller_abs"
    use_wrist_position_for_hand: bool = False
    use_wrist_rotation_for_hand: bool = False
    zero_out_xy_rotation: bool = False


def build_single_arm_pose_gripper_pipeline(input_bundle: QuestInputBundle, config: SingleArmPipelineConfig):
    from isaacteleop.retargeters import GripperRetargeter, GripperRetargeterConfig, Se3AbsRetargeter, Se3RetargeterConfig
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner

    hands = input_bundle.sources.get("hands")
    controllers = input_bundle.sources.get("controllers")
    if hands is None:
        raise ValueError("Quest input bundle is missing hand tracking sources.")

    hand_device = HandsSource.LEFT if config.arm_side == "left" else HandsSource.RIGHT
    controller_device = ControllersSource.LEFT if config.arm_side == "left" else ControllersSource.RIGHT

    gripper = GripperRetargeter(GripperRetargeterConfig(hand_side=config.arm_side), name=f"{config.arm_side}_gripper")
    gripper_inputs = {hand_device: hands.output(hand_device)}
    if controllers is not None:
        gripper_inputs[controller_device] = controllers.output(controller_device)
    connected_gripper = gripper.connect(gripper_inputs)

    if config.pose_input_mode == "controller_abs":
        if controllers is None:
            raise ValueError("controller_abs mode requires controller sources.")
        pose_input_device = controller_device
        pose_source = controllers
        use_wrist_position = False
        use_wrist_rotation = False
    else:
        pose_input_device = hand_device
        pose_source = hands
        use_wrist_position = config.use_wrist_position_for_hand
        use_wrist_rotation = config.use_wrist_rotation_for_hand

    pose = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=pose_input_device,
            use_wrist_position=use_wrist_position,
            use_wrist_rotation=use_wrist_rotation,
            zero_out_xy_rotation=config.zero_out_xy_rotation,
        ),
        name=f"{config.arm_side}_pose",
    )
    connected_pose = pose.connect({pose_input_device: pose_source.output(pose_input_device)})

    outputs = {
        "ee_pose": connected_pose.output("ee_pose"),
        "gripper_command": connected_gripper.output("gripper_command"),
        "raw_hand": hands.output(hand_device),
    }
    if controllers is not None:
        outputs["raw_controller"] = controllers.output(controller_device)
    return OutputCombiner(outputs)
