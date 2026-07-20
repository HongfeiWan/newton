"""Host-side IK and servo abstractions for real-robot teleoperation."""

from teleop_stack.ik.controller import DifferentialIkController, DifferentialIkControllerConfig
from teleop_stack.ik.differential_ik import (
    PositionKinematicsModel,
    SpatialJacobian,
    SyntheticSevenDofPositionKinematics,
)
from teleop_stack.ik.full_pose import (
    FullPoseTask,
    build_full_pose_task,
    solve_full_pose_damped_least_squares_step,
)
from teleop_stack.ik.full_pose_controller import (
    FullPoseDifferentialIkController,
    FullPoseDifferentialIkControllerConfig,
    FullPoseKinematicsModel,
)
from teleop_stack.ik.rokae_kinematics import (
    RokaeHostIkKinematicsBackend,
    RokaeHostIkKinematicsConfig,
    RokaeModelApiLike,
    RokaeModelPositionKinematics,
    RokaeModelProviderConfig,
    build_rokae_host_ik_kinematics,
)
from teleop_stack.ik.types import JointServoStepResult, RobotStateSnapshot, TaskSpaceTarget

__all__ = [
    "DifferentialIkController",
    "DifferentialIkControllerConfig",
    "FullPoseDifferentialIkController",
    "FullPoseDifferentialIkControllerConfig",
    "FullPoseKinematicsModel",
    "FullPoseTask",
    "JointServoStepResult",
    "PositionKinematicsModel",
    "RobotStateSnapshot",
    "RokaeHostIkKinematicsBackend",
    "RokaeHostIkKinematicsConfig",
    "RokaeModelApiLike",
    "RokaeModelPositionKinematics",
    "RokaeModelProviderConfig",
    "SpatialJacobian",
    "SyntheticSevenDofPositionKinematics",
    "TaskSpaceTarget",
    "build_full_pose_task",
    "build_rokae_host_ik_kinematics",
    "solve_full_pose_damped_least_squares_step",
]
