from __future__ import annotations

import inspect
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

from teleop_stack.ik.differential_ik import (
    PositionJacobian,
    PositionKinematicsModel,
    SpatialJacobian,
    SyntheticSevenDofPositionKinematics,
)
from teleop_stack.models import Pose7


RokaeHostIkKinematicsBackend = Literal["synthetic", "rokae_model"]


@dataclass(frozen=True)
class RokaeHostIkKinematicsConfig:
    backend: RokaeHostIkKinematicsBackend = "synthetic"
    provider_spec: str | None = None
    sdk_root: Path | None = None
    xmate_type_name: str = "XMATE3"
    robot_ip: str | None = None
    robot_udp_port: int = 1337
    helper_binary_path: Path | None = None
    helper_build_script_path: Path | None = None


@dataclass(frozen=True)
class RokaeModelProviderConfig:
    sdk_root: Path | None = None
    xmate_type_name: str = "XMATE3"
    robot_ip: str | None = None
    robot_udp_port: int = 1337
    helper_binary_path: Path | None = None
    helper_build_script_path: Path | None = None
    helper_use_sudo: bool = False


class RokaeModelApiLike(Protocol):
    def get_cart_pose(self, joint_positions_rad: tuple[float, ...]) -> Pose7:
        raise NotImplementedError

    def get_position_jacobian(self, joint_positions_rad: tuple[float, ...]) -> PositionJacobian:
        raise NotImplementedError

    def get_spatial_jacobian(self, joint_positions_rad: tuple[float, ...]) -> SpatialJacobian:
        raise NotImplementedError


class RokaeModelPositionKinematics(PositionKinematicsModel):
    def __init__(self, model_api: RokaeModelApiLike):
        self._model_api = model_api

    def forward_pose(self, joint_positions_rad: tuple[float, ...]) -> Pose7:
        return self._model_api.get_cart_pose(joint_positions_rad)

    def position_jacobian(self, joint_positions_rad: tuple[float, ...]) -> PositionJacobian:
        return self._model_api.get_position_jacobian(joint_positions_rad)

    def spatial_jacobian(self, joint_positions_rad: tuple[float, ...]) -> SpatialJacobian:
        return self._model_api.get_spatial_jacobian(joint_positions_rad)


def build_rokae_host_ik_kinematics(config: RokaeHostIkKinematicsConfig) -> PositionKinematicsModel:
    if config.backend == "synthetic":
        return SyntheticSevenDofPositionKinematics()
    if config.backend == "rokae_model":
        return RokaeModelPositionKinematics(
            _load_rokae_model_api(
                provider_spec=config.provider_spec or "teleop_stack.ik.rokae_model_provider:create_rokae_model_provider",
                provider_config=RokaeModelProviderConfig(
                    sdk_root=config.sdk_root,
                    xmate_type_name=config.xmate_type_name,
                    robot_ip=config.robot_ip,
                    robot_udp_port=int(config.robot_udp_port),
                    helper_binary_path=config.helper_binary_path,
                    helper_build_script_path=config.helper_build_script_path,
                ),
            )
        )
    raise ValueError(f"Unsupported Rokae host IK kinematics backend: {config.backend}")


def _load_rokae_model_api(
    *,
    provider_spec: str | None,
    provider_config: RokaeModelProviderConfig,
) -> RokaeModelApiLike:
    if not provider_spec:
        raise RuntimeError(
            "Rokae host IK kinematics backend 'rokae_model' was requested, but no provider factory was configured. "
            "Provide --rokae-host-ik-kinematics-provider module:function or set "
            "RealShadowAssemblyConfig.rokae_host_ik_kinematics_provider_spec. "
            "This repository does not ship a built-in Python binding for RCI model.h."
        )

    module_name, separator, callable_name = provider_spec.partition(":")
    if not separator or not module_name or not callable_name:
        raise ValueError(
            f"Invalid provider spec '{provider_spec}'. Expected format 'module.submodule:factory_name'."
        )

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Failed to import Rokae host IK provider module '{module_name}' from spec '{provider_spec}'."
        ) from exc

    try:
        factory = getattr(module, callable_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Rokae host IK provider spec '{provider_spec}' does not expose callable '{callable_name}'."
        ) from exc
    if not callable(factory):
        raise RuntimeError(
            f"Rokae host IK provider spec '{provider_spec}' resolved to a non-callable object."
        )

    provider = _call_factory(factory, provider_config)
    _validate_provider(provider, provider_spec)
    return provider


def _call_factory(
    factory: Callable[..., object],
    provider_config: RokaeModelProviderConfig,
) -> object:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(provider_config)
    if len(signature.parameters) == 0:
        return factory()
    return factory(provider_config)


def _validate_provider(provider: object, provider_spec: str) -> None:
    for method_name in ("get_cart_pose", "get_position_jacobian"):
        if not callable(getattr(provider, method_name, None)):
            raise RuntimeError(
                f"Rokae host IK provider '{provider_spec}' must return an object with callable '{method_name}()'."
            )
