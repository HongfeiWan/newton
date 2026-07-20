from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from teleop_stack.paths import resolve_isaac_teleop_root, resolve_plugin_root_dir

DEFAULT_PLUGIN_NAME = "controller_synthetic_hands"
DEFAULT_PLUGIN_ROOT_ID = "synthetic_hands"


@dataclass(frozen=True)
class QuestInputConfig:
    enable_hands: bool = True
    enable_controllers: bool = True
    enable_head: bool = False
    enable_synthetic_hands_plugin: bool = True
    plugin_name: str = DEFAULT_PLUGIN_NAME
    plugin_root_id: str = DEFAULT_PLUGIN_ROOT_ID
    isaac_teleop_root: str | None = None


@dataclass(frozen=True)
class QuestInputBundle:
    trackers: list[Any]
    sources: dict[str, Any]
    plugins: list[Any]
    isaac_teleop_root: Path


def build_quest_input_bundle(config: QuestInputConfig) -> QuestInputBundle:
    import isaacteleop.deviceio as deviceio
    from isaacteleop.teleop_session_manager import PluginConfig, create_standard_inputs

    isaac_teleop_root = resolve_isaac_teleop_root(config.isaac_teleop_root)
    trackers: list[Any] = []
    if config.enable_hands:
        trackers.append(deviceio.HandTracker())
    if config.enable_controllers:
        trackers.append(deviceio.ControllerTracker())
    if config.enable_head:
        trackers.append(deviceio.HeadTracker())

    sources = create_standard_inputs(trackers)
    plugins: list[Any] = []
    plugin_root_dir = resolve_plugin_root_dir(isaac_teleop_root)
    if config.enable_synthetic_hands_plugin and plugin_root_dir is not None:
        plugins.append(
            PluginConfig(
                plugin_name=config.plugin_name,
                plugin_root_id=config.plugin_root_id,
                search_paths=[plugin_root_dir],
            )
        )
    return QuestInputBundle(trackers=trackers, sources=sources, plugins=plugins, isaac_teleop_root=isaac_teleop_root)
