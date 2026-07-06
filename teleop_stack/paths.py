from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_isaac_teleop_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_value = os.environ.get("ISAAC_TELEOP_ROOT")
    if env_value:
        candidates.append(Path(env_value))
    root = repo_root()
    candidates.extend((root / "external" / "IsaacTeleop", root.parent / "IsaacTeleop"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "Could not locate IsaacTeleop. Set ISAAC_TELEOP_ROOT or clone it into external/IsaacTeleop or ../IsaacTeleop."
    )


def resolve_cloudxr_env_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get("CLOUDXR_ENV_PATH")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (Path.home() / ".cloudxr" / "run" / "cloudxr.env").resolve()


def resolve_plugin_root_dir(isaac_teleop_root: str | os.PathLike[str] | None = None) -> Path | None:
    root = resolve_isaac_teleop_root(isaac_teleop_root)
    plugin_dir = root / "plugins"
    return plugin_dir if plugin_dir.is_dir() else None


def resolve_linkerhand_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_value = os.environ.get("LINKERHAND_URDF_ROOT")
    if env_value:
        candidates.append(Path(env_value))
    root = repo_root()
    candidates.extend(
        (
            root / "assets" / "linkerhand-urdf",
            root / "external" / "linkerhand-urdf",
            root.parent / "linkerhand-urdf",
        )
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "Could not locate linkerhand-urdf. Set LINKERHAND_URDF_ROOT or keep it under assets/linkerhand-urdf."
    )


def resolve_linkerhand_l10_right_urdf(explicit_root: str | os.PathLike[str] | None = None) -> Path:
    root = resolve_linkerhand_root(explicit_root)
    urdf_path = root / "l10" / "right" / "linkerhand_l10_right.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Could not find Linker Hand L10 right URDF at {urdf_path}")
    return urdf_path
