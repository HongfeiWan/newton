from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

LinkerL10RetargeterKind = Literal[
    "heuristic",
    "dex_vector",
    "dex_position",
    "dex_dexpilot",
    "l10_adaptive",
    "holo_layered",
]
LinkerL10LimitProfile = Literal["urdf", "right_l10_sdk"]


@dataclass(frozen=True)
class LinkerL10RetargeterConfig:
    kind: LinkerL10RetargeterKind = "heuristic"
    low_pass_alpha: float = 0.2
    scaling_factor: float = 1.0
    max_delta_rad: float = 0.20
    fallback_to_heuristic: bool = True
    limit_profile: LinkerL10LimitProfile = "urdf"

    @classmethod
    def from_env(cls) -> LinkerL10RetargeterConfig:
        return cls(
            kind=_parse_kind(os.environ.get("TELEOP_LINKER_L10_RETARGETER", "heuristic")),
            low_pass_alpha=_float_env("TELEOP_LINKER_L10_RETARGETER_LOW_PASS_ALPHA", 0.2),
            scaling_factor=_float_env("TELEOP_LINKER_L10_RETARGETER_SCALING_FACTOR", 1.0),
            max_delta_rad=_float_env("TELEOP_LINKER_L10_RETARGETER_MAX_DELTA_RAD", 0.20),
            fallback_to_heuristic=_bool_env("TELEOP_LINKER_L10_RETARGETER_FALLBACK_HEURISTIC", True),
            limit_profile=_parse_limit_profile(os.environ.get("TELEOP_LINKER_L10_LIMIT_PROFILE", "urdf")),
        )


def _parse_kind(value: str | None) -> LinkerL10RetargeterKind:
    normalized = (value or "heuristic").strip().lower().replace("-", "_")
    aliases = {
        "baseline": "heuristic",
        "linker_heuristic": "heuristic",
        "dex": "dex_vector",
        "vector": "dex_vector",
        "position": "dex_position",
        "dexpilot": "dex_dexpilot",
        "dex_pilot": "dex_dexpilot",
        "adaptive": "l10_adaptive",
        "wuji_adaptive": "l10_adaptive",
        "holo": "holo_layered",
        "holodex": "holo_layered",
        "holo_dex": "holo_layered",
        "layered": "holo_layered",
    }
    normalized = aliases.get(normalized, normalized)
    valid = {"heuristic", "dex_vector", "dex_position", "dex_dexpilot", "l10_adaptive", "holo_layered"}
    if normalized not in valid:
        raise ValueError(
            "TELEOP_LINKER_L10_RETARGETER must be one of "
            "heuristic, dex_vector, dex_position, dex_dexpilot, l10_adaptive, holo_layered."
        )
    return normalized  # type: ignore[return-value]


def _parse_limit_profile(value: str | None) -> LinkerL10LimitProfile:
    normalized = (value or "urdf").strip().lower().replace("-", "_")
    aliases = {
        "default": "urdf",
        "sim": "urdf",
        "simulation": "urdf",
        "real": "right_l10_sdk",
        "hardware": "right_l10_sdk",
        "right_hand": "right_l10_sdk",
        "l10_sdk": "right_l10_sdk",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"urdf", "right_l10_sdk"}:
        raise ValueError("TELEOP_LINKER_L10_LIMIT_PROFILE must be one of urdf or right_l10_sdk.")
    return normalized  # type: ignore[return-value]


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
