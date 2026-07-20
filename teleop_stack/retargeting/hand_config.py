from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from teleop_stack.models import NamedJointValues
from teleop_stack.paths import resolve_linkerhand_l10_right_urdf

LINKER_L10_NON_THUMB_MCP_PITCH_JOINT_NAMES: tuple[str, ...] = (
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)

LINKER_L10_FINGERTIP_LINK_NAMES: tuple[str, ...] = (
    "thumb_distal",
    "index_distal",
    "middle_distal",
    "ring_distal",
    "pinky_distal",
)

LINKER_L10_FINGERTIP_LOCAL_OFFSETS_M: tuple[tuple[float, float, float], ...] = (
    (-0.008709782, -0.000085963, 0.026135302),
    (-0.005600260, -0.000015293, 0.025815126),
    (-0.005638273, -0.000015293, 0.025806858),
    (-0.005600260, -0.000015293, 0.025815126),
    (-0.005600260, -0.000015293, 0.025815126),
)


@dataclass(frozen=True)
class MimicJointSpec:
    joint_name: str
    source_joint_name: str
    multiplier: float
    offset: float


@dataclass(frozen=True)
class DexHandModelSpec:
    name: str
    urdf_path: Path
    mesh_dir: Path
    base_link_name: str
    fingertip_link_names: tuple[str, ...]
    active_joint_names: tuple[str, ...]
    active_joint_limits: tuple[tuple[float, float], ...]
    mimic_joints: tuple[MimicJointSpec, ...]
    default_open_pose: NamedJointValues
    default_close_pose: NamedJointValues

    def interpolate_synergy(self, close_fraction: float) -> NamedJointValues:
        fraction = max(0.0, min(1.0, float(close_fraction)))
        open_positions = self.default_open_pose.joint_positions
        close_positions = self.default_close_pose.joint_positions
        positions = tuple(
            open_value + fraction * (close_value - open_value)
            for open_value, close_value in zip(open_positions, close_positions, strict=True)
        )
        return NamedJointValues(
            joint_names=self.active_joint_names,
            joint_positions=positions,
        )

    def expand_mimic_joint_values(self, joint_values: NamedJointValues) -> NamedJointValues:
        source_positions = dict(zip(joint_values.joint_names, joint_values.joint_positions, strict=True))
        expanded_joint_names = list(joint_values.joint_names)
        expanded_joint_positions = list(joint_values.joint_positions)
        for mimic_joint in self.mimic_joints:
            source_value = source_positions[mimic_joint.source_joint_name]
            expanded_joint_names.append(mimic_joint.joint_name)
            expanded_joint_positions.append(mimic_joint.multiplier * source_value + mimic_joint.offset)
        return NamedJointValues(
            joint_names=tuple(expanded_joint_names),
            joint_positions=tuple(expanded_joint_positions),
        )


def _default_open_ratio(joint_name: str) -> float:
    # The URDF lower limits correspond to mechanical zero, not a visually natural "open hand" pose.
    # Bias the default pose slightly toward finger spread and light flexion so the hand looks relaxed.
    overrides = {
        "thumb_cmc_roll": 0.10,
        "thumb_cmc_yaw": 0.22,
        "thumb_cmc_pitch": 0.14,
        "index_mcp_roll": 0.16,
        "index_mcp_pitch": 0.10,
        "middle_mcp_pitch": 0.10,
        "ring_mcp_roll": 0.12,
        "ring_mcp_pitch": 0.10,
        "pinky_mcp_roll": 0.18,
        "pinky_mcp_pitch": 0.15,
    }
    return overrides.get(joint_name, 0.10)


def _default_close_ratio(joint_name: str) -> float:
    overrides = {
        "thumb_cmc_roll": 0.18,
        "thumb_cmc_yaw": 0.55,
        "thumb_cmc_pitch": 0.52,
        "index_mcp_roll": 0.10,
        "index_mcp_pitch": 0.80,
        "middle_mcp_pitch": 0.80,
        "ring_mcp_roll": 0.10,
        "ring_mcp_pitch": 0.80,
        "pinky_mcp_roll": 0.18,
        "pinky_mcp_pitch": 0.80,
    }
    return overrides.get(joint_name, 0.75)


def _pose_from_joint_ratios(
    joint_names: list[str],
    joint_limits: list[tuple[float, float]],
    *,
    ratio_fn,
) -> tuple[float, ...]:
    return tuple(
        lower + ratio_fn(joint_name) * (upper - lower)
        for joint_name, (lower, upper) in zip(joint_names, joint_limits, strict=True)
    )


def _override_active_joint_limits(
    joint_names: list[str],
    joint_limits: list[tuple[float, float]],
    limit_overrides: dict[str, tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    if not limit_overrides:
        return joint_limits
    return [limit_overrides.get(name, limits) for name, limits in zip(joint_names, joint_limits, strict=True)]


def load_linker_l10_right_hand_spec(
    explicit_root: str | Path | None = None,
    *,
    active_joint_limit_overrides: dict[str, tuple[float, float]] | None = None,
) -> DexHandModelSpec:
    urdf_path = resolve_linkerhand_l10_right_urdf(explicit_root)
    mesh_dir = urdf_path.parent / "meshes"
    root = ET.parse(urdf_path).getroot()

    active_joint_names: list[str] = []
    active_joint_limits: list[tuple[float, float]] = []
    mimic_joints: list[MimicJointSpec] = []

    for child in root:
        if child.tag != "joint":
            continue

        joint_name = child.attrib["name"]
        limit_tag = child.find("limit")
        if limit_tag is None:
            continue

        lower = float(limit_tag.attrib.get("lower", "0"))
        upper = float(limit_tag.attrib.get("upper", "0"))
        mimic_tag = child.find("mimic")
        if mimic_tag is None:
            active_joint_names.append(joint_name)
            active_joint_limits.append((lower, upper))
        else:
            mimic_joints.append(
                MimicJointSpec(
                    joint_name=joint_name,
                    source_joint_name=mimic_tag.attrib["joint"],
                    multiplier=float(mimic_tag.attrib.get("multiplier", "1.0")),
                    offset=float(mimic_tag.attrib.get("offset", "0.0")),
                )
            )

    active_joint_limits = _override_active_joint_limits(
        active_joint_names,
        active_joint_limits,
        active_joint_limit_overrides,
    )

    open_positions = _pose_from_joint_ratios(
        active_joint_names,
        active_joint_limits,
        ratio_fn=_default_open_ratio,
    )
    close_positions = _pose_from_joint_ratios(
        active_joint_names,
        active_joint_limits,
        ratio_fn=_default_close_ratio,
    )

    return DexHandModelSpec(
        name="linker_hand_l10_right",
        urdf_path=urdf_path,
        mesh_dir=mesh_dir,
        base_link_name="hand_base_link",
        fingertip_link_names=LINKER_L10_FINGERTIP_LINK_NAMES,
        active_joint_names=tuple(active_joint_names),
        active_joint_limits=tuple(active_joint_limits),
        mimic_joints=tuple(mimic_joints),
        default_open_pose=NamedJointValues(
            joint_names=tuple(active_joint_names),
            joint_positions=open_positions,
        ),
        default_close_pose=NamedJointValues(
            joint_names=tuple(active_joint_names),
            joint_positions=close_positions,
        ),
    )


def linker_l10_full_open_pose(spec: DexHandModelSpec) -> NamedJointValues:
    relaxed_by_name = dict(zip(spec.default_open_pose.joint_names, spec.default_open_pose.joint_positions, strict=True))
    limits_by_name = dict(zip(spec.active_joint_names, spec.active_joint_limits, strict=True))
    positions = []
    for joint_name in spec.active_joint_names:
        if joint_name in LINKER_L10_NON_THUMB_MCP_PITCH_JOINT_NAMES:
            positions.append(float(limits_by_name[joint_name][0]))
        else:
            positions.append(float(relaxed_by_name[joint_name]))
    return NamedJointValues(
        joint_names=spec.active_joint_names,
        joint_positions=tuple(positions),
    )
