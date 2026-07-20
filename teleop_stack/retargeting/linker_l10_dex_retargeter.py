from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from teleop_stack.models import NamedJointValues
from teleop_stack.retargeting.hand_config import (
    LINKER_L10_FINGERTIP_LINK_NAMES,
    LINKER_L10_FINGERTIP_LOCAL_OFFSETS_M,
    LINKER_L10_NON_THUMB_MCP_PITCH_JOINT_NAMES,
    DexHandModelSpec,
    linker_l10_full_open_pose,
    load_linker_l10_right_hand_spec,
)
from teleop_stack.retargeting.linker_hand_heuristic import (
    _orientation_thumb_ratios,
    _safe_normalize,
    retarget_openxr_joint_positions_to_linker_l10_right,
)
from teleop_stack.retargeting.linker_l10_retargeter_config import (
    LinkerL10RetargeterConfig,
    LinkerL10RetargeterKind,
)

logger = logging.getLogger(__name__)


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


class LinkerL10Retargeter(Protocol):
    def retarget(
        self,
        joint_positions_xyz: np.ndarray,
        *,
        joint_orientations_xyzw: np.ndarray | None = None,
        joint_valid: np.ndarray | None = None,
    ) -> NamedJointValues: ...


@dataclass(frozen=True)
class _PalmFrame:
    origin_xyz: np.ndarray
    across: np.ndarray
    forward: np.ndarray
    normal: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        return np.column_stack((self.across, self.forward, self.normal))

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin_xyz) @ self.matrix


class LinkerL10HeuristicRetargeter:
    def __init__(self, *, spec: DexHandModelSpec | None = None):
        self._spec = spec

    def retarget(
        self,
        joint_positions_xyz: np.ndarray,
        *,
        joint_orientations_xyzw: np.ndarray | None = None,
        joint_valid: np.ndarray | None = None,
    ) -> NamedJointValues:
        return retarget_openxr_joint_positions_to_linker_l10_right(
            joint_positions_xyz,
            joint_orientations_xyzw=joint_orientations_xyzw,
            joint_valid=joint_valid,
            spec=self._spec,
        )


class LinkerL10DexRetargeter:
    def __init__(
        self,
        *,
        mode: LinkerL10RetargeterKind = "dex_vector",
        spec: DexHandModelSpec | None = None,
        low_pass_alpha: float = 0.2,
        scaling_factor: float = 1.0,
        max_delta_rad: float = 0.20,
        fallback: LinkerL10Retargeter | None = None,
    ):
        if mode not in {"dex_vector", "dex_position", "dex_dexpilot"}:
            raise ValueError(f"Unsupported L10 dex retargeting mode: {mode}")
        self.mode = mode
        self.spec = spec or load_linker_l10_right_hand_spec()
        self.low_pass_alpha = float(low_pass_alpha)
        self.scaling_factor = float(scaling_factor)
        self.max_delta_rad = max(0.0, float(max_delta_rad))
        self.fallback = fallback
        self._retargeting = None
        self._debug_model = None
        self._last_output_by_name: dict[str, float] | None = None
        self.last_debug: object | None = None
        self._warned_unavailable = False

    def retarget(
        self,
        joint_positions_xyz: np.ndarray,
        *,
        joint_orientations_xyzw: np.ndarray | None = None,
        joint_valid: np.ndarray | None = None,
    ) -> NamedJointValues:
        points = np.asarray(joint_positions_xyz, dtype=np.float64)
        if points.shape != (26, 3):
            raise ValueError(f"Expected OpenXR hand positions with shape (26, 3), got {points.shape}")

        if joint_valid is not None:
            valid = np.asarray(joint_valid, dtype=np.uint8)
            if valid.shape != (26,):
                return self._fallback(points, joint_orientations_xyzw, joint_valid)
            if int(valid.sum()) < 10:
                return self._fallback(points, joint_orientations_xyzw, joint_valid)

        try:
            retargeting = self._get_retargeting()
            palm = _build_palm_frame(points)
            local_points = _openxr_palm_local_to_l10_urdf_local(palm.to_local(points))
            ref_value = self._reference_value(local_points)
            qpos = retargeting.retarget(ref_value)
            result = self._qpos_to_named_joint_values(qpos, retargeting.joint_names)
            result = self._fuse_thumb_orientation(
                result,
                joint_orientations_xyzw=joint_orientations_xyzw,
                joint_valid=joint_valid,
                palm=palm,
            )
            result = self._clamp_and_rate_limit(result)
            self._last_output_by_name = dict(zip(result.joint_names, result.joint_positions, strict=True))
            self._update_basic_debug(local_points, result)
            return result
        except Exception as exc:
            if not self._warned_unavailable:
                logger.warning("L10 %s retargeter unavailable; falling back to heuristic. %s", self.mode, exc)
                self._warned_unavailable = True
            return self._fallback(points, joint_orientations_xyzw, joint_valid)

    def _get_retargeting(self):
        if self._retargeting is not None:
            return self._retargeting

        from dex_retargeting.retargeting_config import RetargetingConfig

        cfg = self._dex_config()
        retargeting = RetargetingConfig.from_dict(cfg).build()
        self._set_initial_qpos(retargeting)
        self._retargeting = retargeting
        return retargeting

    def _dex_config(self) -> dict[str, object]:
        active_joint_names = list(self.spec.active_joint_names)
        if self.mode == "dex_position":
            return {
                "type": "position",
                "urdf_path": str(self.spec.urdf_path),
                "target_joint_names": active_joint_names,
                "target_link_names": list(self.spec.fingertip_link_names),
                "target_link_human_indices": np.array(_TIP_INDICES, dtype=int),
                "add_dummy_free_joint": False,
                "scaling_factor": self.scaling_factor,
                "low_pass_alpha": self.low_pass_alpha,
            }
        if self.mode == "dex_dexpilot":
            return {
                "type": "dexpilot",
                "urdf_path": str(self.spec.urdf_path),
                "target_joint_names": active_joint_names,
                "wrist_link_name": self.spec.base_link_name,
                "finger_tip_link_names": list(self.spec.fingertip_link_names),
                "scaling_factor": self.scaling_factor,
                "project_dist": 0.025,
                "escape_dist": 0.055,
                "low_pass_alpha": self.low_pass_alpha,
            }
        return {
            "type": "vector",
            "urdf_path": str(self.spec.urdf_path),
            "target_joint_names": active_joint_names,
            "target_origin_link_names": [self.spec.base_link_name] * len(self.spec.fingertip_link_names),
            "target_task_link_names": list(self.spec.fingertip_link_names),
            "target_link_human_indices": np.array(
                [[_WRIST_INDEX] * len(_TIP_INDICES), list(_TIP_INDICES)],
                dtype=int,
            ),
            "scaling_factor": self.scaling_factor,
            "low_pass_alpha": self.low_pass_alpha,
        }

    def _set_initial_qpos(self, retargeting) -> None:
        qpos = np.zeros((len(retargeting.joint_names),), dtype=np.float32)
        open_by_name = dict(
            zip(self.spec.default_open_pose.joint_names, self.spec.default_open_pose.joint_positions, strict=True)
        )
        for index, name in enumerate(retargeting.joint_names):
            if name in open_by_name:
                qpos[index] = float(open_by_name[name])
        retargeting.set_qpos(qpos)

    def _reference_value(self, local_points: np.ndarray) -> np.ndarray:
        if self.mode == "dex_position":
            return local_points[list(_TIP_INDICES), :].astype(np.float32)
        if self.mode == "dex_dexpilot":
            anchors = np.stack(
                (
                    local_points[_WRIST_INDEX],
                    local_points[_THUMB_TIP_INDEX],
                    local_points[_INDEX_TIP_INDEX],
                    local_points[_MIDDLE_TIP_INDEX],
                    local_points[_RING_TIP_INDEX],
                    local_points[_LITTLE_TIP_INDEX],
                ),
                axis=0,
            )
            origin_indices, task_indices = _dexpilot_link_indices(num_fingers=5)
            return (anchors[task_indices, :] - anchors[origin_indices, :]).astype(np.float32)
        wrist = local_points[_WRIST_INDEX]
        return (local_points[list(_TIP_INDICES), :] - wrist[None, :]).astype(np.float32)

    def _qpos_to_named_joint_values(
        self, qpos: np.ndarray, joint_names: tuple[str, ...] | list[str]
    ) -> NamedJointValues:
        by_name = {str(name): float(value) for name, value in zip(joint_names, qpos, strict=True)}
        fallback_by_name = dict(
            zip(self.spec.default_open_pose.joint_names, self.spec.default_open_pose.joint_positions, strict=True)
        )
        return NamedJointValues(
            joint_names=self.spec.active_joint_names,
            joint_positions=tuple(
                float(by_name.get(name, fallback_by_name[name])) for name in self.spec.active_joint_names
            ),
        )

    def _fuse_thumb_orientation(
        self,
        result: NamedJointValues,
        *,
        joint_orientations_xyzw: np.ndarray | None,
        joint_valid: np.ndarray | None,
        palm: _PalmFrame,
    ) -> NamedJointValues:
        orientation_ratios = _orientation_thumb_ratios(
            joint_orientations_xyzw,
            joint_valid,
            palm_across=palm.across,
            palm_forward=palm.forward,
            palm_normal=palm.normal,
        )
        if not orientation_ratios:
            return result

        positions_by_name = dict(zip(result.joint_names, result.joint_positions, strict=True))
        limits_by_name = dict(zip(self.spec.active_joint_names, self.spec.active_joint_limits, strict=True))
        for joint_name, orientation_weight in (("thumb_cmc_roll", 0.65), ("thumb_cmc_pitch", 0.40)):
            ratio = orientation_ratios.get(joint_name)
            if ratio is None or joint_name not in positions_by_name:
                continue
            lower, upper = limits_by_name[joint_name]
            orientation_value = lower + float(ratio) * (upper - lower)
            positions_by_name[joint_name] = (1.0 - orientation_weight) * positions_by_name[
                joint_name
            ] + orientation_weight * orientation_value

        return NamedJointValues(
            joint_names=result.joint_names,
            joint_positions=tuple(float(positions_by_name[name]) for name in result.joint_names),
        )

    def _clamp_and_rate_limit(self, result: NamedJointValues) -> NamedJointValues:
        limits_by_name = dict(zip(self.spec.active_joint_names, self.spec.active_joint_limits, strict=True))
        positions: list[float] = []
        for name, value in zip(result.joint_names, result.joint_positions, strict=True):
            lower, upper = limits_by_name[name]
            clamped = float(np.clip(value, lower, upper))
            if self._last_output_by_name is not None and name in self._last_output_by_name:
                previous = self._last_output_by_name[name]
                delta = float(np.clip(clamped - previous, -self.max_delta_rad, self.max_delta_rad))
                clamped = previous + delta
            positions.append(clamped)
        return NamedJointValues(joint_names=result.joint_names, joint_positions=tuple(positions))

    def _fallback(
        self,
        joint_positions_xyz: np.ndarray,
        joint_orientations_xyzw: np.ndarray | None,
        joint_valid: np.ndarray | None,
    ) -> NamedJointValues:
        if self.fallback is not None:
            self.last_debug = None
            return self.fallback.retarget(
                joint_positions_xyz,
                joint_orientations_xyzw=joint_orientations_xyzw,
                joint_valid=joint_valid,
            )
        if self._last_output_by_name is not None:
            return NamedJointValues(
                joint_names=self.spec.active_joint_names,
                joint_positions=tuple(float(self._last_output_by_name[name]) for name in self.spec.active_joint_names),
            )
        return self.spec.default_open_pose

    def _update_basic_debug(self, local_points: np.ndarray, result: NamedJointValues) -> None:
        try:
            from teleop_stack.retargeting.linker_l10_adaptive_retargeter import (
                L10AdaptiveRetargeterConfig,
                L10RetargetDebugFrame,
                _L10PinocchioModel,
            )

            thumb_tip = local_points[_THUMB_TIP_INDEX]
            finger_tips = local_points[list(_TIP_INDICES[1:])]
            target_distances = np.linalg.norm(finger_tips - thumb_tip[None, :], axis=1) * 100.0
            cfg = L10AdaptiveRetargeterConfig()
            d1 = np.asarray(cfg.pinch_d1_cm, dtype=np.float64)
            d2 = np.asarray(cfg.pinch_d2_cm, dtype=np.float64)
            alphas_4 = np.clip((d2 - target_distances) / (d2 - d1 + 1e-8), 0.0, cfg.alpha_max)
            alphas = np.concatenate(([float(np.max(alphas_4))], alphas_4))

            if self._debug_model is None:
                self._debug_model = _L10PinocchioModel(self.spec)
            model = self._debug_model
            values_by_name = dict(zip(result.joint_names, result.joint_positions, strict=True))
            qpos = np.array([values_by_name[name] for name in self.spec.active_joint_names], dtype=np.float64)
            positions, _ = model.forward(qpos)
            idx = model.link_index_by_name
            robot_thumb_tip = positions[idx["thumb_distal"]]
            robot_finger_tips = np.array(
                [
                    positions[idx["index_distal"]],
                    positions[idx["middle_distal"]],
                    positions[idx["ring_distal"]],
                    positions[idx["pinky_distal"]],
                ],
                dtype=np.float64,
            )
            robot_distances = np.linalg.norm(robot_finger_tips - robot_thumb_tip[None, :], axis=1) * 100.0
            self.last_debug = L10RetargetDebugFrame(
                optimizer_success=True,
                optimizer_cost=float("nan"),
                optimizer_iterations=0,
                pinch_alphas=tuple(float(value) for value in alphas),
                loss_terms={},
                target_pinch_distances_cm=tuple(float(value) for value in target_distances),
                robot_pinch_distances_cm=tuple(float(value) for value in robot_distances),
            )
        except Exception:
            self.last_debug = None


class LinkerL10HoloLayeredRetargeter:
    """Holo-Dex-style layered retargeting for the current Linker L10 setup.

    Non-thumb fingers use the fast direct joint-angle heuristic. The thumb is
    then refined by solving only the L10 thumb joints to match a palm-local
    fingertip target. The target is scaled into the robot hand workspace and,
    during pinches, is anchored to the nearest non-thumb fingertip. This mirrors
    the Holo-Dex split while avoiding Allegro-specific calibration files.
    """

    _THUMB_JOINT_NAMES = ("thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch")
    _PINCH_FINGER_JOINT_NAMES = (
        ("index_mcp_roll", "index_mcp_pitch"),
        ("middle_mcp_pitch",),
        ("ring_mcp_roll", "ring_mcp_pitch"),
        ("pinky_mcp_roll", "pinky_mcp_pitch"),
    )
    _PINCH_FINGER_LINK_NAMES = ("index_distal", "middle_distal", "ring_distal", "pinky_distal")
    _TIP_LINK_LOCAL_OFFSETS_M = {
        name: np.asarray(offset, dtype=np.float64)
        for name, offset in zip(
            LINKER_L10_FINGERTIP_LINK_NAMES,
            LINKER_L10_FINGERTIP_LOCAL_OFFSETS_M,
            strict=True,
        )
    }
    _PINCH_BLEND_NEAR_M = 0.007
    _DEFAULT_PINCH_ACTIVATE_M = 0.015
    _DEFAULT_PINCH_CONTACT_GAP_M = 0.0
    _PINCH_PARTNER_TIP_WEIGHT = 2.0
    _PINCH_PARTNER_LATERAL_WEIGHT = 4.0
    _PINCH_DISTANCE_WEIGHT = 2000.0
    _PINCH_PARTNER_ROLL_REGULARIZATION_SCALE = 0.20
    # Initial guesses only. These are never copied directly into the output; the
    # pair IK still solves from the live hand pose and only changes the thumb and
    # current pinch partner joints.
    _PAIR_IK_SEEDS = (
        {
            "thumb_cmc_roll": 0.120,
            "thumb_cmc_yaw": 1.190,
            "thumb_cmc_pitch": 0.270,
            "index_mcp_roll": 0.000,
            "index_mcp_pitch": 0.760,
        },
        {
            "thumb_cmc_roll": 0.290,
            "thumb_cmc_yaw": 1.190,
            "thumb_cmc_pitch": 0.290,
            "middle_mcp_pitch": 0.775,
        },
        {
            "thumb_cmc_roll": 0.475,
            "thumb_cmc_yaw": 1.190,
            "thumb_cmc_pitch": 0.285,
            "ring_mcp_roll": 0.000,
            "ring_mcp_pitch": 0.760,
        },
        {
            "thumb_cmc_roll": 0.705,
            "thumb_cmc_yaw": 1.130,
            "thumb_cmc_pitch": 0.280,
            "pinky_mcp_roll": 0.000,
            "pinky_mcp_pitch": 0.760,
        },
    )

    def __init__(
        self,
        *,
        spec: DexHandModelSpec | None = None,
        low_pass_alpha: float = 0.2,
        scaling_factor: float = 1.0,
        max_delta_rad: float = 0.20,
        fallback: LinkerL10Retargeter | None = None,
    ):
        self.spec = spec or load_linker_l10_right_hand_spec()
        self.low_pass_alpha = float(low_pass_alpha)
        self.scaling_factor = float(scaling_factor)
        self.max_delta_rad = max(0.0, float(max_delta_rad))
        self.pinch_contact_gap_m = max(
            0.0,
            _float_env("TELEOP_LINKER_L10_HOLO_PINCH_CONTACT_GAP_M", self._DEFAULT_PINCH_CONTACT_GAP_M),
        )
        self.pinch_activate_distance_m = max(
            self._PINCH_BLEND_NEAR_M + 1e-6,
            _float_env("TELEOP_LINKER_L10_HOLO_PINCH_ACTIVATE_M", self._DEFAULT_PINCH_ACTIVATE_M),
        )
        self.tip_offset_scale = max(0.0, _float_env("TELEOP_LINKER_L10_HOLO_TIP_OFFSET_SCALE", 1.0))
        self.fallback = fallback
        self._direct_retargeter = LinkerL10HeuristicRetargeter(spec=self.spec)
        self._model = None
        self._filtered_qpos: np.ndarray | None = None
        self._last_output_by_name: dict[str, float] | None = None
        self._last_optimizer_success = False
        self._last_optimizer_cost = float("nan")
        self._last_optimizer_iterations = 0
        self._last_thumb_target_debug: dict[str, object] = {}
        self._warned_unavailable = False
        self.last_debug: object | None = None

    def retarget(
        self,
        joint_positions_xyz: np.ndarray,
        *,
        joint_orientations_xyzw: np.ndarray | None = None,
        joint_valid: np.ndarray | None = None,
    ) -> NamedJointValues:
        points = np.asarray(joint_positions_xyz, dtype=np.float64)
        if points.shape != (26, 3):
            raise ValueError(f"Expected OpenXR hand positions with shape (26, 3), got {points.shape}")

        if joint_valid is not None:
            valid = np.asarray(joint_valid, dtype=np.uint8)
            if valid.shape != (26,) or int(valid.sum()) < 10:
                return self._fallback(points, joint_orientations_xyzw, joint_valid)

        direct = self._direct_retargeter.retarget(
            points,
            joint_orientations_xyzw=joint_orientations_xyzw,
            joint_valid=joint_valid,
        )
        try:
            qpos = self._direct_qpos(direct)
            qpos = self._solve_pinch_aware_ik(points, qpos)
            if self._pinch_alpha() >= 0.98:
                qpos = self._clip_and_reset_filter(qpos)
            else:
                qpos = self._filter_and_rate_limit(qpos)
            result = NamedJointValues(
                joint_names=self.spec.active_joint_names,
                joint_positions=tuple(float(value) for value in qpos),
            )
            self._last_output_by_name = dict(zip(result.joint_names, result.joint_positions, strict=True))
            self._update_debug(points, qpos, optimizer_success=True, fallback_reason=None)
            return result
        except Exception as exc:
            if not self._warned_unavailable:
                logger.warning("L10 holo_layered retargeter unavailable; falling back. %s", exc)
                self._warned_unavailable = True
            self._update_debug(points, self._direct_qpos(direct), optimizer_success=False, fallback_reason=str(exc))
            return self._fallback(points, joint_orientations_xyzw, joint_valid, direct=direct)

    def _direct_qpos(self, direct: NamedJointValues) -> np.ndarray:
        direct_by_name = dict(zip(direct.joint_names, direct.joint_positions, strict=True))
        open_by_name = dict(
            zip(self.spec.default_open_pose.joint_names, self.spec.default_open_pose.joint_positions, strict=True)
        )
        return np.array(
            [float(direct_by_name.get(name, open_by_name[name])) for name in self.spec.active_joint_names],
            dtype=np.float64,
        )

    def _solve_pinch_aware_ik(self, points: np.ndarray, seed_qpos: np.ndarray) -> np.ndarray:
        from scipy.optimize import minimize

        model = self._get_model()
        palm = _build_palm_frame(points)
        local_points = _openxr_palm_local_to_l10_urdf_local(palm.to_local(points))
        target_tip = self._thumb_target_in_robot_workspace(local_points, seed_qpos)

        nearest_finger_index = int(self._last_thumb_target_debug.get("nearest_finger_index", -1))
        pinch_alpha = float(self._last_thumb_target_debug.get("pinch_alpha", 0.0))
        optimized_joint_names = list(self._THUMB_JOINT_NAMES)
        if 0 <= nearest_finger_index < len(self._PINCH_FINGER_JOINT_NAMES) and pinch_alpha >= 0.15:
            optimized_joint_names.extend(self._PINCH_FINGER_JOINT_NAMES[nearest_finger_index])

        optimized_indices = [self.spec.active_joint_names.index(name) for name in optimized_joint_names]
        limits = np.asarray(self.spec.active_joint_limits, dtype=np.float64)
        bounds = tuple((float(limits[index, 0]), float(limits[index, 1])) for index in optimized_indices)
        seed_x0 = seed_qpos[optimized_indices].copy()
        regularization_weights = np.ones_like(seed_x0)
        partner_roll_joint_name = self._pinch_partner_roll_joint_name(nearest_finger_index)
        if partner_roll_joint_name in optimized_joint_names:
            roll_position = optimized_joint_names.index(partner_roll_joint_name)
            regularization_weights[roll_position] = self._PINCH_PARTNER_ROLL_REGULARIZATION_SCALE

        target_vec = np.asarray(
            self._last_thumb_target_debug.get("nearest_human_pinch_vector_robot_m", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        target_midpoint = np.asarray(
            self._last_thumb_target_debug.get("pinch_midpoint_target_xyz", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        target_partner_tip = target_midpoint + 0.5 * target_vec
        target_distance = self.pinch_contact_gap_m
        partner_lateral_direction = None
        if (
            0 <= nearest_finger_index < len(self._PINCH_FINGER_LINK_NAMES)
            and pinch_alpha > 0.0
            and partner_roll_joint_name is not None
        ):
            partner_lateral_direction = self._joint_tip_sensitivity_direction(
                model,
                seed_qpos,
                joint_name=partner_roll_joint_name,
                link_name=self._PINCH_FINGER_LINK_NAMES[nearest_finger_index],
            )

        def objective(optimized_values: np.ndarray, regularization_anchor: np.ndarray) -> float:
            qpos = seed_qpos.copy()
            qpos[optimized_indices] = optimized_values
            positions, _ = model.forward(qpos)
            robot_thumb_tip = self._robot_tip_position(model, positions, "thumb_distal")
            tip_loss = float(np.sum((robot_thumb_tip - target_tip) ** 2))
            pinch_loss = 0.0
            if 0 <= nearest_finger_index < len(self._PINCH_FINGER_LINK_NAMES) and pinch_alpha > 0.0:
                robot_finger_tip = self._robot_tip_position(
                    model,
                    positions,
                    self._PINCH_FINGER_LINK_NAMES[nearest_finger_index],
                )
                robot_vec = robot_finger_tip - robot_thumb_tip
                robot_midpoint = 0.5 * (robot_thumb_tip + robot_finger_tip)
                vec_loss = float(np.sum((robot_vec - target_vec) ** 2))
                midpoint_loss = float(np.sum((robot_midpoint - target_midpoint) ** 2))
                partner_tip_loss = float(np.sum((robot_finger_tip - target_partner_tip) ** 2))
                lateral_loss = 0.0
                if partner_lateral_direction is not None:
                    lateral_error = float(np.dot(robot_finger_tip - target_partner_tip, partner_lateral_direction))
                    lateral_loss = lateral_error * lateral_error
                robot_distance = float(np.linalg.norm(robot_vec))
                distance_loss = (robot_distance - target_distance) * (robot_distance - target_distance)
                gap_violation = max(0.0, self.pinch_contact_gap_m - robot_distance)
                gap_loss = gap_violation * gap_violation
                pinch_loss = pinch_alpha * (
                    4.0 * vec_loss
                    + 1.5 * midpoint_loss
                    + self._PINCH_PARTNER_TIP_WEIGHT * partner_tip_loss
                    + self._PINCH_PARTNER_LATERAL_WEIGHT * lateral_loss
                    + self._PINCH_DISTANCE_WEIGHT * distance_loss
                    + 6.0 * gap_loss
                )
            tip_loss_weight = 1.0 - 0.85 * float(np.clip(pinch_alpha, 0.0, 1.0))
            regularization_scale = 0.0005 if pinch_alpha >= 0.98 else 0.003
            regularization = regularization_scale * float(
                np.sum(regularization_weights * (optimized_values - regularization_anchor) ** 2)
            )
            return tip_loss_weight * tip_loss + pinch_loss + regularization

        maxiter = 80 if pinch_alpha >= 0.98 else 30
        ftol = 1e-7 if pinch_alpha >= 0.98 else 1e-5
        initial_guesses = [seed_x0]
        if pinch_alpha >= 0.85:
            initial_guesses.append(self._pair_ik_seed_qpos(seed_qpos, nearest_finger_index)[optimized_indices])
        if self._last_output_by_name is not None:
            last_qpos = np.asarray(
                [self._last_output_by_name[name] for name in self.spec.active_joint_names],
                dtype=np.float64,
            )
            initial_guesses.append(last_qpos[optimized_indices])

        best: tuple[float, np.ndarray, bool, int] | None = None
        seen: set[tuple[float, ...]] = set()
        for initial_guess in initial_guesses:
            x0 = np.clip(np.asarray(initial_guess, dtype=np.float64), [b[0] for b in bounds], [b[1] for b in bounds])
            key = tuple(float(f"{value:.6f}") for value in x0)
            if key in seen:
                continue
            seen.add(key)
            result = minimize(
                lambda values: objective(values, seed_x0),
                x0=x0,
                bounds=bounds,
                method="SLSQP",
                options={"maxiter": maxiter, "ftol": ftol, "disp": False},
            )
            cost = float(objective(np.asarray(result.x, dtype=np.float64), seed_x0))
            if best is None or cost < best[0]:
                best = (
                    cost,
                    np.asarray(result.x, dtype=np.float64),
                    bool(result.success),
                    int(getattr(result, "nit", 0)),
                )
        if best is None:
            raise RuntimeError("L10 holo_layered pinch IK did not run any initial guess")

        qpos = seed_qpos.copy()
        qpos[optimized_indices] = best[1]
        self._last_optimizer_success = bool(best[2])
        self._last_optimizer_cost = float(best[0])
        self._last_optimizer_iterations = int(best[3])
        return qpos

    def _pinch_alpha(self) -> float:
        return float(self._last_thumb_target_debug.get("pinch_alpha", 0.0))

    def _pair_ik_seed_qpos(self, seed_qpos: np.ndarray, nearest_finger_index: int) -> np.ndarray:
        if not 0 <= nearest_finger_index < len(self._PAIR_IK_SEEDS):
            return seed_qpos
        seeded = np.asarray(seed_qpos, dtype=np.float64).copy()
        limits_by_name = dict(zip(self.spec.active_joint_names, self.spec.active_joint_limits, strict=True))
        active_index_by_name = {name: index for index, name in enumerate(self.spec.active_joint_names)}
        for joint_name, target_value in self._PAIR_IK_SEEDS[nearest_finger_index].items():
            joint_index = active_index_by_name.get(joint_name)
            if joint_index is None:
                continue
            lower, upper = limits_by_name[joint_name]
            seeded[joint_index] = float(np.clip(float(target_value), lower, upper))
        return seeded

    def _apply_full_open_target(self, points: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        from teleop_stack.retargeting.linker_l10_adaptive_retargeter import _full_open_alpha_from_local_points

        palm = _build_palm_frame(points)
        local_points = _openxr_palm_local_to_l10_urdf_local(palm.to_local(points))
        full_open_alpha = _full_open_alpha_from_local_points(
            local_points,
            max_pinch_alpha=float(self._last_thumb_target_debug.get("pinch_alpha", 0.0)),
        )
        if full_open_alpha <= 0.0:
            return qpos

        full_open_by_name = dict(
            zip(
                self.spec.active_joint_names,
                linker_l10_full_open_pose(self.spec).joint_positions,
                strict=True,
            )
        )
        adjusted = np.asarray(qpos, dtype=np.float64).copy()
        for joint_name in LINKER_L10_NON_THUMB_MCP_PITCH_JOINT_NAMES:
            if joint_name not in self.spec.active_joint_names:
                continue
            joint_index = self.spec.active_joint_names.index(joint_name)
            adjusted[joint_index] = (1.0 - full_open_alpha) * adjusted[
                joint_index
            ] + full_open_alpha * full_open_by_name[joint_name]
        if self._filtered_qpos is not None:
            self._filtered_qpos = adjusted.copy()
        return adjusted

    def _thumb_target_in_robot_workspace(self, local_points: np.ndarray, seed_qpos: np.ndarray) -> np.ndarray:
        model = self._get_model()
        positions, _ = model.forward(seed_qpos)
        idx = model.link_index_by_name
        robot_base = positions[idx["hand_base_link"]]
        robot_non_thumb_tips = np.asarray(
            [
                self._robot_tip_position(model, positions, "index_distal"),
                self._robot_tip_position(model, positions, "middle_distal"),
                self._robot_tip_position(model, positions, "ring_distal"),
                self._robot_tip_position(model, positions, "pinky_distal"),
            ],
            dtype=np.float64,
        )

        human_wrist = local_points[_WRIST_INDEX]
        human_thumb_tip = local_points[_THUMB_TIP_INDEX]
        human_non_thumb_tips = local_points[list(_TIP_INDICES[1:])]
        human_tip_vectors = human_non_thumb_tips - human_wrist[None, :]
        robot_tip_vectors = robot_non_thumb_tips - robot_base[None, :]
        human_scale = float(np.median(np.linalg.norm(human_tip_vectors, axis=1)))
        robot_scale = float(np.median(np.linalg.norm(robot_tip_vectors, axis=1)))
        scale = 1.0 if human_scale <= 1e-6 or robot_scale <= 1e-6 else robot_scale / human_scale
        scale *= self.scaling_factor

        base_target = robot_base + (human_thumb_tip - human_wrist) * scale
        distances = np.linalg.norm(human_non_thumb_tips - human_thumb_tip[None, :], axis=1)
        nearest_index = int(np.argmin(distances))
        weights = 1.0 / np.maximum(distances, 1e-4) ** 2
        weights = weights / float(np.sum(weights))
        human_anchor = np.sum(human_non_thumb_tips * weights[:, None], axis=0)
        robot_anchor = np.sum(robot_non_thumb_tips * weights[:, None], axis=0)
        nearest_raw_vec = (human_non_thumb_tips[nearest_index] - human_thumb_tip) * scale
        nearest_seed_vec = robot_non_thumb_tips[nearest_index] - self._robot_tip_position(
            model,
            positions,
            "thumb_distal",
        )
        nearest_distance = float(np.min(distances))
        if nearest_distance <= self._PINCH_BLEND_NEAR_M:
            nearest_target_vec = np.zeros((3,), dtype=np.float64)
        else:
            nearest_target_vec = self._with_contact_gap(nearest_raw_vec, fallback_direction=nearest_seed_vec)
        nearest_human_midpoint = 0.5 * (human_thumb_tip + human_non_thumb_tips[nearest_index])
        nearest_midpoint_target = robot_base + (nearest_human_midpoint - human_wrist) * scale
        weighted_pinched_thumb_target = robot_anchor + (human_thumb_tip - human_anchor) * scale
        nearest_pinched_thumb_target = nearest_midpoint_target - 0.5 * nearest_target_vec
        pinch_target = 0.35 * weighted_pinched_thumb_target + 0.65 * nearest_pinched_thumb_target

        denom = self.pinch_activate_distance_m - self._PINCH_BLEND_NEAR_M
        pinch_alpha = float(np.clip((self.pinch_activate_distance_m - nearest_distance) / max(denom, 1e-6), 0.0, 1.0))
        target = (1.0 - pinch_alpha) * base_target + pinch_alpha * pinch_target
        self._last_thumb_target_debug = {
            "target_tip_xyz": [float(value) for value in target],
            "base_target_xyz": [float(value) for value in base_target],
            "pinch_target_xyz": [float(value) for value in pinch_target],
            "pinch_alpha": pinch_alpha,
            "nearest_human_pinch_distance_m": nearest_distance,
            "nearest_finger_index": nearest_index,
            "nearest_human_pinch_vector_robot_m": [float(value) for value in nearest_target_vec],
            "pinch_midpoint_target_xyz": [float(value) for value in nearest_midpoint_target],
            "pinch_activate_distance_m": self.pinch_activate_distance_m,
            "pinch_contact_gap_m": self.pinch_contact_gap_m,
            "raw_nearest_human_pinch_vector_norm_m": float(np.linalg.norm(nearest_raw_vec)),
            "workspace_scale": float(scale),
            "non_thumb_anchor_weights": [float(value) for value in weights],
        }
        return target

    def _robot_tip_position(self, model, positions: np.ndarray, link_name: str) -> np.ndarray:
        origin = np.asarray(positions[model.link_index_by_name[link_name]], dtype=np.float64)
        offset = self._TIP_LINK_LOCAL_OFFSETS_M.get(link_name)
        if offset is None or self.tip_offset_scale <= 0.0:
            return origin
        offset = offset * self.tip_offset_scale
        if hasattr(model, "data") and hasattr(model, "link_frame_ids") and hasattr(model, "link_names"):
            try:
                frame_id = model.link_frame_ids[model.link_names.index(link_name)]
                rotation = np.asarray(model.data.oMf[frame_id].rotation, dtype=np.float64)
                return origin + rotation @ offset
            except Exception:
                pass
        return origin + offset

    def _pinch_partner_roll_joint_name(self, finger_index: int) -> str | None:
        if not 0 <= finger_index < len(self._PINCH_FINGER_JOINT_NAMES):
            return None
        for joint_name in self._PINCH_FINGER_JOINT_NAMES[finger_index]:
            if joint_name.endswith("_mcp_roll"):
                return joint_name
        return None

    def _joint_tip_sensitivity_direction(
        self,
        model,
        seed_qpos: np.ndarray,
        *,
        joint_name: str,
        link_name: str,
    ) -> np.ndarray | None:
        if joint_name not in self.spec.active_joint_names:
            return None
        joint_index = self.spec.active_joint_names.index(joint_name)
        lower, upper = self.spec.active_joint_limits[joint_index]
        span = float(upper - lower)
        if span <= 1e-9:
            return None

        center = float(seed_qpos[joint_index])
        eps = min(1e-3, max(1e-5, 0.01 * span))
        plus = min(float(upper), center + eps)
        minus = max(float(lower), center - eps)
        if plus - minus <= 1e-9:
            return None

        q_plus = seed_qpos.copy()
        q_minus = seed_qpos.copy()
        q_plus[joint_index] = plus
        q_minus[joint_index] = minus
        positions_plus, _ = model.forward(q_plus)
        positions_minus, _ = model.forward(q_minus)
        direction = self._robot_tip_position(model, positions_plus, link_name) - self._robot_tip_position(
            model,
            positions_minus,
            link_name,
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return None
        return direction / norm

    def _with_contact_gap(self, vector: np.ndarray, *, fallback_direction: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm >= self.pinch_contact_gap_m:
            return vector
        direction = _safe_normalize(vector)
        if not np.any(direction):
            direction = _safe_normalize(np.asarray(fallback_direction, dtype=np.float64))
        if not np.any(direction):
            direction = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        return direction * self.pinch_contact_gap_m

    def _get_model(self):
        if self._model is None:
            from teleop_stack.retargeting.linker_l10_adaptive_retargeter import _L10PinocchioModel

            self._model = _L10PinocchioModel(self.spec)
        return self._model

    def _filter_and_rate_limit(self, qpos: np.ndarray) -> np.ndarray:
        limits = np.asarray(self.spec.active_joint_limits, dtype=np.float64)
        filtered = np.clip(np.asarray(qpos, dtype=np.float64), limits[:, 0], limits[:, 1])
        if self._filtered_qpos is None:
            self._filtered_qpos = filtered.copy()
        else:
            alpha = float(np.clip(self.low_pass_alpha, 0.0, 1.0))
            self._filtered_qpos = self._filtered_qpos + alpha * (filtered - self._filtered_qpos)
        filtered = self._filtered_qpos.copy()
        if self._last_output_by_name is not None and self.max_delta_rad > 0.0:
            previous = np.array(
                [self._last_output_by_name[name] for name in self.spec.active_joint_names],
                dtype=np.float64,
            )
            delta = np.clip(filtered - previous, -self.max_delta_rad, self.max_delta_rad)
            filtered = previous + delta
            self._filtered_qpos = filtered.copy()
        return np.clip(filtered, limits[:, 0], limits[:, 1])

    def _clip_and_reset_filter(self, qpos: np.ndarray) -> np.ndarray:
        limits = np.asarray(self.spec.active_joint_limits, dtype=np.float64)
        filtered = np.clip(np.asarray(qpos, dtype=np.float64), limits[:, 0], limits[:, 1])
        self._filtered_qpos = filtered.copy()
        return filtered

    def _fallback(
        self,
        joint_positions_xyz: np.ndarray,
        joint_orientations_xyzw: np.ndarray | None,
        joint_valid: np.ndarray | None,
        *,
        direct: NamedJointValues | None = None,
    ) -> NamedJointValues:
        if self.fallback is not None:
            return self.fallback.retarget(
                joint_positions_xyz,
                joint_orientations_xyzw=joint_orientations_xyzw,
                joint_valid=joint_valid,
            )
        if direct is not None:
            return direct
        if self._last_output_by_name is not None:
            return NamedJointValues(
                joint_names=self.spec.active_joint_names,
                joint_positions=tuple(float(self._last_output_by_name[name]) for name in self.spec.active_joint_names),
            )
        return self.spec.default_open_pose

    def _update_debug(
        self,
        points: np.ndarray,
        qpos: np.ndarray,
        *,
        optimizer_success: bool,
        fallback_reason: str | None,
    ) -> None:
        try:
            from teleop_stack.retargeting.linker_l10_adaptive_retargeter import L10RetargetDebugFrame

            palm = _build_palm_frame(points)
            local_points = _openxr_palm_local_to_l10_urdf_local(palm.to_local(points))
            thumb_tip = local_points[_THUMB_TIP_INDEX]
            finger_tips = local_points[list(_TIP_INDICES[1:])]
            target_distances = np.linalg.norm(finger_tips - thumb_tip[None, :], axis=1) * 100.0

            model = self._get_model()
            positions, _ = model.forward(qpos)
            robot_thumb_tip = self._robot_tip_position(model, positions, "thumb_distal")
            robot_finger_tips = np.array(
                [
                    self._robot_tip_position(model, positions, "index_distal"),
                    self._robot_tip_position(model, positions, "middle_distal"),
                    self._robot_tip_position(model, positions, "ring_distal"),
                    self._robot_tip_position(model, positions, "pinky_distal"),
                ],
                dtype=np.float64,
            )
            robot_distances = np.linalg.norm(robot_finger_tips - robot_thumb_tip[None, :], axis=1) * 100.0
            pinch_alpha = self._pinch_alpha()
            nearest_finger_index = int(self._last_thumb_target_debug.get("nearest_finger_index", -1))
            pinch_alphas = [0.0] * 5
            if 0 <= nearest_finger_index < 4:
                pinch_alphas[0] = pinch_alpha
                pinch_alphas[nearest_finger_index + 1] = pinch_alpha
            self.last_debug = L10RetargetDebugFrame(
                optimizer_success=optimizer_success,
                optimizer_cost=float(self._last_optimizer_cost),
                optimizer_iterations=int(self._last_optimizer_iterations),
                pinch_alphas=tuple(pinch_alphas),
                loss_terms={
                    "thumb_tip_ik": float(self._last_optimizer_cost),
                    **{
                        f"holo_thumb_{name}": float(value)
                        for name, value in self._last_thumb_target_debug.items()
                        if isinstance(value, (int, float))
                    },
                },
                target_pinch_distances_cm=tuple(float(value) for value in target_distances),
                robot_pinch_distances_cm=tuple(float(value) for value in robot_distances),
                fallback_reason=fallback_reason,
                qpos_by_name=dict(zip(self.spec.active_joint_names, (float(value) for value in qpos), strict=True)),
            )
        except Exception:
            self.last_debug = None


def build_linker_l10_retargeter(
    config: LinkerL10RetargeterConfig | None = None,
    *,
    spec: DexHandModelSpec | None = None,
) -> LinkerL10Retargeter:
    resolved_config = config or LinkerL10RetargeterConfig.from_env()
    hand_spec = spec or _load_l10_spec_for_limit_profile(resolved_config.limit_profile)
    heuristic = LinkerL10HeuristicRetargeter(spec=hand_spec)
    if resolved_config.kind == "heuristic":
        return heuristic
    if resolved_config.kind == "l10_adaptive":
        from teleop_stack.retargeting.linker_l10_adaptive_retargeter import (
            L10AdaptiveRetargeterConfig,
            LinkerL10AdaptiveRetargeter,
        )

        return LinkerL10AdaptiveRetargeter(
            spec=hand_spec,
            config=L10AdaptiveRetargeterConfig(
                low_pass_alpha=resolved_config.low_pass_alpha,
                scaling_factor=resolved_config.scaling_factor,
                max_delta_rad=resolved_config.max_delta_rad,
            ),
            fallback=heuristic.retarget if resolved_config.fallback_to_heuristic else None,
        )
    if resolved_config.kind == "holo_layered":
        return LinkerL10HoloLayeredRetargeter(
            spec=hand_spec,
            low_pass_alpha=resolved_config.low_pass_alpha,
            scaling_factor=resolved_config.scaling_factor,
            max_delta_rad=resolved_config.max_delta_rad,
            fallback=heuristic if resolved_config.fallback_to_heuristic else None,
        )
    return LinkerL10DexRetargeter(
        mode=resolved_config.kind,
        spec=hand_spec,
        low_pass_alpha=resolved_config.low_pass_alpha,
        scaling_factor=resolved_config.scaling_factor,
        max_delta_rad=resolved_config.max_delta_rad,
        fallback=heuristic if resolved_config.fallback_to_heuristic else None,
    )


def _load_l10_spec_for_limit_profile(limit_profile: str) -> DexHandModelSpec:
    if limit_profile == "right_l10_sdk":
        from teleop_stack.robots.linker_hand_l10_sdk import L10_RIGHT_CALIBRATION

        return load_linker_l10_right_hand_spec(
            active_joint_limit_overrides={
                joint.joint_name: (joint.lower_rad, joint.upper_rad) for joint in L10_RIGHT_CALIBRATION
            }
        )
    return load_linker_l10_right_hand_spec()


def retarget_openxr_hand_to_linker_l10_right(
    joint_positions_xyz: np.ndarray,
    *,
    joint_orientations_xyzw: np.ndarray | None = None,
    joint_valid: np.ndarray | None = None,
    retargeter: LinkerL10Retargeter | None = None,
) -> NamedJointValues:
    active_retargeter = retargeter or _get_cached_env_retargeter()
    return active_retargeter.retarget(
        joint_positions_xyz,
        joint_orientations_xyzw=joint_orientations_xyzw,
        joint_valid=joint_valid,
    )


def reset_linker_l10_retargeter_cache() -> None:
    global _CACHED_ENV_KEY, _CACHED_ENV_RETARGETER
    _CACHED_ENV_KEY = None
    _CACHED_ENV_RETARGETER = None


def get_cached_linker_l10_retargeter_debug() -> object | None:
    if _CACHED_ENV_RETARGETER is None:
        return None
    return getattr(_CACHED_ENV_RETARGETER, "last_debug", None)


def _get_cached_env_retargeter() -> LinkerL10Retargeter:
    global _CACHED_ENV_KEY, _CACHED_ENV_RETARGETER
    config = LinkerL10RetargeterConfig.from_env()
    key = (
        config.kind,
        config.low_pass_alpha,
        config.scaling_factor,
        config.max_delta_rad,
        config.fallback_to_heuristic,
        config.limit_profile,
    )
    if _CACHED_ENV_RETARGETER is None or _CACHED_ENV_KEY != key:
        _CACHED_ENV_RETARGETER = build_linker_l10_retargeter(config)
        _CACHED_ENV_KEY = key
    return _CACHED_ENV_RETARGETER


def _build_palm_frame(points: np.ndarray) -> _PalmFrame:
    wrist = points[_WRIST_INDEX]
    palm_forward = _safe_normalize(points[_MIDDLE_PROXIMAL_INDEX] - wrist)
    palm_across = _safe_normalize(points[_INDEX_PROXIMAL_INDEX] - points[_LITTLE_PROXIMAL_INDEX])
    palm_normal = _safe_normalize(np.cross(palm_across, palm_forward))
    palm_across = _safe_normalize(np.cross(palm_forward, palm_normal))
    if not np.any(palm_forward) or not np.any(palm_across) or not np.any(palm_normal):
        raise ValueError("OpenXR hand points do not define a valid palm frame")
    return _PalmFrame(origin_xyz=wrist, across=palm_across, forward=palm_forward, normal=palm_normal)


def _openxr_palm_local_to_l10_urdf_local(local_points: np.ndarray) -> np.ndarray:
    # OpenXR palm local columns are (across, forward, normal). The L10 URDF uses
    # +Z as the main finger-forward direction and +Y for index-side spread.
    return np.asarray(local_points, dtype=np.float64)[:, (2, 0, 1)]


def _dexpilot_link_indices(num_fingers: int) -> tuple[np.ndarray, np.ndarray]:
    origin_link_index: list[int] = []
    task_link_index: list[int] = []
    for i in range(1, num_fingers):
        for j in range(i + 1, num_fingers + 1):
            origin_link_index.append(j)
            task_link_index.append(i)
    for i in range(1, num_fingers + 1):
        origin_link_index.append(0)
        task_link_index.append(i)
    return np.asarray(origin_link_index, dtype=int), np.asarray(task_link_index, dtype=int)


_CACHED_ENV_KEY: tuple[object, ...] | None = None
_CACHED_ENV_RETARGETER: LinkerL10Retargeter | None = None

_WRIST_INDEX = 1
_THUMB_TIP_INDEX = 5
_INDEX_PROXIMAL_INDEX = 7
_INDEX_TIP_INDEX = 10
_MIDDLE_PROXIMAL_INDEX = 12
_MIDDLE_TIP_INDEX = 15
_RING_TIP_INDEX = 20
_LITTLE_PROXIMAL_INDEX = 22
_LITTLE_TIP_INDEX = 25
_TIP_INDICES = (
    _THUMB_TIP_INDEX,
    _INDEX_TIP_INDEX,
    _MIDDLE_TIP_INDEX,
    _RING_TIP_INDEX,
    _LITTLE_TIP_INDEX,
)
