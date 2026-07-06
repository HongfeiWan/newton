from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Callable

import numpy as np

from teleop_stack.models import NamedJointValues
from teleop_stack.retargeting.hand_config import (
    LINKER_L10_NON_THUMB_MCP_PITCH_JOINT_NAMES,
    DexHandModelSpec,
    linker_l10_full_open_pose,
    load_linker_l10_right_hand_spec,
)
from teleop_stack.retargeting.linker_hand_heuristic import _orientation_thumb_ratios


logger = logging.getLogger(__name__)


_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_WRIST_INDEX = 1
_THUMB_PROXIMAL_INDEX = 3
_THUMB_DISTAL_INDEX = 4
_THUMB_TIP_INDEX = 5
_INDEX_PROXIMAL_INDEX = 7
_INDEX_DISTAL_INDEX = 9
_INDEX_TIP_INDEX = 10
_MIDDLE_PROXIMAL_INDEX = 12
_MIDDLE_DISTAL_INDEX = 14
_MIDDLE_TIP_INDEX = 15
_RING_PROXIMAL_INDEX = 17
_RING_DISTAL_INDEX = 19
_RING_TIP_INDEX = 20
_LITTLE_PROXIMAL_INDEX = 22
_LITTLE_DISTAL_INDEX = 24
_LITTLE_TIP_INDEX = 25

_PIP_INDICES = np.array(
    (
        _THUMB_PROXIMAL_INDEX,
        _INDEX_PROXIMAL_INDEX,
        _MIDDLE_PROXIMAL_INDEX,
        _RING_PROXIMAL_INDEX,
        _LITTLE_PROXIMAL_INDEX,
    ),
    dtype=int,
)
_DIP_INDICES = np.array(
    (
        _THUMB_DISTAL_INDEX,
        _INDEX_DISTAL_INDEX,
        _MIDDLE_DISTAL_INDEX,
        _RING_DISTAL_INDEX,
        _LITTLE_DISTAL_INDEX,
    ),
    dtype=int,
)
_TIP_INDICES = np.array(
    (
        _THUMB_TIP_INDEX,
        _INDEX_TIP_INDEX,
        _MIDDLE_TIP_INDEX,
        _RING_TIP_INDEX,
        _LITTLE_TIP_INDEX,
    ),
    dtype=int,
)


@dataclass(frozen=True)
class L10AdaptiveRetargeterConfig:
    low_pass_alpha: float = 0.2
    scaling_factor: float = 1.0
    max_delta_rad: float = 0.20
    max_iterations: int = 50
    huber_delta_m: float = 0.02
    huber_delta_dir: float = 0.5
    norm_delta: float = 0.0
    w_full_hand: float = 1.0
    w_tip_pos: float = 1.0
    w_tip_dir: float = 10.0
    w_pinch_vec: float = 0.4
    w_thumb_roll_prior: float = 0.25
    w_thumb_pitch_prior: float = 0.25
    pinch_d1_cm: tuple[float, float, float, float] = (2.0, 2.0, 2.0, 2.0)
    pinch_d2_cm: tuple[float, float, float, float] = (5.0, 5.0, 5.0, 5.0)
    alpha_max: float = 0.8
    segment_scaling: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {name: (1.0, 1.0, 1.0) for name in _FINGER_NAMES}
    )


@dataclass(frozen=True)
class L10RetargetDebugFrame:
    optimizer_success: bool
    optimizer_cost: float
    optimizer_iterations: int
    pinch_alphas: tuple[float, float, float, float, float]
    loss_terms: dict[str, float]
    target_pinch_distances_cm: tuple[float, float, float, float]
    robot_pinch_distances_cm: tuple[float, float, float, float]
    fallback_reason: str | None = None
    qpos_by_name: dict[str, float] = field(default_factory=dict)
    target_pip_vectors_cm: tuple[tuple[float, float, float], ...] = ()
    target_dip_vectors_cm: tuple[tuple[float, float, float], ...] = ()
    target_tip_vectors_cm: tuple[tuple[float, float, float], ...] = ()
    robot_pip_vectors_cm: tuple[tuple[float, float, float], ...] = ()
    robot_dip_vectors_cm: tuple[tuple[float, float, float], ...] = ()
    robot_tip_vectors_cm: tuple[tuple[float, float, float], ...] = ()
    target_tip_dirs: tuple[tuple[float, float, float], ...] = ()
    robot_tip_dirs: tuple[tuple[float, float, float], ...] = ()


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


@dataclass(frozen=True)
class _ObjectiveTargets:
    pip_vectors: np.ndarray
    dip_vectors: np.ndarray
    tip_vectors: np.ndarray
    tip_dirs: np.ndarray
    pinch_vectors: np.ndarray
    pinch_alphas: np.ndarray
    target_pinch_distances_cm: np.ndarray
    thumb_prior: dict[str, float]
    full_open_alpha: float


class _L10PinocchioModel:
    def __init__(self, spec: DexHandModelSpec):
        import pinocchio as pin

        self.pin = pin
        self.spec = spec
        self.model = pin.buildModelFromUrdf(str(spec.urdf_path))
        self.data = self.model.createData()
        if self.model.nq != self.model.nv:
            raise NotImplementedError("L10 adaptive retargeter requires nq == nv")

        self.active_joint_names = tuple(spec.active_joint_names)
        self.active_index_by_name = {name: i for i, name in enumerate(self.active_joint_names)}
        self.mimic_by_name = {joint.joint_name: joint for joint in spec.mimic_joints}

        self.expansion = np.zeros((self.model.nq, len(self.active_joint_names)), dtype=np.float64)
        self.offset = np.zeros((self.model.nq,), dtype=np.float64)
        for joint_name in self.active_joint_names:
            q_index = self._joint_q_index(joint_name)
            self.expansion[q_index, self.active_index_by_name[joint_name]] = 1.0
        for mimic in spec.mimic_joints:
            q_index = self._joint_q_index(mimic.joint_name)
            self.expansion[q_index, self.active_index_by_name[mimic.source_joint_name]] = mimic.multiplier
            self.offset[q_index] = mimic.offset

        self.link_names = (
            "hand_base_link",
            "thumb_proximal",
            "thumb_distal",
            "index_proximal",
            "index_middle",
            "index_distal",
            "middle_proximal",
            "middle_middle",
            "middle_distal",
            "ring_proximal",
            "ring_middle",
            "ring_distal",
            "pinky_proximal",
            "pinky_middle",
            "pinky_distal",
        )
        self.link_frame_ids = tuple(self._frame_id(name) for name in self.link_names)
        self.link_index_by_name = {name: i for i, name in enumerate(self.link_names)}

    def _frame_id(self, name: str) -> int:
        frame_id = self.model.getFrameId(name, self.pin.BODY)
        if frame_id >= self.model.nframes:
            raise RuntimeError(f"L10 link frame {name!r} not found in {self.spec.urdf_path}")
        return int(frame_id)

    def _joint_q_index(self, name: str) -> int:
        joint_id = self.model.getJointId(name)
        if joint_id >= self.model.njoints:
            raise RuntimeError(f"L10 joint {name!r} not found in {self.spec.urdf_path}")
        # Avoid model.idx_qs here: in the Isaac runtime its std::vector<int>
        # binding is not always registered. JointModel.idx_q is stable.
        return int(self.model.joints[joint_id].idx_q)

    def expand_qpos(self, active_qpos: np.ndarray) -> np.ndarray:
        return self.expansion @ np.asarray(active_qpos, dtype=np.float64) + self.offset

    def forward(self, active_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_full = self.expand_qpos(active_qpos)
        self.pin.forwardKinematics(self.model, self.data, q_full)
        self.pin.computeJointJacobians(self.model, self.data, q_full)
        self.pin.updateFramePlacements(self.model, self.data)

        positions: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        for frame_id in self.link_frame_ids:
            placement = self.data.oMf[frame_id]
            positions.append(np.asarray(placement.translation, dtype=np.float64))
            j_full = self.pin.getFrameJacobian(
                self.model,
                self.data,
                frame_id,
                self.pin.LOCAL_WORLD_ALIGNED,
            )[:3, :]
            jacobians.append(np.asarray(j_full, dtype=np.float64) @ self.expansion)
        return np.stack(positions, axis=0), np.stack(jacobians, axis=0)


class LinkerL10AdaptiveRetargeter:
    def __init__(
        self,
        *,
        spec: DexHandModelSpec | None = None,
        config: L10AdaptiveRetargeterConfig | None = None,
        fallback: Callable[..., NamedJointValues] | None = None,
    ):
        self.spec = spec or load_linker_l10_right_hand_spec()
        self.config = config or L10AdaptiveRetargeterConfig()
        self.fallback = fallback
        self._model: _L10PinocchioModel | None = None
        self._last_qpos: np.ndarray | None = None
        self._filtered_qpos: np.ndarray | None = None
        self._last_output_by_name: dict[str, float] | None = None
        self._warned_unavailable = False
        self.last_debug: L10RetargetDebugFrame | None = None

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
                return self._fallback(points, joint_orientations_xyzw, joint_valid, "invalid_tracking")

        try:
            palm = _build_palm_frame(points)
            local_points = _openxr_palm_local_to_l10_urdf_local(palm.to_local(points))
            targets = self._build_targets(local_points, joint_orientations_xyzw, joint_valid, palm)
            qpos, success, cost, iterations, terms = self._solve(targets)
            qpos = self._filter_and_rate_limit(qpos)
            qpos = self._apply_full_open_target(qpos, targets.full_open_alpha)
            self._last_qpos = qpos.copy()
            result = self._named_values_from_qpos(qpos)
            self._last_output_by_name = dict(zip(result.joint_names, result.joint_positions, strict=True))
            self.last_debug = self._build_debug(targets, qpos, success, cost, iterations, terms)
            return result
        except Exception as exc:
            if not self._warned_unavailable:
                logger.warning("L10 adaptive retargeter unavailable; falling back. %s", exc)
                self._warned_unavailable = True
            return self._fallback(points, joint_orientations_xyzw, joint_valid, str(exc))

    def _get_model(self) -> _L10PinocchioModel:
        if self._model is None:
            self._model = _L10PinocchioModel(self.spec)
        return self._model

    def _build_targets(
        self,
        local_points: np.ndarray,
        joint_orientations_xyzw: np.ndarray | None,
        joint_valid: np.ndarray | None,
        palm: _PalmFrame,
    ) -> _ObjectiveTargets:
        wrist = local_points[_WRIST_INDEX]
        scaling = self.config.scaling_factor
        segment_scaling = np.array(
            [self.config.segment_scaling.get(name, (1.0, 1.0, 1.0)) for name in _FINGER_NAMES],
            dtype=np.float64,
        )

        pip_vectors = (local_points[_PIP_INDICES] - wrist[None, :]) * segment_scaling[:, 0:1] * scaling
        dip_vectors = (local_points[_DIP_INDICES] - wrist[None, :]) * segment_scaling[:, 1:2] * scaling
        tip_vectors = (local_points[_TIP_INDICES] - wrist[None, :]) * segment_scaling[:, 2:3] * scaling

        target_dir_vecs = local_points[_TIP_INDICES] - local_points[_DIP_INDICES]
        target_dir_vecs[0] = local_points[_THUMB_TIP_INDEX] - local_points[_THUMB_PROXIMAL_INDEX]
        tip_dirs = _safe_normalize_rows(target_dir_vecs)

        thumb_tip = local_points[_THUMB_TIP_INDEX]
        finger_tips = local_points[_TIP_INDICES[1:]]
        target_pinch_vectors = (finger_tips - thumb_tip[None, :]) * scaling
        target_pinch_distances_cm = np.linalg.norm(finger_tips - thumb_tip[None, :], axis=1) * 100.0
        d1 = np.asarray(self.config.pinch_d1_cm, dtype=np.float64)
        d2 = np.asarray(self.config.pinch_d2_cm, dtype=np.float64)
        alphas_4 = np.clip((d2 - target_pinch_distances_cm) / (d2 - d1 + 1e-8), 0.0, self.config.alpha_max)
        pinch_alphas = np.concatenate(([float(np.max(alphas_4))], alphas_4))

        thumb_prior = self._thumb_orientation_prior(joint_orientations_xyzw, joint_valid, palm)
        return _ObjectiveTargets(
            pip_vectors=pip_vectors,
            dip_vectors=dip_vectors,
            tip_vectors=tip_vectors,
            tip_dirs=tip_dirs,
            pinch_vectors=target_pinch_vectors,
            pinch_alphas=pinch_alphas,
            target_pinch_distances_cm=target_pinch_distances_cm,
            thumb_prior=thumb_prior,
            full_open_alpha=_full_open_alpha_from_local_points(local_points, max_pinch_alpha=float(np.max(alphas_4))),
        )

    def _apply_full_open_target(self, qpos: np.ndarray, full_open_alpha: float) -> np.ndarray:
        alpha = float(np.clip(full_open_alpha, 0.0, 1.0))
        if alpha <= 0.0:
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
            adjusted[joint_index] = (1.0 - alpha) * adjusted[joint_index] + alpha * full_open_by_name[joint_name]
        if self._filtered_qpos is not None:
            self._filtered_qpos = adjusted.copy()
        return adjusted

    def _thumb_orientation_prior(
        self,
        joint_orientations_xyzw: np.ndarray | None,
        joint_valid: np.ndarray | None,
        palm: _PalmFrame,
    ) -> dict[str, float]:
        ratios = _orientation_thumb_ratios(
            joint_orientations_xyzw,
            joint_valid,
            palm_across=palm.across,
            palm_forward=palm.forward,
            palm_normal=palm.normal,
        )
        if not ratios:
            return {}

        limits_by_name = dict(zip(self.spec.active_joint_names, self.spec.active_joint_limits, strict=True))
        prior: dict[str, float] = {}
        for joint_name, ratio in ratios.items():
            if joint_name not in limits_by_name:
                continue
            lower, upper = limits_by_name[joint_name]
            prior[joint_name] = lower + float(ratio) * (upper - lower)
        return prior

    def _solve(self, targets: _ObjectiveTargets) -> tuple[np.ndarray, bool, float, int, dict[str, float]]:
        from scipy.optimize import minimize

        limits = np.asarray(self.spec.active_joint_limits, dtype=np.float64)
        bounds = tuple((float(lower), float(upper)) for lower, upper in limits)

        default_open = np.asarray(self.spec.default_open_pose.joint_positions, dtype=np.float64)
        initial_guesses: list[np.ndarray] = [default_open]
        if self._last_qpos is not None:
            initial_guesses.insert(0, self._last_qpos.copy())

        # Pinch frames benefit from warm-start smoothness. Open/non-pinch frames must be
        # allowed to escape a previously closed local optimum.
        max_alpha = float(np.max(targets.pinch_alphas)) if targets.pinch_alphas.size else 0.0
        reg_qpos = self._last_qpos if max_alpha >= 0.2 else None

        best: tuple[float, np.ndarray, bool, int] | None = None
        seen: set[tuple[float, ...]] = set()
        for init in initial_guesses:
            init = np.clip(init, limits[:, 0], limits[:, 1])
            key = tuple(float(f"{value:.6f}") for value in init)
            if key in seen:
                continue
            seen.add(key)

            def objective(qpos: np.ndarray) -> tuple[float, np.ndarray]:
                loss, grad, _ = self._loss_grad_terms(qpos, targets, reg_qpos)
                return loss, grad

            result = minimize(
                fun=lambda q: objective(q)[0],
                x0=init,
                jac=lambda q: objective(q)[1],
                bounds=bounds,
                method="SLSQP",
                options={"maxiter": self.config.max_iterations, "ftol": 1e-4, "disp": False},
            )
            q_candidate = np.asarray(result.x, dtype=np.float64)
            task_cost, _, _ = self._loss_grad_terms(q_candidate, targets, None)
            if best is None or task_cost < best[0]:
                best = (float(task_cost), q_candidate, bool(result.success), int(result.nit))

        if best is None:
            raise RuntimeError("L10 adaptive optimizer did not run any initial guess")
        _, qpos, success, iterations = best
        cost, _, terms = self._loss_grad_terms(qpos, targets, None)
        return qpos, success, float(cost), iterations, terms

    def _loss_grad_terms(
        self,
        qpos: np.ndarray,
        targets: _ObjectiveTargets,
        last_qpos: np.ndarray | None,
    ) -> tuple[float, np.ndarray, dict[str, float]]:
        model = self._get_model()
        positions, jacobians = model.forward(qpos)
        idx = model.link_index_by_name

        base_pos = positions[idx["hand_base_link"]]
        j_base = jacobians[idx["hand_base_link"]]
        pip_names = ("thumb_proximal", "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal")
        dip_names = ("thumb_distal", "index_middle", "middle_middle", "ring_middle", "pinky_middle")
        tip_names = ("thumb_distal", "index_distal", "middle_distal", "ring_distal", "pinky_distal")

        total_loss = 0.0
        total_grad = np.zeros((len(self.spec.active_joint_names),), dtype=np.float64)
        terms = {
            "full_hand": 0.0,
            "tip_pos": 0.0,
            "tip_dir": 0.0,
            "pinch": 0.0,
            "thumb_prior": 0.0,
            "reg": 0.0,
        }

        tip_positions: list[np.ndarray] = []
        tip_jacobians: list[np.ndarray] = []
        for i, (pip_name, dip_name, tip_name) in enumerate(zip(pip_names, dip_names, tip_names, strict=True)):
            alpha = float(targets.pinch_alphas[i])
            pip_pos = positions[idx[pip_name]]
            dip_pos = positions[idx[dip_name]]
            tip_pos = positions[idx[tip_name]]
            j_pip = jacobians[idx[pip_name]]
            j_dip = jacobians[idx[dip_name]]
            j_tip = jacobians[idx[tip_name]]
            tip_positions.append(tip_pos)
            tip_jacobians.append(j_tip)

            full_weight = (1.0 - alpha) * self.config.w_full_hand / 3.0
            if full_weight > 0.0:
                for robot_vec, target_vec, j_vec in (
                    (pip_pos - base_pos, targets.pip_vectors[i], j_pip - j_base),
                    (dip_pos - base_pos, targets.dip_vectors[i], j_dip - j_base),
                    (tip_pos - base_pos, targets.tip_vectors[i], j_tip - j_base),
                ):
                    loss, grad = _vector_huber_loss_grad(robot_vec - target_vec, j_vec, self.config.huber_delta_m)
                    total_loss += full_weight * loss
                    total_grad += full_weight * grad
                    terms["full_hand"] += full_weight * loss

            tip_pos_weight = alpha * self.config.w_tip_pos
            if tip_pos_weight > 0.0:
                loss, grad = _vector_huber_loss_grad(
                    (tip_pos - base_pos) - targets.tip_vectors[i],
                    j_tip - j_base,
                    self.config.huber_delta_m,
                )
                total_loss += tip_pos_weight * loss
                total_grad += tip_pos_weight * grad
                terms["tip_pos"] += tip_pos_weight * loss

            dir_origin_pos = pip_pos if i == 0 else dip_pos
            j_dir_origin = j_pip if i == 0 else j_dip
            dir_weight = alpha * self.config.w_tip_dir
            if dir_weight > 0.0:
                loss, grad = _direction_huber_loss_grad(
                    tip_pos - dir_origin_pos,
                    j_tip - j_dir_origin,
                    targets.tip_dirs[i],
                    self.config.huber_delta_dir,
                )
                total_loss += dir_weight * loss
                total_grad += dir_weight * grad
                terms["tip_dir"] += dir_weight * loss

        thumb_tip_pos = tip_positions[0]
        j_thumb_tip = tip_jacobians[0]
        for finger_index in range(1, 5):
            pinch_weight = float(targets.pinch_alphas[finger_index]) * self.config.w_pinch_vec
            if pinch_weight <= 0.0:
                continue
            robot_vec = tip_positions[finger_index] - thumb_tip_pos
            j_vec = tip_jacobians[finger_index] - j_thumb_tip
            target_vec = targets.pinch_vectors[finger_index - 1]
            loss, grad = _vector_huber_loss_grad(robot_vec - target_vec, j_vec, self.config.huber_delta_m)
            total_loss += pinch_weight * loss
            total_grad += pinch_weight * grad
            terms["pinch"] += pinch_weight * loss

        prior_weights = {
            "thumb_cmc_roll": self.config.w_thumb_roll_prior,
            "thumb_cmc_pitch": self.config.w_thumb_pitch_prior,
        }
        active_index_by_name = {name: i for i, name in enumerate(self.spec.active_joint_names)}
        for joint_name, prior_value in targets.thumb_prior.items():
            weight = prior_weights.get(joint_name, 0.0)
            if weight <= 0.0 or joint_name not in active_index_by_name:
                continue
            joint_index = active_index_by_name[joint_name]
            diff = qpos[joint_index] - prior_value
            loss = weight * diff * diff
            total_loss += loss
            total_grad[joint_index] += 2.0 * weight * diff
            terms["thumb_prior"] += loss

        if last_qpos is not None:
            diff = qpos - last_qpos
            loss = self.config.norm_delta * float(np.sum(diff * diff))
            total_loss += loss
            total_grad += 2.0 * self.config.norm_delta * diff
            terms["reg"] += loss

        return float(total_loss), total_grad, terms

    def _filter_and_rate_limit(self, qpos: np.ndarray) -> np.ndarray:
        limits = np.asarray(self.spec.active_joint_limits, dtype=np.float64)
        filtered = np.clip(np.asarray(qpos, dtype=np.float64), limits[:, 0], limits[:, 1])
        if self._filtered_qpos is None:
            self._filtered_qpos = filtered.copy()
        else:
            alpha = float(np.clip(self.config.low_pass_alpha, 0.0, 1.0))
            self._filtered_qpos = self._filtered_qpos + alpha * (filtered - self._filtered_qpos)
        filtered = self._filtered_qpos.copy()
        if self._last_output_by_name is not None and self.config.max_delta_rad > 0.0:
            previous = np.array(
                [self._last_output_by_name[name] for name in self.spec.active_joint_names],
                dtype=np.float64,
            )
            delta = np.clip(filtered - previous, -self.config.max_delta_rad, self.config.max_delta_rad)
            filtered = previous + delta
            self._filtered_qpos = filtered.copy()
        return np.clip(filtered, limits[:, 0], limits[:, 1])

    def _named_values_from_qpos(self, qpos: np.ndarray) -> NamedJointValues:
        return NamedJointValues(
            joint_names=self.spec.active_joint_names,
            joint_positions=tuple(float(value) for value in qpos),
        )

    def _build_debug(
        self,
        targets: _ObjectiveTargets,
        qpos: np.ndarray,
        success: bool,
        cost: float,
        iterations: int,
        terms: dict[str, float],
    ) -> L10RetargetDebugFrame:
        model = self._get_model()
        positions, _ = model.forward(qpos)
        idx = model.link_index_by_name
        base_pos = positions[idx["hand_base_link"]]
        pip_names = ("thumb_proximal", "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal")
        dip_names = ("thumb_distal", "index_middle", "middle_middle", "ring_middle", "pinky_middle")
        tip_names = ("thumb_distal", "index_distal", "middle_distal", "ring_distal", "pinky_distal")
        pip_positions = np.array([positions[idx[name]] for name in pip_names], dtype=np.float64)
        dip_positions = np.array([positions[idx[name]] for name in dip_names], dtype=np.float64)
        tip_positions = np.array([positions[idx[name]] for name in tip_names], dtype=np.float64)
        robot_pip_vectors = (pip_positions - base_pos[None, :]) * 100.0
        robot_dip_vectors = (dip_positions - base_pos[None, :]) * 100.0
        robot_tip_vectors = (tip_positions - base_pos[None, :]) * 100.0
        robot_dir_vecs = tip_positions - dip_positions
        robot_dir_vecs[0] = tip_positions[0] - pip_positions[0]
        robot_tip_dirs = _safe_normalize_rows(robot_dir_vecs)

        thumb_tip = positions[idx["thumb_distal"]]
        finger_tips = np.array(
            [
                positions[idx["index_distal"]],
                positions[idx["middle_distal"]],
                positions[idx["ring_distal"]],
                positions[idx["pinky_distal"]],
            ],
            dtype=np.float64,
        )
        robot_distances = np.linalg.norm(finger_tips - thumb_tip[None, :], axis=1) * 100.0
        return L10RetargetDebugFrame(
            optimizer_success=success,
            optimizer_cost=cost,
            optimizer_iterations=iterations,
            pinch_alphas=tuple(float(value) for value in targets.pinch_alphas),
            loss_terms={name: float(value) for name, value in terms.items()},
            target_pinch_distances_cm=tuple(float(value) for value in targets.target_pinch_distances_cm),
            robot_pinch_distances_cm=tuple(float(value) for value in robot_distances),
            qpos_by_name=dict(zip(self.spec.active_joint_names, (float(value) for value in qpos), strict=True)),
            target_pip_vectors_cm=_rows_to_tuple(targets.pip_vectors * 100.0),
            target_dip_vectors_cm=_rows_to_tuple(targets.dip_vectors * 100.0),
            target_tip_vectors_cm=_rows_to_tuple(targets.tip_vectors * 100.0),
            robot_pip_vectors_cm=_rows_to_tuple(robot_pip_vectors),
            robot_dip_vectors_cm=_rows_to_tuple(robot_dip_vectors),
            robot_tip_vectors_cm=_rows_to_tuple(robot_tip_vectors),
            target_tip_dirs=_rows_to_tuple(targets.tip_dirs),
            robot_tip_dirs=_rows_to_tuple(robot_tip_dirs),
        )

    def _fallback(
        self,
        joint_positions_xyz: np.ndarray,
        joint_orientations_xyzw: np.ndarray | None,
        joint_valid: np.ndarray | None,
        reason: str,
    ) -> NamedJointValues:
        self.last_debug = L10RetargetDebugFrame(
            optimizer_success=False,
            optimizer_cost=float("nan"),
            optimizer_iterations=0,
            pinch_alphas=(0.0, 0.0, 0.0, 0.0, 0.0),
            loss_terms={},
            target_pinch_distances_cm=(float("nan"),) * 4,
            robot_pinch_distances_cm=(float("nan"),) * 4,
            fallback_reason=reason,
        )
        if self.fallback is not None:
            return self.fallback(
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


def _full_open_alpha_from_local_points(local_points: np.ndarray, *, max_pinch_alpha: float) -> float:
    if max_pinch_alpha > 0.05:
        return 0.0
    chain_indices = (
        (_INDEX_PROXIMAL_INDEX, _INDEX_DISTAL_INDEX, _INDEX_TIP_INDEX),
        (_MIDDLE_PROXIMAL_INDEX, _MIDDLE_DISTAL_INDEX, _MIDDLE_TIP_INDEX),
        (_RING_PROXIMAL_INDEX, _RING_DISTAL_INDEX, _RING_TIP_INDEX),
        (_LITTLE_PROXIMAL_INDEX, _LITTLE_DISTAL_INDEX, _LITTLE_TIP_INDEX),
    )
    curl_ratios = []
    for proximal_index, distal_index, tip_index in chain_indices:
        proximal = local_points[proximal_index]
        distal = local_points[distal_index]
        tip = local_points[tip_index]
        finger_forward = _safe_normalize(tip - proximal)
        distal_forward = _safe_normalize(tip - distal)
        if not np.any(finger_forward) or not np.any(distal_forward):
            return 0.0
        curl_ratios.append(_vector_angle_rad(finger_forward, distal_forward) / np.pi)
    max_curl = max(curl_ratios)
    return float(np.clip((0.18 - max_curl) / 0.13, 0.0, 1.0))


def _vector_angle_rad(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_unit = _safe_normalize(lhs)
    rhs_unit = _safe_normalize(rhs)
    if not np.any(lhs_unit) or not np.any(rhs_unit):
        return float(np.pi)
    dot = float(np.clip(np.dot(lhs_unit, rhs_unit), -1.0, 1.0))
    return float(np.arccos(dot))


def _safe_normalize(vector: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _safe_normalize_rows(vectors: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def _rows_to_tuple(rows: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in np.asarray(rows, dtype=np.float64))


def _huber_loss_grad(distance: float, delta: float) -> tuple[float, float]:
    abs_distance = abs(float(distance))
    if abs_distance <= delta:
        return 0.5 * distance * distance, distance
    return delta * (abs_distance - 0.5 * delta), delta


def _vector_huber_loss_grad(diff: np.ndarray, jacobian: np.ndarray, delta: float) -> tuple[float, np.ndarray]:
    distance = float(np.linalg.norm(diff))
    loss, grad_distance = _huber_loss_grad(distance, delta)
    direction = diff / (distance + 1e-9)
    grad = grad_distance * (direction @ jacobian)
    return float(loss), np.asarray(grad, dtype=np.float64)


def _direction_huber_loss_grad(
    robot_vec: np.ndarray,
    robot_vec_jacobian: np.ndarray,
    target_dir: np.ndarray,
    delta: float,
) -> tuple[float, np.ndarray]:
    norm = float(np.linalg.norm(robot_vec))
    robot_dir = robot_vec / (norm + 1e-9)
    diff = robot_dir - target_dir
    distance = float(np.linalg.norm(diff))
    loss, grad_distance = _huber_loss_grad(distance, delta)
    diff_dir = diff / (distance + 1e-9)
    j_norm = (np.eye(3) - np.outer(robot_dir, robot_dir)) / (norm + 1e-9)
    grad = grad_distance * (diff_dir @ j_norm @ robot_vec_jacobian)
    return float(loss), np.asarray(grad, dtype=np.float64)
