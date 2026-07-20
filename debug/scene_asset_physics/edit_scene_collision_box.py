# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Editor for a scene-local box collision proxy over a visual scene GLB."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

try:
    from debug.scene_asset_physics.edit_dynamic_bottle_body import load_glb_mesh_parts
except ModuleNotFoundError:
    from edit_dynamic_bottle_body import load_glb_mesh_parts


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_GLB = REPO_ROOT / "scene" / "scene.glb"
DEFAULT_OUTPUT = REPO_ROOT / "debug" / "scene_collision_boxes.json"
SPEC_FORMAT = "newton_scene_collision_boxes_v1"
BOX_POSITION_MIN_M = -1.0
BOX_POSITION_MAX_M = 1.0
BOX_POSITION_STEP_M = 0.01


@dataclass
class SceneCollisionBoxSpec:
    scene_glb: Path
    scene_pos: list[float]
    scene_rpy_deg: list[float]
    scene_scale: float
    box_name: str
    box_pos: list[float]
    box_rpy_deg: list[float]
    box_size: list[float]
    friction: float
    visible: bool


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value greater than 0")
    return result


def _quantize_box_position(value: float) -> float:
    value = min(max(float(value), BOX_POSITION_MIN_M), BOX_POSITION_MAX_M)
    snapped = round(value / BOX_POSITION_STEP_M) * BOX_POSITION_STEP_M
    return min(max(snapped, BOX_POSITION_MIN_M), BOX_POSITION_MAX_M)


def _resolve_path(path: Path, *, base: Path | None = None) -> Path:
    resolved = path if path.is_absolute() else ((base or Path.cwd()) / path)
    return resolved.resolve()


def _rotation_from_euler_deg(euler_deg: tuple[float, float, float]) -> np.ndarray:
    x, y, z = (np.deg2rad(v) for v in euler_deg)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)), dtype=np.float64)
    ry = np.asarray(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), dtype=np.float64)
    rz = np.asarray(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    return rz @ ry @ rx


def _quat_xyzw_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(rotation)))
        if i == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray((x, y, z, w), dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)


def _wp_transform_from_pos_quat(pos: np.ndarray, quat_xyzw: tuple[float, float, float, float]) -> wp.transform:
    qx, qy, qz, qw = quat_xyzw
    return wp.transform(wp.vec3(*pos.tolist()), wp.quat(qx, qy, qz, qw))


def _visual_cfg() -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    cfg.is_visible = True
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    cfg.collision_group = 0
    return cfg


def _collision_cfg(friction: float, *, visible: bool) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    cfg.mu = float(friction)
    cfg.restitution = 0.0
    cfg.ke = 5.0e4
    cfg.kd = 5.0e2
    cfg.kf = 1.0e3
    cfg.is_visible = bool(visible)
    cfg.has_shape_collision = True
    cfg.has_particle_collision = True
    return cfg


def _relative_path(path: Path, *, output_path: Path) -> str:
    try:
        path.relative_to(REPO_ROOT)
        output_path.parent.relative_to(REPO_ROOT)
        return os.path.relpath(path, output_path.parent)
    except ValueError:
        return str(path)


def _spec_to_dict(spec: SceneCollisionBoxSpec, *, output_path: Path) -> dict[str, object]:
    return {
        "format": SPEC_FORMAT,
        "scene_glb": _relative_path(spec.scene_glb, output_path=output_path),
        "scene": {
            "position": [float(v) for v in spec.scene_pos],
            "rpy_deg": [float(v) for v in spec.scene_rpy_deg],
            "scale": float(spec.scene_scale),
        },
        "collision_boxes": [
            {
                "name": spec.box_name,
                "type": "box",
                "frame": "scene",
                "position": [float(v) for v in spec.box_pos],
                "rpy_deg": [float(v) for v in spec.box_rpy_deg],
                "size": [float(v) for v in spec.box_size],
                "friction": float(spec.friction),
                "visible": bool(spec.visible),
            }
        ],
    }


def _spec_from_dict(data: dict[str, object], *, base_dir: Path, fallback_scene_glb: Path) -> SceneCollisionBoxSpec:
    if data.get("format") != SPEC_FORMAT:
        raise ValueError(f"Unsupported scene collision spec format: {data.get('format')!r}")
    scene_data = data.get("scene", {})
    if not isinstance(scene_data, dict):
        scene_data = {}
    boxes = data.get("collision_boxes", [])
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("Scene collision spec must contain at least one collision box")
    box = boxes[0]
    if not isinstance(box, dict):
        raise ValueError("Scene collision box entry must be an object")
    scene_glb = Path(str(data.get("scene_glb", fallback_scene_glb)))
    if not scene_glb.is_absolute():
        scene_glb = (base_dir / scene_glb).resolve()
    return SceneCollisionBoxSpec(
        scene_glb=scene_glb,
        scene_pos=[float(v) for v in scene_data.get("position", (0.0, -0.0184, 0.129))],
        scene_rpy_deg=[float(v) for v in scene_data.get("rpy_deg", (0.0, 180.0, 0.0))],
        scene_scale=float(scene_data.get("scale", 1.0)),
        box_name=str(box.get("name", "scene_collision_box")),
        box_pos=[float(v) for v in box.get("position", (0.0, 0.0, 0.0))],
        box_rpy_deg=[float(v) for v in box.get("rpy_deg", (0.0, 0.0, 0.0))],
        box_size=[float(v) for v in box.get("size", (0.5, 0.5, 0.02))],
        friction=float(box.get("friction", 1.0)),
        visible=bool(box.get("visible", False)),
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = float(args.fps)
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.output_path = _resolve_path(args.output)

        input_path = _resolve_path(args.input) if args.input is not None else None
        if input_path is not None and input_path.exists():
            self.spec = _spec_from_dict(
                json.loads(input_path.read_text(encoding="utf-8")),
                base_dir=input_path.parent,
                fallback_scene_glb=_resolve_path(args.scene_glb),
            )
        else:
            self.spec = SceneCollisionBoxSpec(
                scene_glb=_resolve_path(args.scene_glb),
                scene_pos=[args.scene_pos_x, args.scene_pos_y, args.scene_pos_z],
                scene_rpy_deg=[args.scene_roll, args.scene_pitch, args.scene_yaw],
                scene_scale=float(args.scene_scale),
                box_name=str(args.box_name),
                box_pos=[args.box_pos_x, args.box_pos_y, args.box_pos_z],
                box_rpy_deg=[args.box_roll, args.box_pitch, args.box_yaw],
                box_size=[args.box_size_x, args.box_size_y, args.box_size_z],
                friction=float(args.friction),
                visible=bool(args.box_visible),
            )

        self.status = ""
        self._dirty = True
        self._preview_box_mesh: str | None = None
        self._preview_color = wp.array([wp.vec3(1.0, 0.72, 0.15)], dtype=wp.vec3)
        self._preview_material = wp.array([wp.vec4(0.55, 0.0, 0.0, 0.0)], dtype=wp.vec4)

        builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
        scene_xform = self.scene_transform()
        visual_cfg = _visual_cfg()
        for index, part in enumerate(load_glb_mesh_parts(self.spec.scene_glb)):
            builder.add_shape_mesh(
                body=-1,
                mesh=part.mesh,
                xform=scene_xform,
                scale=(self.spec.scene_scale, self.spec.scene_scale, self.spec.scene_scale),
                cfg=visual_cfg,
                color=part.color,
                label=f"scene_glb_part_{index:02d}",
            )

        self.box_shape = builder.add_shape_box(
            body=-1,
            xform=self.box_world_transform(),
            hx=0.5 * self.spec.box_size[0] * self.spec.scene_scale,
            hy=0.5 * self.spec.box_size[1] * self.spec.scene_scale,
            hz=0.5 * self.spec.box_size[2] * self.spec.scene_scale,
            cfg=_collision_cfg(self.spec.friction, visible=False),
            color=(1.0, 0.72, 0.15),
            label=self.spec.box_name,
        )

        self.model = builder.finalize(device=args.device)
        self.state_0 = self.model.state()
        self.contacts = self.model.contacts()
        self._shape_transform_host = self.model.shape_transform.numpy().copy()
        self._shape_scale_host = self.model.shape_scale.numpy().copy()
        self.apply_preview_update()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(1.8, -2.4, 1.2), pitch=-20.0, yaw=135.0)

        if args.save_spec:
            self.save_spec()

    def scene_transform(self) -> wp.transform:
        quat = _quat_xyzw_from_rotation(_rotation_from_euler_deg(tuple(self.spec.scene_rpy_deg)))
        return _wp_transform_from_pos_quat(np.asarray(self.spec.scene_pos, dtype=np.float64), quat)

    def box_world_pose(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        scene_rotation = _rotation_from_euler_deg(tuple(self.spec.scene_rpy_deg))
        box_rotation = _rotation_from_euler_deg(tuple(self.spec.box_rpy_deg))
        scene_pos = np.asarray(self.spec.scene_pos, dtype=np.float64)
        box_pos = np.asarray(self.spec.box_pos, dtype=np.float64)
        world_pos = scene_pos + scene_rotation @ (box_pos * float(self.spec.scene_scale))
        world_quat = _quat_xyzw_from_rotation(scene_rotation @ box_rotation)
        return world_pos, world_quat

    def box_world_transform(self) -> wp.transform:
        pos, quat = self.box_world_pose()
        return _wp_transform_from_pos_quat(pos, quat)

    def apply_preview_update(self) -> None:
        self._dirty = False
        self.spec.box_size = [max(float(v), 1.0e-4) for v in self.spec.box_size]
        self.spec.box_pos = [_quantize_box_position(v) for v in self.spec.box_pos]
        self.spec.scene_scale = max(float(self.spec.scene_scale), 1.0e-6)
        pos, quat = self.box_world_pose()
        self._shape_transform_host[self.box_shape, :] = (*pos.tolist(), *quat)
        half_extents = 0.5 * np.asarray(self.spec.box_size, dtype=np.float32) * float(self.spec.scene_scale)
        self._shape_scale_host[self.box_shape, :] = half_extents
        self.model.shape_transform = wp.array(self._shape_transform_host, dtype=wp.transform, device=self.model.device)
        self.model.shape_scale = wp.array(self._shape_scale_host, dtype=wp.vec3, device=self.model.device)
        self.model.bvh_refit_shapes(self.state_0)

    def save_spec(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = _spec_to_dict(self.spec, output_path=self.output_path)
        self.output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.status = f"Saved {self.output_path}"
        print(self.status, flush=True)

    def step(self) -> None:
        self.sim_time += self.frame_dt

    def render(self) -> None:
        if self._dirty:
            self.apply_preview_update()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.log_collision_box_preview()
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def log_collision_box_preview(self) -> None:
        box_size = tuple(float(v) * float(self.spec.scene_scale) for v in self.spec.box_size)
        box_transform = wp.array([self.box_world_transform()], dtype=wp.transform, device=self.model.device)

        if not hasattr(self.viewer, "_populate_geometry") or not hasattr(self.viewer, "log_instances"):
            self.viewer.log_shapes(
                "/scene_collision_box_preview",
                int(newton.GeoType.BOX),
                box_size,
                box_transform,
                self._preview_color,
                self._preview_material,
            )
            return

        if self._preview_box_mesh is None:
            self._preview_box_mesh = self.viewer._populate_geometry(
                int(newton.GeoType.BOX),
                (1.0, 1.0, 1.0),
                0.0,
                True,
            )

        self.viewer.log_instances(
            "/scene_collision_box_preview",
            self._preview_box_mesh,
            box_transform,
            wp.array([wp.vec3(*box_size)], dtype=wp.vec3, device=self.model.device),
            self._preview_color,
            self._preview_material,
        )

    def gui(self, imgui) -> None:
        changed = False
        imgui.text("Scene collision box")
        imgui.separator()

        for label, index in (("Size X [m]", 0), ("Size Y [m]", 1), ("Size Z [m]", 2)):
            updated, value = imgui.slider_float(label, float(self.spec.box_size[index]), 0.001, 4.0, "%.4f")
            if updated:
                self.spec.box_size[index] = value
                changed = True

        imgui.separator()
        for axis, index in (("Position X [m]", 0), ("Position Y [m]", 1), ("Position Z [m]", 2)):
            updated, value = imgui.slider_float(
                axis,
                float(self.spec.box_pos[index]),
                BOX_POSITION_MIN_M,
                BOX_POSITION_MAX_M,
                "%.2f",
            )
            if updated:
                self.spec.box_pos[index] = _quantize_box_position(value)
                changed = True

        imgui.separator()
        for axis, index in (("Roll [deg]", 0), ("Pitch [deg]", 1), ("Yaw [deg]", 2)):
            updated, value = imgui.slider_float(axis, float(self.spec.box_rpy_deg[index]), -180.0, 180.0, "%.2f")
            if updated:
                self.spec.box_rpy_deg[index] = value
                changed = True

        imgui.separator()
        updated, value = imgui.slider_float("Friction", float(self.spec.friction), 0.0, 15.0, "%.3f")
        if updated:
            self.spec.friction = value
            changed = True

        if changed:
            self._dirty = True

        imgui.separator()
        if imgui.button("Save Scene Collision JSON"):
            self.save_spec()
        if self.status:
            imgui.text(self.status)

    def test_final(self) -> None:
        if self.model.shape_scale is not None:
            shape_scale = self.model.shape_scale.numpy()
            if not np.isfinite(shape_scale).all():
                raise RuntimeError("Scene collision box editor produced non-finite shape scales")

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = newton.examples.create_parser()
        parser.description = "Edit a scene-local box collision proxy for scene.glb."
        parser.add_argument("--scene-glb", type=Path, default=DEFAULT_SCENE_GLB, help="Scene GLB visual asset.")
        parser.add_argument("--input", type=Path, default=None, help="Existing scene collision JSON to edit.")
        parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output scene collision JSON.")
        parser.add_argument(
            "--save-spec", action="store_true", help="Write the current spec immediately after startup."
        )
        parser.add_argument("--scene-pos-x", type=float, default=0.0, help="Scene visual X offset [m].")
        parser.add_argument("--scene-pos-y", type=float, default=-0.0184, help="Scene visual Y offset [m].")
        parser.add_argument("--scene-pos-z", type=float, default=0.129, help="Scene visual Z offset [m].")
        parser.add_argument("--scene-roll", type=float, default=0.0, help="Scene visual roll [deg].")
        parser.add_argument("--scene-pitch", type=float, default=180.0, help="Scene visual pitch [deg].")
        parser.add_argument("--scene-yaw", type=float, default=0.0, help="Scene visual yaw [deg].")
        parser.add_argument("--scene-scale", type=_positive_float, default=1.0, help="Scene visual uniform scale.")
        parser.add_argument("--box-name", type=str, default="scene_collision_box", help="Collision box name.")
        parser.add_argument("--box-size-x", type=_positive_float, default=0.6, help="Collision box X length [m].")
        parser.add_argument("--box-size-y", type=_positive_float, default=0.6, help="Collision box Y length [m].")
        parser.add_argument("--box-size-z", type=_positive_float, default=0.02, help="Collision box Z length [m].")
        parser.add_argument("--box-pos-x", type=float, default=0.0, help="Box X position in scene frame [m].")
        parser.add_argument("--box-pos-y", type=float, default=0.0, help="Box Y position in scene frame [m].")
        parser.add_argument("--box-pos-z", type=float, default=0.0, help="Box Z position in scene frame [m].")
        parser.add_argument("--box-roll", type=float, default=0.0, help="Box roll in scene frame [deg].")
        parser.add_argument("--box-pitch", type=float, default=0.0, help="Box pitch in scene frame [deg].")
        parser.add_argument("--box-yaw", type=float, default=0.0, help="Box yaw in scene frame [deg].")
        parser.add_argument("--friction", type=float, default=10.0, help="Collision box friction coefficient.")
        parser.add_argument(
            "--box-visible",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Save box as visible in the runtime model.",
        )
        parser.add_argument("--fps", type=float, default=60.0, help="Viewer frame rate [Hz].")
        return parser


def main() -> None:
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
