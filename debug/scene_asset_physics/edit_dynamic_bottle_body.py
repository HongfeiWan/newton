# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Editor for a dynamic cylindrical bottle proxy with a GLB visual."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOTTLE_GLB = REPO_ROOT / "assets" / "bottle.glb"
DEFAULT_OUTPUT = REPO_ROOT / "debug" / "dynamic_bottle_body.json"
SPEC_FORMAT = "newton_dynamic_bottle_v1"
WATER_DENSITY_KG_PER_M3 = 1000.0
DEFAULT_WATER_CYLINDER_HEIGHT_M = 0.03
BOTTLE_CONTACT_MARGIN_M = 0.0
BOTTLE_CONTACT_GAP_M = 1.0e-4
BOTTLE_CONTACT_TORSIONAL_FRICTION = 0.02
BOTTLE_CONTACT_ROLLING_FRICTION = 0.002
BOTTLE_CONTACT_KE = 6.0e3
BOTTLE_CONTACT_KD = 1.2e3
BOTTLE_CONTACT_KF = 2.0e2


@dataclass
class GlbMeshPart:
    mesh: newton.Mesh
    texture: np.ndarray | None
    color: tuple[float, float, float]


@dataclass
class VisualFit:
    local_pos: tuple[float, float, float]
    local_quat_xyzw: tuple[float, float, float, float]
    radius: float
    height: float
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]


@dataclass
class DynamicBottleSpec:
    visual_glb: Path
    pos: list[float]
    rpy_deg: list[float]
    radius: float
    height: float
    red_cylinder_height: float
    water_density: float
    mass: float
    friction: float
    visual_local_pos: list[float]
    visual_local_quat_xyzw: list[float]
    fit_margin: float


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value greater than 0")
    return result


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


def _wp_transform_from_pos_rpy(pos: list[float], rpy_deg: list[float]) -> wp.transform:
    qx, qy, qz, qw = _quat_xyzw_from_rotation(_rotation_from_euler_deg(tuple(rpy_deg)))
    return wp.transform(wp.vec3(*pos), wp.quat(qx, qy, qz, qw))


def _wp_transform_from_pos_quat(pos: list[float], quat_xyzw: list[float]) -> wp.transform:
    qx, qy, qz, qw = quat_xyzw
    return wp.transform(wp.vec3(*pos), wp.quat(qx, qy, qz, qw))


def _mesh_color(mesh, texture: np.ndarray | None) -> tuple[float, float, float]:
    if texture is not None:
        material = getattr(getattr(mesh, "visual", None), "material", None)
        base_color = getattr(material, "baseColorFactor", None)
        if base_color is None:
            return (1.0, 1.0, 1.0)

    default = (0.65, 0.65, 0.65)
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    candidates = [
        getattr(material, "baseColorFactor", None),
        getattr(material, "main_color", None),
        getattr(visual, "main_color", None),
        getattr(visual, "vertex_colors", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        color = np.asarray(candidate, dtype=np.float32)
        if color.ndim == 2:
            color = color[:, :3].mean(axis=0)
        color = color.reshape(-1)
        if color.size >= 3:
            if np.max(color[:3]) > 1.0:
                color = color / 255.0
            return tuple(np.clip(color[:3], 0.0, 1.0).tolist())
    return default


def _mesh_texture(mesh) -> np.ndarray | None:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        texture = getattr(material, "base_color_texture", None)
    if texture is None:
        texture = getattr(material, "image", None)
    if texture is None:
        return None
    if hasattr(texture, "convert"):
        return np.asarray(texture.convert("RGBA"))
    return np.asarray(texture)


def load_glb_mesh_parts(glb_path: Path) -> list[GlbMeshPart]:
    import trimesh

    glb_path = _resolve_path(glb_path)
    scene = trimesh.load(glb_path, force="scene")
    parts: list[GlbMeshPart] = []
    for node_name in scene.graph.nodes_geometry:
        node_transform, geometry_name = scene.graph.get(node_name)
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(node_transform)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
        if vertices.size == 0 or faces.size == 0:
            continue

        normals_np = None
        if getattr(mesh, "vertex_normals", None) is not None and len(mesh.vertex_normals) == len(mesh.vertices):
            normals_np = np.asarray(mesh.vertex_normals, dtype=np.float32)
            if not normals_np.size:
                normals_np = None

        uvs_np = None
        visual_uvs = getattr(getattr(mesh, "visual", None), "uv", None)
        if visual_uvs is not None:
            uvs_np = np.asarray(visual_uvs, dtype=np.float32)
            if uvs_np.shape != (len(mesh.vertices), 2):
                uvs_np = None

        texture = _mesh_texture(mesh)
        color = _mesh_color(mesh, texture)
        parts.append(
            GlbMeshPart(
                mesh=newton.Mesh(
                    vertices,
                    faces,
                    normals=normals_np,
                    uvs=uvs_np,
                    compute_inertia=False,
                    is_solid=False,
                    color=color,
                    texture=texture,
                ),
                texture=texture,
                color=color,
            )
        )

    if not parts:
        raise ValueError(f"No renderable meshes found in GLB: {glb_path}")
    return parts


def _axis_to_z_rotation(axis: int) -> np.ndarray:
    if axis == 0:
        return _rotation_from_euler_deg((0.0, -90.0, 0.0))
    if axis == 1:
        return _rotation_from_euler_deg((90.0, 0.0, 0.0))
    return np.eye(3, dtype=np.float64)


def fit_cylinder_envelope(parts: list[GlbMeshPart], *, margin: float) -> VisualFit:
    vertices = np.concatenate([part.mesh.vertices for part in parts], axis=0).astype(np.float64, copy=False)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    extents = bounds_max - bounds_min
    rotation = _axis_to_z_rotation(int(np.argmax(extents)))
    local_vertices = (rotation @ (vertices - center).T).T

    radial = np.linalg.norm(local_vertices[:, :2], axis=1)
    radius = float(radial.max() + margin)
    height = float(local_vertices[:, 2].max() - local_vertices[:, 2].min() + 2.0 * margin)
    local_pos = -rotation @ center
    return VisualFit(
        local_pos=tuple(float(v) for v in local_pos),
        local_quat_xyzw=_quat_xyzw_from_rotation(rotation),
        radius=radius,
        height=height,
        bounds_min=tuple(float(v) for v in bounds_min),
        bounds_max=tuple(float(v) for v in bounds_max),
    )


def _shape_cfg(
    *,
    density: float,
    friction: float,
    visible: bool = True,
    colliding: bool = True,
) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = float(density)
    cfg.mu = float(friction)
    cfg.restitution = 0.0
    cfg.mu_torsional = BOTTLE_CONTACT_TORSIONAL_FRICTION
    cfg.mu_rolling = BOTTLE_CONTACT_ROLLING_FRICTION
    cfg.ke = BOTTLE_CONTACT_KE
    cfg.kd = BOTTLE_CONTACT_KD
    cfg.kf = BOTTLE_CONTACT_KF
    cfg.is_visible = bool(visible)
    cfg.has_shape_collision = bool(colliding)
    cfg.has_particle_collision = bool(colliding)
    if colliding:
        cfg.margin = BOTTLE_CONTACT_MARGIN_M
        cfg.gap = BOTTLE_CONTACT_GAP_M
    if not colliding:
        cfg.collision_group = 0
    return cfg


def _water_mass_for_cylinder(radius: float, height: float, density: float) -> float:
    # Bottle water level uses the same half-height convention as Newton cylinders.
    volume = math.pi * radius * radius * (2.0 * height)
    return float(density * max(volume, 0.0))


def _effective_cylinder_height(height: float) -> float:
    # Spec height fields are stored as cylinder half-height parameters for this bottle.
    return 2.0 * float(height)


def _water_cylinder_local_pos(blue_height: float, red_height: float) -> list[float]:
    blue_full_height = _effective_cylinder_height(blue_height)
    red_full_height = _effective_cylinder_height(red_height)
    return [0.0, 0.0, 0.5 * (red_full_height - blue_full_height)]


def build_dynamic_bottle(
    builder: newton.ModelBuilder,
    spec: DynamicBottleSpec | dict[str, Any],
    *,
    device: str | None = None,
) -> dict[str, int | list[int]]:
    if isinstance(spec, dict):
        spec = dynamic_bottle_spec_from_dict(spec, base_dir=Path.cwd())

    parts = load_glb_mesh_parts(spec.visual_glb)
    body = builder.add_body(xform=_wp_transform_from_pos_rpy(spec.pos, spec.rpy_deg), label="dynamic_bottle")
    collision_cfg = _shape_cfg(density=0.0, friction=spec.friction, visible=False, colliding=True)
    cylinder_shape = builder.add_shape_cylinder(
        body=body,
        radius=spec.radius,
        half_height=spec.height,
        cfg=collision_cfg,
        color=(0.1, 0.55, 1.0),
        label="dynamic_bottle_collision_cylinder",
    )

    # Water is a mass-only fill volume: it shifts COM but the outer bottle handles contact.
    water_cfg = _shape_cfg(density=spec.water_density, friction=spec.friction, visible=False, colliding=False)
    water_cylinder_shape = builder.add_shape_cylinder(
        body=body,
        xform=_wp_transform_from_pos_quat(
            _water_cylinder_local_pos(spec.height, spec.red_cylinder_height),
            [0.0, 0.0, 0.0, 1.0],
        ),
        radius=spec.radius,
        half_height=spec.red_cylinder_height,
        cfg=water_cfg,
        color=(1.0, 0.05, 0.03),
        label="dynamic_bottle_water_mass_cylinder",
    )

    visual_cfg = _shape_cfg(density=0.0, friction=spec.friction, visible=True, colliding=False)
    visual_shapes = []
    visual_xform = _wp_transform_from_pos_quat(spec.visual_local_pos, spec.visual_local_quat_xyzw)
    for index, part in enumerate(parts):
        visual_shapes.append(
            builder.add_shape_mesh(
                body=body,
                mesh=part.mesh,
                xform=visual_xform,
                cfg=visual_cfg,
                scale=(1.0, 1.0, 1.0),
                color=part.color,
                label=f"dynamic_bottle_visual_{index:02d}",
            )
        )

    return {
        "body": body,
        "collision_shape": cylinder_shape,
        "water_collision_shape": water_cylinder_shape,
        "visual_shapes": visual_shapes,
    }


def dynamic_bottle_spec_from_dict(data: dict[str, Any], *, base_dir: Path) -> DynamicBottleSpec:
    visual_glb = Path(data["visual_glb"])
    if not visual_glb.is_absolute():
        visual_glb = (base_dir / visual_glb).resolve()
    body = data["body"]
    collision = data["collision"]
    visual = data["visual"]
    return DynamicBottleSpec(
        visual_glb=visual_glb,
        pos=[float(v) for v in body["position"]],
        rpy_deg=[float(v) for v in body["rpy_deg"]],
        radius=float(collision["radius"]),
        height=float(collision["height"]),
        red_cylinder_height=float(
            collision.get("red_cylinder_height", collision.get("red_height", DEFAULT_WATER_CYLINDER_HEIGHT_M))
        ),
        water_density=float(collision.get("water_density", WATER_DENSITY_KG_PER_M3)),
        mass=float(body["mass"]),
        friction=float(collision["friction"]),
        visual_local_pos=[float(v) for v in visual["local_position"]],
        visual_local_quat_xyzw=[float(v) for v in visual["local_quat_xyzw"]],
        fit_margin=float(visual.get("fit_margin", 0.0)),
    )


def load_dynamic_bottle_spec(path: Path) -> DynamicBottleSpec:
    path = _resolve_path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != SPEC_FORMAT:
        raise ValueError(f"Unsupported dynamic bottle spec format: {data.get('format')!r}")
    return dynamic_bottle_spec_from_dict(data, base_dir=path.parent)


def dynamic_bottle_spec_to_dict(spec: DynamicBottleSpec, *, output_path: Path) -> dict[str, Any]:
    water_mass = _water_mass_for_cylinder(spec.radius, spec.red_cylinder_height, spec.water_density)
    try:
        spec.visual_glb.relative_to(REPO_ROOT)
        output_path.parent.relative_to(REPO_ROOT)
        visual_glb = os.path.relpath(spec.visual_glb, output_path.parent)
    except ValueError:
        visual_glb = str(spec.visual_glb)
    return {
        "format": SPEC_FORMAT,
        "visual_glb": visual_glb,
        "body": {
            "position": [float(v) for v in spec.pos],
            "rpy_deg": [float(v) for v in spec.rpy_deg],
            "mass": float(water_mass),
            "dynamic": True,
            "gravity": True,
        },
        "collision": {
            "type": "cylinder",
            "radius": float(spec.radius),
            "height": float(spec.height),
            "red_cylinder_height": float(spec.red_cylinder_height),
            "water_density": float(spec.water_density),
            "friction": float(spec.friction),
        },
        "visual": {
            "type": "glb",
            "local_position": [float(v) for v in spec.visual_local_pos],
            "local_quat_xyzw": [float(v) for v in spec.visual_local_quat_xyzw],
            "fit_margin": float(spec.fit_margin),
            "collision_enabled": False,
        },
    }


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = float(args.fps)
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.output_path = _resolve_path(args.output)
        input_path = _resolve_path(args.input) if args.input is not None else self.output_path
        if input_path.exists():
            self.spec = load_dynamic_bottle_spec(input_path)
            self.status = f"Loaded {input_path}"
            print(self.status, flush=True)
            parts = load_glb_mesh_parts(self.spec.visual_glb)
        else:
            visual_glb = _resolve_path(args.visual_glb)
            parts = load_glb_mesh_parts(visual_glb)
            fit = fit_cylinder_envelope(parts, margin=float(args.fit_margin))
            radius = float(args.radius if args.radius is not None else fit.radius)
            height = float(args.height if args.height is not None else fit.height)

            self.spec = DynamicBottleSpec(
                visual_glb=visual_glb,
                pos=[args.pos_x, args.pos_y, args.pos_z],
                rpy_deg=[args.roll, args.pitch, args.yaw],
                radius=radius,
                height=height,
                red_cylinder_height=float(
                    args.red_cylinder_height
                    if args.red_cylinder_height is not None
                    else DEFAULT_WATER_CYLINDER_HEIGHT_M
                ),
                water_density=float(args.water_density),
                mass=0.0,
                friction=float(args.friction),
                visual_local_pos=list(fit.local_pos),
                visual_local_quat_xyzw=list(fit.local_quat_xyzw),
                fit_margin=float(args.fit_margin),
            )
            self.status = ""

        self.visual_glb = self.spec.visual_glb
        self.fit = fit_cylinder_envelope(parts, margin=float(self.spec.fit_margin))
        self._dirty = True
        self._preview_cylinder_mesh: str | None = None
        self._preview_red_cylinder_mesh: str | None = None
        self._preview_cylinder_color = wp.array([wp.vec3(0.1, 0.55, 1.0)], dtype=wp.vec3)
        self._preview_cylinder_material = wp.array([wp.vec4(0.45, 0.0, 0.0, 0.0)], dtype=wp.vec4)
        self._preview_red_cylinder_color = wp.array([wp.vec3(1.0, 0.05, 0.03)], dtype=wp.vec3)
        self._preview_red_cylinder_material = wp.array([wp.vec4(0.55, 0.0, 0.0, 0.0)], dtype=wp.vec4)

        builder = newton.ModelBuilder(up_axis="Z", gravity=args.gravity)
        handles = build_dynamic_bottle(builder, self.spec)
        if args.add_ground:
            builder.add_ground_plane()

        self.body_index = int(handles["body"])
        self.cylinder_shape = int(handles["collision_shape"])
        self.water_cylinder_shape = int(handles["water_collision_shape"])
        self.model = builder.finalize(device=args.device)
        self.state_0 = self.model.state()
        self.contacts = self.model.contacts()
        self._body_q_host = self.state_0.body_q.numpy().copy()
        self._shape_transform_host = self.model.shape_transform.numpy().copy()
        self._shape_scale_host = self.model.shape_scale.numpy().copy()
        self.apply_preview_update()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.8, -1.2, 0.7), pitch=-20.0, yaw=135.0)

        if args.save_spec:
            self.save_spec()

    def current_body_transform_row(self) -> np.ndarray:
        qx, qy, qz, qw = _quat_xyzw_from_rotation(_rotation_from_euler_deg(tuple(self.spec.rpy_deg)))
        return np.asarray((*self.spec.pos, qx, qy, qz, qw), dtype=np.float32)

    def apply_preview_update(self) -> None:
        self._dirty = False
        self.spec.radius = max(float(self.spec.radius), 1.0e-4)
        self.spec.height = max(float(self.spec.height), 1.0e-4)
        self.spec.red_cylinder_height = min(max(float(self.spec.red_cylinder_height), 1.0e-4), self.spec.height)
        self.spec.water_density = max(float(self.spec.water_density), 0.0)
        self.spec.mass = _water_mass_for_cylinder(
            self.spec.radius,
            self.spec.red_cylinder_height,
            self.spec.water_density,
        )

        self._body_q_host[self.body_index, :] = self.current_body_transform_row()
        self.state_0.body_q = wp.array(self._body_q_host, dtype=wp.transform, device=self.model.device)
        self._shape_scale_host[self.cylinder_shape, :] = (self.spec.radius, self.spec.height, 0.0)
        self._shape_transform_host[self.water_cylinder_shape, :] = (
            *_water_cylinder_local_pos(self.spec.height, self.spec.red_cylinder_height),
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self._shape_scale_host[self.water_cylinder_shape, :] = (
            self.spec.radius,
            self.spec.red_cylinder_height,
            0.0,
        )
        self.model.shape_transform = wp.array(
            self._shape_transform_host,
            dtype=wp.transform,
            device=self.model.device,
        )
        self.model.shape_scale = wp.array(self._shape_scale_host, dtype=wp.vec3, device=self.model.device)
        self.model.bvh_refit_shapes(self.state_0)

    def save_spec(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = dynamic_bottle_spec_to_dict(self.spec, output_path=self.output_path)
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
        self.log_collision_cylinder_preview()
        self.log_red_cylinder_preview()
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def red_cylinder_transform(self) -> wp.transform:
        rotation = _rotation_from_euler_deg(tuple(self.spec.rpy_deg))
        pos = np.asarray(self.spec.pos, dtype=np.float64)
        pos += rotation[:, 2] * _water_cylinder_local_pos(self.spec.height, self.spec.red_cylinder_height)[2]
        return _wp_transform_from_pos_rpy(pos.tolist(), self.spec.rpy_deg)

    def log_collision_cylinder_preview(self) -> None:
        if not hasattr(self.viewer, "_populate_geometry") or not hasattr(self.viewer, "log_instances"):
            self.viewer.log_shapes(
                "/dynamic_bottle_collision_preview",
                newton.GeoType.CYLINDER,
                (self.spec.radius, self.spec.height),
                wp.array([_wp_transform_from_pos_rpy(self.spec.pos, self.spec.rpy_deg)], dtype=wp.transform),
                self._preview_cylinder_color,
                self._preview_cylinder_material,
            )
            return

        if self._preview_cylinder_mesh is None:
            self._preview_cylinder_mesh = self.viewer._populate_geometry(
                int(newton.GeoType.CYLINDER),
                (1.0, 1.0),
                0.0,
                True,
            )

        self.viewer.log_instances(
            "/dynamic_bottle_collision_preview",
            self._preview_cylinder_mesh,
            wp.array([_wp_transform_from_pos_rpy(self.spec.pos, self.spec.rpy_deg)], dtype=wp.transform),
            wp.array([wp.vec3(self.spec.radius, self.spec.radius, self.spec.height)], dtype=wp.vec3),
            self._preview_cylinder_color,
            self._preview_cylinder_material,
        )

    def log_red_cylinder_preview(self) -> None:
        red_transform = wp.array([self.red_cylinder_transform()], dtype=wp.transform)
        if not hasattr(self.viewer, "_populate_geometry") or not hasattr(self.viewer, "log_instances"):
            self.viewer.log_shapes(
                "/dynamic_bottle_red_cylinder_preview",
                newton.GeoType.CYLINDER,
                (self.spec.radius, self.spec.red_cylinder_height),
                red_transform,
                self._preview_red_cylinder_color,
                self._preview_red_cylinder_material,
            )
            return

        if self._preview_red_cylinder_mesh is None:
            self._preview_red_cylinder_mesh = self.viewer._populate_geometry(
                int(newton.GeoType.CYLINDER),
                (1.0, 1.0),
                0.0,
                True,
            )

        self.viewer.log_instances(
            "/dynamic_bottle_red_cylinder_preview",
            self._preview_red_cylinder_mesh,
            red_transform,
            wp.array([wp.vec3(self.spec.radius, self.spec.radius, self.spec.red_cylinder_height)], dtype=wp.vec3),
            self._preview_red_cylinder_color,
            self._preview_red_cylinder_material,
        )

    def gui(self, imgui) -> None:
        changed = False
        imgui.text("Dynamic bottle body")
        imgui.separator()

        updated, value = imgui.slider_float("Cylinder Radius [m]", float(self.spec.radius), 0.005, 0.2, "%.4f")
        if updated:
            self.spec.radius = value
            changed = True
        updated, value = imgui.slider_float("Cylinder Height [m]", float(self.spec.height), 0.02, 0.5, "%.4f")
        if updated:
            self.spec.height = value
            changed = True
        updated, value = imgui.slider_float(
            "Red Cylinder Height [m]",
            float(self.spec.red_cylinder_height),
            0.001,
            max(float(self.spec.height), 0.001),
            "%.4f",
        )
        if updated:
            self.spec.red_cylinder_height = value
            changed = True

        imgui.separator()
        for axis, index in (("X", 0), ("Y", 1), ("Z", 2)):
            updated, value = imgui.slider_float(f"Position {axis} [m]", float(self.spec.pos[index]), -2.0, 2.0, "%.4f")
            if updated:
                self.spec.pos[index] = value
                changed = True

        imgui.separator()
        for axis, index in (("Roll", 0), ("Pitch", 1), ("Yaw", 2)):
            updated, value = imgui.slider_float(f"{axis} [deg]", float(self.spec.rpy_deg[index]), -180.0, 180.0, "%.2f")
            if updated:
                self.spec.rpy_deg[index] = value
                changed = True

        imgui.separator()
        water_mass = _water_mass_for_cylinder(self.spec.radius, self.spec.red_cylinder_height, self.spec.water_density)
        imgui.text(f"Water Density [kg/m^3]: {self.spec.water_density:.1f}")
        imgui.text(f"Computed Water Mass [kg]: {water_mass:.4f}")
        updated, value = imgui.slider_float("Friction", float(self.spec.friction), 0.0, 8.0, "%.3f")
        if updated:
            self.spec.friction = value
            changed = True

        if changed:
            self._dirty = True

        imgui.separator()
        if imgui.button("Save Dynamic Bottle Spec"):
            self.save_spec()
        if self.status:
            imgui.text(self.status)

    def test_final(self) -> None:
        if self.state_0.body_q is not None:
            body_q = self.state_0.body_q.numpy()
            if not np.isfinite(body_q).all():
                raise RuntimeError("Dynamic bottle editor produced non-finite body transforms")

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = newton.examples.create_parser()
        parser.description = "Edit a dynamic cylindrical bottle body with a GLB visual."
        parser.add_argument("--visual-glb", type=Path, default=DEFAULT_BOTTLE_GLB, help="Bottle GLB visual asset.")
        parser.add_argument(
            "--input",
            type=Path,
            default=None,
            help="Existing dynamic bottle JSON spec to edit. Defaults to --output if it already exists.",
        )
        parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output dynamic bottle JSON spec.")
        parser.add_argument(
            "--save-spec", action="store_true", help="Write the current spec immediately after startup."
        )
        parser.add_argument("--radius", type=_positive_float, default=None, help="Initial cylinder radius [m].")
        parser.add_argument("--height", type=_positive_float, default=None, help="Initial cylinder height [m].")
        parser.add_argument(
            "--red-cylinder-height",
            type=_positive_float,
            default=DEFAULT_WATER_CYLINDER_HEIGHT_M,
            help="Initial red cylinder height [m].",
        )
        parser.add_argument(
            "--water-density",
            type=_positive_float,
            default=WATER_DENSITY_KG_PER_M3,
            help="Water cylinder density [kg/m^3].",
        )
        parser.add_argument(
            "--fit-margin", type=float, default=-0.001, help="Extra envelope margin around the GLB [m]."
        )
        parser.add_argument("--pos-x", type=float, default=0.42, help="Initial body X position [m].")
        parser.add_argument("--pos-y", type=float, default=0.07, help="Initial body Y position [m].")
        parser.add_argument("--pos-z", type=float, default=-0.57, help="Initial body Z position [m].")
        parser.add_argument("--roll", type=float, default=-90.0, help="Initial body roll [deg].")
        parser.add_argument("--pitch", type=float, default=0.0, help="Initial body pitch [deg].")
        parser.add_argument("--yaw", type=float, default=0.0, help="Initial body yaw [deg].")
        parser.add_argument("--mass", type=_positive_float, default=0.22, help="Deprecated; ignored.")
        parser.add_argument("--friction", type=float, default=3.0, help="Cylinder collision friction coefficient.")
        parser.add_argument("--gravity", type=float, default=-9.81, help="Gravity acceleration along Z [m/s^2].")
        parser.add_argument("--fps", type=float, default=60.0, help="Viewer frame rate [Hz].")
        parser.add_argument(
            "--add-ground", action=argparse.BooleanOptionalAction, default=True, help="Add a ground plane."
        )
        return parser


def main() -> None:
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
