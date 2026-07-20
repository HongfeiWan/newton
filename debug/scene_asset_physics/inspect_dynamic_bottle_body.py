# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interactive dynamics inspector for ``dynamic_bottle_body.json``."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

try:
    from debug.scene_asset_physics.edit_dynamic_bottle_body import (
        DEFAULT_OUTPUT,
        _rotation_from_euler_deg,
        _water_mass_for_cylinder,
        _wp_transform_from_pos_quat,
        build_dynamic_bottle,
        load_dynamic_bottle_spec,
    )
except ModuleNotFoundError:
    from edit_dynamic_bottle_body import (
        DEFAULT_OUTPUT,
        _rotation_from_euler_deg,
        _water_mass_for_cylinder,
        _wp_transform_from_pos_quat,
        build_dynamic_bottle,
        load_dynamic_bottle_spec,
    )


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _quat_to_rotation(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        ),
        dtype=np.float64,
    )


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = (float(v) for v in left)
    rx, ry, rz, rw = (float(v) for v in right)
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float64,
    )


def _compose_transform_rows(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    rotation = _quat_to_rotation(parent[3:7])
    pos = parent[:3] + rotation @ child[:3]
    quat = _quat_multiply_xyzw(parent[3:7], child[3:7])
    norm = np.linalg.norm(quat)
    if norm > 0.0:
        quat = quat / norm
    return np.asarray((*pos, *quat), dtype=np.float32)


def _cylinder_vertical_extent(radius: float, height: float, rpy_deg: list[float]) -> float:
    axis = _rotation_from_euler_deg(tuple(rpy_deg))[:, 2]
    radial_z = float(np.sqrt(max(0.0, 1.0 - axis[2] * axis[2])) * radius)
    axial_z = float(abs(axis[2]) * height)
    return radial_z + axial_z


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = float(args.fps)
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.spec_path = _resolve_path(args.input)
        self.spec = load_dynamic_bottle_spec(self.spec_path)

        if args.place_on_ground:
            self.spec.pos[0] = float(args.pos_x)
            self.spec.pos[1] = float(args.pos_y)
            self.spec.pos[2] = _cylinder_vertical_extent(self.spec.radius, self.spec.height, self.spec.rpy_deg)

        builder = newton.ModelBuilder(up_axis="Z", gravity=float(args.gravity))
        handles = build_dynamic_bottle(builder, self.spec)
        if args.add_ground:
            builder.add_ground_plane()

        self.body_index = int(handles["body"])
        self.collision_shape = int(handles["collision_shape"])
        self.water_collision_shape = int(handles["water_collision_shape"])
        self.model = builder.finalize(device=args.device)
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=int(args.iterations),
            enable_restitution=bool(args.enable_restitution),
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self._initial_body_q = self.state_0.body_q.numpy().copy()
        self._initial_body_qd = self.state_0.body_qd.numpy().copy()
        self._body_qd_host = self._initial_body_qd.copy()
        self._com_color = wp.array([wp.vec3(1.0, 0.95, 0.05)], dtype=wp.vec3, device=self.model.device)
        self._com_radius = max(0.01, 0.25 * float(self.spec.radius))
        self._proxy_meshes: dict[int, str] = {}
        self._proxy_blue_color = wp.array([wp.vec3(0.1, 0.55, 1.0)], dtype=wp.vec3, device=self.model.device)
        self._proxy_water_color = wp.array([wp.vec3(1.0, 0.05, 0.03)], dtype=wp.vec3, device=self.model.device)
        self._proxy_blue_material = wp.array([wp.vec4(0.35, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self._proxy_water_material = wp.array([wp.vec4(0.55, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)

        self.viewer.picking_enabled = True
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.5, -1.0, 0.35), pitch=-15.0, yaw=125.0)

        self.model.collide(self.state_0, self.contacts)
        blue_contacts, water_contacts = self.contact_shape_counts()

        print(
            "Dynamic bottle inspector: right-click and drag the bottle to apply mouse forces.",
            flush=True,
        )
        print(
            f"Loaded {self.spec_path} mass={self.model.body_mass.numpy()[self.body_index]:.6g} kg "
            f"body_com={self.model.body_com.numpy()[self.body_index].tolist()}",
            flush=True,
        )
        print(
            f"Initial rigid contacts: blue_collision={blue_contacts} water_collision={water_contacts}",
            flush=True,
        )

    def reset(self) -> None:
        self.state_0.body_q = wp.array(self._initial_body_q, dtype=wp.transform, device=self.model.device)
        self.state_0.body_qd = wp.array(self._initial_body_qd, dtype=wp.spatial_vector, device=self.model.device)
        self.state_1.body_q = wp.array(self._initial_body_q, dtype=wp.transform, device=self.model.device)
        self.state_1.body_qd = wp.array(self._initial_body_qd, dtype=wp.spatial_vector, device=self.model.device)
        self._body_qd_host = self._initial_body_qd.copy()
        self.model.collide(self.state_0, self.contacts)
        self.sim_time = 0.0

    def apply_velocity_impulse(self, velocity: tuple[float, float, float]) -> None:
        self._body_qd_host = self.state_0.body_qd.numpy().copy()
        self._body_qd_host[self.body_index, :3] += np.asarray(velocity, dtype=np.float32)
        self.state_0.body_qd = wp.array(self._body_qd_host, dtype=wp.spatial_vector, device=self.model.device)

    def apply_angular_impulse(self, angular_velocity: tuple[float, float, float]) -> None:
        self._body_qd_host = self.state_0.body_qd.numpy().copy()
        self._body_qd_host[self.body_index, 3:] += np.asarray(angular_velocity, dtype=np.float32)
        self.state_0.body_qd = wp.array(self._body_qd_host, dtype=wp.spatial_vector, device=self.model.device)

    def bottle_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        body_q = self.state_0.body_q.numpy()[self.body_index]
        body_qd = self.state_0.body_qd.numpy()[self.body_index]
        rotation = _quat_to_rotation(body_q[3:7])
        world_com = body_q[:3] + rotation @ self.model.body_com.numpy()[self.body_index]
        return body_q, body_qd, rotation, world_com

    def contact_shape_counts(self) -> tuple[int, int]:
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        if count <= 0:
            return 0, 0

        shape0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:count]
        blue = int(np.count_nonzero((shape0 == self.collision_shape) | (shape1 == self.collision_shape)))
        water = int(np.count_nonzero((shape0 == self.water_collision_shape) | (shape1 == self.water_collision_shape)))
        return blue, water

    def collision_proxy_transform(self, shape_index: int) -> wp.transform:
        body_q = self.state_0.body_q.numpy()[self.body_index]
        shape_q = self.model.shape_transform.numpy()[shape_index]
        world_q = _compose_transform_rows(body_q, shape_q)
        return _wp_transform_from_pos_quat(world_q[:3].tolist(), world_q[3:7].tolist())

    def log_collision_proxy(
        self,
        shape_index: int,
        name: str,
        color: wp.array[wp.vec3],
        material: wp.array[wp.vec4],
    ) -> None:
        if not self.args.show_collision_proxies:
            if hasattr(self.viewer, "log_instances") and shape_index in self._proxy_meshes:
                self.viewer.log_instances(name, self._proxy_meshes[shape_index], None, None, None, None, hidden=True)
            return

        scale = self.model.shape_scale.numpy()[shape_index]
        radius = float(scale[0])
        half_height = float(scale[1])
        xform = wp.array([self.collision_proxy_transform(shape_index)], dtype=wp.transform, device=self.model.device)

        if not hasattr(self.viewer, "_populate_geometry") or not hasattr(self.viewer, "log_instances"):
            self.viewer.log_shapes(
                name,
                newton.GeoType.CYLINDER,
                (radius, half_height),
                xform,
                color,
                material,
            )
            return

        mesh = self._proxy_meshes.get(shape_index)
        if mesh is None:
            mesh = f"/dynamic_bottle/collision_proxy_mesh_{shape_index}"
            mesh = self.viewer._populate_geometry(
                int(newton.GeoType.CYLINDER),
                (1.0, 1.0),
                0.0,
                True,
            )
            self._proxy_meshes[shape_index] = mesh

        self.viewer.log_instances(
            name,
            mesh,
            xform,
            wp.array([wp.vec3(radius, radius, half_height)], dtype=wp.vec3, device=self.model.device),
            color,
            material,
        )

    def log_collision_proxies(self) -> None:
        self.log_collision_proxy(
            self.collision_shape,
            "/dynamic_bottle/blue_collision_proxy",
            self._proxy_blue_color,
            self._proxy_blue_material,
        )
        self.log_collision_proxy(
            self.water_collision_shape,
            "/dynamic_bottle/water_collision_proxy",
            self._proxy_water_color,
            self._proxy_water_material,
        )

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        _, _, _, world_com = self.bottle_state()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_points(
            "/dynamic_bottle/com",
            wp.array([wp.vec3(*world_com.tolist())], dtype=wp.vec3, device=self.model.device),
            radii=self._com_radius,
            colors=self._com_color,
        )
        self.viewer.end_frame()

    def gui(self, imgui) -> None:
        body_q, body_qd, _, world_com = self.bottle_state()
        mass = float(self.model.body_mass.numpy()[self.body_index])
        body_com = self.model.body_com.numpy()[self.body_index]
        water_mass = _water_mass_for_cylinder(self.spec.radius, self.spec.red_cylinder_height, self.spec.water_density)
        blue_contacts, water_contacts = self.contact_shape_counts()

        imgui.text("Dynamic bottle inspector")
        imgui.separator()
        imgui.text("Right-click drag applies mouse picking forces.")
        imgui.text(f"Mass [kg]: {mass:.6f}")
        imgui.text(f"Water mass [kg]: {water_mass:.6f}")
        imgui.text(f"Body COM local [m]: {np.round(body_com, 5).tolist()}")
        imgui.text(f"World COM [m]: {np.round(world_com, 5).tolist()}")
        imgui.text(f"Body pos [m]: {np.round(body_q[:3], 5).tolist()}")
        imgui.text(f"Linear vel [m/s]: {np.round(body_qd[:3], 5).tolist()}")
        imgui.text(f"Angular vel [rad/s]: {np.round(body_qd[3:], 5).tolist()}")
        imgui.text(f"Blue height [m]: {self.spec.height:.4f}")
        imgui.text(f"Water height [m]: {self.spec.red_cylinder_height:.4f}")
        imgui.text(f"Water density [kg/m^3]: {self.spec.water_density:.1f}")
        imgui.text(f"Rigid contacts blue/water: {blue_contacts} / {water_contacts}")
        imgui.separator()
        if imgui.button("Reset"):
            self.reset()
        impulse = float(self.args.impulse_velocity)
        if imgui.button("+X Velocity Kick"):
            self.apply_velocity_impulse((impulse, 0.0, 0.0))
        if imgui.button("+Y Velocity Kick"):
            self.apply_velocity_impulse((0.0, impulse, 0.0))
        if imgui.button("+Yaw Spin Kick"):
            self.apply_angular_impulse((0.0, 0.0, float(self.args.angular_impulse)))

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.isfinite(body_q).all() or not np.isfinite(body_qd).all():
            raise RuntimeError("Dynamic bottle inspector produced non-finite body state")

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = newton.examples.create_parser()
        parser.description = "Interactively inspect dynamic_bottle_body.json with mouse picking."
        parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT, help="Dynamic bottle JSON spec.")
        parser.add_argument("--fps", type=float, default=60.0, help="Viewer frame rate [Hz].")
        parser.add_argument("--substeps", type=int, default=8, help="Simulation substeps per frame.")
        parser.add_argument("--iterations", type=int, default=12, help="XPBD solver iterations.")
        parser.add_argument("--gravity", type=float, default=-9.81, help="Gravity acceleration along Z [m/s^2].")
        parser.add_argument(
            "--add-ground", action=argparse.BooleanOptionalAction, default=True, help="Add ground plane."
        )
        parser.add_argument(
            "--place-on-ground",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Move the JSON pose vertically so the bottle starts on the ground.",
        )
        parser.add_argument("--pos-x", type=float, default=0.0, help="Initial X when --place-on-ground is enabled [m].")
        parser.add_argument("--pos-y", type=float, default=0.0, help="Initial Y when --place-on-ground is enabled [m].")
        parser.add_argument("--impulse-velocity", type=float, default=0.4, help="Velocity kick button magnitude [m/s].")
        parser.add_argument(
            "--angular-impulse",
            type=float,
            default=4.0,
            help="Angular velocity kick button magnitude [rad/s].",
        )
        parser.add_argument(
            "--enable-restitution",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable XPBD restitution handling.",
        )
        parser.add_argument(
            "--show-collision-proxies",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Render blue/red collision proxy cylinders in the viewer.",
        )
        return parser


def main() -> None:
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
