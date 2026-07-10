#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a quick material-color preview of the right L10 hand in a URDF."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "assets" / "generated" / "dual_nero_linker_l10_combined.urdf"
DEFAULT_OUTPUT = REPO_ROOT / "debug" / "l10_urdf_color_preview"


def _rpy_matrix(rpy: str) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy.split())
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def _origin_matrix(origin: ET.Element | None) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    if origin is None:
        return mat
    xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()], dtype=np.float64)
    mat[:3, :3] = _rpy_matrix(origin.get("rpy", "0 0 0"))
    mat[:3, 3] = xyz
    return mat


def _rgba(visual: ET.Element) -> tuple[float, float, float, float]:
    color = visual.find("material/color")
    if color is None:
        return (0.7, 0.7, 0.7, 1.0)
    vals = [float(v) for v in color.get("rgba", "0.7 0.7 0.7 1").split()]
    return tuple(vals[:4])


def _resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return (urdf_path.parent / path).resolve()


def _box_mesh(size: str) -> trimesh.Trimesh:
    extents = [float(v) for v in size.split()]
    return trimesh.creation.box(extents=extents)


def _load_visual_mesh(urdf_path: Path, visual: ET.Element) -> trimesh.Trimesh | None:
    mesh_elem = visual.find("geometry/mesh")
    if mesh_elem is not None:
        mesh = trimesh.load(_resolve_mesh_path(urdf_path, mesh_elem.get("filename")), force="mesh")
        scale = mesh_elem.get("scale")
        if scale:
            mesh.apply_scale([float(v) for v in scale.split()])
        return mesh
    box_elem = visual.find("geometry/box")
    if box_elem is not None:
        return _box_mesh(box_elem.get("size"))
    return None


def _sample_faces(mesh: trimesh.Trimesh, max_faces: int) -> np.ndarray:
    faces = np.asarray(mesh.faces)
    if len(faces) <= max_faces:
        return faces
    stride = max(1, len(faces) // max_faces)
    return faces[::stride][:max_faces]


def _link_transforms(root: ET.Element, root_link: str) -> dict[str, np.ndarray]:
    children = defaultdict(list)
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if not child_name.startswith("right_l10_"):
            continue
        children[parent_name].append((child_name, _origin_matrix(joint.find("origin"))))

    transforms = {root_link: np.eye(4, dtype=np.float64)}
    queue = deque([root_link])
    while queue:
        parent = queue.popleft()
        for child, joint_tf in children[parent]:
            transforms[child] = transforms[parent] @ joint_tf
            queue.append(child)
    return transforms


def render_view(
    root: ET.Element,
    urdf_path: Path,
    output_path: Path,
    *,
    elev: float,
    azim: float,
    max_faces_per_visual: int,
) -> None:
    transforms = _link_transforms(root, "right_l10_hand_base_link")

    fig = plt.figure(figsize=(12, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    all_points = []

    for link in root.findall("link"):
        link_name = link.get("name", "")
        if link_name not in transforms:
            continue
        link_tf = transforms[link_name]
        for visual in link.findall("visual"):
            mesh = _load_visual_mesh(urdf_path, visual)
            if mesh is None:
                continue
            visual_tf = _origin_matrix(visual.find("origin"))
            mesh.apply_transform(link_tf @ visual_tf)
            vertices = np.asarray(mesh.vertices)
            faces = _sample_faces(mesh, max_faces_per_visual)
            collection = Poly3DCollection(vertices[faces], linewidths=0.0, alpha=1.0)
            color = _rgba(visual)[:3]
            collection.set_facecolor(color)
            collection.set_edgecolor("none")
            ax.add_collection3d(collection)
            all_points.append(vertices)

    points = np.vstack(all_points)
    center = points.mean(axis=0)
    span = float(np.max(points.max(axis=0) - points.min(axis=0)))
    for setter, c in ((ax.set_xlim, center[0]), (ax.set_ylim, center[1]), (ax.set_zlim, center[2])):
        setter(c - 0.55 * span, c + 0.55 * span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=(0.78, 0.78, 0.76))
    print(f"Wrote {output_path}")


def render(urdf_path: Path, output_prefix: Path, *, max_faces_per_visual: int) -> None:
    root = ET.parse(urdf_path).getroot()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, elev, azim in (
        ("palm", 12.0, -78.0),
        ("side", 6.0, -8.0),
        ("three_quarter", 18.0, -42.0),
    ):
        render_view(
            root,
            urdf_path,
            output_prefix.with_name(f"{output_prefix.name}_{suffix}.png"),
            elev=elev,
            azim=azim,
            max_faces_per_visual=max_faces_per_visual,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-faces-per-visual", type=int, default=2500)
    args = parser.parse_args()
    render(args.urdf.resolve(), args.output_prefix.resolve(), max_faces_per_visual=max(100, args.max_faces_per_visual))


if __name__ == "__main__":
    main()
