#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = REPO_ROOT / "tools" / "camera_streamer_overlay"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patch IsaacTeleop camera_streamer for Newton sim-screen XR output.",
    )
    parser.add_argument(
        "--camera-streamer-root",
        required=True,
        help="Path to IsaacTeleop/examples/camera_streamer.",
    )
    parser.add_argument(
        "--dockerfile-syntax-image",
        default=None,
        help=(
            "Optional BuildKit frontend image for Dockerfile # syntax. "
            "Use a mirror such as docker.1ms.run/docker/dockerfile:1 when Docker Hub is unreachable."
        ),
    )
    return parser


def _replace_once(text: str, old: str, new: str, path: Path) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Expected snippet not found while patching {path}")
    return text.replace(old, new, 1)


def _copy_overlay_tree(root: Path) -> None:
    if not OVERLAY_ROOT.is_dir():
        return
    for overlay_path in OVERLAY_ROOT.rglob("*"):
        if overlay_path.is_dir():
            continue
        if "__pycache__" in overlay_path.parts or overlay_path.suffix == ".pyc":
            continue
        relative_path = overlay_path.relative_to(OVERLAY_ROOT)
        target_path = root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(overlay_path, target_path)


def _patch_camera_config(root: Path) -> None:
    path = root / "camera_config.py"
    text = path.read_text(encoding="utf-8")
    if "roi_zoom: float" not in text:
        text = _replace_once(
            text,
            "    device: str | None = None\n",
            """    device: str | None = None
    roi_zoom: float = 1.0
    \"\"\"ROI digital zoom factor. 1.0 disables ROI crop.\"\"\"
    roi_center_x: float = 0.5
    \"\"\"ROI center X in normalized image coordinates.\"\"\"
    roi_center_y: float = 0.5
    \"\"\"ROI center Y in normalized image coordinates.\"\"\"
""",
            path,
        )

    if '        "roi_zoom",\n' not in text:
        text = _replace_once(
            text,
            """        "device",
        "video_dir",
""",
            """        "device",
        "roi_zoom",
        "roi_center_x",
        "roi_center_y",
        "video_dir",
""",
            path,
        )

    roi_validation = """        if self.roi_zoom < 1.0:
            raise ValueError(
                f"Camera '{self.name}': roi_zoom must be >= 1.0 (got {self.roi_zoom!r})"
            )
        if not 0.0 <= self.roi_center_x <= 1.0:
            raise ValueError(
                f"Camera '{self.name}': roi_center_x must be in [0, 1] (got {self.roi_center_x!r})"
            )
        if not 0.0 <= self.roi_center_y <= 1.0:
            raise ValueError(
                f"Camera '{self.name}': roi_center_y must be in [0, 1] (got {self.roi_center_y!r})"
            )
"""
    if roi_validation not in text:
        text = _replace_once(
            text,
            "        self._validate_rgb_fields()\n\n    def _validate_rgb_fields(self):\n",
            "        self._validate_rgb_fields()\n" + roi_validation + "\n    def _validate_rgb_fields(self):\n",
            path,
        )

    if 'roi_zoom=float(data.get("roi_zoom", 1.0))' not in text:
        text = _replace_once(
            text,
            """            device=data.get("device"),
            video_dir=data.get("video_dir"),
""",
            """            device=data.get("device"),
            roi_zoom=float(data.get("roi_zoom", 1.0)),
            roi_center_x=float(data.get("roi_center_x", 0.5)),
            roi_center_y=float(data.get("roi_center_y", 0.5)),
            video_dir=data.get("video_dir"),
""",
            path,
        )
    path.write_text(text, encoding="utf-8")


def _patch_camera_sources(root: Path) -> None:
    path = root / "camera_sources.py"
    text = path.read_text(encoding="utf-8")
    if "v4l2_kwargs = dict(" not in text:
        text = _replace_once(
            text,
            """    v4l2_source = V4L2VideoCaptureOp(
        fragment,
        name=f"{cam_name}_source",
        allocator=allocator,
        device=device,
        width=cam_cfg.width,
        height=cam_cfg.height,
        frame_rate=cam_cfg.fps,
        pass_through=True,
    )
""",
            """    v4l2_kwargs = dict(
        allocator=allocator,
        device=device,
        width=cam_cfg.width,
        height=cam_cfg.height,
        pass_through=True,
    )
    if cam_cfg.fps > 0:
        v4l2_kwargs["frame_rate"] = cam_cfg.fps

    v4l2_source = V4L2VideoCaptureOp(
        fragment,
        name=f"{cam_name}_source",
        **v4l2_kwargs,
    )
""",
            path,
        )
    if "RoiCropZoomOp" not in text:
        text = _replace_once(
            text,
            """    if color_format == "rgb":
        result = CameraSourceResult(
            operators=[v4l2_source, yuyv_to_rgb],
            flows=[
                (v4l2_source, yuyv_to_rgb, {("signal", "source_video")}),
            ],
            frame_outputs={"mono": (yuyv_to_rgb, "tensor")},
        )
""",
            """    if color_format == "rgb":
        result = CameraSourceResult(
            operators=[v4l2_source, yuyv_to_rgb],
            flows=[
                (v4l2_source, yuyv_to_rgb, {("signal", "source_video")}),
            ],
            frame_outputs={"mono": (yuyv_to_rgb, "tensor")},
        )
        if getattr(cam_cfg, "roi_zoom", 1.0) > 1.0:
            from operators.roi_crop_zoom.roi_crop_zoom_op import RoiCropZoomOp

            roi_crop_zoom = RoiCropZoomOp(
                fragment,
                name=f"{cam_name}_roi_crop_zoom",
                zoom=cam_cfg.roi_zoom,
                center_x=cam_cfg.roi_center_x,
                center_y=cam_cfg.roi_center_y,
                tensor_name=cam_name,
            )
            result.operators.append(roi_crop_zoom)
            result.flows.append((yuyv_to_rgb, roi_crop_zoom, {("tensor", "frame_in")}))
            result.frame_outputs["mono"] = (roi_crop_zoom, "frame_out")
            logger.info(
                f"  V4L2 ROI crop zoom: {cam_name} {cam_cfg.roi_zoom:.2f}x "
                f"center=({cam_cfg.roi_center_x:.2f}, {cam_cfg.roi_center_y:.2f})"
            )
""",
            path,
        )
    path.write_text(text, encoding="utf-8")


def _patch_teleop_camera_subgraph(root: Path) -> None:
    path = root / "teleop_camera_subgraph.py"
    text = path.read_text(encoding="utf-8")

    text = _replace_once(
        text,
        "from dataclasses import dataclass\nfrom enum import Enum\nfrom typing import Any\n",
        "from dataclasses import dataclass\nfrom enum import Enum\nimport os\nfrom typing import Any\n",
        path,
    )

    text = _replace_once(
        text,
        """            if cam_cfg.fps <= 0:
                errors.append(
                    f"Camera '{cam_name}': fps must be positive (got {cam_cfg.fps})"
                )
""",
        """            if cam_cfg.fps <= 0 and not (
                self.source == "local"
                and cam_cfg.camera_type == "v4l2"
                and cam_cfg.fps == 0
            ):
                errors.append(
                    f"Camera '{cam_name}': fps must be positive (got {cam_cfg.fps})"
                )
""",
        path,
    )

    text = _replace_once(
        text,
        """        xr_session = self._xr_session

        # XR frame timing
""",
        """        xr_session = self._xr_session

        hand_overlay_disabled = os.environ.get(
            "TELEOP_CAMERA_DISABLE_HAND_OVERLAY", ""
        ).lower() in ("1", "true", "yes", "on")
        xr_left_hand_tracker = None
        xr_right_hand_tracker = None
        if hand_overlay_disabled:
            logger.info(
                "XR hand skeleton overlay disabled via TELEOP_CAMERA_DISABLE_HAND_OVERLAY"
            )
        elif hasattr(xr, "XrHandTracker") and hasattr(xr, "XrHandEXT"):
            xr_left_hand_tracker = xr.XrHandTracker(
                self.fragment,
                xr_session=xr_session,
                hand=xr.XrHandEXT.XR_HAND_LEFT_EXT,
                name=self._create_name("xr_left_hand_tracker"),
            )
            xr_right_hand_tracker = xr.XrHandTracker(
                self.fragment,
                xr_session=xr_session,
                hand=xr.XrHandEXT.XR_HAND_RIGHT_EXT,
                name=self._create_name("xr_right_hand_tracker"),
            )
        else:
            logger.warning(
                "XR hand skeleton overlay disabled: holohub.xr bindings do not expose XrHandTracker"
            )

        # XR frame timing
""",
        path,
    )

    text = _replace_once(
        text,
        """        xr_renderer = XrPlaneRendererOp(
            self.fragment,
            name=self._create_name("xr_plane_renderer"),
            xr_session=xr_session,
            planes=plane_configs,
            verbose=verbose,
        )
""",
        """        xr_renderer = XrPlaneRendererOp(
            self.fragment,
            name=self._create_name("xr_plane_renderer"),
            xr_session=xr_session,
            planes=plane_configs,
            left_hand_tracker=xr_left_hand_tracker,
            right_hand_tracker=xr_right_hand_tracker,
            verbose=verbose,
        )
""",
        path,
    )

    text = _replace_once(
        text,
        """        logger.info(
            f"XR mode: {len(plane_configs)} camera planes (single Vulkan context)"
        )
""",
        """        hand_overlay_state = (
            "enabled"
            if xr_left_hand_tracker is not None and xr_right_hand_tracker is not None
            else "disabled"
        )
        logger.info(
            "XR mode: "
            f"{len(plane_configs)} camera planes (single Vulkan context), "
            f"hand skeleton overlay={hand_overlay_state}"
        )
""",
        path,
    )

    path.write_text(text, encoding="utf-8")


def _patch_camera_streamer_script(root: Path) -> None:
    path = root / "camera_streamer.sh"
    text = path.read_text(encoding="utf-8")
    insert = '        -e TELEOP_CAMERA_DISABLE_HAND_OVERLAY="${TELEOP_CAMERA_DISABLE_HAND_OVERLAY:-}" \\\n'
    if insert not in text:
        anchor = '        -e NV_CXR_RUNTIME_DIR="$NV_CXR_RUNTIME_DIR" \\\n'
        if anchor not in text:
            raise RuntimeError(f"Expected snippet not found while patching {path}")
        text = text.replace(anchor, anchor + insert, 1)
    path.write_text(text, encoding="utf-8")


def _patch_dockerfile_syntax(root: Path, syntax_image: str | None) -> None:
    if not syntax_image:
        return
    path = root / "Dockerfile"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if not lines or not lines[0].startswith("# syntax="):
        raise RuntimeError(f"Expected Dockerfile # syntax directive at first line in {path}")
    newline = "\n" if lines[0].endswith("\n") else ""
    lines[0] = f"# syntax={syntax_image}{newline}"
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    root = Path(args.camera_streamer_root).expanduser().resolve()
    if not (root / "teleop_camera_subgraph.py").is_file():
        raise SystemExit(f"camera_streamer root not found or invalid: {root}")

    _patch_dockerfile_syntax(root, args.dockerfile_syntax_image)
    _copy_overlay_tree(root)
    _patch_camera_config(root)
    _patch_camera_sources(root)
    _patch_teleop_camera_subgraph(root)
    _patch_camera_streamer_script(root)
    print(f"[camera-streamer-overlay] patched {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
