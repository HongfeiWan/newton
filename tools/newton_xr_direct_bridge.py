# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""In-process Newton viewer frame bridge for Quest XR output."""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import warp as wp


@wp.kernel
def _copy_rgb_flip_x(input_img: wp.array3d[wp.uint8], output_img: wp.array3d[wp.uint8], width: int):
    x, y = wp.tid()
    source_x = width - x - 1
    output_img[y, x, 0] = input_img[y, source_x, 0]
    output_img[y, x, 1] = input_img[y, source_x, 1]
    output_img[y, x, 2] = input_img[y, source_x, 2]


class NewtonXrBridgeUnavailable(RuntimeError):
    """Raised when the direct GPU/XR runtime is not available."""


@dataclass
class NewtonXrBridgeConfig:
    """Configuration for the in-process Newton to XR bridge."""

    width: int = 1280
    height: int = 720
    backend: str = "auto"
    gpu: int = 0
    plane_distance: float = 1.4
    plane_width: float = 1.35
    plane_offset_x: float = 0.0
    plane_offset_y: float = 0.0
    lock_mode: str = "head"
    look_away_angle: float = 55.0
    reposition_distance: float = 0.35
    reposition_delay: float = 0.5
    transition_duration: float = 0.25
    verbose: bool = False
    scheduler_threads: int = 3
    capture_fps: float = 60.0
    flip_x: bool = False


class _LatestFrameStore:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self._lock = threading.Lock()
        self._frame: Any | None = None
        self._sequence = 0
        self._resize_count = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def update(self, frame: Any) -> int:
        shape = tuple(int(v) for v in getattr(frame, "shape", ()))
        if len(shape) != 3 or shape[2] != 3:
            raise RuntimeError(f"direct-gpu frame shape mismatch: got {shape}, expected (height, width, 3)")
        with self._lock:
            if shape[0] != self._height or shape[1] != self._width:
                self._height = shape[0]
                self._width = shape[1]
                self._resize_count += 1
            self._frame = frame
            self._sequence += 1
            return self._sequence

    def latest(self) -> tuple[Any | None, tuple[int, int], int]:
        with self._lock:
            return self._frame, (self._width, self._height), self._resize_count


def _add_camera_streamer_paths() -> None:
    candidates = [
        Path("/workspace/IsaacTeleop/examples/camera_streamer"),
        Path("/workspace/IsaacTeleop/examples/camera_streamer/build"),
        Path("/camera_streamer"),
        Path("/camera_streamer/build"),
    ]
    if os.environ.get("CAMERA_STREAMER_ROOT"):
        candidates.insert(0, Path(os.environ["CAMERA_STREAMER_ROOT"]))
    if os.environ.get("ISAAC_TELEOP_ROOT"):
        candidates.insert(1, Path(os.environ["ISAAC_TELEOP_ROOT"]) / "examples" / "camera_streamer")
    for path in candidates:
        if not str(path):
            continue
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _load_direct_gpu_modules() -> dict[str, Any]:
    _add_camera_streamer_paths()
    missing: list[str] = []
    modules: dict[str, Any] = {}
    for name in (
        "holoscan",
        "holoscan.core",
        "holoscan.schedulers",
        "holohub.xr",
        "cupy",
        "xr_plane_renderer",
    ):
        try:
            modules[name] = __import__(name, fromlist=["*"])
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        joined = "\n  ".join(missing)
        raise NewtonXrBridgeUnavailable(
            "direct-gpu VR output requires a runtime image containing Holoscan, "
            "holohub.xr, CuPy, and the camera_streamer xr_plane_renderer extension.\n"
            f"Missing imports:\n  {joined}\n"
            "Build/use docker/Dockerfile.direct_gpu or run with "
            "--vr-output-mode legacy-v4l2."
        )
    return modules


class NewtonXrBridge:
    """Bridge Newton's GL viewer frame into the existing XR plane renderer."""

    def __init__(self, config: NewtonXrBridgeConfig):
        self.config = config
        self._store = _LatestFrameStore(config.width, config.height)
        self._target_image: Any | None = None
        self._app: Any | None = None
        self._thread: threading.Thread | None = None
        self._flipped_image: Any | None = None
        self._started = False
        self._captured_frames = 0
        self._last_log_time = 0.0
        self._last_resize_count = 0
        self._next_capture_time = 0.0

    def start(self) -> None:
        if self._started:
            return
        modules = _load_direct_gpu_modules()
        backend = self.config.backend
        if backend == "external-texture":
            raise NewtonXrBridgeUnavailable(
                "direct-gpu backend external-texture is not implemented in this Python bridge yet; "
                "use --direct-gpu-xr-backend auto or pbo-cuda."
            )
        if backend == "auto":
            print(
                "[newton-xr-direct] external-texture backend is unavailable; using pbo-cuda GPU frame source.",
                flush=True,
            )
        self._app = _create_holoscan_app(self.config, self._store, modules)
        self._thread = threading.Thread(target=self._run_app, name="newton-xr-direct", daemon=True)
        self._thread.start()
        self._started = True
        print(
            "[newton-xr-direct] started "
            f"backend=pbo-cuda gpu={self.config.gpu} frame={self.config.width}x{self.config.height}",
            flush=True,
        )

    def _run_app(self) -> None:
        assert self._app is not None
        try:
            self._app.run()
        except Exception as exc:
            print(f"[newton-xr-direct] error: XR Holoscan app stopped: {exc}", flush=True)

    def capture_viewer(self, viewer: Any) -> None:
        if not self._started:
            return
        now = time.monotonic()
        if self.config.capture_fps > 0.0 and now < self._next_capture_time:
            return
        if self.config.capture_fps > 0.0:
            self._next_capture_time = now + 1.0 / self.config.capture_fps
        frame = viewer.get_frame(target_image=self._target_image, render_ui=False)
        self._target_image = frame
        frame_for_xr = self._maybe_flip_x(frame)
        sequence = self._store.update(frame_for_xr)
        self._captured_frames = sequence
        _, (width, height), resize_count = self._store.latest()
        if resize_count != self._last_resize_count:
            print(
                f"[newton-xr-direct] viewer frame size {width}x{height}; using actual GL capture size for XR plane.",
                flush=True,
            )
            self._last_resize_count = resize_count
        if self.config.verbose and now - self._last_log_time >= 5.0:
            print(f"[newton-xr-direct] captured_frames={sequence}", flush=True)
            self._last_log_time = now

    def _maybe_flip_x(self, frame: Any) -> Any:
        if not self.config.flip_x:
            return frame

        if self._flipped_image is None or self._flipped_image.shape != frame.shape:
            self._flipped_image = wp.empty(shape=frame.shape, dtype=wp.uint8, device=frame.device)
        height, width, _channels = tuple(int(v) for v in frame.shape)
        wp.launch(
            _copy_rgb_flip_x,
            dim=(width, height),
            inputs=[frame, self._flipped_image, width],
            device=frame.device,
        )
        return self._flipped_image

    def stop(self) -> None:
        app = self._app
        if app is not None and hasattr(app, "stop"):
            try:
                app.stop()
            except Exception:
                pass
        self._started = False


def _create_holoscan_app(config: NewtonXrBridgeConfig, store: _LatestFrameStore, modules: dict[str, Any]) -> Any:
    cp = modules["cupy"]
    holoscan = modules["holoscan"]
    core = modules["holoscan.core"]
    schedulers = modules["holoscan.schedulers"]
    xr = modules["holohub.xr"]
    xr_plane = modules["xr_plane_renderer"]

    Application = core.Application
    MetadataPolicy = core.MetadataPolicy
    Operator = core.Operator
    OperatorSpec = core.OperatorSpec
    EventBasedScheduler = schedulers.EventBasedScheduler
    as_tensor = holoscan.as_tensor
    XrPlaneConfig = getattr(xr_plane, "XrPlaneConfig", None) or getattr(xr_plane, "CppXrPlaneConfig", None)
    if XrPlaneConfig is None:
        raise NewtonXrBridgeUnavailable(
            "xr_plane_renderer does not expose XrPlaneConfig/CppXrPlaneConfig; "
            f"available symbols: {', '.join(name for name in dir(xr_plane) if 'Plane' in name)}"
        )
    XrPlaneRendererOp = xr_plane.XrPlaneRendererOp

    class NewtonFrameSourceOp(Operator):
        def __init__(self, fragment: Any, *args: Any, frame_store: _LatestFrameStore, verbose: bool, **kwargs: Any):
            self._frame_store = frame_store
            self._verbose = verbose
            super().__init__(fragment, *args, **kwargs)

        def setup(self, spec: OperatorSpec) -> None:
            spec.output("frame")

        def start(self) -> None:
            self._placeholder = cp.zeros((self._frame_store.height, self._frame_store.width, 3), dtype=cp.uint8)
            self._last_frame = self._placeholder
            self._emitted = 0

        def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
            frame, _, _ = self._frame_store.latest()
            if frame is not None:
                self._last_frame = cp.asarray(frame)
            self._emitted += 1
            op_output.emit(as_tensor(self._last_frame), "frame", emitter_name="holoscan::Tensor")

    class NewtonDirectXrApp(Application):
        def __init__(self, bridge_config: NewtonXrBridgeConfig, frame_store: _LatestFrameStore):
            self._bridge_config = bridge_config
            self._frame_store = frame_store
            super().__init__()
            self.metadata_policy = MetadataPolicy.UPDATE

        def compose(self) -> None:
            xr_session = xr.XrSession(self)
            left_hand_tracker = None
            right_hand_tracker = None
            if hasattr(xr, "XrHandTracker") and hasattr(xr, "XrHandEXT"):
                left_hand_tracker = xr.XrHandTracker(
                    self,
                    xr_session=xr_session,
                    hand=xr.XrHandEXT.XR_HAND_LEFT_EXT,
                    name="newton_xr_left_hand_tracker",
                )
                right_hand_tracker = xr.XrHandTracker(
                    self,
                    xr_session=xr_session,
                    hand=xr.XrHandEXT.XR_HAND_RIGHT_EXT,
                    name="newton_xr_right_hand_tracker",
                )

            frame_source = NewtonFrameSourceOp(
                self,
                name="newton_frame_source",
                frame_store=self._frame_store,
                verbose=self._bridge_config.verbose,
            )
            plane = XrPlaneConfig(
                name="newton_viewer",
                distance=self._bridge_config.plane_distance,
                width=self._bridge_config.plane_width,
                offset_x=self._bridge_config.plane_offset_x,
                offset_y=self._bridge_config.plane_offset_y,
                lock_mode=self._bridge_config.lock_mode,
                look_away_angle=self._bridge_config.look_away_angle,
                reposition_distance=self._bridge_config.reposition_distance,
                reposition_delay=self._bridge_config.reposition_delay,
                transition_duration=self._bridge_config.transition_duration,
                is_stereo=False,
            )
            xr_begin = xr.XrBeginFrameOp(self, xr_session=xr_session, name="newton_xr_begin_frame")
            xr_end = xr.XrEndFrameOp(self, xr_session=xr_session, name="newton_xr_end_frame")
            xr_renderer = XrPlaneRendererOp(
                self,
                name="newton_xr_plane_renderer",
                xr_session=xr_session,
                planes=[plane],
                left_hand_tracker=left_hand_tracker,
                right_hand_tracker=right_hand_tracker,
                verbose=self._bridge_config.verbose,
            )
            self.add_flow(frame_source, xr_renderer, {("frame", "camera_frame_0")})
            self.add_flow(self.start_op(), xr_begin)
            self.add_flow(xr_begin, xr_renderer, {("xr_frame_state", "xr_frame_state")})
            self.add_flow(xr_renderer, xr_end, {("xr_composition_layer", "xr_composition_layers")})
            self.add_flow(xr_begin, xr_end, {("xr_frame_state", "xr_frame_state")})
            self.add_flow(xr_end, xr_begin)
            self.add_operator(frame_source)
            self.add_operator(xr_begin)
            self.add_operator(xr_renderer)
            self.add_operator(xr_end)
            scheduler = EventBasedScheduler(
                self,
                name="newton_xr_scheduler",
                worker_thread_number=max(1, int(self._bridge_config.scheduler_threads)),
                stop_on_deadlock=False,
            )
            self.scheduler(scheduler)

    return NewtonDirectXrApp(config, store)
