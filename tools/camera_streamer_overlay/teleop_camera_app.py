#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Teleop Camera App: Multi-Camera Display Application

Displays multiple camera streams in either monitor mode (tiled 2D) or XR mode (3D planes).

Supports camera sources:
  - RTP receiver: Receives H.264 streams from teleop_camera_sender
  - Direct: Connect cameras directly without sender/receiver
"""

import argparse
import os
import sys
import threading
import time

from holoscan.core import Application, MetadataPolicy
from holoscan.schedulers import EventBasedScheduler
from loguru import logger
from teleop_camera_subgraph import (
    DisplayMode,
    TeleopCameraSubgraph,
    TeleopCameraSubgraphConfig,
)


class TeleopCameraApp(Application):
    """Multi-camera display application.

    This is a thin wrapper around TeleopCameraSubgraph that provides
    the application entry point and YAML configuration loading.
    """

    def __init__(
        self,
        config: TeleopCameraSubgraphConfig,
        scheduler_threads: int = 4,
        *args,
        **kwargs,
    ):
        self._config = config
        self._scheduler_threads = scheduler_threads
        super().__init__(*args, **kwargs)
        self.metadata_policy = MetadataPolicy.UPDATE

    def compose(self):
        """Compose the application using the camera subgraph."""
        xr_session = None
        if self._config.display_mode == DisplayMode.XR:
            try:
                import holohub.xr as xr

                xr_session = xr.XrSession(self)
            except ImportError:
                logger.error("XR mode requires holohub.xr module")
                raise

        TeleopCameraSubgraph(
            self,
            name="teleop_camera",
            config=self._config,
            xr_session=xr_session,
        )

        scheduler = EventBasedScheduler(
            self,
            name="scheduler",
            worker_thread_number=self._scheduler_threads,
            stop_on_deadlock=False,
        )
        self.scheduler(scheduler)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Ignoring invalid integer env {name}={value!r}")
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Ignoring invalid float env {name}={value!r}")
        return default


def _reexec_self(*, reason: str, counter_env: str, count: int, delay_s: float) -> None:
    logger.warning(f"{reason} Retrying in {delay_s:.1f}s... (attempt {count})")
    time.sleep(delay_s)
    logger.info("Re-executing for clean process state...")
    os.environ[counter_env] = str(count)
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _status_path() -> str:
    return os.environ.get("TELEOP_CAMERA_STATUS_PATH", "/tmp/teleop_camera_status.txt")


def _write_status(state: str, detail: str = "") -> None:
    payload = state if not detail else f"{state} {detail}"
    try:
        with open(_status_path(), "w", encoding="utf-8") as status_file:
            status_file.write(f"{payload}\n")
    except OSError as exc:
        logger.warning(f"Failed to write status file: {exc}")


def _promote_running_after_delay(delay_s: float, detail: str) -> threading.Timer:
    def _mark_running() -> None:
        _write_status("running", detail)

    timer = threading.Timer(delay_s, _mark_running)
    timer.daemon = True
    timer.start()
    return timer


def main():
    parser = argparse.ArgumentParser(
        description="Teleop Camera App: Multi-camera display for teleoperation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config/multi_camera.yaml"),
        help="Path to camera configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["monitor", "xr"],
        default=None,
        help="Override display mode",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["rtp", "local"],
        default=None,
        help="Override camera source (rtp: receive streams, local: open cameras directly)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    logger.info(f"Loading config from: {args.config}")

    try:
        config = TeleopCameraSubgraphConfig.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load config '{args.config}': {e}")
        sys.exit(1)

    if args.source:
        config.source = args.source
    if args.mode:
        config.display_mode = DisplayMode(args.mode)
    if args.verbose:
        config.verbose = True

    errors = config.validate()
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Teleop Camera App")
    logger.info("=" * 60)
    logger.info(f"Source: {config.source}")
    logger.info(f"Display mode: {config.display_mode.value}")
    logger.info(f"Cameras: {len(config.cameras)}")
    for cam_name, cam_cfg in config.cameras.items():
        cam_type = "stereo" if cam_cfg.stereo else "mono"
        streams = ", ".join(f"{s}:{cfg.port}" for s, cfg in cam_cfg.streams.items())
        logger.info(f"  {cam_name} ({cam_type}): {streams}")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")
    _write_status("starting", f"mode={config.display_mode.value} source={config.source}")

    xr_retry_errors = (
        "ErrorFormFactorUnavailable",
        "ErrorLimitReached",
        "ErrorRuntimeUnavailable",
        "XR_ERROR_RUNTIME_UNAVAILABLE",
    )
    recoverable_graph_errors = (
        "Tensor '' not found in message",
        "GXF_ENTITY_COMPONENT_NOT_FOUND",
    )
    xr_recovery_count = _env_int("TELEOP_CAMERA_XR_RECOVERY_COUNT", 0)
    max_xr_retries = _env_int("TELEOP_CAMERA_XR_RECOVERY_MAX_RETRIES", 30)
    graph_recovery_count = _env_int("TELEOP_CAMERA_GRAPH_RECOVERY_COUNT", 0)
    max_graph_retries = _env_int("TELEOP_CAMERA_GRAPH_RECOVERY_MAX_RETRIES", 8)
    xr_recovery_delay_s = _env_float("TELEOP_CAMERA_XR_RECOVERY_DELAY_S", 2.0)
    recovery_delay_s = _env_float("TELEOP_CAMERA_GRAPH_RECOVERY_DELAY_S", 2.0)
    running_status_delay_s = _env_float("TELEOP_CAMERA_RUNNING_STATUS_DELAY_S", 5.0)

    while True:
        app = TeleopCameraApp(config)
        running_detail = f"xr_retries={xr_recovery_count} graph_retries={graph_recovery_count}"
        running_timer: threading.Timer | None = None
        try:
            _write_status("activating", running_detail)
            running_timer = _promote_running_after_delay(running_status_delay_s, running_detail)
            app.run()
            if running_timer is not None:
                running_timer.cancel()
            break
        except KeyboardInterrupt:
            if running_timer is not None:
                running_timer.cancel()
            _write_status("stopped", "keyboard_interrupt")
            logger.info("Interrupted by user")
            break
        except Exception as e:
            if running_timer is not None:
                running_timer.cancel()
            msg = str(e)
            if any(err in msg for err in xr_retry_errors) and xr_recovery_count < max_xr_retries:
                _write_status("xr_retry", f"next_attempt={xr_recovery_count + 1} message={msg}")
                del app
                _reexec_self(
                    reason=f"XR runtime not ready yet. {msg}",
                    counter_env="TELEOP_CAMERA_XR_RECOVERY_COUNT",
                    count=xr_recovery_count + 1,
                    delay_s=xr_recovery_delay_s,
                )
            if any(err in msg for err in recoverable_graph_errors) and graph_recovery_count < max_graph_retries:
                _write_status("graph_retry", f"next_attempt={graph_recovery_count + 1} message={msg}")
                del app
                _reexec_self(
                    reason=f"Recoverable camera graph failure: {msg}",
                    counter_env="TELEOP_CAMERA_GRAPH_RECOVERY_COUNT",
                    count=graph_recovery_count + 1,
                    delay_s=recovery_delay_s,
                )
            if any(err in msg for err in xr_retry_errors):
                _write_status("failed", f"xr_retries_exhausted={xr_recovery_count} message={msg}")
                logger.error(
                    "XR runtime recovery exhausted after "
                    f"{xr_recovery_count} retry attempt(s): {msg}"
                )
            else:
                _write_status("failed", f"message={msg}")
            logger.error(f"Error: {e}")
            raise

    _write_status("stopped", "clean_exit")
    logger.info("Shutdown complete")
    os._exit(0)


if __name__ == "__main__":
    main()
