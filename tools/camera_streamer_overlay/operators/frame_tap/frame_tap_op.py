from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cupy as cp
import numpy as np
from holoscan.core import IOSpec, Operator, OperatorSpec


class FrameTapOp(Operator):
    """Persist camera tensors to a filesystem spool without owning the camera."""

    def __init__(
        self,
        fragment,
        *args,
        output_dir: str,
        camera_name: str,
        tensor_name: str = "",
        stride: int = 1,
        **kwargs,
    ):
        self._output_dir = Path(output_dir)
        self._camera_name = str(camera_name)
        self._tensor_name = str(tensor_name)
        self._stride = max(1, int(stride))
        self._frame_index = 0
        self._index_path = self._output_dir / "index.jsonl"
        self._frames_dir = self._output_dir / "frames" / self._camera_name
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("frame_in", size=1, policy=IOSpec.QueuePolicy.POP)
        spec.output("frame_out")

    def start(self):
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # The camera app runs inside a root-owned Docker container. Make the
        # bind-mounted spool removable by the host operator after tmux/docker
        # is killed abruptly.
        os.chmod(self._output_dir, 0o777)
        os.chmod(self._output_dir / "frames", 0o777)
        os.chmod(self._frames_dir, 0o777)

    def compute(self, op_input, op_output, context):
        frame_dict = op_input.receive("frame_in")
        if not frame_dict:
            return

        self._frame_index += 1
        if self._frame_index % self._stride != 0:
            op_output.emit(frame_dict, "frame_out")
            return

        tensor_key = (
            self._tensor_name
            if self._tensor_name and self._tensor_name in frame_dict
            else next(iter(frame_dict.keys()))
        )
        frame = cp.asnumpy(frame_dict[tensor_key])
        if frame.ndim != 3 or frame.shape[2] < 3:
            op_output.emit(frame_dict, "frame_out")
            return

        rgb = np.ascontiguousarray(frame[:, :, :3])
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        output_path = self._frames_dir / f"{self._frame_index:06d}.ppm"
        tmp_path = output_path.with_suffix(".ppm.tmp")
        with tmp_path.open("wb") as handle:
            handle.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii"))
            handle.write(rgb.tobytes())
        os.chmod(tmp_path, 0o666)
        os.replace(tmp_path, output_path)

        now = time.monotonic()
        record = {
            "schema_version": "teleop_stack.camera_frame_tap.v1",
            "monotonic_ts_s": now,
            "capture_ts_s": now,
            "source_ts_s": None,
            "camera_name": self._camera_name,
            "frame_index": int(self._frame_index),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "encoding": "ppm_rgb8",
            "relative_path": str(output_path.relative_to(self._output_dir)),
        }
        with self._index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")
        os.chmod(self._index_path, 0o666)

        op_output.emit(frame_dict, "frame_out")
