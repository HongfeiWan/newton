from __future__ import annotations

import cupy as cp
import cv2
from holoscan.core import IOSpec, Operator, OperatorSpec
import numpy as np


class RoiCropZoomOp(Operator):
    """Center-crop a camera tensor and resize it back to the original shape."""

    def __init__(
        self,
        fragment,
        *args,
        zoom: float,
        center_x: float = 0.5,
        center_y: float = 0.5,
        tensor_name: str = "",
        **kwargs,
    ):
        self._zoom = float(zoom)
        self._center_x = min(1.0, max(0.0, float(center_x)))
        self._center_y = min(1.0, max(0.0, float(center_y)))
        self._tensor_name = tensor_name
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("frame_in", size=1, policy=IOSpec.QueuePolicy.POP)
        spec.output("frame_out")

    def compute(self, op_input, op_output, context):
        frame_dict = op_input.receive("frame_in")
        if not frame_dict:
            return

        if self._tensor_name and self._tensor_name in frame_dict:
            tensor_key = self._tensor_name
        else:
            tensor_key = next(iter(frame_dict.keys()))

        tensor_value = frame_dict[tensor_key]
        if self._zoom <= 1.0:
            op_output.emit(frame_dict, "frame_out")
            return

        frame = cp.asnumpy(tensor_value)
        if frame.ndim < 2:
            op_output.emit(frame_dict, "frame_out")
            return

        height, width = int(frame.shape[0]), int(frame.shape[1])
        crop_width = max(1, min(width, int(round(width / self._zoom))))
        crop_height = max(1, min(height, int(round(height / self._zoom))))
        crop_x = int(round(self._center_x * width - crop_width / 2.0))
        crop_y = int(round(self._center_y * height - crop_height / 2.0))
        crop_x = min(max(0, crop_x), max(0, width - crop_width))
        crop_y = min(max(0, crop_y), max(0, height - crop_height))

        cropped = frame[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        output = cp.asarray(np.ascontiguousarray(resized))

        out_dict = dict(frame_dict)
        out_dict[tensor_key] = output
        op_output.emit(out_dict, "frame_out")
