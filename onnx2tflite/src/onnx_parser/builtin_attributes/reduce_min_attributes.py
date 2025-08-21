#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class ReduceMin(meta.ONNXOperatorAttributes):
    axes: meta.ONNXIntListAttribute | None  # Used in versions < 18.
    keepdims: int
    noop_with_empty_axes: int  # Used in versions >= 18.

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.axes = None
        self.keepdims = 1
        self.noop_with_empty_axes = 0

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "axes":
                self.axes = meta.ONNXIntListAttribute(attr)
            elif attr.name == "keepdims":
                self.keepdims = attr.i
            elif attr.name == "noop_with_empty_axes":
                self.noop_with_empty_axes = attr.i
            else:
                logger.w(f"ONNX `ReduceMin` attribute '{attr.name}' is not supported!")
