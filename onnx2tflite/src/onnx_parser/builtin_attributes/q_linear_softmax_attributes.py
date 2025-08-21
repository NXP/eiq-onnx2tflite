#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""QLinearSoftmax

Representation of an ONNX 'QLinearSoftmax' operator.
Initialized from a protobuf descriptor object.
"""

from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class QLinearSoftmax(meta.ONNXOperatorAttributes):
    axis: int | None
    opset: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.axis = -1
        self.opset = -1

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "axis":
                self.axis = attr.i
            elif attr.name == "opset":
                self.opset = attr.i
            else:
                logger.w(f"ONNX QLinearSoftmax attribute '{attr.name}' is not supported!")
