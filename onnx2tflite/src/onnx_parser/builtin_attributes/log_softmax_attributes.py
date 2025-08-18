#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    LogSoftmax

Representation of an ONNX 'LogSoftmax' operator.
Initialized from a protobuf descriptor object.
"""

from typing import Iterable

import onnx

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.meta.meta as meta


class LogSoftmax(meta.ONNXOperatorAttributes):
    axis: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.axis = -1

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "axis":
                self.axis = attr.i
            else:
                logger.w(f"ONNX Softmax attribute '{attr.name}' is not supported!")
