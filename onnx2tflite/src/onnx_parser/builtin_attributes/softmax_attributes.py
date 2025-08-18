#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    Softmax

Representation of an ONNX 'Softmax' operator.
Initialized from a protobuf descriptor object.
"""

from typing import Iterable

import onnx

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.meta.meta as meta


class Softmax(meta.ONNXOperatorAttributes):
    axis: int | None  # '-1' for v1 & v11 and '1' for v13 (handled in convert_softmax.py)

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.axis = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "axis":
                self.axis = attr.i
            else:
                logger.w(f"ONNX Softmax attribute '{attr.name}' is not supported!")
