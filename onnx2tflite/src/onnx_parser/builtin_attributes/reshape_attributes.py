#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    Reshape

Representation of an ONNX 'Reshape' operator.
Initialized from a protobuf descriptor object.
"""

from typing import Iterable

import onnx

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.meta.meta as meta


class Reshape(meta.ONNXOperatorAttributes):
    allow_zero: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.allow_zero = 0

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "allowzero":  # Not tested!
                self.allow_zero = attr.i
            else:
                logger.w(f"ONNX Reshape attribute '{attr.name}' is not supported!")
