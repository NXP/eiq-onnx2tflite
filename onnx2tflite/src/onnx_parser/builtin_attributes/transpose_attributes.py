#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    Transpose

Representation of an ONNX 'Transpose' operator.
Initialized from a protobuf descriptor object.
"""

from typing import Iterable, Optional

import onnx

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.meta.meta as meta


class Transpose(meta.ONNXOperatorAttributes):
    perm: Optional[meta.ONNXIntListAttribute]

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.perm = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "perm":
                self.perm = meta.ONNXIntListAttribute(attr)
            else:
                logger.w(f"ONNX Transpose attribute '{attr.name}' is not supported!")
