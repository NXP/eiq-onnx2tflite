#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""Transpose

Representation of an ONNX 'Transpose' operator.
Initialized from a protobuf descriptor object.
"""

from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class Transpose(meta.ONNXOperatorAttributes):
    perm: meta.ONNXIntListAttribute | None

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
