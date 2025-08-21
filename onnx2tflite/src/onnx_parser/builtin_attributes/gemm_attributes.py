#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""Gemm

Representation of an ONNX 'Gemm' operator.
Initialized from a protobuf descriptor object.
"""

from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class Gemm(meta.ONNXOperatorAttributes):
    alpha: float
    beta: float
    transA: int
    transB: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.alpha = 1.0
        self.beta = 1.0
        self.transA = 0
        self.transB = 0

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "alpha":
                self.alpha = attr.f
            elif attr.name == "beta":
                self.beta = attr.f
            elif attr.name == "transA":
                self.transA = attr.i
            elif attr.name == "transB":
                self.transB = attr.i
            else:
                logger.w(f"ONNX Gemm attribute '{attr.name}' is not supported!")
