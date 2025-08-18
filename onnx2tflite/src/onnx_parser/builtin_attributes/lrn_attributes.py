#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    LRN

Representation of an ONNX 'LRN' operator.
Initialized from a protobuf descriptor object.
"""

from typing import Iterable

import numpy as np
import onnx

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.meta.meta as meta


class LRN(meta.ONNXOperatorAttributes):
    alpha: float
    beta: float
    bias: float
    size: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.alpha = np.float32(1e-4)  # ~= 9.99999974737e-05. Corresponds to onnxruntime/core/providers/cpu/nn/lrn.h:23
        self.beta = 0.75
        self.bias = 1.0

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "alpha":
                self.alpha = attr.f
            elif attr.name == "beta":
                self.beta = attr.f
            elif attr.name == "bias":
                self.bias = attr.f
            elif attr.name == "size":
                self.size = attr.i
            else:
                logger.w(f"ONNX LRN attribute '{attr.name}' is not supported!")

        # Attribute 'size' is required
        if not hasattr(self, "size"):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     "ONNX 'LRN' is missing the required 'size' attribute!")
