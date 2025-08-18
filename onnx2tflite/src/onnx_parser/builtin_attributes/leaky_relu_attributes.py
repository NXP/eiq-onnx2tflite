#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from typing import Iterable

import numpy as np
import onnx

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.meta.meta as meta


class LeakyRelu(meta.ONNXOperatorAttributes):
    alpha: float

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.alpha = np.float32(0.01)

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "alpha":
                self.alpha = attr.f
            else:
                logger.w(f"ONNX LeakyRelu attribute '{attr.name}' is not supported!")
