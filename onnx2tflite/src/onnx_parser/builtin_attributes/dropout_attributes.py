#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from typing import Optional, Iterable

import onnx

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.meta.meta as meta


class Dropout(meta.ONNXOperatorAttributes):
    seed: Optional[int]
    ratio: float
    is_test: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.seed = None
        self.ratio = 0.5
        self.is_test = 0  # Appears to be unsupported by ONNX Runtime.

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "seed":
                self.seed = attr.i
            elif attr.name == 'ratio':
                self.ratio = attr.f
            elif attr.name == 'is_test':
                self.is_test = attr.i
            else:
                logger.w(f"ONNX Dropout attribute '{attr.name}' is not supported!")
