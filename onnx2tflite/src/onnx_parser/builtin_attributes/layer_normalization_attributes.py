#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import Iterable

import numpy as np
import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class LayerNormalization(meta.ONNXOperatorAttributes):
    axis: int
    epsilon: float
    stash_type: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.axis = -1
        self.epsilon = np.float32(1e-5)
        self.stash_type = 1

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "axis":
                self.axis = attr.i
            elif attr.name == "epsilon":
                self.epsilon = attr.f
            elif attr.name == "stash_type":
                self.stash_type = attr.i
            else:
                logger.w(f"ONNX LayerNormalization attribute '{attr.name}' is not supported!")
