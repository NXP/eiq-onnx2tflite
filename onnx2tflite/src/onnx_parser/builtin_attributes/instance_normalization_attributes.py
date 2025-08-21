#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from collections.abc import Iterable

import numpy as np
import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class InstanceNormalization(meta.ONNXOperatorAttributes):
    epsilon: float

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.epsilon = np.float32(1.e-5)

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "epsilon":
                self.epsilon = attr.f
            else:
                logger.w(f"ONNX InstanceNormalization attribute '{attr.name}' is not supported!")
