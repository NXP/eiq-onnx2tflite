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


class HardSigmoid(meta.ONNXOperatorAttributes):
    alpha: float
    beta: float

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self) -> None:
        self.alpha = np.float32(0.2)
        self.beta = np.float32(0.5)

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "alpha":
                self.alpha = attr.f
            elif attr.name == "beta":
                self.beta = attr.f
            else:
                logger.w(f"ONNX HardSigmoid attribute '{attr.name}' is not supported!")
