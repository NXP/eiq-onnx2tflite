#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import Iterable

import numpy as np
import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class QuickGelu(meta.ONNXOperatorAttributes):
    alpha: float

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        # The default `alpha` is not mentioned in the documentation and I also couldn't find it in ONNX Runtime kernels.
        # These articles:
        #   https://zeta.apac.ai/en/latest/zeta/nn/modules/quickgeluactivation/
        #   https://paperswithcode.com/method/gelu
        #  mention the value 1.702. I experimented with slightly larger and smaller values. Using exactly 1.702, the
        #  error was the smallest. It got up to 5.e-7, which is larger than our regular absolute tolerance of 1.e-8.
        self.alpha = np.float32(1.702)

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "alpha":
                self.alpha = attr.f
            else:
                logger.w(f"ONNX QuickGelu attribute '{attr.name}' is not supported!")
