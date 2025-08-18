#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""
    q_linear_global_average_pool

    Representation of an ONNX Runtime 'QLinearGlobalAveragePool' operator.
"""

from typing import Iterable

import onnx

import onnx2tflite.src.onnx_parser.meta.meta as meta
from onnx2tflite.src import logger


class QLinearGlobalAveragePool(meta.ONNXOperatorAttributes):
    channels_last: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.channels_last = 0

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "channels_last":
                self.channels_last = attr.i
            else:
                logger.w(f"ONNX QLinearGlobalAveragePool attribute '{attr.name}' is not supported!")

        if self.channels_last != 0:
            # The converter doesn't really consider this option right now.
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX QLinearGlobalAveragePool with channels_last = "
                                                  f"'{self.channels_last}' is not yet implemented.")
