#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import Iterable, Optional

import onnx

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.meta.meta as meta


class QLinearConcat(meta.ONNXOperatorAttributes):
    axis: Optional[int]

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.axis = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "axis":
                self.axis = attr.i
            else:
                logger.w(f"ONNX QLinearConcat attribute '{attr.name}' is not supported!")

        if self.axis is None:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX QLinearConcat has no 'axis' attribute!")
