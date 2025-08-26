#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class Concat(meta.ONNXOperatorAttributes):
    axis: int  # Required since Opset 4. Previously optional with default value 1.

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self) -> None:
        self.axis = 1

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "axis":
                self.axis = attr.i
            else:
                logger.w(f"ONNX Concat attribute '{attr.name}' is not supported!")
