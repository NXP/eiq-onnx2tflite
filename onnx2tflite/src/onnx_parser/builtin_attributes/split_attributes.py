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


class Split(meta.ONNXOperatorAttributes):
    axis: int
    num_outputs: int | None  # V 18
    split: meta.ONNXIntListAttribute | None  # V 2, 11

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.axis = 0
        self.num_outputs = None
        self.split = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "axis":
                self.axis = attr.i
            elif attr.name == "num_outputs":
                self.num_outputs = attr.i
            elif attr.name == "split":
                self.split = meta.ONNXIntListAttribute(attr)
            else:
                logger.w(f"ONNX Split attribute '{attr.name}' is not supported!")
