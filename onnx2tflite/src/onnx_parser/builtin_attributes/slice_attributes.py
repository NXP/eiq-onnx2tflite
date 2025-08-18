#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""
    slice_attributes

    Representation of an ONNX 'Slice' operator.
"""

from typing import Iterable, Optional

import onnx

import onnx2tflite.src.onnx_parser.meta.meta as meta
from onnx2tflite.src import logger


class Slice(meta.ONNXOperatorAttributes):
    starts: Optional[meta.ONNXIntListAttribute]
    ends: Optional[meta.ONNXIntListAttribute]
    axes: Optional[meta.ONNXIntListAttribute]

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.starts = None
        self.ends = None
        self.axes = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "starts":
                self.starts = meta.ONNXIntListAttribute(attr)
            elif attr.name == "ends":
                self.ends = meta.ONNXIntListAttribute(attr)
            elif attr.name == "axes":
                self.axes = meta.ONNXIntListAttribute(attr)
            else:
                logger.w(f"ONNX Slice attribute '{attr.name}' is not supported!")
