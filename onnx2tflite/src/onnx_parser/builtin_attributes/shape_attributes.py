#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""
    shape_attributes

    Representation of an ONNX 'Shape' operator.
"""

from typing import Iterable, Optional

import onnx

import onnx2tflite.src.onnx_parser.meta.meta as meta
from onnx2tflite.src import logger


class Shape(meta.ONNXOperatorAttributes):
    start: Optional[int]
    end: Optional[int]

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.start = None
        self.end = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "start":
                self.start = attr.i
            elif attr.name == "end":
                self.end = attr.i
            else:
                logger.w(f"ONNX 'Shape' attribute '{attr.name}' is not supported!")
