#
# Copyright 2023, 2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""quantize_linear_attributes

Representation of an ONNX 'QuantizeLinear' operator.
"""

from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class QuantizeLinear(meta.ONNXOperatorAttributes):
    axis: int
    block_size: int
    output_dtype: int
    precision: int
    saturate: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self) -> None:
        self.axis = 1
        self.block_size = 0
        self.output_dtype = 0
        self.precision = 0
        self.saturate = 1

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "axis":
                self.axis = attr.i
            elif attr.name == "block_size":
                self.block_size = attr.i
            elif attr.name == "output_dtype" or attr.name == "precision":
                self.output_dtype = attr.i
            elif attr.name == "saturate":
                self.saturate = attr.i
            else:
                logger.w(f"ONNX QuantizeLinear attribute '{attr.name}' is not supported!")
