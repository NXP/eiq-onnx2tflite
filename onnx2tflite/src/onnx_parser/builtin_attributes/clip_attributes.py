#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class Clip(meta.ONNXOperatorAttributes):
    # Only v6 uses attributes. Newer versions use input tensors.
    max: float
    min: float

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self) -> None:
        # All parsed Clip versions will have these default attributes. Use node.version to distinguish between versions.
        self.max = 3.402823e+38
        self.min = -3.402823e+38

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "max":
                self.max = attr.f
            elif attr.name == "min":
                self.min = attr.f
            else:
                logger.w(f"ONNX Clip attribute '{attr.name}' is not supported!")
