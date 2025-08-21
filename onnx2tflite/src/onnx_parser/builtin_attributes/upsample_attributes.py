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
from onnx2tflite.src.onnx_parser.meta.meta import ONNXFloatListAttribute


# noinspection SpellCheckingInspection
class Upsample(meta.ONNXOperatorAttributes):
    mode: str

    # Removed in v9
    scales: ONNXFloatListAttribute | None

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.mode = "nearest"
        self.scales = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "mode":
                self.mode = attr.s.decode("utf-8")
            elif attr.name == "scales":
                self.scales = ONNXFloatListAttribute(attr)
            else:
                logger.w(f"ONNX `Upsample` attribute `{attr.name}` is not supported!")
