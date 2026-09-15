#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

import logging
from collections.abc import Iterable

import onnx

from onnx2tflite.src.onnx_parser.meta import meta

logger = logging.getLogger(__name__)


class Pad(meta.ONNXOperatorAttributes):
    mode: str

    # Opset Version 2
    pads: meta.ONNXIntListAttribute | None
    value: float

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self) -> None:
        self.mode = "constant"
        self.pads = None
        self.value = 0.0

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "mode":
                self.mode = attr.s.decode("UTF-8")
            elif attr.name == "pads":
                self.pads = meta.ONNXIntListAttribute(attr)
            elif attr.name == "value":
                self.value = attr.f
            else:
                logger.warning(f"ONNX Pad attribute '{attr.name}' is not supported!")
