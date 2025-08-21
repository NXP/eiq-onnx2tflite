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


class Unsqueeze(meta.ONNXOperatorAttributes):
    # Unused since version 13
    axes: meta.ONNXIntListAttribute | None

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.axes = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "axes":
                self.axes = meta.ONNXIntListAttribute(attr)
            else:
                logger.w(f"ONNX Unsqueeze attribute '{attr.name}' is not supported!")
