#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class TopK(meta.ONNXOperatorAttributes):
    axis: int
    largest: int
    sorted: int

    def _default_values(self) -> None:
        self.axis = -1
        self.largest = 1
        self.sorted = 1

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "axis":
                self.axis = attr.i
            elif attr.name == "largest":
                self.largest = attr.i
            elif attr.name == "sorted":
                self.sorted = attr.i
            else:
                logger.w(f"ONNX `TopK` attribute '{attr.name}' is not supported.")
