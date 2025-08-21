#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class ArgMin(meta.ONNXOperatorAttributes):
    axis: int
    keepdims: int
    select_last_index: int

    def _default_values(self):
        self.axis = 0
        self.keepdims = 1
        self.select_last_index = 0

    def _init_attributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "axis":
                    self.axis = attr.i
                case "keepdims":
                    self.keepdims = attr.i
                case "select_last_index":
                    self.select_last_index = attr.i
                case _:
                    logger.w(f"ONNX `ArgMin` attribute '{attr.name}' is not supported!")
