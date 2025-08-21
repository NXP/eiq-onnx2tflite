#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class ReverseSequence(meta.ONNXOperatorAttributes):
    batch_axis: int
    time_axis: int

    def _default_values(self):
        self.batch_axis = 1
        self.time_axis = 0

    def _init_attributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "batch_axis":
                    self.batch_axis = attr.i
                case "time_axis":
                    self.time_axis = attr.i
                case _:
                    logger.w(f"ONNX `ReverseSequence` attribute `{attr.name}` is not supported!")
