#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class CumSum(meta.ONNXOperatorAttributes):
    exclusive: int
    reverse: int

    def _default_values(self):
        self.exclusive = 0
        self.reverse = 0

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == 'exclusive':
                self.exclusive = attr.i
            if attr.name == 'reverse':
                self.reverse = attr.i
            else:
                logger.w(f"ONNX `CumSum` attribute '{attr.name}' is not supported!")
