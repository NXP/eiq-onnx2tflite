#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class Einsum(meta.ONNXOperatorAttributes):
    equation: str | None

    def _default_values(self):
        self.equation = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "equation":
                self.equation = attr.s.decode("utf-8")
            else:
                logger.w(f"ONNX `Einsum` attribute '{attr.name}' is not supported!")
