#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class Gelu(meta.ONNXOperatorAttributes):
    approximate: str

    def _default_values(self) -> None:
        self.approximate = "none"

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "approximate":
                self.approximate = attr.s.decode("utf-8")
            else:
                logger.w(f"ONNX `Gelu` attribute '{attr.name}' is not supported!")
