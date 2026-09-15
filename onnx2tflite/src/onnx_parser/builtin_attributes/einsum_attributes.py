#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src.onnx_parser.meta import meta
import logging

logger = logging.getLogger(__name__)


class Einsum(meta.ONNXOperatorAttributes):
    equation: str | None

    def _default_values(self) -> None:
        self.equation = None

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "equation":
                self.equation = attr.s.decode("utf-8")
            else:
                logger.warning(f"ONNX `Einsum` attribute '{attr.name}' is not supported!")
