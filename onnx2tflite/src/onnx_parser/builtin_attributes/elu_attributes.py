#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import logging

from onnx2tflite.src.onnx_parser.meta import meta

logger = logging.getLogger(__name__)


class Elu(meta.ONNXOperatorAttributes):
    alpha: float

    def _default_values(self) -> None:
        self.alpha = 1.0

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "alpha":
                self.alpha = attr.f
            else:
                logger.warning(f"ONNX `Elu` attribute `{attr.name}` is not supported!")
