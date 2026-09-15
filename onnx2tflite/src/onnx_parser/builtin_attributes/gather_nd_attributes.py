#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src.onnx_parser.meta import meta
import logging

logger = logging.getLogger(__name__)


class GatherND(meta.ONNXOperatorAttributes):
    batch_dims: int

    def _default_values(self) -> None:
        self.batch_dims = 0

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "batch_dims":
                self.batch_dims = attr.i

            else:
                logger.warning(f"ONNX GatherND attribute '{attr.name}' is not supported!")
