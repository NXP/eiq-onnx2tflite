#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class GatherND(meta.ONNXOperatorAttributes):
    batch_dims: int

    def _default_values(self):
        self.batch_dims = 0

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "batch_dims":
                self.batch_dims = attr.i

            else:
                logger.w(f"ONNX GatherND attribute '{attr.name}' is not supported!")
