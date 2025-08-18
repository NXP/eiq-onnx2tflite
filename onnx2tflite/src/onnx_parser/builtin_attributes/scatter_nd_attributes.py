#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class ScatterND(meta.ONNXOperatorAttributes):
    reduction: str

    def _default_values(self):
        self.reduction = 'none'

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == 'reduction':
                self.reduction = attr.s.decode('utf-8')
            else:
                logger.w(f"ONNX `ScatterND` attribute '{attr.name}' is not supported.")
