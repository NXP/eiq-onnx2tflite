#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class SpaceToDepth(meta.ONNXOperatorAttributes):
    block_size: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "blocksize":
                self.block_size = attr.i
            else:
                logger.w(f"ONNX `SpaceToDepth` attribute `{attr.name}` is not supported!")

        if not hasattr(self, 'block_size'):
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     'ONNX `SpaceToDepth` is missing the required `blocksize` attribute.')
