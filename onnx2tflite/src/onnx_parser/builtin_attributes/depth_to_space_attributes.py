#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class DepthToSpace(meta.ONNXOperatorAttributes):
    block_size: int
    mode: str

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self) -> None:
        self.mode = "DCR"

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "blocksize":
                self.block_size = attr.i
            elif attr.name == "mode":
                self.mode = attr.s.decode("utf-8")
            else:
                logger.w(f"ONNX `DepthToSpace` attribute `{attr.name}` is not supported!")

        if not hasattr(self, "block_size"):
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     "ONNX `DepthToSpace` is missing the required `blocksize` attribute.")
