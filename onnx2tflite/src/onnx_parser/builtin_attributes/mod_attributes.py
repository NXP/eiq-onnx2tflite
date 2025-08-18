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


class Mod(meta.ONNXOperatorAttributes):
    fmod: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.fmod = 0

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "fmod":
                self.fmod = attr.i
            else:
                logger.w(f"ONNX `Mod` attribute `{attr.name}` is not supported!")
