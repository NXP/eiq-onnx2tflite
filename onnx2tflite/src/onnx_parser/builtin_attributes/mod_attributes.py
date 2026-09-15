#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import logging
from collections.abc import Iterable

import onnx

from onnx2tflite.src.onnx_parser.meta import meta

logger = logging.getLogger(__name__)


class Mod(meta.ONNXOperatorAttributes):
    fmod: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self) -> None:
        self.fmod = 0

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "fmod":
                self.fmod = attr.i
            else:
                logger.warning(f"ONNX `Mod` attribute `{attr.name}` is not supported!")
