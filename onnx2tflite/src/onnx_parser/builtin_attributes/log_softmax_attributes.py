#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""LogSoftmax

Representation of an ONNX 'LogSoftmax' operator.
Initialized from a protobuf descriptor object.
"""

import logging
from collections.abc import Iterable

import onnx

from onnx2tflite.src.onnx_parser.meta import meta

logger = logging.getLogger(__name__)


class LogSoftmax(meta.ONNXOperatorAttributes):
    axis: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self) -> None:
        self.axis = -1

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "axis":
                self.axis = attr.i
            else:
                logger.warning(f"ONNX Softmax attribute '{attr.name}' is not supported!")
