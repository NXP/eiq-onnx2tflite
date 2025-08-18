#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    MatMul

Representation of an ONNX 'MatMul' operator. 
Initialized from a protobuf descriptor object.
"""

from typing import Iterable

import onnx

import onnx2tflite.src.onnx_parser.meta.meta as meta


class MatMul(meta.ONNXOperatorAttributes):
    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)
