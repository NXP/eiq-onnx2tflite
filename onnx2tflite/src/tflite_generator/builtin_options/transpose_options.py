#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""Transpose

Representation of the TFLite operator 'Transpose'.
"""

import flatbuffers as fb

import onnx2tflite.lib.tflite.BuiltinOperator as libBuiltinOperator
import onnx2tflite.lib.tflite.BuiltinOptions as libBuiltinOptions
import onnx2tflite.lib.tflite.TransposeOptions as libTransposeOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Transpose(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.TransposeOptions,
                         libBuiltinOperator.BuiltinOperator.TRANSPOSE)

    def gen_tflite(self, builder: fb.Builder):
        libTransposeOptions.Start(builder)
        return libTransposeOptions.End(builder)
