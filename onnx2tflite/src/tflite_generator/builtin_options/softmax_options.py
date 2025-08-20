#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""Softmax

Representation of the TFLite operator 'Softmax'.
"""

import flatbuffers as fb

import onnx2tflite.lib.tflite.BuiltinOperator as libBuiltinOperator
import onnx2tflite.lib.tflite.BuiltinOptions as libBuiltinOptions
import onnx2tflite.lib.tflite.SoftmaxOptions as libSoftmaxOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Softmax(meta.BuiltinOptions):
    beta: float

    def __init__(self, beta: float) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.SoftmaxOptions,
                         libBuiltinOperator.BuiltinOperator.SOFTMAX)
        self.beta = beta

    def gen_tflite(self, builder: fb.Builder) -> int:
        libSoftmaxOptions.Start(builder)

        libSoftmaxOptions.AddBeta(builder, self.beta)

        return libSoftmaxOptions.End(builder)
