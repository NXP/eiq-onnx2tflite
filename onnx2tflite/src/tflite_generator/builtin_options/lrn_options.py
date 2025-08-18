#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    LRN

Representation of the TFLite operator 'LocalResponseNormalization'.
"""

import flatbuffers as fb

import onnx2tflite.lib.tflite.BuiltinOperator as libBuiltinOperator
import onnx2tflite.lib.tflite.BuiltinOptions as libBuiltinOptions
import onnx2tflite.lib.tflite.LocalResponseNormalizationOptions as libLocalResponseNormalizationOptions
import onnx2tflite.src.tflite_generator.meta.meta as meta


class LRN(meta.BuiltinOptions):
    radius: int
    bias: float
    alpha: float
    beta: float

    def __init__(self, radius: int, bias: float, alpha: float, beta: float) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.LocalResponseNormalizationOptions,
                         libBuiltinOperator.BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION)
        self.radius = radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def gen_tflite(self, builder: fb.Builder):
        libLocalResponseNormalizationOptions.Start(builder)

        libLocalResponseNormalizationOptions.AddRadius(builder, self.radius)
        libLocalResponseNormalizationOptions.AddBias(builder, self.bias)
        libLocalResponseNormalizationOptions.AddAlpha(builder, self.alpha)
        libLocalResponseNormalizationOptions.AddBeta(builder, self.beta)

        return libLocalResponseNormalizationOptions.End(builder)
