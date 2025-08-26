#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#


import flatbuffers as fb

import onnx2tflite.lib.tflite.BuiltinOperator as libBuiltinOperator
import onnx2tflite.lib.tflite.BuiltinOptions as libBuiltinOptions
import onnx2tflite.lib.tflite.LeakyReluOptions as libLeakyReluOptions
from onnx2tflite.src.tflite_generator.meta import meta


class LeakyRelu(meta.BuiltinOptions):
    alpha: float

    def __init__(self, alpha: float) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.LeakyReluOptions,
                         libBuiltinOperator.BuiltinOperator.LEAKY_RELU)
        self.alpha = alpha

    def gen_tflite(self, builder: fb.Builder) -> int:
        libLeakyReluOptions.Start(builder)

        libLeakyReluOptions.AddAlpha(builder, self.alpha)

        return libLeakyReluOptions.End(builder)
