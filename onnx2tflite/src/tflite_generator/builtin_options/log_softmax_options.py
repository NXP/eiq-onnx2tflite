#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""LogSoftmax

Representation of the TFLite operator 'LogSoftmax'.
"""

import flatbuffers as fb

import onnx2tflite.lib.tflite.BuiltinOperator as libBuiltinOperator
import onnx2tflite.lib.tflite.BuiltinOptions as libBuiltinOptions
import onnx2tflite.lib.tflite.LogSoftmaxOptions as libLogSoftmaxOptions
from onnx2tflite.src.tflite_generator.meta import meta


class LogSoftmax(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.LogSoftmaxOptions,
                         libBuiltinOperator.BuiltinOperator.LOG_SOFTMAX)

    def gen_tflite(self, builder: fb.Builder):
        libLogSoftmaxOptions.Start(builder)
        return libLogSoftmaxOptions.End(builder)
