#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""AddN

Representation of the TFLite operator 'AddN'.
"""

import flatbuffers as fb

from onnx2tflite.lib.tflite import AddNOptions as libAddNOptions
from onnx2tflite.lib.tflite import BuiltinOperator as libBuiltinOperator
from onnx2tflite.lib.tflite import BuiltinOptions as libBuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class AddN(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.AddNOptions,
                         libBuiltinOperator.BuiltinOperator.ADD_N)

    def gen_tflite(self, builder: fb.Builder):
        libAddNOptions.Start(builder)
        return libAddNOptions.End(builder)
