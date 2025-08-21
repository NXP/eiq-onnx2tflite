#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""Add

Representation of the TFLite operator 'Add'.
"""

import flatbuffers as fb

from onnx2tflite.lib.tflite import ActivationFunctionType as libActivationFunctionType
from onnx2tflite.lib.tflite import AddOptions as libAddOptions
from onnx2tflite.lib.tflite import BuiltinOperator as libBuiltinOperator
from onnx2tflite.lib.tflite import BuiltinOptions as libBuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Add(meta.BuiltinOptions):
    fused_activation_function: libActivationFunctionType.ActivationFunctionType

    # TODO potScaleInt16

    def __init__(self,
                 fused_activation_function: libActivationFunctionType.ActivationFunctionType = libActivationFunctionType.ActivationFunctionType.NONE
                 ) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.AddOptions,
                         libBuiltinOperator.BuiltinOperator.ADD)
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libAddOptions.Start(builder)

        libAddOptions.AddFusedActivationFunction(builder, self.fused_activation_function)

        return libAddOptions.End(builder)
