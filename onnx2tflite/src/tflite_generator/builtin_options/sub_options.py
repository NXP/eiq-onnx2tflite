#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""Sub

Representation of the TFLite operator 'Sub'.
"""

import flatbuffers as fb

from onnx2tflite.lib.tflite import ActivationFunctionType as libActivationFunctionType
from onnx2tflite.lib.tflite import BuiltinOperator as libBuiltinOperator
from onnx2tflite.lib.tflite import BuiltinOptions as libBuiltinOptions
from onnx2tflite.lib.tflite import SubOptions as libSubOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Sub(meta.BuiltinOptions):
    fused_activation_function: libActivationFunctionType.ActivationFunctionType

    # TODO potScaleInt16

    def __init__(self,
                 fused_activation_function: libActivationFunctionType.ActivationFunctionType = libActivationFunctionType.ActivationFunctionType.NONE
                 ) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.SubOptions,
                         libBuiltinOperator.BuiltinOperator.SUB)
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libSubOptions.Start(builder)

        libSubOptions.AddFusedActivationFunction(builder, self.fused_activation_function)

        return libSubOptions.End(builder)
