#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

import onnx2tflite.lib.tflite.ActivationFunctionType as libActivationFunctionType
import onnx2tflite.lib.tflite.BuiltinOperator as libBuiltinOperator
import onnx2tflite.lib.tflite.BuiltinOptions as libBuiltinOptions
import onnx2tflite.lib.tflite.ConcatenationOptions as libConcatenationOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Concatenation(meta.BuiltinOptions):
    axis: int
    fused_activation_function: libActivationFunctionType.ActivationFunctionType

    def __init__(self, axis: int,
                 fused_activation_function: libActivationFunctionType.ActivationFunctionType = libActivationFunctionType.ActivationFunctionType.NONE) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.ConcatenationOptions,
                         libBuiltinOperator.BuiltinOperator.CONCATENATION)
        self.axis = axis
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libConcatenationOptions.Start(builder)

        libConcatenationOptions.AddAxis(builder, self.axis)
        libConcatenationOptions.AddFusedActivationFunction(builder, self.fused_activation_function)

        return libConcatenationOptions.End(builder)
