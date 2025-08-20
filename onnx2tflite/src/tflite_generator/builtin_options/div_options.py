#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""Div

Representation of the TFLite operator 'Div'.
"""

import flatbuffers as fb

from onnx2tflite.lib.tflite import DivOptions as libDivOptions
from onnx2tflite.lib.tflite.ActivationFunctionType import ActivationFunctionType
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Div(meta.BuiltinOptions):
    fused_activation_function: ActivationFunctionType

    def __init__(self, fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE) -> None:
        super().__init__(BuiltinOptions.DivOptions, BuiltinOperator.DIV)
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder) -> int:
        libDivOptions.Start(builder)

        libDivOptions.AddFusedActivationFunction(builder, self.fused_activation_function)

        return libDivOptions.End(builder)
