#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

import onnx2tflite.lib.tflite.TransposeConvOptions as libTransposeConvOptions
from onnx2tflite.lib.tflite.ActivationFunctionType import ActivationFunctionType
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.lib.tflite.Padding import Padding
from onnx2tflite.src.tflite_generator.meta import meta


class TransposeConv(meta.BuiltinOptions):
    padding: Padding
    stride_w: int
    stride_h: int
    fused_activation_function: ActivationFunctionType

    def __init__(self, padding: Padding = Padding.SAME,
                 stride_w: int = 1, stride_h: int = 1,
                 fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE) -> None:
        super().__init__(BuiltinOptions.TransposeConvOptions, BuiltinOperator.TRANSPOSE_CONV)
        self.padding = padding
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libTransposeConvOptions.Start(builder)

        libTransposeConvOptions.AddPadding(builder, self.padding)
        libTransposeConvOptions.AddStrideW(builder, self.stride_w)
        libTransposeConvOptions.AddStrideH(builder, self.stride_h)
        libTransposeConvOptions.AddFusedActivationFunction(builder, self.fused_activation_function)

        return libTransposeConvOptions.End(builder)
