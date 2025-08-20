#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""MaxPool2D

Representation of the TFLite operator 'MaxPool2D'.
"""

import flatbuffers as fb

import onnx2tflite.lib.tflite.ActivationFunctionType as libActivationFunctionType
import onnx2tflite.lib.tflite.BuiltinOperator as libBuiltinOperator
import onnx2tflite.lib.tflite.BuiltinOptions as libBuiltinOptions
import onnx2tflite.lib.tflite.Padding as libPadding
import onnx2tflite.lib.tflite.Pool2DOptions as libPool2DOptions
from onnx2tflite.src.tflite_generator.meta import meta


class MaxPool2D(meta.BuiltinOptions):
    padding: libPadding.Padding
    stride_w: int
    stride_h: int
    filter_w: int
    filter_h: int
    fused_activation_function: libActivationFunctionType.ActivationFunctionType

    def __init__(self, padding: libPadding.Padding = libPadding.Padding.SAME,
                 stride_w: int = 1, stride_h: int = 1,
                 filter_w: int = 1, filter_h: int = 1,
                 fused_activation_function: libActivationFunctionType.ActivationFunctionType = libActivationFunctionType.ActivationFunctionType.NONE) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.Pool2DOptions,
                         libBuiltinOperator.BuiltinOperator.MAX_POOL_2D)
        self.padding = padding
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.filter_w = filter_w
        self.filter_h = filter_h
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder) -> int:
        libPool2DOptions.Start(builder)

        libPool2DOptions.AddPadding(builder, self.padding)
        libPool2DOptions.AddStrideW(builder, self.stride_w)
        libPool2DOptions.AddStrideH(builder, self.stride_h)
        libPool2DOptions.AddFilterHeight(builder, self.filter_h)
        libPool2DOptions.AddFilterWidth(builder, self.filter_w)
        libPool2DOptions.AddFusedActivationFunction(builder, self.fused_activation_function)

        return libPool2DOptions.End(builder)
