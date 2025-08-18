#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""
    depthwise_conv_2d_options

Representation of the TFLite operator 'DepthwiseConv2D'.
"""

import flatbuffers as fb

import onnx2tflite.lib.tflite.DepthwiseConv2DOptions as libDepthwiseConv2DOptions
import onnx2tflite.src.tflite_generator.meta.meta as meta
from onnx2tflite.lib.tflite.ActivationFunctionType import ActivationFunctionType
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.lib.tflite.Padding import Padding


class DepthwiseConv2D(meta.BuiltinOptions):
    padding: Padding
    stride_w: int = 1
    stride_h: int = 1
    fused_activation_function: ActivationFunctionType
    dilation_w_factor: int = 1
    dilation_h_factor: int = 1
    depth_multiplier: int = 1  # Redundant according to schema.fbs (line 597)

    def __init__(self, padding: Padding = Padding.SAME,
                 stride_w: int = 1, stride_h: int = 1,
                 dilation_w_factor: int = 1, dilation_h_factor: int = 1,
                 fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE,
                 depth_multiplier: int = 1) -> None:
        super().__init__(BuiltinOptions.DepthwiseConv2DOptions,
                         BuiltinOperator.DEPTHWISE_CONV_2D)
        self.padding = padding
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.fused_activation_function = fused_activation_function
        self.dilation_w_factor = dilation_w_factor
        self.dilation_h_factor = dilation_h_factor
        self.depth_multiplier = depth_multiplier

    def gen_tflite(self, builder: fb.Builder):
        libDepthwiseConv2DOptions.Start(builder)

        libDepthwiseConv2DOptions.AddPadding(builder, self.padding)
        libDepthwiseConv2DOptions.AddStrideW(builder, self.stride_w)
        libDepthwiseConv2DOptions.AddStrideH(builder, self.stride_h)
        libDepthwiseConv2DOptions.AddFusedActivationFunction(builder, self.fused_activation_function)
        libDepthwiseConv2DOptions.AddDilationWFactor(builder, self.dilation_w_factor)
        libDepthwiseConv2DOptions.AddDilationHFactor(builder, self.dilation_h_factor)

        libDepthwiseConv2DOptions.AddDepthMultiplier(builder, self.depth_multiplier)

        return libDepthwiseConv2DOptions.End(builder)
