#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


import flatbuffers as fb

import onnx2tflite.src.tflite_generator.meta.meta as meta
from onnx2tflite.lib.tflite import GeluOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions


class Gelu(meta.BuiltinOptions):
    approximate: bool

    def __init__(self, approximate: bool) -> None:
        super().__init__(BuiltinOptions.GeluOptions, BuiltinOperator.GELU)
        self.approximate = approximate

    def gen_tflite(self, builder: fb.Builder):
        GeluOptions.Start(builder)

        GeluOptions.AddApproximate(builder, self.approximate)

        return GeluOptions.End(builder)
