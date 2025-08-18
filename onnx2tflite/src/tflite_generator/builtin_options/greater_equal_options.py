#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

import onnx2tflite.src.tflite_generator.meta.meta as meta
from onnx2tflite.lib.tflite import GreaterEqualOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions


class GreaterEqual(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.GreaterEqualOptions, BuiltinOperator.GREATER_EQUAL)

    def gen_tflite(self, builder: fb.Builder):
        GreaterEqualOptions.Start(builder)

        return GreaterEqualOptions.End(builder)
