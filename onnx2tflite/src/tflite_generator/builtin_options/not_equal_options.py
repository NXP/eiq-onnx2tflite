#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

import onnx2tflite.src.tflite_generator.meta.meta as meta
from onnx2tflite.lib.tflite import NotEqualOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions


class NotEqual(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.NotEqualOptions, BuiltinOperator.NOT_EQUAL)

    def gen_tflite(self, builder: fb.Builder):
        NotEqualOptions.Start(builder)

        return NotEqualOptions.End(builder)
