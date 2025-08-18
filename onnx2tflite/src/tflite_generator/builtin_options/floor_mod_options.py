#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

import onnx2tflite.src.tflite_generator.meta.meta as meta
from onnx2tflite.lib.tflite import FloorModOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions


class FloorMod(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.FloorModOptions, BuiltinOperator.FLOOR_MOD)

    def gen_tflite(self, builder: fb.Builder):
        FloorModOptions.Start(builder)

        return FloorModOptions.End(builder)
