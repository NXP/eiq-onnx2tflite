#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

import onnx2tflite.lib.tflite.LessOptions as libLessOptions
import onnx2tflite.src.tflite_generator.meta.meta as meta
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions


class Less(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.LessOptions, BuiltinOperator.LESS)

    def gen_tflite(self, builder: fb.Builder):
        libLessOptions.Start(builder)

        return libLessOptions.End(builder)
