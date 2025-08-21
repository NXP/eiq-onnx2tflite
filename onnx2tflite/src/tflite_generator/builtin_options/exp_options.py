#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

import onnx2tflite.lib.tflite.ExpOptions as libExpOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Exp(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.ExpOptions, BuiltinOperator.EXP)

    def gen_tflite(self, builder: fb.Builder):
        libExpOptions.Start(builder)

        return libExpOptions.End(builder)
