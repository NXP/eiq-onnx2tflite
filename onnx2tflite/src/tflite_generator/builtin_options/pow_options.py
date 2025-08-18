#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers

from onnx2tflite.lib.tflite import PowOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Pow(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.PowOptions, BuiltinOperator.POW)

    def gen_tflite(self, builder: flatbuffers.Builder):
        PowOptions.Start(builder)

        return PowOptions.End(builder)
