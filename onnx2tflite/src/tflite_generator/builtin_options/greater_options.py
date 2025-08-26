#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

from onnx2tflite.lib.tflite import GreaterOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Greater(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.GreaterOptions, BuiltinOperator.GREATER)

    def gen_tflite(self, builder: fb.Builder) -> int:
        GreaterOptions.Start(builder)

        return GreaterOptions.End(builder)
