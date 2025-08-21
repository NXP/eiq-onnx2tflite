#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

from onnx2tflite.lib.tflite import CumsumOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class CumSum(meta.BuiltinOptions):
    exclusive: bool
    reverse: bool

    def __init__(self, exclusive: bool, reverse: bool) -> None:
        super().__init__(BuiltinOptions.CumsumOptions, BuiltinOperator.CUMSUM)
        self.exclusive = exclusive
        self.reverse = reverse

    def gen_tflite(self, builder: fb.Builder):
        CumsumOptions.Start(builder)

        CumsumOptions.AddExclusive(builder, self.exclusive)
        CumsumOptions.AddReverse(builder, self.reverse)

        return CumsumOptions.End(builder)
