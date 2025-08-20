#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

from onnx2tflite.lib.tflite import OneHotOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class OneHot(meta.BuiltinOptions):
    axis: int

    def __init__(self, axis: int) -> None:
        super().__init__(BuiltinOptions.OneHotOptions, BuiltinOperator.ONE_HOT)
        self.axis = axis

    def gen_tflite(self, builder: fb.Builder) -> int:
        OneHotOptions.Start(builder)

        OneHotOptions.AddAxis(builder, self.axis)

        return OneHotOptions.End(builder)
