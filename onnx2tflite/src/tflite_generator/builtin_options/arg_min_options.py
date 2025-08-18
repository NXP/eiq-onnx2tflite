#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

import onnx2tflite.src.tflite_generator.meta.meta as meta
from onnx2tflite.lib.tflite import ArgMinOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.lib.tflite.TensorType import TensorType


class ArgMin(meta.BuiltinOptions):
    output_type: TensorType

    def __init__(self, output_type: TensorType) -> None:
        super().__init__(BuiltinOptions.ArgMinOptions, BuiltinOperator.ARG_MIN)
        self.output_type = output_type

    def gen_tflite(self, builder: fb.Builder):
        ArgMinOptions.Start(builder)

        ArgMinOptions.AddOutputType(builder, self.output_type)

        return ArgMinOptions.End(builder)
