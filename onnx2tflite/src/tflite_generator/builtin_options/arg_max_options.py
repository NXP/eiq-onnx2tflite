#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

from onnx2tflite.lib.tflite import ArgMaxOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.tflite_generator.meta import meta


class ArgMax(meta.BuiltinOptions):
    output_type: TensorType

    def __init__(self, output_type: TensorType) -> None:
        super().__init__(BuiltinOptions.ArgMaxOptions, BuiltinOperator.ARG_MAX)
        self.output_type = output_type

    def gen_tflite(self, builder: fb.Builder) -> int:
        ArgMaxOptions.Start(builder)

        ArgMaxOptions.AddOutputType(builder, self.output_type)

        return ArgMaxOptions.End(builder)
