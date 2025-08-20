#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import flatbuffers as fb

from onnx2tflite.lib.tflite import ReducerOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Mean(meta.BuiltinOptions):
    keep_dims: bool

    def __init__(self, keep_dims: bool) -> None:
        super().__init__(BuiltinOptions.ReducerOptions, BuiltinOperator.MEAN)
        self.keep_dims = keep_dims

    def gen_tflite(self, builder: fb.Builder) -> int:
        ReducerOptions.Start(builder)

        ReducerOptions.AddKeepDims(builder, self.keep_dims)

        return ReducerOptions.End(builder)
