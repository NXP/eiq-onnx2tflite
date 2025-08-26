#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import flatbuffers as fb

import onnx2tflite.lib.tflite.SplitVOptions as libSplitVOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class SplitV(meta.BuiltinOptions):
    num_splits: int

    def __init__(self, num_splits: int) -> None:
        super().__init__(BuiltinOptions.SplitVOptions, BuiltinOperator.SPLIT_V)
        self.num_splits = num_splits

    def gen_tflite(self, builder: fb.Builder) -> int:
        libSplitVOptions.Start(builder)

        libSplitVOptions.AddNumSplits(builder, self.num_splits)

        return libSplitVOptions.End(builder)
