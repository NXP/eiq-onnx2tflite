#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

import onnx2tflite.src.tflite_generator.meta.meta as meta
from onnx2tflite.lib.tflite import ReverseSequenceOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions


class ReverseSequence(meta.BuiltinOptions):
    seq_dim: int
    batch_dim: int

    def __init__(self, seq_dim: int, batch_dim: int) -> None:
        super().__init__(BuiltinOptions.ReverseSequenceOptions, BuiltinOperator.REVERSE_SEQUENCE)
        self.seq_dim = seq_dim
        self.batch_dim = batch_dim

    def gen_tflite(self, builder: fb.Builder):
        ReverseSequenceOptions.Start(builder)

        ReverseSequenceOptions.AddSeqDim(builder, self.seq_dim)
        ReverseSequenceOptions.AddBatchDim(builder, self.batch_dim)

        return ReverseSequenceOptions.End(builder)
