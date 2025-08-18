#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""
    batch_mat_mul_options

Representation of the TFLite operator 'BatchMatMul'.
"""
import flatbuffers as fb

import onnx2tflite.src.tflite_generator.meta.meta as meta
from onnx2tflite.lib.tflite import BuiltinOperator, BuiltinOptions, BatchMatMulOptions


class BatchMatMul(meta.BuiltinOptions):
    adj_x: bool
    adj_y: bool
    asymmetric_quantize_inputs: bool

    def __init__(self, adj_x: bool, adj_y: bool, asymmetric_quantize_inputs: bool) -> None:
        super().__init__(BuiltinOptions.BuiltinOptions.BatchMatMulOptions,
                         BuiltinOperator.BuiltinOperator.BATCH_MATMUL)
        self.adj_x = adj_x
        self.adj_y = adj_y
        self.asymmetric_quantize_inputs = asymmetric_quantize_inputs

    def gen_tflite(self, builder: fb.Builder):
        BatchMatMulOptions.Start(builder)

        BatchMatMulOptions.AddAdjX(builder, self.adj_x)
        BatchMatMulOptions.AddAdjY(builder, self.adj_y)
        BatchMatMulOptions.AddAsymmetricQuantizeInputs(builder, self.asymmetric_quantize_inputs)

        return BatchMatMulOptions.End(builder)
