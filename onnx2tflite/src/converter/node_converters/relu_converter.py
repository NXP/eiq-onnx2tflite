#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS


class ReluConverter(NodeConverter):
    node = 'Relu'

    onnx_supported_types = FLOATS + INTS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/activations.cc#L738-L763
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT16]
    # TFLite supports only quantized `int8`. Other overlapping types are not supported by ORT.
    verified_types = [TensorType.FLOAT32]

    def convert(self, _, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        t_op.builtin_options = None
        t_op.opcode_index = self.context.tflite_builder.op_code_index_for_op_type(BuiltinOperator.RELU)

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(t_op.tmp_inputs[0].type)

        return [t_op]
