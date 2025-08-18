#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


import numpy as np

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class PReluConverter(NodeConverter):
    node = 'PRelu'

    onnx_supported_types = FLOATS + [TensorType.INT32, TensorType.INT64, TensorType.UINT32, TensorType.UINT64]
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/activations.cc#L1462-L1549
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT8, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def convert(self, _, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX `PRelu` to TFLite `PRelu`. """
        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"ONNX 'PRelu' has '{len(t_op.tmp_inputs)}' inputs! Expected '2'.")

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(t_op.tmp_inputs[0].type)

        extra_ops = self.context.tflite_builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0])

        a, b = t_op.tmp_inputs
        if a.shape.is_well_defined() and b.shape.is_well_defined():
            try:
                np.broadcast_shapes(a.shape.vector, b.shape.vector)

            except ValueError:
                # Shapes are not broadcastable. This shouldn't happen. Possible cause is an invalid ONNX model.
                logger.e(logger.Code.INTERNAL_ERROR,
                         f"Failed to broadcast shapes '{a.shape.vector}' and '{b.shape.vector}' during PRelu conversion.")

        t_op.builtin_options = None  # TFLite PRelu has no builtin options.
        t_op.opcode_index = self.context.tflite_builder.op_code_index_for_op_type(BuiltinOperator.PRELU)

        return extra_ops + [t_op]
