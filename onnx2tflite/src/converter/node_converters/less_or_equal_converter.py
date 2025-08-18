#
# Copyright 2024-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import less_equal_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


class LessOrEqualConverter(NodeConverter):
    node = 'LessOrEqual'

    onnx_supported_types = FLOATS + INTS + UINTS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/comparisons.cc#L408-L434
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64] # UINT8/INT8 not supported by TFLite (SIGABRT)
    verified_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX `LessOrEqual` to TFLite `LessEqual`. """

        ops = OpsList(middle_op=t_op)

        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `LessOrEqual` has invalid number of inputs.')

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_inputs[1]

        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     'ONNX `LessOrEqual` has input tensors with different data types.')

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        # ONNX LessOrEqual supports shape broadcasting.
        ops.add_pre(self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0]))

        t_op.builtin_options = less_equal_options.LessEqual()

        return ops.flatten()
