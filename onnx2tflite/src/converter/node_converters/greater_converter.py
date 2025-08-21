#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList, exactly_one_is_none
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.greater_options import Greater
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


class GreaterConverter(NodeConverter):
    node = "Greater"

    onnx_supported_types = FLOATS + INTS + UINTS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/comparisons.cc#L277-L303
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT32, TensorType.INT64]
    verified_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `Greater` to TFLite `Greater`."""
        ops = OpsList(middle_op=t_op)

        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX `Greater` has invalid number of inputs.")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_inputs[1]

        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX `Greater` has input tensors with different data types.")

        if exactly_one_is_none(x.quantization, y.quantization):
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX `Greater` has only 1 input quantized.")

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        # ONNX Greater supports shape broadcasting.
        ops.add_pre(self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0]))

        t_op.builtin_options = Greater()

        return ops.flatten()
