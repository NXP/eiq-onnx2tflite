#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.node_converters.shared import reduce_utils
from onnx2tflite.src.converter.node_converters.shared.reduce_utils import (
    convert_axes_from_attribute,
    convert_axes_from_input_tensor,
)
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes.reduce_sum_attributes import ReduceSum
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import sum_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS


class ReduceSumConverter(NodeConverter):
    node = "ReduceSum"

    onnx_supported_types = FLOATS + [TensorType.INT32, TensorType.INT64, TensorType.UINT32, TensorType.UINT64]
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/reduce.cc#L874-L905
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/reduce.cc#L935-L944
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8, TensorType.BOOL]
    verified_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]

    def _convert_v_11(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX ReduceSum version 1 / 11 to TFLite Sum.

        The `axes` is in the form of an attribute.
        """
        attrs = cast(ReduceSum, node.attributes)

        convert_axes_from_attribute(t_op, self.builder, attrs.axes)

        t_op.builtin_options = sum_options.Sum(bool(attrs.keepdims))

        ops = OpsList(middle_op=t_op)
        reduce_utils.ensure_reduce_transposition(self.builder, ops)

        return ops.flatten()

    def _convert_v_13(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX ReduceSum version 13+ to TFLite Sum.

        The `axes` are an optional input tensor.
        """
        attrs = cast(ReduceSum, node.attributes)

        ops = OpsList(middle_op=t_op)
        convert_axes_from_input_tensor(t_op, self.builder, self.inspector, ops, attrs.noop_with_empty_axes,
                                       node.op_type)

        t_op.builtin_options = sum_options.Sum(bool(attrs.keepdims))

        reduce_utils.ensure_reduce_transposition(self.builder, ops)

        return ops.flatten()

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX ReduceSum operator to TFLite Sum."""
        if not (1 <= len(t_op.tmp_inputs) <= 2):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX `ReduceSum` has unexpected number of inputs.")

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(t_op.tmp_inputs[0].type)

        if node.version < 13:
            # Version 1 / 11  -> axes are passed as attribute.
            return self._convert_v_11(node, t_op)

        # Version 13+ -> axes are an optional input tensor.
        return self._convert_v_13(node, t_op)
