#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import topk_attributes
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import neg_options, topk_v2_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS
from onnx2tflite.src.tflite_generator.tflite_model import Operator, Tensor


class TopKConverter(NodeConverter):
    node = "TopK"

    onnx_supported_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]  # ORT supports only these types
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]
    verified_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]

    # noinspection PyMethodMayBeStatic
    def _move_last_dimension_to_idx(self, shape, dim_index) -> list:
        new_shape = list(shape)
        new_shape.insert(dim_index, new_shape[-1])
        del new_shape[-1]

        return new_shape

    # noinspection PyMethodMayBeStatic
    def _normalize_axis(self, axis: int, rank: int) -> int:
        if axis < -rank or axis > rank - 1:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f"ONNX attribute 'axis' ({axis}) must be in range [{-rank}, {rank - 1}]!!")

        # convert negative index to positive
        if axis < 0:
            axis += rank
        return axis

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'TopK' operator to TFLite 'TopK_V2'."""
        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX 'TopK' has unexpected number of inputs. Got '{len(t_op.tmp_inputs)}', expected '2'.")

        x = t_op.tmp_inputs[0]
        k = t_op.tmp_inputs[1]
        rank = len(x.shape.vector)

        self.assert_type_allowed(x.type)

        if not tensor_has_data(k):
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, "Conversion of 'TopK' with dynamic 'k' tensor in not possible.")

        attrs = cast(topk_attributes.TopK, node.attributes)

        if attrs.sorted == 0:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of 'TopK' with 'sorted=0' is not possible.")

        if attrs.largest == 0:
            # Case with 'sorted=1' and 'largest=0' can be implemented with additional Gather.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of 'TopK' with 'largest=0' and 'sorted=1' is not yet implemented.")

        k.tmp_buffer.data = k.tmp_buffer.data.astype(np.int32)
        k.type = TensorType.INT32

        ops = OpsList(middle_op=t_op)

        t_op.builtin_options = topk_v2_options.TopKV2()

        # TFLite indices output tensor supports only INT32
        if t_op.tmp_outputs[1].type != TensorType.INT32:
            cast_op = self.builder.create_cast_after(t_op, 1, TensorType.INT32)
            ops.add_post(cast_op)

        axis = self._normalize_axis(attrs.axis, rank)

        # tensor has format -> permute axis to TFLite format
        if x.tensor_format.is_channels_last():
            axis = translator.create_channels_last_to_channels_first_permutation(rank)[axis]

        # axis is the last dimension -> no need for transposing
        if axis == rank - 1:
            return ops.flatten()

        # move axis as a last dimension
        input_perm = translator.create_axis_to_last_perm(axis, rank)

        # move axis back where it was before TopK
        output_perm = self._move_last_dimension_to_idx(range(rank), axis)

        ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, input_perm))
        ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, output_perm, keep_output_shape=True))
        ops.add_post(self.builder.create_transpose_operator_after(t_op, 1, output_perm, keep_output_shape=True))

        return ops.flatten()
