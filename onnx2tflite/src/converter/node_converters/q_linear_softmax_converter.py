#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math
from typing import Tuple, cast

import numpy as np

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.builtin_attributes.q_linear_softmax_attributes as onnx_ql_softmax_attribs
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.quantization_utils import set_quantization_parameters_to_tensor
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import try_get_input, OpsList
from onnx2tflite.src.converter.conversion.translator import tf_lite_type_to_numpy
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator.builtin_options import (
    softmax_options as tfl_softmax_options,
    reshape_options as tfl_reshape_options,
)


# noinspection PyMethodMayBeStatic
class QLinearSoftmaxConverter(NodeConverter):
    node = 'QLinearSoftmax'

    def _move_last_dimension_to_idx(self, shape, dim_index) -> list:
        new_shape = list(shape)
        new_shape.insert(dim_index, new_shape[-1])
        del new_shape[-1]

        return new_shape

    def _normalize_axis(self, axis, rank):
        if axis < -rank or axis > rank - 1:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f"ONNX attribute 'axis' ({axis}) must be in range [{-rank}, {rank - 1}]!!")

        # convert negative index to positive
        if axis < 0:
            axis += rank
        return axis

    def _convert_v13(self, o_ql_softmax: onnx_ql_softmax_attribs.QLinearSoftmax,
                     t_op: tflite_model.Operator) -> OpsList:
        x = t_op.tmp_inputs[0]
        rank = len(x.shape.vector)
        axis = self._normalize_axis(o_ql_softmax.axis, rank)

        # Avoid applying QLinearSoftmax when axis points to dimension of value 1
        if x.shape.vector[axis] == 1:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     "ONNX QLinearSoftmax attribute 'axis' points to dimension of value 1, "
                     f"providing undesired results. Input shape: {x.shape.vector}, Axis: {axis}")

        # tensor has format -> permute axis to TFLite format
        if x.tensor_format.is_channels_last():
            axis = translator.create_channels_last_to_channels_first_permutation(rank)[axis]

        # axis is the last dimension -> no need for transposing
        if axis == rank - 1:
            return OpsList(middle_op=t_op)

        # move axis as a last dimension
        input_perm = translator.create_axis_to_last_perm(axis, rank)

        # move axis back where it was before softmax
        output_perm = self._move_last_dimension_to_idx(range(rank), axis)

        transpose_pre = self.builder.create_transpose_operator_before(t_op, 0, input_perm)
        transpose_post = self.builder.create_transpose_operator_after(t_op, 0, output_perm,
                                                                      keep_output_shape=True)

        return OpsList(pre_ops=[transpose_pre], middle_op=t_op, post_ops=[transpose_post])

    def _convert_v1_reshaped(self, op_ql_softmax: tflite_model.Operator, axis) -> OpsList:
        old_shape = op_ql_softmax.tmp_inputs[0].shape.vector
        new_shape = [math.prod(old_shape[:axis]), math.prod(old_shape[axis:])]

        reshape_pre, reshape_post = self._wrap_in_reshape(op_ql_softmax, new_shape, old_shape)

        return OpsList(pre_ops=[reshape_pre], middle_op=op_ql_softmax, post_ops=[reshape_post])

    def _convert_v1_transposed_and_reshaped(self, op_ql_softmax: tflite_model.Operator, axis) -> OpsList:
        input_shape = op_ql_softmax.tmp_inputs[0].shape.vector
        rank = len(input_shape)

        to_channel_first_perm = list(translator.create_channels_last_to_channels_first_permutation(rank))
        to_channel_last_perm = list(translator.create_channels_first_to_channels_last_permutation(rank))

        reshape_outer_shape = translator.apply_permutation_to(input_shape, to_channel_first_perm)
        reshape_inner_shape = [math.prod(reshape_outer_shape[:axis]), math.prod(reshape_outer_shape[axis:])]

        # Reshape to two-dimension shape
        reshape_pre, reshape_post = self._wrap_in_reshape(op_ql_softmax, reshape_inner_shape, reshape_outer_shape)

        # We have to transpose before reshaping because input shape doesn't match original ONNX shape
        transpose_pre = self.builder.create_transpose_operator_before(reshape_pre, 0, to_channel_first_perm)
        transpose_post = self.builder.create_transpose_operator_after(reshape_post, 0, to_channel_last_perm,
                                                                      keep_output_shape=True)

        return OpsList(
            pre_ops=[transpose_pre, reshape_pre],
            middle_op=op_ql_softmax,
            post_ops=[reshape_post, transpose_post])

    def _wrap_in_reshape(self, op_ql_softmax: tflite_model.Operator,
                         reshape_inner_shape, reshape_outer_shape
                         ) -> Tuple[tflite_model.Operator, tflite_model.Operator]:
        """
        Surround passed QLinearSoftmax operator by Reshape operators.
    
        (reshape_outer_shape)
                  ↓
              [Reshape] (reshape_pre)
                  ↓
        (reshape_inner_shape)
                  ↓
            [QLinearSoftmax]
                  ↓
        (reshape_inner_shape)
                  ↓
              [Reshape] (reshape_post)
                  ↓
        (reshape_outer_shape)
    
        :param op_ql_softmax: Surrounded Softmax operator.
        :param reshape_inner_shape: Inner shape of reshaped block. New input shape of Softmax operator.
        :param reshape_outer_shape: Outer shape of reshaped block. Input shape of the first reshape operator.
        :return: Returns tuple with created preceding and succeeding Reshape operators.
        """
        x = op_ql_softmax.tmp_inputs[0]
        y = op_ql_softmax.tmp_outputs[0]

        t1 = self.builder.duplicate_tensor(x, "q_linear_softmax_reshape_1_")
        t1.shape = tflite_model.Shape(reshape_inner_shape)
        t1.tensor_format = TensorFormat.FORMATLESS

        t2 = self.builder.duplicate_tensor(y, "q_linear_softmax_reshape_2_")
        t2.shape = tflite_model.Shape(reshape_inner_shape)
        t2.tensor_format = TensorFormat.FORMATLESS

        # Create first Reshape operator
        reshape_pre = tflite_model.Operator(
            builtin_options=tfl_reshape_options.Reshape(reshape_inner_shape)
        )
        reshape_pre.tmp_inputs = [x]
        reshape_pre.tmp_outputs = [t1]

        # Connect softmax to outer reshapes
        op_ql_softmax.tmp_inputs = [t1]
        op_ql_softmax.tmp_outputs = [t2]

        # Create second Reshape operator
        reshape_post = tflite_model.Operator(
            builtin_options=tfl_reshape_options.Reshape(reshape_outer_shape)
        )
        reshape_post.tmp_inputs = [t2]
        reshape_post.tmp_outputs = [y]

        return reshape_pre, reshape_post

    def _convert_v1(self, o_ql_softmax: onnx_ql_softmax_attribs.QLinearSoftmax,
                    t_op: tflite_model.Operator) -> OpsList:
        x = t_op.tmp_inputs[0]
        rank = len(x.shape.vector)
        axis = self._normalize_axis(o_ql_softmax.axis, rank)

        # Avoid applying QLinearSoftmax when axis points to dimension of value 1
        if x.shape.vector[axis] == 1:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     "ONNX QLinearSoftmax attribute 'axis' points to dimension of value 1, "
                     f"providing undesired results. Input shape: {x.shape.vector}, Axis: {axis}")

        if x.tensor_format == TensorFormat.FORMATLESS and axis == rank - 1:
            # We don't need to reshape/transpose input when we compute over last dimension with formatless
            return OpsList(middle_op=t_op)
        elif x.tensor_format.is_channels_last() and axis == 1:
            # Input internally in ONNX reshaped to [d0, d1-dn] -> shape is the same also for TFLite
            return self._convert_v1_reshaped(t_op, axis)
        elif x.tensor_format.is_channels_last():
            # We have to reshape and also transpose because channel dimension is not represented by same axis
            return self._convert_v1_transposed_and_reshaped(t_op, axis)
        else:
            return self._convert_v1_reshaped(t_op, axis)

    def _ensure_correct_output_quant_params(self, ops):
        t_op = ops.middle_op
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        # Check if output quantization params are defined correctly. If not => re-quantize
        zp = [0 if (x.type == TensorType.UINT8) else -128]
        scale = [1.0 / 256.0]

        output_scale = y.quantization.scale.vector
        output_zp = y.quantization.zero_point.vector

        if not math.isclose(output_scale[0], scale[0]) or not math.isclose(output_zp[0], zp[0]):
            logger.w(f"Re-quantizing output tensor '{y.name}' of QLinearSoftmax op to satisfy TFLite's "
                     "q-param requirements. This can decrease accuracy of the model.")
            quantize_op = self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp)
            ops.post_ops.insert(0, quantize_op)

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX Runtime QLinearSoftmax operator to TFLite Softmax.

        :param node: ONNX QLinearSoftmax operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator
        :return: A list of TFLite operators to be added to the TFLite model.
        """

        t_op.builtin_options = tfl_softmax_options.Softmax(1.0)

        # quantize stage
        x_tensor = t_op.tmp_inputs[0]
        x_scale_tensor = t_op.tmp_inputs[1]
        x_zp_tensor = try_get_input(t_op, 2)
        y_scale_tensor = t_op.tmp_inputs[3]
        y_zp_tensor = t_op.tmp_inputs[4]
        y_tensor = t_op.tmp_outputs[0]

        # Input and output types must be the same
        if not (x_tensor.type == y_tensor.type):
            logger.e(logger.Code.INVALID_TYPE,
                     "ONNX QLinearSoftmax input and output tensors must have the same data type!")

        # ONNX only supports INT8 and UINT8
        if y_tensor.type not in {TensorType.INT8, TensorType.UINT8}:
            logger.e(logger.Code.INVALID_TYPE,
                     f"ONNX QLinearSoftmax supports only INT8 and UINT8 data types. Got '{y_tensor.type}'.")

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [x_tensor]
        t_op.tmp_outputs = [y_tensor]

        # Add the quantization parameters to the tensors
        x_scale = x_scale_tensor.tmp_buffer.data
        x_zero_point = x_zp_tensor.tmp_buffer.data if x_zp_tensor is not None else np.zeros(
            x_scale.shape, tf_lite_type_to_numpy(x_tensor.type))
        set_quantization_parameters_to_tensor(x_tensor, x_scale, x_zero_point)

        y_scale = y_scale_tensor.tmp_buffer.data
        y_zero_point = y_zp_tensor.tmp_buffer.data

        set_quantization_parameters_to_tensor(y_tensor, y_scale, y_zero_point)

        o_ql_softmax = cast(onnx_ql_softmax_attribs.QLinearSoftmax, node.attributes)

        if o_ql_softmax.opset < 13:
            ops = self._convert_v1(o_ql_softmax, t_op)
        else:
            ops = self._convert_v13(o_ql_softmax, t_op)

        self._ensure_correct_output_quant_params(ops)

        return ops.flatten()
