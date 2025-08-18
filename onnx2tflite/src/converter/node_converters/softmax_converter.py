#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

import math
from typing import Tuple, Union

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.builtin_attributes.log_softmax_attributes as onnx_log_softmax_attributes
import onnx2tflite.src.onnx_parser.builtin_attributes.softmax_attributes as onnx_softmax_attributes
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator.builtin_options import (log_softmax_options as log_softmax_options,
                                                              reshape_options as tfl_reshape_options,
                                                              softmax_options as tfl_softmax_options)
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


# noinspection PyMethodMayBeStatic
class SoftmaxConverter(NodeConverter):
    node = 'Softmax'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/activations.cc#L1329-L1376
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/activations.cc#L1387-L1445
    # TFLite `Softmax` also supports `int16` over `LogSoftmax`, but it's not supported by ONNX anyway, so it doesn't
    #  affect the type checking.
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8]
    verified_types = [TensorType.FLOAT32]

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

    def _convert_v13(self, o_softmax: Union[onnx_softmax_attributes.Softmax, onnx_log_softmax_attributes.LogSoftmax],
                     t_op: tflite_model.Operator) -> OpsList:
        x = t_op.tmp_inputs[0]
        rank = len(x.shape.vector)

        if o_softmax.axis is None:
            o_softmax.axis = -1
        axis = self._normalize_axis(o_softmax.axis, rank)

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

    def _convert_v1_reshaped(self, op_softmax: tflite_model.Operator, axis) -> OpsList:
        old_shape = op_softmax.tmp_inputs[0].shape.vector
        new_shape = [math.prod(old_shape[:axis]), math.prod(old_shape[axis:])]

        reshape_pre, reshape_post = self._wrap_in_reshape(op_softmax, new_shape, old_shape)

        return OpsList(pre_ops=[reshape_pre], middle_op=op_softmax, post_ops=[reshape_post])

    def _convert_v1_transposed_and_reshaped(self, op_softmax: tflite_model.Operator, axis) -> OpsList:
        input_shape = op_softmax.tmp_inputs[0].shape.vector
        rank = len(input_shape)

        to_channel_first_perm = list(translator.create_channels_last_to_channels_first_permutation(rank))
        to_channel_last_perm = list(translator.create_channels_first_to_channels_last_permutation(rank))

        reshape_outer_shape = translator.apply_permutation_to(input_shape, to_channel_first_perm)
        reshape_inner_shape = [math.prod(reshape_outer_shape[:axis]), math.prod(reshape_outer_shape[axis:])]

        # Reshape to two-dimension shape
        reshape_pre, reshape_post = self._wrap_in_reshape(op_softmax, reshape_inner_shape, reshape_outer_shape)

        # We have to transpose before reshaping because input shape doesn't match original ONNX shape
        transpose_pre = self.builder.create_transpose_operator_before(reshape_pre, 0, to_channel_first_perm)
        transpose_post = self.builder.create_transpose_operator_after(reshape_post, 0, to_channel_last_perm,
                                                                      keep_output_shape=True)

        return OpsList(pre_ops=[transpose_pre, reshape_pre], middle_op=op_softmax,
                       post_ops=[reshape_post, transpose_post])

    def _wrap_in_reshape(self, op_softmax: tflite_model.Operator,
                         reshape_inner_shape, reshape_outer_shape
                         ) -> Tuple[tflite_model.Operator, tflite_model.Operator]:
        """
        Surround passed Softmax operator by Reshape operators.
    
        (reshape_outer_shape)
                  ↓
              [Reshape] (reshape_pre)
                  ↓
        (reshape_inner_shape)
                  ↓
              [Softmax]
                  ↓
        (reshape_inner_shape)
                  ↓
              [Reshape] (reshape_post)
                  ↓
        (reshape_outer_shape)
    
        :param op_softmax: Surrounded Softmax operator.
        :param reshape_inner_shape: Inner shape of reshaped block. New input shape of Softmax operator.
        :param reshape_outer_shape: Outer shape of reshaped block. Input shape of the first reshape operator.
        :return: Returns tuple with created preceding and succeeding Reshape operators.
        """
        x = op_softmax.tmp_inputs[0]
        y = op_softmax.tmp_outputs[0]

        t1 = self.builder.duplicate_tensor(x, "softmax_reshape_1_")
        t1.shape = tflite_model.Shape(reshape_inner_shape)
        t1.tensor_format = TensorFormat.FORMATLESS

        t2 = self.builder.duplicate_tensor(y, "softmax_reshape_2_")
        t2.shape = tflite_model.Shape(reshape_inner_shape)
        t2.tensor_format = TensorFormat.FORMATLESS

        # Create first Reshape operator
        reshape_pre = tflite_model.Operator(
            builtin_options=tfl_reshape_options.Reshape(reshape_inner_shape)
        )
        reshape_pre.tmp_inputs = [x]
        reshape_pre.tmp_outputs = [t1]

        # Connect softmax to outer reshapes
        op_softmax.tmp_inputs = [t1]
        op_softmax.tmp_outputs = [t2]

        # Create second Reshape operator
        reshape_post = tflite_model.Operator(
            builtin_options=tfl_reshape_options.Reshape(reshape_outer_shape)
        )
        reshape_post.tmp_inputs = [t2]
        reshape_post.tmp_outputs = [y]

        return reshape_pre, reshape_post

    def _convert_v1(self, o_softmax: Union[onnx_softmax_attributes.Softmax, onnx_log_softmax_attributes.LogSoftmax],
                    t_op: tflite_model.Operator) -> OpsList:
        x = t_op.tmp_inputs[0]
        rank = len(x.shape.vector)

        if o_softmax.axis is None:
            o_softmax.axis = 1
        axis = self._normalize_axis(o_softmax.axis, rank)

        if x.tensor_format == TensorFormat.FORMATLESS and axis == rank - 1:
            # We don't need to reshape/transpose input when we compute over last dimension with format-less.
            return OpsList(middle_op=t_op)
        elif x.tensor_format.is_channels_last() and axis == 1:
            # Input internally in ONNX reshaped to [d0, d1-dn] -> shape is the same also for TFLite.
            return self._convert_v1_reshaped(t_op, axis)
        elif x.tensor_format.is_channels_last():
            # We have to reshape and also transpose because channel dimension is not represented by same axis.
            return self._convert_v1_transposed_and_reshaped(t_op, axis)
        else:
            return self._convert_v1_reshaped(t_op, axis)

    def _handle_logsoftmax_quantization(self, ops, t_op):
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if x.type in [TensorType.INT8, TensorType.UINT8] and y.quantization is not None:
            zp = [255 if (x.type == TensorType.UINT8) else 127]
            scale = [16. / 256.]

            # Check if output quantization params are defined correctly. If not => re-quantize
            output_scale = y.quantization.scale.vector[0]
            output_zp = y.quantization.zero_point.vector[0]

            if not math.isclose(output_scale, scale[0]) or not math.isclose(output_zp, zp[0]):
                logger.w(f"Re-quantizing output tensor '{y.name}' of LogSoftmax op to satisfy TFLite's "
                         "q-param requirements. This can decrease accuracy of the model.")
                quantize_op = self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp)
                ops.post_ops.insert(0, quantize_op)

    def _handle_softmax_quantization(self, ops, t_op):
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if x.type in [TensorType.INT8, TensorType.UINT8] and y.quantization is not None:
            zp = [0 if (x.type == TensorType.UINT8) else -128]
            scale = [1.0 / 256.0]

            # Check if output quantization params are defined correctly. If not => re-quantize
            output_scale = y.quantization.scale.vector[0]
            output_zp = y.quantization.zero_point.vector[0]

            if not math.isclose(output_scale, scale[0]) or not math.isclose(output_zp, zp[0]):
                logger.w(f"Re-quantizing output tensor '{y.name}' of Softmax op to satisfy TFLite's "
                         "q-param requirements. This can decrease accuracy of the model.")
                quantize_op = self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp)
                ops.post_ops.insert(0, quantize_op)

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX Runtime (Log)Softmax operator to TFLite (Log)Softmax. """

        assert (isinstance(node.attributes, onnx_softmax_attributes.Softmax) or
                isinstance(node.attributes, onnx_log_softmax_attributes.LogSoftmax))

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(t_op.tmp_inputs[0].type)

        if node.version < 13:
            ops = self._convert_v1(node.attributes, t_op)
        else:
            ops = self._convert_v13(node.attributes, t_op)

        if isinstance(node.attributes, onnx_softmax_attributes.Softmax):
            t_op.builtin_options = tfl_softmax_options.Softmax(1.0)

            self._handle_softmax_quantization(ops, t_op)
        elif isinstance(node.attributes, onnx_log_softmax_attributes.LogSoftmax):
            t_op.builtin_options = log_softmax_options.LogSoftmax()

            self._handle_logsoftmax_quantization(ops, t_op)

        return ops.flatten()
