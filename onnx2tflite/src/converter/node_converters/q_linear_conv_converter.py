#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import common, translator
from onnx2tflite.src.converter.conversion.common import try_get_input
from onnx2tflite.src.converter.conversion.translator import apply_permutation_to
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.node_converters.shared import conv_utils
from onnx2tflite.src.converter.quantization_utils import (
    calculate_uint_to_int_re_quantization_zero_point,
    get_symmetric_zero_point_for_type,
    re_quantize_static_tensor,
    set_quantization_parameters_to_tensor,
)
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import conv_attributes, q_linear_conv_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import conv_2d_options, depthwise_conv_2d_options

ConvAttributes = conv_attributes.Conv | q_linear_conv_attributes.QLinearConv


# noinspection PyMethodMayBeStatic
class QLinearConvConverter(NodeConverter):
    node = "QLinearConv"

    def _get_input_with_quant_params(
        self, attrs: ConvAttributes, t_op: tflite_model.Operator, input_idx: int
    ) -> tuple[tflite_model.Tensor, np.ndarray, np.ndarray]:
        """Get input tensor and corresponding q-params from other inputs (QLinearConv) or
        directly from tensor's quantization property.

        :param attrs: ONNX node attributes of QLinearConv or (U)INT8 Conv.
        :param t_op: TFLite representation of ONNX QLinearConv or (U)INT8 Conv NodeProto.
        :param input_idx: Input tensor index - 0 or 1.
        :return: Tuple with input tensor, quant. scale and quant. zero point
        """
        input_mapping = {
            q_linear_conv_attributes.QLinearConv: [0, 3],
            conv_attributes.Conv: [0, 1],
        }

        tensor_idx = input_mapping[type(attrs)][input_idx]
        x = t_op.tmp_inputs[tensor_idx]

        if x.quantization is None:
            x_scale = t_op.tmp_inputs[tensor_idx + 1].tmp_buffer.data
            x_zero_point = t_op.tmp_inputs[tensor_idx + 2].tmp_buffer.data
        else:
            x_scale = np.array(x.quantization.scale.vector)
            x_zero_point = np.array(x.quantization.zero_point.vector)

        return x, x_scale, x_zero_point

    def _get_bias_tensor_idx(self, attrs: ConvAttributes) -> int:
        if isinstance(attrs, q_linear_conv_attributes.QLinearConv):
            return 8
        if isinstance(attrs, conv_attributes.Conv):
            return 2
        logger.e(logger.Code.INTERNAL_ERROR, "Attempt to get index of bias tensor of incorrect ONNX operator.")

    def get_weight_tensor_index(self, attrs: ConvAttributes) -> int:
        if isinstance(attrs, q_linear_conv_attributes.QLinearConv):
            return 3
        if isinstance(attrs, conv_attributes.Conv):
            return 1
        logger.e(logger.Code.INTERNAL_ERROR, "Attempt to get index of weight tensor of incorrect ONNX operator.")

    def _get_output_quant_params(self, t_op) -> tuple[tflite_model.Tensor, np.ndarray, np.ndarray]:
        """Get output tensor and corresponding q-params from other inputs (QLinearConv) or
        directly from tensor's quantization property.

        :param t_op: TFLite representation of ONNX QLinearConv or (U)INT8 Conv NodeProto.
        :return: Tuple with output tensor, quant. scale and quant. zero point
        """
        y = t_op.tmp_outputs[0]

        if y.quantization is None:
            y_scale = t_op.tmp_inputs[6].tmp_buffer.data
            y_zero_point = t_op.tmp_inputs[7].tmp_buffer.data
        else:
            y_scale = np.array(y.quantization.scale.vector)
            y_zero_point = np.array(y.quantization.zero_point.vector)

        return y, y_scale, y_zero_point

    # noinspection PyPep8Naming
    def _convert_unpadded_2D(self, q_linear_conv, t_op) -> conv_utils.ConvConversionResult:  # noqa: N802
        # Prepare the input and output tensors. To replace them, assign to t_op.tmp_inputs/tmp_outputs directly.
        input_tensor, input_scale, input_zero_point = self._get_input_with_quant_params(q_linear_conv, t_op, 0)
        weight_tensor, weight_scale, weight_zero_point = self._get_input_with_quant_params(q_linear_conv, t_op, 1)
        output_tensor, output_scale, output_zero_point = self._get_output_quant_params(t_op)

        if (bias_tensor := try_get_input(t_op, self._get_bias_tensor_idx(q_linear_conv))) is None:
            # Operator has no bias. ONNX model can omit it, TFLite can't.
            # The bias cannot be reused, because quantization parameters may be set to it.
            bias_tensor = self.builder.create_zeros_tensor([weight_tensor.shape.get(0)], "quantized_conv2d_bias_",
                                                           translator.tf_lite_type_to_numpy(TensorType.INT32),
                                                           can_reuse=False)

        if (bias_tensor.tmp_buffer.data is not None and  # Bias tensor has static data...
                not bias_tensor.tmp_buffer.data.any() and  # ... which are all zero ...
                self.context.onnx_inspector.get_number_of_onnx_consumers_safe(bias_tensor.name) > 1
                # is shared btw. multiple nodes
        ):
            # In ONNX the bias tensor quantization parameters are derived implicitly, in tflite has to be set
            # explicitly.This make problems when ONNX decides to share the bias tensor (only possible if bias = 0)
            # instead of omitting it. Therefore, here we create another tensor instead of using the original one.
            bias_tensor = self.builder.create_zeros_tensor([weight_tensor.shape.get(0)], bias_tensor.name,
                                                           translator.tf_lite_type_to_numpy(TensorType.INT32),
                                                           can_reuse=False)

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [input_tensor, weight_tensor, bias_tensor]
        t_op.tmp_outputs = [output_tensor]

        conversion_result = conv_utils.ConvConversionResult(input_tensor, weight_tensor, bias_tensor, output_tensor)
        conversion_result.ops_list.middle_op = t_op

        # Add the quantization parameters to the tensors
        if input_tensor.quantization is None:
            set_quantization_parameters_to_tensor(input_tensor, input_scale, input_zero_point)
        if weight_tensor.quantization is None:
            set_quantization_parameters_to_tensor(weight_tensor, weight_scale, weight_zero_point)
        if bias_tensor.quantization is None:
            bias_scale = input_scale * weight_scale
            bias_zero_point = np.zeros(weight_scale.shape, dtype=np.int64)
            set_quantization_parameters_to_tensor(bias_tensor, bias_scale, bias_zero_point, quantized_dimension=0)
        if output_tensor.quantization is None:
            set_quantization_parameters_to_tensor(output_tensor, output_scale, output_zero_point)

        if np.all(weight_zero_point != get_symmetric_zero_point_for_type(weight_tensor.type)):
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "TFLite Conv2D (DepthwiseConv2D) op doesn't support 'zero_point' "
                     "of weight tensor to be non-zero.")

        # ONNX only supports INT8 and UINT8 input/output/weight types
        if input_tensor.type not in {TensorType.UINT8, TensorType.INT8} or output_tensor.type not in \
                {TensorType.UINT8, TensorType.INT8} or weight_tensor.type not in {TensorType.UINT8, TensorType.INT8}:
            logger.e(logger.Code.INVALID_TYPE, "ONNX QLinearConv with an unexpected input/output/weight data type. "
                                               "Only INT8 and UINT8 are supported!")
        if bias_tensor.type != TensorType.INT32:
            logger.e(logger.Code.INVALID_TYPE,
                     "ONNX QLinearConv with an unexpected bias data type. Only INT32 is supported!")

        # Ensure equal input and output type
        if input_tensor.type != output_tensor.type:
            # https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/quantization/qlinearconv.cc#L306
            logger.e(logger.Code.INVALID_TYPE, "ONNX QLinearConv with mismatching input/output tensor types!")

        # Ensure supported quantization and data type combination, if possible.
        if weight_tensor.quantization.is_per_channel():  # Per CHANNEL quantization
            if input_tensor.type == TensorType.UINT8:
                # Re-quantize operator to int8

                # INPUT
                if tensor_has_data(input_tensor):
                    input_tensor = re_quantize_static_tensor(self.builder, input_tensor, TensorType.INT8)
                    conversion_result.conv_input_tensor = input_tensor
                    t_op.tmp_inputs[0] = input_tensor
                else:
                    new_zero_point = calculate_uint_to_int_re_quantization_zero_point(1, input_zero_point)
                    quantize_op = self.builder.create_quantize_operator_before(t_op, 0, TensorType.INT8, None,
                                                                               [new_zero_point.item()])
                    conversion_result.ops_list.add_pre(quantize_op)
                    conversion_result.conv_input_tensor = quantize_op.tmp_outputs[0]

                # WEIGHTS
                if weight_tensor.type == TensorType.UINT8:
                    new_zero_point = calculate_uint_to_int_re_quantization_zero_point(1, weight_zero_point)
                    if not all([zp == 0 for zp in new_zero_point]):
                        logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                                 "ONNX QLinearConv uses UINT8 weights, which cannot be"
                                 " re-quantized to INT8, because their zero point != 128.")

                    if tensor_has_data(weight_tensor):
                        weight_tensor = re_quantize_static_tensor(self.builder, weight_tensor, TensorType.INT8)
                        conversion_result.conv_weight_tensor = weight_tensor
                        t_op.tmp_inputs[1] = weight_tensor
                    else:
                        logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                                 "ONNX QLinearConv uses dynamic UINT8 weights with per"
                                 " channel quantization what is not supported in TFLite.")

                # OUTPUT
                new_zero_point = calculate_uint_to_int_re_quantization_zero_point(1, output_zero_point)
                quantize_op = self.builder.create_quantize_operator_after(t_op, 0, TensorType.INT8, None,
                                                                          [new_zero_point.item()])
                conversion_result.ops_list.add_post(quantize_op)
                conversion_result.conv_output_tensor = quantize_op.tmp_inputs[0]

        elif input_tensor.type == TensorType.UINT8:
            if weight_tensor.type == TensorType.INT8:
                # Re-quantize operator to INT8

                # INPUT
                if tensor_has_data(input_tensor):
                    input_tensor = re_quantize_static_tensor(self.builder, input_tensor, TensorType.INT8)
                    conversion_result.conv_input_tensor = input_tensor
                    t_op.tmp_inputs[0] = input_tensor

                else:
                    new_zero_point = calculate_uint_to_int_re_quantization_zero_point(1, input_zero_point)
                    quantize_op = self.builder.create_quantize_operator_before(t_op, 0, TensorType.INT8, None,
                                                                               [new_zero_point.item()])
                    conversion_result.ops_list.add_pre(quantize_op)
                    conversion_result.conv_input_tensor = quantize_op.tmp_outputs[0]

                # OUTPUT
                new_zero_point = calculate_uint_to_int_re_quantization_zero_point(1, output_zero_point)
                quantize_op = self.builder.create_quantize_operator_after(t_op, 0, TensorType.INT8, None,
                                                                          [new_zero_point.item()])
                conversion_result.ops_list.add_post(quantize_op)
                conversion_result.conv_output_tensor = quantize_op.tmp_inputs[0]

        # Convert the builtin options
        common.assign_2d_strides(t_op.builtin_options, q_linear_conv.strides)
        common.assign_2d_dilations(t_op.builtin_options, q_linear_conv.dilations)

        return conversion_result

    def _convert_2d_q_linear_conv(self, attrs: ConvAttributes, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        if conv_utils.group_conv_convertible_as_depthwise(attrs, t_op, self.get_weight_tensor_index(attrs)):
            t_op.builtin_options = depthwise_conv_2d_options.DepthwiseConv2D()

            conversion_result = self._convert_unpadded_2D(attrs, t_op)
            padding, pad_op = conv_utils.build_input_tensor_padding(attrs, t_op, self.context.tflite_builder)
            t_op.builtin_options.padding = padding

            if pad_op is not None:
                conversion_result.ops_list.add_pre(pad_op)

            # DepthwiseConv2D expects weights in format [kernel_channels, kernel_height, kernel_width, output_channels]
            perm = [3, 1, 2, 0]
            weight_tensor = conversion_result.conv_weight_tensor

            if tensor_has_data(weight_tensor):
                # Transpose the tensor statically.

                # Duplicate the weight_tensor, in case it is used by other operators.
                weight_tensor = self.builder.duplicate_tensor(weight_tensor)
                t_op.tmp_inputs[1] = weight_tensor

                weight_tensor.tmp_buffer.data = np.transpose(weight_tensor.tmp_buffer.data, perm)
                weight_tensor.shape = tflite_model.Shape(apply_permutation_to(weight_tensor.shape.vector, perm))
                weight_tensor.quantization.quantized_dimension = 3
            else:
                # Insert a Transpose operator
                transpose_op = self.context.tflite_builder.create_transpose_operator_before(t_op, 1, perm)
                transpose_op.tmp_outputs[0].quantization.quantized_dimension = 3
                conversion_result.ops_list.add_pre(transpose_op)

            return conversion_result.ops_list.flatten()

        if conv_utils.group_conv_convertible_into_multiple_convolutions(attrs, t_op):
            t_op.builtin_options = conv_2d_options.Conv2D()

            return conv_utils.create_separated_convolutions_based_on_group(
                attrs, t_op, self.context.tflite_builder, self._convert_unpadded_2D, conv_utils.conv_op_factory, 0)

        t_op.builtin_options = conv_2d_options.Conv2D()
        conversion_result = self._convert_unpadded_2D(attrs, t_op)
        padding, pad_op = conv_utils.build_input_tensor_padding(attrs, t_op, self.context.tflite_builder)
        t_op.builtin_options.padding = padding

        if pad_op is not None:
            conversion_result.ops_list.add_pre(pad_op)

        return conversion_result.ops_list.flatten()

    def convert(self, q_linear_conv_node: onnx_model.NodeProto,
                t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX QLinearConv or quantized Conv operator to TFLite."""
        attrs = cast(ConvAttributes, q_linear_conv_node.attributes)
        weight_tensor, _, _ = self._get_input_with_quant_params(attrs, t_op, 1)

        if attrs.kernel_shape is None:
            attrs.kernel_shape = translator.infer_kernel_shape(weight_tensor)
        elif attrs.kernel_shape != translator.infer_kernel_shape(weight_tensor):
            logger.e(logger.Code.INVALID_ONNX_MODEL, "Weight tensor shape not corresponds to kernel_shape attribute")

        kernel_rank = len(attrs.kernel_shape)
        if kernel_rank == 1:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of 1D QLinearConv is not yet implemented!")
        elif kernel_rank == 2:
            return self._convert_2d_q_linear_conv(attrs, t_op)
        else:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"Conversion of ONNX QLinearConv with '{kernel_rank}' dimensions is not possible!")
