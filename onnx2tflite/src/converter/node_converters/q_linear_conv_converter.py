#
# Copyright 2023-2024,2026 NXP
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
    propagate_quantization,
)
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import conv_attributes, q_linear_conv_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import conv_2d_options, depthwise_conv_2d_options, reshape_options

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

    def _convert_1d_q_linear_conv(
            self, o_conv_attributes: conv_attributes.Conv, t_op: tflite_model.Operator
    ) -> list[tflite_model.Operator]:
        weight_tensor_index = self.get_weight_tensor_index(o_conv_attributes)

        # -- Calculate the shapes for equivalent 2D convolution --
        conv_2d_input_shape = translator.nhc_dimensions_to_nhwc(t_op.tmp_inputs[0].shape.vector)
        conv_2d_weight_shape = translator.nhc_dimensions_to_nhwc(t_op.tmp_inputs[weight_tensor_index].shape.vector)
        conv_2d_output_shape = translator.nhc_dimensions_to_nhwc(t_op.tmp_outputs[0].shape.vector)

        # -- Generate tensors taking part in the conversion --
        input_reshape_x = t_op.tmp_inputs[0]

        input_reshape_y = self.builder.duplicate_tensor(input_reshape_x, name_suffix="_4D_")
        input_reshape_y.shape = tflite_model.Shape(conv_2d_input_shape)

        output_reshape_x = self.builder.duplicate_tensor(t_op.tmp_outputs[0], name_suffix="_4D_")
        output_reshape_x.shape = tflite_model.Shape(conv_2d_output_shape)

        output_reshape_y = t_op.tmp_outputs[0]

        pre_reshapes = []

        # Extend the weights tensor to 4D
        weights_tensor = t_op.tmp_inputs[weight_tensor_index]
        if tensor_has_data(weights_tensor):
            # Do it statically
            weights_tensor.shape = tflite_model.Shape(conv_2d_weight_shape)
            weights_tensor.tmp_buffer.data = weights_tensor.tmp_buffer.data.reshape(conv_2d_weight_shape)
        else:
            # Add a Reshape before the weights tensor
            new_weights_tensor = self.builder.duplicate_tensor(weights_tensor, name_suffix="_4D_")
            new_weights_tensor.shape = tflite_model.Shape(conv_2d_weight_shape)

            weight_reshape = tflite_model.Operator(builtin_options=reshape_options.Reshape(conv_2d_weight_shape))
            weight_reshape.tmp_inputs = [weights_tensor]
            weight_reshape.tmp_outputs = [new_weights_tensor]

            pre_reshapes.append(weight_reshape)

            # Save the new weights tensor, to assign it later.
            weights_tensor = new_weights_tensor

        # -- Create the new operators --
        input_reshape = tflite_model.Operator(builtin_options=reshape_options.Reshape(conv_2d_input_shape))
        input_reshape.tmp_inputs = [input_reshape_x]
        input_reshape.tmp_outputs = [input_reshape_y]
        pre_reshapes.append(input_reshape)

        output_reshape = tflite_model.Operator(builtin_options=reshape_options.Reshape(output_reshape_y.shape.vector))
        output_reshape.tmp_inputs = [output_reshape_x]
        output_reshape.tmp_outputs = [output_reshape_y]

        # Assign the new input and output of the Conv2D
        t_op.tmp_inputs[0] = input_reshape_y
        t_op.tmp_inputs[weight_tensor_index] = weights_tensor
        t_op.tmp_outputs = [output_reshape_x]

        # Extend all ONNX attributes to 2D
        common.extend_1d_kernel_shape_to_2d(o_conv_attributes.kernel_shape)
        common.extend_1d_strides_to_2d(o_conv_attributes.strides)
        common.extend_1d_dilations_to_2d(o_conv_attributes.dilations)
        common.extend_1d_pads_to_2d(o_conv_attributes.pads)

        # Convert the now 2D Conv
        converted_conv_ops = self._convert_2d_q_linear_conv(o_conv_attributes, t_op)

        # Propagate inner (Conv) quantization parameters to outer tensors of added Reshapes
        for reshape in pre_reshapes:
            propagate_quantization(reshape.tmp_outputs[0], reshape.tmp_inputs[0])
        propagate_quantization(output_reshape_x, output_reshape_y)

        return pre_reshapes + converted_conv_ops + [output_reshape]

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
            return self._convert_1d_q_linear_conv(attrs, t_op)
        elif kernel_rank == 2:
            return self._convert_2d_q_linear_conv(attrs, t_op)
        else:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"Conversion of ONNX QLinearConv with '{kernel_rank}' dimensions is not possible!")
