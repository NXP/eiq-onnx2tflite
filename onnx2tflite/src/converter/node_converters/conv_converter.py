#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import common, translator
from onnx2tflite.src.converter.conversion.common import try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.node_converters.q_linear_conv_converter import QLinearConvConverter
from onnx2tflite.src.converter.node_converters.shared import conv_utils
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import conv_attributes
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import (
    conv_2d_options,
    conv_3d_options,
    depthwise_conv_2d_options,
    reshape_options,
)
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, name_for_type


# noinspection PyPep8Naming
class ConvConverter(NodeConverter):
    node = "Conv"

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/conv.cc#L359-L361
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT8, TensorType.INT16, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def _convert_1d_conv(self, o_conv_attributes: conv_attributes.Conv,
                         t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'Conv' operator with a 1D kernel to TFLite 'Conv2D'.
        TFLite doesn't support 1D convolution, but this behaviour can be represented using
               Reshape -> Conv2D -> Reshape.
        The first reshape introduces a 4th dimension with size 1. The second Reshape removes the temporary dimension.

        :param o_conv_attributes: Attributes of the ONNX Conv operator.
        :param t_op: A TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        # -- Calculate the shapes for equivalent 2D convolution --
        conv_2d_input_shape = translator.nhc_dimensions_to_nhwc(t_op.tmp_inputs[0].shape.vector)
        conv_2d_weight_shape = translator.nhc_dimensions_to_nhwc(t_op.tmp_inputs[1].shape.vector)
        conv_2d_output_shape = translator.nhc_dimensions_to_nhwc(t_op.tmp_outputs[0].shape.vector)

        # -- Generate tensors taking part in the conversion --
        reshape1_input = t_op.tmp_inputs[0]

        reshape1_output = self.builder.duplicate_tensor(reshape1_input, name_suffix="_4D_")
        reshape1_output.shape = tflite_model.Shape(conv_2d_input_shape)

        reshape2_input = self.builder.duplicate_tensor(t_op.tmp_outputs[0], name_suffix="_4D_")
        reshape2_input.shape = tflite_model.Shape(conv_2d_output_shape)

        reshape2_output = t_op.tmp_outputs[0]

        pre_reshapes = []

        # Extend the weights tensor to 4D
        weights_tensor = t_op.tmp_inputs[1]
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
        reshape1 = tflite_model.Operator(builtin_options=reshape_options.Reshape(conv_2d_input_shape))
        reshape1.tmp_inputs = [reshape1_input]
        reshape1.tmp_outputs = [reshape1_output]
        pre_reshapes.append(reshape1)

        reshape2 = tflite_model.Operator(builtin_options=reshape_options.Reshape(reshape2_output.shape.vector))
        reshape2.tmp_inputs = [reshape2_input]
        reshape2.tmp_outputs = [reshape2_output]

        # Assign the new input and output of the Conv2D
        t_op.tmp_inputs = [reshape1_output, weights_tensor] + t_op.tmp_inputs[2:]  # Add bias as well, if present
        t_op.tmp_outputs = [reshape2_input]

        # Extend all ONNX attributes to 2D
        common.extend_1d_kernel_shape_to_2d(o_conv_attributes.kernel_shape)
        common.extend_1d_strides_to_2d(o_conv_attributes.strides)
        common.extend_1d_dilations_to_2d(o_conv_attributes.dilations)
        common.extend_1d_pads_to_2d(o_conv_attributes.pads)

        # Convert the now 2D Conv
        converted_conv_ops = self._convert_2d_conv(o_conv_attributes, t_op)

        return pre_reshapes + converted_conv_ops + [reshape2]

    def _convert_unpadded_2D(self, q_linear_conv, t_op) -> conv_utils.ConvConversionResult:
        # Prepare the input and output tensors. To replace them, assign to t_op.tmp_inputs/tmp_outputs directly.
        output_tensor = t_op.tmp_outputs[0]
        input_tensor = t_op.tmp_inputs[0]
        weight_tensor = t_op.tmp_inputs[1]

        if (bias_tensor := try_get_input(t_op, 2)) is None:
            # Operator has no bias. ONNX model can omit it, TFLite can't.
            output_channels = weight_tensor.shape.vector[0]
            bias_type = translator.tf_lite_type_to_numpy(weight_tensor.type)
            bias_tensor = self.builder.create_zeros_tensor([output_channels], "zero_bias", bias_type, True)

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [input_tensor, weight_tensor, bias_tensor]
        t_op.tmp_outputs = [output_tensor]

        conversion_result = conv_utils.ConvConversionResult(input_tensor, weight_tensor, bias_tensor, output_tensor)
        conversion_result.ops_list.middle_op = t_op

        # Convert the builtin options
        common.assign_2d_strides(t_op.builtin_options, q_linear_conv.strides)
        common.assign_2d_dilations(t_op.builtin_options, q_linear_conv.dilations)

        return conversion_result

    def _convert_2d_conv(self, conv_attributes: conv_attributes.Conv,
                         t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        if conv_utils.group_conv_convertible_as_depthwise(conv_attributes, t_op, weight_tensor_index=1):
            t_op.builtin_options = depthwise_conv_2d_options.DepthwiseConv2D()

            conversion_result = self._convert_unpadded_2D(conv_attributes, t_op)
            padding, pad_op = conv_utils.build_input_tensor_padding(conv_attributes, t_op, self.builder)
            t_op.builtin_options.padding = padding

            if pad_op is not None:
                conversion_result.ops_list.add_pre(pad_op)

            # DepthwiseConv2D expects weights in format [kernel_channels, kernel_height, kernel_width, output_channels]
            perm = [3, 1, 2, 0]
            weight_tensor = conversion_result.conv_weight_tensor

            if tensor_has_data(weight_tensor):
                # Transpose cloned tensor statically
                t_op.tmp_inputs[1] = self.builder.create_transposed_tensor(weight_tensor, perm)
            else:
                # Insert a Transpose operator
                transpose_op = self.builder.create_transpose_operator_before(t_op, 1, perm)
                conversion_result.ops_list.add_pre(transpose_op)

            return conversion_result.ops_list.flatten()

        if conv_utils.group_conv_convertible_into_multiple_convolutions(conv_attributes, t_op):
            t_op.builtin_options = conv_2d_options.Conv2D()

            return conv_utils.create_separated_convolutions_based_on_group(
                conv_attributes, t_op, self.builder, self._convert_unpadded_2D, conv_utils.conv_op_factory, 0)

        t_op.builtin_options = conv_2d_options.Conv2D()
        conversion_result = self._convert_unpadded_2D(conv_attributes, t_op)
        padding, pad_op = conv_utils.build_input_tensor_padding(conv_attributes, t_op, self.builder)
        t_op.builtin_options.padding = padding

        if pad_op is not None:
            conversion_result.ops_list.add_pre(pad_op)

        return conversion_result.ops_list.flatten()

    def _convert_unpadded_3D(self, o_conv_attributes, t_op) -> conv_utils.ConvConversionResult:
        """Convert the ONNX 'Conv' operator with a 3D kernel to TFLite 'Conv3D' without padding.

        TFLite Conv2D uses the [output_channels, H, W, input_channels] format for its weights.
        TFLite Conv3D is different, for some reason it needs the weights to be in the format [D, H, W, in_c, out_c].
        ONNX 3D Conv uses [out_c, in_c, D, H, W] which the converter would normally change to [out_c, D, H, W, in_c].
        Therefore, we need to move the fist dimension to the end explicitly.

        :param o_conv_attributes: Attributes of the ONNX Conv operator.
        :param t_op: A TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: ConvConversionResult with added ops and references to IO tensors.
        """
        t_conv = t_op.builtin_options
        output_tensor = t_op.tmp_outputs[0]
        input_tensor = t_op.tmp_inputs[0]
        weight_tensor = t_op.tmp_inputs[1]

        if (bias_tensor := try_get_input(t_op, 2)) is None:
            # Operator has no bias. ONNX model can omit it, TFLite can't.
            t_op.tmp_inputs[2:] = []  # Remove the bias, if it was passed with name "".
            output_channels = weight_tensor.shape.vector[0]
            bias_type = translator.tf_lite_type_to_numpy(weight_tensor.type)
            bias_tensor = self.builder.create_zeros_tensor([output_channels], "zero_bias", bias_type, True)

        if o_conv_attributes.strides is None:
            # Default strides
            t_conv.stride_d = t_conv.stride_h = t_conv.stride_w = 1
        elif len(o_conv_attributes.strides) == 3:
            t_conv.stride_d, t_conv.stride_h, t_conv.stride_w = o_conv_attributes.strides
        else:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX Conv with a 3D kernel doesn't have 3D strides!")

        if o_conv_attributes.dilations is None:
            # Default dilations
            t_conv.dilation_d_factor = t_conv.dilation_h_factor = t_conv.dilation_w_factor = 1
        elif len(o_conv_attributes.dilations) == 3:
            t_conv.dilation_d_factor, t_conv.dilation_h_factor, t_conv.dilation_w_factor = o_conv_attributes.dilations
        else:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     "ONNX Conv with a 3D kernel doesn't have 3D dilations!")

        conversion_result = conv_utils.ConvConversionResult(input_tensor, weight_tensor, bias_tensor, output_tensor)
        conversion_result.ops_list.middle_op = t_op

        # Move the first dimension of the weights tensor to the end, just like described in the function description.
        if tensor_has_data(t_op.tmp_inputs[1]):
            # Transpose the tensor statically
            weight_tensor.tmp_buffer.data = np.moveaxis(weight_tensor.tmp_buffer.data, 0, -1)
            weight_tensor.shape.vector.append(weight_tensor.shape.vector.pop(0))
        else:
            # Insert a Transpose operator
            perm = [1, 2, 3, 4, 0]  # First dimension becomes last.
            transpose_op = self.builder.create_transpose_operator_before(t_op, 1, perm)
            conversion_result.conv_weight_tensor = transpose_op.tmp_outputs[0]
            conversion_result.ops_list.add_post(transpose_op)

        # The Conv3D weights tensor uses its own special format: [output_channels, input_channels, D, H, W]
        t_op.tmp_inputs[1].tensor_format = TensorFormat.CONV_3D_WEIGHT_FORMAT

        return conversion_result

    def _convert_3d_conv(self, o_conv_attributes: conv_attributes.Conv,
                         t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        t_op.builtin_options = conv_3d_options.Conv3D()

        if conv_utils.group_conv_convertible_into_multiple_convolutions(o_conv_attributes, t_op):
            return conv_utils.create_separated_convolutions_based_on_group(
                o_conv_attributes, t_op, self.builder, self._convert_unpadded_3D, conv_utils.conv_op_factory, 4)
        if o_conv_attributes.group != 1:
            logger.e(logger.Code.NOT_IMPLEMENTED, "ONNX Conv with a 3D kernel and unsupported 'group' value!")

        conversion_result = self._convert_unpadded_3D(o_conv_attributes, t_op)
        padding, pad_op = conv_utils.build_input_tensor_padding(o_conv_attributes, t_op, self.builder)
        t_op.builtin_options.padding = padding

        if pad_op is not None:
            conversion_result.ops_list.add_pre(pad_op)

        return conversion_result.ops_list.flatten()

    def convert(self, conv_node: onnx_model.NodeProto,
                t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'Conv' operator to TFLite 'Conv2D' and potential 'Reshape' operators.

        :param conv_node: ONNX Conv node.
        :param t_op: A TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        o_conv_attributes = cast(conv_attributes.Conv, conv_node.attributes)

        if not (2 <= len(t_op.tmp_inputs) <= 3):
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX Conv has invalid number of inputs. Got '{len(t_op.tmp_inputs)}', expected 2 or 3.")

        input_tensor = t_op.tmp_inputs[0]
        weight_tensor = t_op.tmp_inputs[1]

        if o_conv_attributes.kernel_shape is None:
            o_conv_attributes.kernel_shape = translator.infer_kernel_shape(weight_tensor)
        elif o_conv_attributes.kernel_shape != translator.infer_kernel_shape(weight_tensor):
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     "Weight tensor shape not corresponds to kernel_shape attribute")

        if t_op.is_quantized_without_qdq():
            # Not supported by ONNX. Keep this check just in case.
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX `Conv` has quantized inputs.")

        if input_tensor.quantization is None:
            self.assert_type_allowed(input_tensor.type)

        else:
            # Only INT8 and UINT8 quantization is supported.
            if input_tensor.type not in [TensorType.INT8, TensorType.UINT8]:
                logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX `Conv` quantized with type "
                                                      f"`{name_for_type(input_tensor.type)}` is not supported.")

            # (U)INT8 Conv is basically QLinearConv -> use already prepared conversion code
            return QLinearConvConverter(self.context).convert(conv_node, t_op)

        kernel_rank = len(o_conv_attributes.kernel_shape)

        if kernel_rank == 1:
            return self._convert_1d_conv(o_conv_attributes, t_op)

        if kernel_rank == 2:
            return self._convert_2d_conv(o_conv_attributes, t_op)

        if kernel_rank == 3:
            return self._convert_3d_conv(o_conv_attributes, t_op)

        logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                 f"Conversion of ONNX Conv with a {kernel_rank}D kernel is not possible!")
