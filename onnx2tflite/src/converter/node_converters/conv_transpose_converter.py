#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.onnx_model as onnx_model
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.conversion import translator, common
from onnx2tflite.src.converter.conversion.common import OpsList, try_get_input
from onnx2tflite.src.converter.conversion.translator import convert_padding
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser.builtin_attributes import conv_transpose_attributes
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator.builtin_options import transpose_conv_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, name_for_type

ConvAttributes = conv_transpose_attributes.ConvTranspose


# noinspection PyMethodMayBeStatic
class ConvTransposeConverter(NodeConverter):
    node = 'ConvTranspose'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/transpose_conv.cc#L274-L276
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT8, TensorType.INT16, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def _compute_slicing_params(self, output_shape, explicit_padding) -> tuple[list[int], list[int]]:
        begins = []
        sizes = []

        for axis in range(len(output_shape)):
            (start, end) = explicit_padding[axis]

            begins.append(start)
            sizes.append(output_shape[axis] - start - end)

        return begins, sizes

    def _convert_2d_conv(self, o_conv_attributes: ConvAttributes,
                         t_op: tflite_model.Operator) -> list[tflite_model.Operator]:

        if o_conv_attributes.dilations is not None and any(dilation != 1 for dilation in o_conv_attributes.dilations):
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"ONNX ConvTranspose with 'dilations' other than '1' cannot be converted to TFLite.")

        if o_conv_attributes.group is not None and o_conv_attributes.group != 1:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"ONNX ConvTranspose with 'group' other than '1' cannot be converted to TFLite.")

        # Prepare the input and output tensors. To replace them, assign to t_op.tmp_inputs/tmp_outputs directly.
        input_tensor = t_op.tmp_inputs[0]
        weight_tensor = t_op.tmp_inputs[1]
        output_tensor = t_op.tmp_outputs[0]
        bias_tensor = try_get_input(t_op, 2)

        if weight_tensor.tensor_format.is_channels_last():
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX 'ConvTranspose' with dynamic weight originating in Conv/Pool-like layer "
                     "is currently not implemented.")

        # Compute padding from output to input, because TransposeConv is up-sampling
        padding, explicit_padding = convert_padding(o_conv_attributes.auto_pad, o_conv_attributes.pads,
                                                    t_op.tmp_outputs[0].shape.vector,
                                                    t_op.tmp_inputs[0].shape.vector,
                                                    o_conv_attributes.kernel_shape, o_conv_attributes.strides)

        output_shape_tensor_data = np.asarray(output_tensor.shape.vector, dtype=np.int32)
        output_shape_tensor = self.builder.create_tensor_for_data(output_shape_tensor_data, "output_shape")

        t_op.builtin_options = transpose_conv_options.TransposeConv()
        t_op.tmp_inputs = [output_shape_tensor, weight_tensor, input_tensor]
        t_op.tmp_outputs = [output_tensor]
        t_op.builtin_options.padding = padding
        common.assign_2d_strides(t_op.builtin_options, o_conv_attributes.strides)

        if bias_tensor is not None:
            bias_shape = bias_tensor.shape.vector
            if len(bias_shape) > 1 or bias_shape[0] != weight_tensor.shape.vector[1]:
                logger.e(logger.Code.INVALID_ONNX_MODEL, "Bias tensor is not 1D or its length doesn't equal to "
                                                         "the length of feature maps (weights_tensor.shape[1] or 'M').")

            t_op.tmp_inputs.append(bias_tensor)

        ops = OpsList(middle_op=t_op)

        # Weight tensor format in ONNX: [C, M/group, kH, kW]
        # Weight tensor format in TFLite: [M/group, kH, kW, C]
        # (C = input channels, M = feature maps (output channels), kW = kernel width, kH = kernel height)
        if tensor_has_data(weight_tensor):
            translator.permute_static_tensor(weight_tensor, [1, 2, 3, 0])
        else:
            ops.add_post(self.builder.create_transpose_operator_before(t_op, 1, [1, 2, 3, 0]))
        weight_tensor.tensor_format = TensorFormat.TRANSPOSE_CONV_2D_WEIGHT_FORMAT

        if explicit_padding:
            # Add padding to output shape to make sure we have computed all the data we need
            for idx, padding in enumerate(explicit_padding):
                output_shape_tensor_data[idx] += padding[0] + padding[1]
            output_tensor.shape = tflite_model.Shape(output_shape_tensor_data.tolist())

            # We need to "cut" produced tensor by size of explicit padding
            begins, sizes = self._compute_slicing_params(output_shape_tensor_data.tolist(), explicit_padding)
            ops.add_post(self.builder.create_slice_after(t_op, 0, begins, sizes))

        return ops.flatten()

    def _verify_kernel_shape_attribute(self, o_conv_attributes, t_op):
        weight_tensor = t_op.tmp_inputs[1]
        derived_kernel_shape = weight_tensor.shape.vector[2:]

        if o_conv_attributes.kernel_shape is None:
            o_conv_attributes.kernel_shape = derived_kernel_shape
        elif o_conv_attributes.kernel_shape != derived_kernel_shape:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     "Weight tensor shape not corresponds to kernel_shape attribute")

    def convert(self, conv_node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        o_conv_attributes = cast(ConvAttributes, conv_node.attributes)

        if not (2 <= len(t_op.tmp_inputs) <= 3):
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX ConvTranspose has invalid number of inputs. Got "
                                                     f"'{len(t_op.tmp_inputs)}', expected 2 or 3.")

        if t_op.is_quantized_without_qdq():
            # This isn't supported by ONNX. Keep this check just in case.
            logger.e(logger.Code.INVALID_ONNX_MODEL, 'ONNX `ConvTranspose` with a quantized input is not supported.')

        x = t_op.tmp_inputs[0]
        if x.quantization is None:
            self.assert_type_allowed(x.type)

        elif x.type not in [TensorType.INT8]:
            logger.e(logger.Code.NOT_IMPLEMENTED, 'Conversion of ONNX `ConvTranspose` with quantized input of '
                                                  f'type {name_for_type(x.type)} is not supported.')

        self._verify_kernel_shape_attribute(o_conv_attributes, t_op)
        kernel_rank = len(o_conv_attributes.kernel_shape)

        if kernel_rank == 1 or kernel_rank == 3:
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     f"Conversion of ONNX ConvTranspose with a {kernel_rank}D kernel is not implemented.")
        elif kernel_rank == 2:
            return self._convert_2d_conv(o_conv_attributes, t_op)
        else:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"Conversion of ONNX ConvTranspose with a {kernel_rank}D kernel is not possible!")
