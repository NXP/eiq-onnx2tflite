#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from collections.abc import Callable
from copy import copy
from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.Padding import Padding
from onnx2tflite.src import logger
from onnx2tflite.src.converter.builder.model_builder import ModelBuilder
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser.builtin_attributes import conv_attributes, q_linear_conv_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import (
    concatenation_options,
    conv_2d_options,
    conv_3d_options,
    split_options,
)

ConvAttributes = conv_attributes.Conv | q_linear_conv_attributes.QLinearConv
TFTensor = tflite_model.Tensor
TFOperator = tflite_model.Operator


def group_conv_convertible_as_depthwise(o_conv_attributes: ConvAttributes, t_op: TFOperator,
                                        weight_tensor_index: int) -> bool:
    """Check whether provided Conv/QLinearConv ONNX operator could be converted into TFLite DepthwiseConv2D.

    :param o_conv_attributes: ONNX Conv/QLinearConv operator attributes.
    :param t_op: A TFLite operator with inputs and outputs corresponding to the ONNX operator.
    :param weight_tensor_index: Index of weight tensor within operator inputs ('1' for Conv, '3' for QLinearConv).
    :return: True if operator can be converted into DepthwiseConv2D.
    """
    input_channels = t_op.tmp_inputs[0].shape.vector[-1]
    output_channels = t_op.tmp_inputs[weight_tensor_index].shape.vector[0]

    return input_channels == output_channels == o_conv_attributes.group


def group_conv_convertible_into_multiple_convolutions(o_conv_attributes: ConvAttributes, t_op: TFOperator) -> bool:
    """Check whether provided Conv/QLinearConv ONNX operator with group > 1 could be converted into
    multiple non-group Conv2D/Conv3D operator.

    :param o_conv_attributes: ONNX Conv/QLinearConv operator attributes.
    :param t_op: A TFLite operator with inputs and outputs corresponding to the ONNX operator.
    :return: True if operator can be converted into multiple non-group Conv2D/Conv3D operators.
    """
    if isinstance(o_conv_attributes, q_linear_conv_attributes.QLinearConv):
        weight_tensor = t_op.tmp_inputs[3]
        if weight_tensor.shape.vector[0] % o_conv_attributes.group != 0:
            logger.d("Unable to split group QLinearConv into separated convolutions because out_channels % group != 0.")
            return False

    elif isinstance(o_conv_attributes, conv_attributes.Conv):
        weight_tensor = t_op.tmp_inputs[1]

        if weight_tensor.shape.vector[0] % o_conv_attributes.group != 0:
            logger.d("Unable to split group Conv into separated convolutions because out_channels % group != 0.")
            return False

    if o_conv_attributes.group == 1:
        return False

    is_supported_group_count = 2 <= o_conv_attributes.group <= 10
    if not is_supported_group_count:
        logger.d("Attribute 'group' of Conv operator not in range 2 <= 'group' <= 10. Splitting not performed.")

    return is_supported_group_count


class ConvConversionResult:
    """Holds references to the direct I/O tensors of the Conv operator
    and list of surrounding operators (Quantize, Transpose, etc.).
    """

    def __init__(self, input_tensor: TFTensor, weight_tensor: TFTensor,
                 bias_tensor: TFTensor, output_tensor: TFTensor):
        self.conv_input_tensor = input_tensor
        self.conv_weight_tensor = weight_tensor
        self.conv_bias_tensor = bias_tensor
        self.conv_output_tensor = output_tensor
        self.ops_list = OpsList()


ConvBuiltinOptions = conv_2d_options.Conv2D | conv_3d_options.Conv3D
ConvOpFactory = Callable[
    [ConvAttributes, TFTensor, TFTensor, TFTensor, TFTensor, ModelBuilder, ConvBuiltinOptions], OpsList
]
ConvConversionFn = Callable[[ConvAttributes, TFOperator], ConvConversionResult]


class _InputTensorsSplitter:
    input_tensors: list[TFTensor]
    weight_tensors: list[TFTensor]
    bias_tensors: list[TFTensor]
    split_ops: list[TFOperator]

    def __init__(self, input_tensor: TFTensor, weight_tensor: TFTensor, bias_tensor: TFTensor, groups: int,
                 builder: ModelBuilder, weight_out_channels_idx: int):
        self.input_tensors = []
        self.weight_tensors = []
        self.bias_tensors = []
        self.split_ops = []

        inputs = [
            # input tensor, split by axis, output tensors container
            (input_tensor, -1, self.input_tensors),
            (weight_tensor, weight_out_channels_idx, self.weight_tensors),
            (bias_tensor, 0, self.bias_tensors),
        ]

        for i in inputs:
            if tensor_has_data(i[0]):
                self._generate_static_tensors(builder, groups, i[0], i[1], i[2])
            else:
                self._generate_dynamic_tensors(builder, groups, i[0], i[1], i[2])

    def _generate_dynamic_tensors(self, builder, groups, split_tensor, axis, target_list):
        quantization = None
        if split_tensor.quantization is not None:
            if split_tensor.quantization.is_per_channel():
                scale = np.split(np.array(split_tensor.quantization.scale.vector, "float32"), groups)
                zero_point = np.split(np.array(split_tensor.quantization.zero_point.vector, "int32"), groups)
                quantization = [
                    tflite_model.Quantization(scale=tflite_model.Scale(s), zero_point=tflite_model.ZeroPoint(zp))
                    for s, zp in zip(scale, zero_point, strict=False)
                ]
            else:
                quantization = [split_tensor.quantization] * groups

        split_op = self._create_split_op(builder, groups, split_tensor, axis)

        new_tensor_shape = split_tensor.shape.vector.copy()
        new_tensor_shape[axis] = new_tensor_shape[axis] // groups

        for i in range(groups):
            conv_split_tensor = builder.duplicate_tensor(split_tensor, name_suffix="_group_" + str(i))
            conv_split_tensor.shape = tflite_model.Shape(new_tensor_shape)
            if quantization is not None:
                conv_split_tensor.quantization = copy(quantization[i])

            split_op.tmp_outputs.append(conv_split_tensor)
            target_list.append(conv_split_tensor)
        self.split_ops.append(split_op)

    def _generate_static_tensors(self, builder, groups, split_tensor, axis, target_list):
        quantization = None
        if split_tensor.quantization is not None:
            if split_tensor.quantization.is_per_channel():
                scale = np.split(np.array(split_tensor.quantization.scale.vector, "float32"), groups)
                zero_point = np.split(np.array(split_tensor.quantization.zero_point.vector, "int32"), groups)
                quantization = [
                    tflite_model.Quantization(scale=tflite_model.Scale(s), zero_point=tflite_model.ZeroPoint(zp))
                    for s, zp in zip(scale, zero_point, strict=False)
                ]
            else:
                quantization = [split_tensor.quantization] * groups

        input_data = np.split(split_tensor.tmp_buffer.data, groups, axis)

        for i in range(len(input_data)):
            tensor_name = split_tensor.name + "_group_" + str(i)
            conv_input_tensor = builder.create_tensor_for_data(input_data[i], tensor_name)
            if quantization is not None:
                conv_input_tensor.quantization = copy(quantization[i])

            target_list.append(conv_input_tensor)

    def _create_split_op(self, builder, groups, input_tensor, axis):
        axis_tensor = builder.create_tensor_for_data(np.asarray([axis], np.int32), "split_dim_")
        input_split_op = TFOperator(builtin_options=split_options.Split(groups))
        input_split_op.tmp_inputs = [axis_tensor, input_tensor]

        return input_split_op

    def get_input_tensor(self, idx) -> TFTensor:
        return self.input_tensors[idx]

    def get_weight_tensor(self, idx) -> TFTensor:
        return self.weight_tensors[idx]

    def get_bias_tensor(self, idx) -> TFTensor:
        return self.bias_tensors[idx]

    def get_ops(self) -> list[TFOperator]:
        return self.split_ops


class _OutputTensorsCombiner:
    """Handles creation and aggregation of the TFLite Conv2D/Conv3D output tensors.
    Aggregation is done with 'Concatenation' op.
    """

    output_tensors: list[TFTensor]
    concat_op: TFOperator

    def __init__(self, output_tensor, groups, builder):
        self.output_tensors = []
        combine_axis = -1

        new_conv_output_shape = output_tensor.shape.vector.copy()
        new_conv_output_shape[combine_axis] = new_conv_output_shape[combine_axis] // groups
        conv_output_shape = tflite_model.Shape(new_conv_output_shape)

        self.concat_op = tflite_model.Operator(builtin_options=concatenation_options.Concatenation(combine_axis))
        self.concat_op.tmp_outputs = [output_tensor]

        for i in range(groups):
            tensor_name = output_tensor.name + "_group_" + str(i)
            output_tensor = builder.duplicate_tensor(output_tensor, tensor_name)
            output_tensor.shape = conv_output_shape

            self.output_tensors.append(output_tensor)
            self.concat_op.tmp_inputs.append(output_tensor)

    def get_output_tensor(self, idx):
        return self.output_tensors[idx]

    def get_ops(self):
        return [self.concat_op]


def build_input_tensor_padding(conv_attributes, t_op, builder, input_idx=0) -> tuple[
    Padding, (tflite_model.Operator | None)]:
    """Build padding for input tensor of Conv/QLinearConv op 't_op'.

    :param conv_attributes: Attributes of converted ONNX 'Conv' operator.
    :param t_op: A TFLite 'ConvXD' operator.
    :param builder: ModelBuilder object.
    :param input_idx: Padded input tensor index.
    :return: Tuple with Padding object and optional 'Pad' operator that should be prepended before 't_op' by caller.
    """
    padding, explicit_padding = translator.convert_padding(
        conv_attributes.auto_pad,
        conv_attributes.pads,
        t_op.tmp_inputs[input_idx].shape.vector,
        t_op.tmp_outputs[0].shape.vector,
        conv_attributes.kernel_shape,
        conv_attributes.strides,
        conv_attributes.dilations)

    if explicit_padding is not None:
        # Must add extra 'Pad' operator
        return padding, builder.create_pad_operator_before(t_op, input_idx, explicit_padding)

    return padding, None


def conv_op_factory(o_conv_attributes, input_tensor: tflite_model.Tensor,
                    weight_tensor: tflite_model.Tensor, bias_tensor: tflite_model.Tensor,
                    output_tensor: tflite_model.Tensor, builder,
                    builtin_options) -> OpsList:
    """Build padded 'Conv(2D|3D)' TFLite operator. Padding is realized by 'builtin_options.padding'
    definition and by optional prepended 'Pad' operator.

    :param o_conv_attributes: Attributes attached to converted ONNX 'Conv' operator.
    :param input_tensor: 'Conv(2D|3D)' input (x) tensor.
    :param weight_tensor: 'Conv(2D|3D)' Weight tensor.
    :param bias_tensor: 'Conv(2D|3D)' bias tensor.
    :param output_tensor: 'Conv(2D|3D)' output tensor.
    :param builder: ModelBuilder object.
    :param builtin_options: Returned 'Conv(2D|3D)' op builtin options.
    :return: OpsList with definition of 'Conv(2D|3D)' operator and optional 'Pad' operator (input tensor padding).
    """
    conv_op = tflite_model.Operator(builtin_options=copy(builtin_options))
    conv_op.tmp_inputs = [input_tensor, weight_tensor, bias_tensor]
    conv_op.tmp_outputs = [output_tensor]

    padding, pad_op = build_input_tensor_padding(o_conv_attributes, conv_op, builder)
    conv_op.builtin_options.padding = padding

    if pad_op is not None:
        return OpsList(pre_ops=[pad_op], middle_op=conv_op)
    return OpsList(middle_op=conv_op)


def create_separated_convolutions_based_on_group(o_conv_attributes: ConvAttributes, t_op: TFOperator,
                                                 builder: ModelBuilder,
                                                 conv_conversion_fn: ConvConversionFn,
                                                 conv_op_factory: ConvOpFactory,
                                                 weight_out_channels_idx: int) -> list[TFOperator]:
    """Build subgraph with multiple TFLite Conv2D/Conv3D operators that replace Conv/QLinearConv
    ONNX operator with 'group' attribute higher than one. Number of new Conv2D/Conv3D operators
    correspond to number of groups. Input tensors of ONNX operator are split and distributed
    into related convolution operators. Outputs are then concatenated back.

    Example: 'Conv' ONNX operator with group=2 converted into TFLite subgraph will have
    the following structure (tensor dimensions are just for illustrative purposes):


                  │ (1,4,4,48)
              ┌───▼──┐
              │Split │
              └┬────┬┘
    (1,4,4,24) │    │ (1,4,4,24)
         ┌─────▼┐  ┌▼─────┐
         │Conv2D│  │Conv2D│
         └────┬─┘  └─┬────┘
    (1,4,4,18)│      │(1,4,4,18)
            ┌─▼──────▼──┐
            │Concatenate│
            └─────┬─────┘
                  │ (1,4,4,36)
                  ▼

    :param o_conv_attributes: ONNX Conv/QLinearConv operator attributes.
    :param t_op: A TFLite operator with inputs and outputs corresponding to the ONNX operator.
    :param builder: ModelBuilder instance.
    :param conv_conversion_fn: Function for conversion of Conv/QLinearConv ONNX op into its unpadded
        TFLite variant. It returns conversion metadata (IO tensors and all generated ops).
    :param conv_op_factory: Factory function for creation of Conv2D/Conv3D TFLite operator based
        on provided IO tensors, attributes nad builtin_options.
    :param weight_out_channels_idx: Index of weight tensor within operator's 'tmp_inputs'.
    :return: List of TFLite operators representing subgraph.
    """
    groups = o_conv_attributes.group
    conversion_result = conv_conversion_fn(o_conv_attributes, t_op)

    splitter = _InputTensorsSplitter(conversion_result.conv_input_tensor, conversion_result.conv_weight_tensor,
                                     conversion_result.conv_bias_tensor, groups, builder, weight_out_channels_idx)
    combiner = _OutputTensorsCombiner(conversion_result.conv_output_tensor, groups, builder)

    conv_ops = []
    for i in range(groups):
        input_tensor = splitter.get_input_tensor(i)
        weight_tensor = splitter.get_weight_tensor(i)
        bias_tensor = splitter.get_bias_tensor(i)
        output_tensor = combiner.get_output_tensor(i)

        conv_builtin_options = cast(ConvBuiltinOptions, conversion_result.ops_list.middle_op.builtin_options)
        conv_ops_list = conv_op_factory(o_conv_attributes, input_tensor, weight_tensor, bias_tensor, output_tensor,
                                        builder, conv_builtin_options)

        conv_ops.extend(conv_ops_list.flatten())

    return (conversion_result.ops_list.pre_ops +  # Transpose, Quantize ops
            splitter.get_ops() + conv_ops + combiner.get_ops() +  # Split, Conv2D/Conv3D, Pad, Concatenate ops
            conversion_result.ops_list.post_ops)  # Transpose, Quantize ops
