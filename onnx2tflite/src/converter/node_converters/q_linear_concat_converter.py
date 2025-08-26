#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

import onnx2tflite.src.tflite_generator.builtin_options.concatenation_options as tflite_concatenation_options
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import (
    quantization_params_to_lists,
    re_quantize_static_tensor,
    set_quantization_parameters_to_tensor,
)
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes.q_linear_concat_attributes import QLinearConcat
from onnx2tflite.src.tflite_generator import tflite_model


class QLinearConcatConverter(NodeConverter):
    node = "QLinearConcat"

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX Runtime 'QLinearConcat' operator to TFLite 'Concat'."""
        attrs = cast(QLinearConcat, node.attributes)

        axis = attrs.axis
        rank = len(t_op.tmp_inputs[2].shape.vector)

        if axis < -rank or axis > rank - 1:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f"ONNX attribute 'axis' ({axis}) must be in range [{-rank}, {rank - 1}]!!")

        if axis < 0:  # convert negative index to positive
            axis += rank

        if t_op.tmp_inputs[2].tensor_format.is_channels_last():
            axis = translator.create_channels_last_to_channels_first_permutation(rank)[axis]

        output_tensor = t_op.tmp_outputs[0]
        output_scale = t_op.tmp_inputs[0].tmp_buffer.data
        output_zero_point = t_op.tmp_inputs[1].tmp_buffer.data

        # ONNX only supports INT8 and UINT8
        if output_tensor.type not in {TensorType.INT8, TensorType.UINT8}:
            logger.e(logger.Code.INVALID_TYPE,
                     f"ONNX QLinearConcat supports only INT8 and UINT8 data types. Got '{output_tensor.type}'.")

        # Input types match with output type
        io_types = [t.type for t in t_op.tmp_inputs[2::3]] + [output_tensor.type]
        if len(set(io_types)) != 1:
            logger.e(logger.Code.INVALID_TYPE,
                     "ONNX QLinearConcat at least one input type doesn't match with output type!")

        input_tensors = []
        ops_to_add = []

        for idx in range(2, len(t_op.tmp_inputs), 3):
            input_tensor = t_op.tmp_inputs[idx]

            input_scale = t_op.tmp_inputs[idx + 1].tmp_buffer.data
            input_zero_point = t_op.tmp_inputs[idx + 2].tmp_buffer.data

            output_scale_match = np.allclose(output_scale, input_scale)
            output_zero_point_match = np.allclose(output_zero_point, input_zero_point)
            is_int8_input = input_tensor.type == TensorType.INT8

            # TFLite doesn't support different quantization params for type INT8. Need to requantize.
            if is_int8_input and (not output_scale_match or not output_zero_point_match):
                output_scale_list, output_zero_point_list = quantization_params_to_lists(output_scale,
                                                                                         output_zero_point)

                if tensor_has_data(input_tensor):
                    set_quantization_parameters_to_tensor(input_tensor, input_scale, input_zero_point)
                    input_tensor = re_quantize_static_tensor(self.builder, input_tensor, TensorType.INT8,
                                                             output_scale_list, output_zero_point_list)

                    input_tensors.append(input_tensor)
                else:
                    # Prepend input with non-matching quant parameters by 'Quantize' operator
                    set_quantization_parameters_to_tensor(input_tensor, input_scale, input_zero_point)
                    quantize_op = self.builder.create_quantize_operator_before(
                        t_op, idx, TensorType.INT8,
                        new_input_scale=output_scale_list,
                        new_input_zero_point=output_zero_point_list)
                    ops_to_add.append(quantize_op)

                    input_tensors.append(quantize_op.tmp_outputs[0])
            else:
                set_quantization_parameters_to_tensor(input_tensor, input_scale, input_zero_point)
                input_tensors.append(input_tensor)

        set_quantization_parameters_to_tensor(output_tensor, output_scale, output_zero_point)

        concatenation_op = tflite_model.Operator(
            builtin_options=tflite_concatenation_options.Concatenation(axis)
        )

        concatenation_op.tmp_inputs = input_tensors
        concatenation_op.tmp_outputs = [output_tensor]

        return ops_to_add + [concatenation_op]
