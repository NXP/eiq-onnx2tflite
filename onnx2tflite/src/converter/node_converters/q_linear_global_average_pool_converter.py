#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np

import onnx2tflite.lib.tflite.Padding as tflPadding
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.quantization_utils import set_quantization_parameters_to_tensor
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator.builtin_options import average_pool_2d_options, reshape_options


class QLinearGlobalAveragePoolConverter(NodeConverter):
    node = 'QLinearGlobalAveragePool'

    def _convert_2d_q_linear_global_average_pool(self, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the 2D ONNX Runtime QLinearGlobalAveragePool operator to TFLite `AveragePool2D`. """

        # Prepare the input and output tensors.
        output_tensor = t_op.tmp_outputs[0]
        input_tensor = t_op.tmp_inputs[0]
        input_scale_tensor = t_op.tmp_inputs[1]
        input_zero_point_tensor = t_op.tmp_inputs[2]
        output_scale_tensor = t_op.tmp_inputs[3]
        output_zero_point_tensor = t_op.tmp_inputs[4]

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [input_tensor]
        t_op.tmp_outputs = [output_tensor]

        # # Add the quantization parameters to the tensors
        input_scale = input_scale_tensor.tmp_buffer.data
        input_zero_point = input_zero_point_tensor.tmp_buffer.data
        set_quantization_parameters_to_tensor(input_tensor, input_scale, input_zero_point)

        output_scale = output_scale_tensor.tmp_buffer.data
        output_zero_point = output_zero_point_tensor.tmp_buffer.data
        set_quantization_parameters_to_tensor(output_tensor, output_scale, output_zero_point)

        # Input and output types must be the same
        if input_tensor.type != output_tensor.type:
            logger.e(logger.Code.INVALID_TYPE, f"ONNX QLinearGlobalAveragePool input and output tensors must have the "
                                               f"same data type! '{input_tensor.type}' != '{output_tensor.type}'.")

        # ONNX only supports INT8 and UINT8
        if input_tensor.type not in {TensorType.INT8, TensorType.UINT8}:
            logger.e(logger.Code.INVALID_TYPE,
                     f"ONNX QLinearGlobalAveragePool supports only INT8 and UINT8 data types. Got '{input_tensor.type}'.")

        input_height = input_tensor.shape.get(1)
        input_width = input_tensor.shape.get(2)

        # Create the AveragePool2D
        t_op.builtin_options = average_pool_2d_options.AveragePool2D(
            tflPadding.Padding.VALID,
            stride_w=1, stride_h=1,
            filter_w=input_width, filter_h=input_height
        )

        ops_to_return = [t_op]

        # TFLite AveragePool2D silently doesn't support different input and output quantization parameters
        if np.not_equal(input_scale, output_scale) or np.not_equal(input_zero_point, output_zero_point):
            # Add a Quantize operator, to re-quantize the output tensor.

            new_output_scale = [input_scale.item()]
            new_output_zero_point = [input_zero_point.item()]
            quantize = self.builder.create_quantize_operator_after(t_op, 0, output_tensor.type, new_output_scale,
                                                                   new_output_zero_point)
            ops_to_return.append(quantize)

        return ops_to_return

    def _convert_q_linear_global_average_pool_and_surround_with_reshapes(
            self, t_op: tflite_model.Operator, reshape_1_output_shape: list[int],
            reshape_2_input_shape: list[int], reshape_2_output_shape: list[int]) -> list[tflite_model.Operator]:
        """ Convert an ONNX QLinearGlobalAveragePool to TFLite AveragePool2D and surround it with 2 Reshape operators.
            The Reshape operators will change the shape to 4D, according to input parameters.

        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator
        :param reshape_1_output_shape: The shape of the output tensor of the first Reshape operator.
        :param reshape_2_input_shape: The shape of the input tensor of the second Reshape operator.
        :param reshape_2_output_shape: The shape of the output tensor of the first Reshape operator.
        :return: A list of TFLite operators, to add to the model.
        """
        original_input_tensor = t_op.tmp_inputs[0]
        original_output_tensor = t_op.tmp_outputs[0]

        # Create new tensors for the QLinearGlobalAveragePool operator
        new_input_tensor = self.builder.duplicate_tensor(original_input_tensor, "QLinearGlobalAveragePool_input")
        new_input_tensor.shape = tflite_model.Shape(reshape_1_output_shape)

        new_output_tensor = self.builder.duplicate_tensor(original_output_tensor, "QLinearGlobalAveragePool_output")
        new_output_tensor.shape = tflite_model.Shape(reshape_2_input_shape)

        t_op.tmp_inputs[0] = new_input_tensor
        t_op.tmp_outputs[0] = new_output_tensor

        # Convert the '2D' QLinearGlobalAveragePool
        converted_operators = self._convert_2d_q_linear_global_average_pool(t_op)

        # Add a Reshape operator, to reshape the input of the first operator to 4 dimensions (2D kernel)
        reshape_1 = tflite_model.Operator(
            builtin_options=reshape_options.Reshape(reshape_1_output_shape)
        )
        reshape_1.tmp_inputs = [original_input_tensor]
        reshape_1.tmp_outputs = [converted_operators[0].tmp_inputs[0]]

        # Add a Reshape operator, to reshape the output of the last operator back from 4 dimensions
        reshape_2 = tflite_model.Operator(
            builtin_options=reshape_options.Reshape(reshape_2_output_shape)
        )
        reshape_2.tmp_inputs = [converted_operators[-1].tmp_outputs[0]]
        reshape_2.tmp_outputs = [original_output_tensor]

        return [reshape_1] + converted_operators + [reshape_2]

    def _convert_more_than_2d_q_linear_global_average_pool(self, t_op: tflite_model.Operator
                                                           ) -> list[tflite_model.Operator]:
        """ Convert the ONNX Runtime QLinearGlobalAveragePool operator with a kernel of at least 3 dimensions, to TFLite
             `AveragePool2D` and extra `Reshape` operators.
        """

        input_rank = t_op.tmp_inputs[0].rank
        if input_rank < 5:
            logger.e(logger.Code.INTERNAL_ERROR,
                     f"Input tensor has only '{input_rank}' dimensions! Expected at least 5.")

        original_input_tensor = t_op.tmp_inputs[0]
        original_output_tensor = t_op.tmp_outputs[0]

        # Add the quantization parameters to the input and output tensors
        input_scale = t_op.tmp_inputs[1].tmp_buffer.data
        input_zero_point = t_op.tmp_inputs[2].tmp_buffer.data
        set_quantization_parameters_to_tensor(original_input_tensor, input_scale, input_zero_point)

        output_scale = t_op.tmp_inputs[3].tmp_buffer.data
        output_zero_point = t_op.tmp_inputs[4].tmp_buffer.data
        set_quantization_parameters_to_tensor(original_output_tensor, output_scale, output_zero_point)

        # Calculate the shapes, that the Reshape operators will use
        reshape_1_output_shape = original_input_tensor.shape.vector.copy()
        reshape_1_output_shape[2:-1] = [np.prod(reshape_1_output_shape[2:-1]).item()]

        reshape_2_input_shape = original_input_tensor.shape.vector.copy()
        reshape_2_input_shape[1:-1] = [1] * (input_rank - 3)

        reshape_2_output_shape = original_input_tensor.shape.vector.copy()
        reshape_2_output_shape[1:-1] = [1] * (input_rank - 2)

        return self._convert_q_linear_global_average_pool_and_surround_with_reshapes(t_op,
                                                                                     reshape_1_output_shape,
                                                                                     reshape_2_input_shape,
                                                                                     reshape_2_output_shape)

    def _convert_1d_q_linear_global_average_pool(self, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX Runtime QLinearGlobalAveragePool operator with a 1D kernel, to TFLite `AveragePool2D` and
             extra `Reshape` operators.
        """

        input_rank = t_op.tmp_inputs[0].rank
        if input_rank != 3:
            logger.e(logger.Code.INTERNAL_ERROR, f"Input tensor has '{input_rank}' dimensions! Expected 3.")

        original_input_tensor = t_op.tmp_inputs[0]
        original_output_tensor = t_op.tmp_outputs[0]

        # Add the quantization parameters to the input and output tensors
        input_scale = t_op.tmp_inputs[1].tmp_buffer.data
        input_zero_point = t_op.tmp_inputs[2].tmp_buffer.data
        set_quantization_parameters_to_tensor(original_input_tensor, input_scale, input_zero_point)

        output_scale = t_op.tmp_inputs[3].tmp_buffer.data
        output_zero_point = t_op.tmp_inputs[4].tmp_buffer.data
        set_quantization_parameters_to_tensor(original_output_tensor, output_scale, output_zero_point)

        # Calculate the shapes, that the Reshape operators will use
        reshape_1_output_shape = original_input_tensor.shape.vector.copy()
        reshape_1_output_shape.insert(2, 1)

        reshape_2_input_shape = reshape_1_output_shape.copy()
        reshape_2_input_shape[1] = 1

        reshape_2_output_shape = original_input_tensor.shape.vector.copy()
        reshape_2_output_shape[1] = 1

        return self._convert_q_linear_global_average_pool_and_surround_with_reshapes(t_op,
                                                                                     reshape_1_output_shape,
                                                                                     reshape_2_input_shape,
                                                                                     reshape_2_output_shape)

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX Runtime QLinearGlobalAveragePool operator to TFLite `AveragePool2D`. """

        rank = t_op.tmp_inputs[0].rank

        if rank == 3:
            # 1D kernel
            return self._convert_1d_q_linear_global_average_pool(t_op)

        elif rank == 4:
            # 2D kernel
            return self._convert_2d_q_linear_global_average_pool(t_op)

        elif rank >= 5:
            # kernel with at least 3 dimensions
            return self._convert_more_than_2d_q_linear_global_average_pool(t_op)

        else:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, f"Conversion of ONNX QLinearGlobalAveragePool with '{rank}' "
                                                        f"dimensions is not possible!")
