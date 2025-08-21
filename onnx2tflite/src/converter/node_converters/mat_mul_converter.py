#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.node_converters.q_linear_mat_mul_converter import QLinearMatMulConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import batch_mat_mul_options, fully_connected_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class MatMulConverter(NodeConverter):
    node = "MatMul"

    onnx_supported_types = FLOATS + [TensorType.INT32, TensorType.INT64, TensorType.UINT32, TensorType.UINT64]
    # `tflite_supported_types` depend on which TFLite op the `MatMul` gets converted to.
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/batch_matmul.cc#L777-L804
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/fully_connected.cc#L1562-L1605
    # Either way, the only overlapping type is float32.
    verified_types = [TensorType.FLOAT32]

    # noinspection PyPep8Naming
    def _convert_formatless_1D(self, t_op) -> OpsList:  # noqa: N802
        """Convert 'MatMul' operator with formatless input tensor when there's exactly one 1D input tensor.

        1D input is not supported by TFLite. We have to prepend input with 'Reshape' and append
        'Reshape' to output as well to remove unwanted dimension with value '1'.
        If the first input is 1D (x), it is promoted to a matrix by prepending a 1 to its dimensions.
        If the second input is 2D (y), it is promoted to a matrix by appending a 1 to its dimensions.
        """
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_inputs[1]
        x_rank = len(x.shape.vector)
        y_rank = len(y.shape.vector)

        ops = OpsList(middle_op=t_op)

        assert not (x_rank == 1 and y_rank == 1), "Both inputs are 1D. This should be prohibited!"

        if x_rank == 1:
            if tensor_has_data(x):
                # 1D tensor is static -> just change a shape
                x.shape = tflite_model.Shape([1, x.shape.vector[0]])
            else:
                # Reshape 1D tensor to 2D with prepended '1'
                ops.pre_ops.append(self.context.tflite_builder.create_reshape_before(t_op, 0, [1, x.shape.vector[0]]))

            # Remove penultimate dim (lost during matrix multiplication) and reshape
            reshape_output_shape = y.shape.vector.copy()
            reshape_output_shape.pop(-2)
            ops.post_ops.append(self.context.tflite_builder.create_reshape_after(t_op, 0, reshape_output_shape))

            # TFLite doesn't remove artificial '1' dim as ONNX do. Add it to output tensor of BatchMatMul operator.
            matmul_output_shape = reshape_output_shape.copy()
            matmul_output_shape.insert(-1, 1)
            t_op.tmp_outputs[0].shape = tflite_model.Shape(matmul_output_shape)

            t_op.builtin_options = batch_mat_mul_options.BatchMatMul(False, False, False)

        if y_rank == 1:
            if tensor_has_data(y):
                # 1D tensor is static -> just change a shape
                y.shape = tflite_model.Shape([1, y.shape.vector[0]])
            else:
                # Reshape 1D tensor to 2D with appended '1'
                ops.pre_ops.append(self.context.tflite_builder.create_reshape_before(t_op, 1, [1, y.shape.vector[0]]))

            # Remove last dim (lost during matrix multiplication) and reshape
            reshape_output_shape = x.shape.vector.copy()[:-1]
            ops.post_ops.append(self.context.tflite_builder.create_reshape_after(t_op, 0, reshape_output_shape))

            # TFLite doesn't remove artificial '1' dim as ONNX do. Add it to output tensor of BatchMatMul operator.
            t_op.tmp_outputs[0].shape = tflite_model.Shape(reshape_output_shape + [1])

            t_op.builtin_options = fully_connected_options.FullyConnected(keep_num_dims=True)

        return ops

    # noinspection PyPep8Naming
    def _convert_formatless_2D(self, t_op) -> OpsList:  # noqa: N802
        """Convert 'MatMul' with formatless input tensors where 'y' tensor is 2D and 'x' is in range <2D, 5D>.
        """
        y = t_op.tmp_inputs[1]

        ops = OpsList(middle_op=t_op)

        # Switch dimensions to conform FC specification
        if tensor_has_data(y):
            t_op.tmp_inputs[1] = self.builder.create_transposed_tensor(y, [1, 0])
        else:
            ops.pre_ops.append(self.context.tflite_builder.create_transpose_operator_before(t_op, 1, [1, 0]))

        t_op.builtin_options = fully_connected_options.FullyConnected(keep_num_dims=True)

        return ops

    # noinspection PyPep8Naming
    def _convert_formatless_ND(self, t_op) -> OpsList:  # noqa: N802
        """Convert 'MatMul' with formatless input tensors of rank 2 up to 5. Broadcasting is supported out of box.
        """
        t_op.builtin_options = batch_mat_mul_options.BatchMatMul(False, False, False)

        return OpsList(middle_op=t_op)

    # noinspection PyPep8Naming
    def _convert_channel_last_1D(self, t_op: tflite_model.Operator) -> OpsList:  # noqa: N802
        """Convert 'MatMul' with channel first input tensors with exactly one 1D input tensor.
        Non-1D tensor is prepended by 'Transpose' operator to ensure matrix multiplication is computed correctly.
        Similarly, 'Transpose' operator is appended to the end of the chained operators.
        """
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_inputs[1]
        x_rank = len(t_op.tmp_inputs[0].shape.vector)
        y_rank = len(t_op.tmp_inputs[1].shape.vector)

        def _wrap_in_transpose(tensor, tensor_rank, input_index) -> OpsList:
            ops = OpsList(middle_op=t_op)

            if tensor_has_data(tensor) and tensor.tensor_format.is_channels_last():
                t_op.tmp_inputs[input_index] = self._build_transposed_static_tensor(tensor, tensor_rank)
            elif tensor.tensor_format.is_channels_last():
                permutation = list(translator.create_channels_last_to_channels_first_permutation(tensor_rank))
                transpose_op = self.context.tflite_builder.create_transpose_operator_before(t_op, input_index,
                                                                                            permutation)

                ops.pre_ops.append(transpose_op)

            wrapped_ops = self._convert_formatless_1D(t_op)
            ops.pre_ops.extend(wrapped_ops.pre_ops)
            ops.post_ops.extend(wrapped_ops.post_ops)

            last_op = wrapped_ops.post_ops[0] if len(wrapped_ops.post_ops) > 0 else t_op

            output_rank = len(last_op.tmp_outputs[0].shape.vector)
            post_permutation = list(translator.create_channels_first_to_channels_last_permutation(output_rank))
            post_transpose = self.context.tflite_builder.create_transpose_operator_after(last_op, 0,
                                                                                         post_permutation)
            ops.post_ops.append(post_transpose)

            return ops

        if x_rank != 1:
            return _wrap_in_transpose(x, x_rank, input_index=0)

        if y_rank != 1:
            return _wrap_in_transpose(y, y_rank, input_index=1)

    # noinspection PyPep8Naming
    def _convert_channel_last_2D(self, t_op: tflite_model.Operator) -> OpsList:  # noqa: N802
        """Convert 'MatMul' operator with 'x' input in channel last format and 'y' as a 2D tensor. Static
        input tensors with data are transposed. Dynamic input tensor are prepended with 'Transpose' operator.
        """
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_inputs[1]
        x_rank = len(x.shape.vector)
        output_rank = len(t_op.tmp_outputs[0].shape.vector)

        ops = OpsList(middle_op=t_op)

        if tensor_has_data(x) and x.tensor_format.is_channels_last():
            t_op.tmp_inputs[0] = self._build_transposed_static_tensor(x, x_rank)
        elif x.tensor_format.is_channels_last():
            permutation = list(translator.create_channels_last_to_channels_first_permutation(x_rank))
            transpose_op = self.context.tflite_builder.create_transpose_operator_before(t_op, 0, permutation)
            ops.pre_ops.append(transpose_op)

        # Switch dimensions to conform FC specification
        if tensor_has_data(y):
            t_op.tmp_inputs[1] = self.builder.create_transposed_tensor(y, [1, 0])
        else:
            ops.pre_ops.append(self.context.tflite_builder.create_transpose_operator_before(t_op, 1, [1, 0]))

        post_permutation = list(translator.create_channels_first_to_channels_last_permutation(output_rank))
        post_transpose = self.context.tflite_builder.create_transpose_operator_after(t_op, 0, post_permutation)
        ops.post_ops.append(post_transpose)

        t_op.builtin_options = fully_connected_options.FullyConnected(keep_num_dims=True)

        return ops

    # noinspection PyPep8Naming
    def _convert_channel_last_ND(self, t_op: tflite_model.Operator) -> OpsList:  # noqa: N802
        """Convert 'MatMul' operator with at least one input in channel last format and zero 1D inputs. Static
        input tensors with data are transposed. Dynamic input tensor are prepended with 'Transpose' operator.
        """
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_inputs[1]
        x_rank = len(x.shape.vector)
        y_rank = len(y.shape.vector)
        output_rank = len(t_op.tmp_outputs[0].shape.vector)

        ops = OpsList(middle_op=t_op)

        def process_input_tensor(tensor, tensor_rank, input_index) -> None:
            if tensor_has_data(tensor) and tensor.tensor_format.is_channels_last():
                t_op.tmp_inputs[input_index] = self._build_transposed_static_tensor(tensor, tensor_rank)
            elif tensor.tensor_format.is_channels_last():
                permutation = list(translator.create_channels_last_to_channels_first_permutation(tensor_rank))
                transpose_op = self.context.tflite_builder.create_transpose_operator_before(t_op, input_index,
                                                                                            permutation)
                ops.pre_ops.append(transpose_op)

        process_input_tensor(x, x_rank, 0)
        process_input_tensor(y, y_rank, 1)

        post_permutation = list(translator.create_channels_first_to_channels_last_permutation(output_rank))
        post_transpose = self.context.tflite_builder.create_transpose_operator_after(t_op, 0, post_permutation)
        ops.post_ops.append(post_transpose)

        t_op.builtin_options = batch_mat_mul_options.BatchMatMul(False, False, False)

        return ops

    def _build_transposed_static_tensor(self, tensor, tensor_rank) -> tflite_model.Tensor:
        permutation = list(translator.create_channels_last_to_channels_first_permutation(tensor_rank))

        new_tensor = self.context.tflite_builder.duplicate_tensor(tensor, tensor.name + "_transposed")
        new_tensor.tmp_buffer.data = np.transpose(new_tensor.tmp_buffer.data, permutation)
        new_tensor.shape = tflite_model.Shape(list(new_tensor.tmp_buffer.data.shape))

        return new_tensor

    def convert(self, o_matmul: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'MatMul' operator to TFLite 'BatchMatMul' or 'FullyConnected'.

        Conversion mapping:
        +-------------++-----+-----+-----+
        |             ||    Input rank   |
        +-------------++-----+-----+-----+
        | Output rank ||  1D |  2D |  ND |
        +-------------++-----+-----+-----+
        |     1D      ||  -  |  FC |  FC |
        +-------------++-----+-----+-----+
        |     2D      || BMM |  FC |  FC |
        +-------------++-----+-----+-----+
        |     ND      || BMM | BMM | BMM |
        +-------------++-----+-----+-----+
        FC = FullyConnected, BMM = BatchMatMul

        :param o_matmul: MatMul NodeProto.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator
        :return: A list of TFLite operators, to add to the model.
        """
        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, f"ONNX operator 'MatMul' has '{len(t_op.tmp_inputs)}' inputs!")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_inputs[1]
        x_rank = len(x.shape.vector)
        y_rank = len(y.shape.vector)

        if x_rank > 5:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"Input of 'MatMul' must have rank in range <1,5>. Got: {x_rank}!")

        if y_rank > 5:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"Input of 'MatMul' must have rank in range <1,5>. Got: {y_rank}!")

        # Rare scenario - both inputs 1D - ignore for now
        if x_rank == 1 and y_rank == 1:
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Both inputs of 'MatMul' has 1D rank what is currently not supported!")

        if t_op.is_qdq_quantized():
            # (U)INT8 MatMul is basically QLinearMatMul -> use already prepared conversion code
            return QLinearMatMulConverter(self.context).convert(o_matmul, t_op)

        if x.type != y.type:
            # If the `MatMul` is QDQ quantized, the inputs *can* have different types (int8 and uint8). This is
            #  because in the ONNX model, the `MatMul` is in float32 and surrounded with `QuantizeLinear` and
            #  `DequantizeLinear`, which convert between float32 and (u)int8. Our approach to QDQ conversion removes
            #  these extra ops and makes the `MatMul` inputs (u)int8, which can cause the type mismatch.
            # If the `MatMul` is not QDQ quantized, the input types must be the same.
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     "ONNX operator 'MatMul' has inputs with different data types!")
        self.assert_type_allowed(x.type)

        is_channels_last = x.tensor_format.is_channels_last() or y.tensor_format.is_channels_last()

        if is_channels_last:
            if x_rank == 1 or y_rank == 1:
                return self._convert_channel_last_1D(t_op).flatten()
            if y_rank == 2:
                return self._convert_channel_last_2D(t_op).flatten()
            return self._convert_channel_last_ND(t_op).flatten()
        if x_rank == 1 or y_rank == 1:
            return self._convert_formatless_1D(t_op).flatten()
        if y_rank == 2:
            return self._convert_formatless_2D(t_op).flatten()
        return self._convert_formatless_ND(t_op).flatten()
