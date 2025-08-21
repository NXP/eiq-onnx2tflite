#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""convert_q_linear_mat_mul.

Convert ONNX operator QLinearMatMul to TFLite.
"""


import numpy as np

import onnx2tflite.src.tflite_generator.meta.types as tfl_types
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import (
    calculate_uint_to_int_re_quantization_zero_point,
    re_quantize_static_tensor,
    set_quantization_parameters_to_tensor,
)
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import batch_mat_mul_options, fully_connected_options


# noinspection PyMethodMayBeStatic
class QLinearMatMulConverter(NodeConverter):
    node = "QLinearMatMul"

    def _is_unsupported_type(self, data_type: TensorType) -> bool:
        """Determine if given TensorType is unsupported by the ONNX QLinearMatMul."""
        return data_type not in {
            TensorType.UINT8,
            TensorType.INT8
        }

    def _per_channel_zero_points_are_supported(self, data_type: TensorType, zero_point: np.ndarray) -> bool:
        if data_type == TensorType.INT8:
            return all(zp == 0 for zp in zero_point)

        if data_type == TensorType.UINT8:
            return all(zp == 128 for zp in zero_point)

        return False

    def _ensure_signed_quantized_input(self, input_tensor: tflite_model.Tensor, t_op: tflite_model.Operator,
                                       input_index: int, input_zero_point: np.ndarray,
                                       ops: OpsList) -> None:
        if input_tensor.type == TensorType.UINT8:
            if tensor_has_data(input_tensor):
                # Tensor can be re-quantized statically
                input_tensor = re_quantize_static_tensor(self.builder, input_tensor, TensorType.INT8)
                t_op.tmp_inputs[input_index] = input_tensor

            else:
                # Need to prepend a Quantize operator
                new_zero_point = calculate_uint_to_int_re_quantization_zero_point(1, input_zero_point)
                if new_zero_point.size == 1:
                    new_zero_point = [new_zero_point.item()]
                else:
                    new_zero_point = list(new_zero_point)
                quantize_op = self.context.tflite_builder.create_quantize_operator_before(t_op, input_index,
                                                                                          TensorType.INT8, None,
                                                                                          new_zero_point)
                ops.add_pre(quantize_op)

    def _at_least_1_is_none(self, *args) -> bool:
        """Determine if at least 1 of provided arguments is 'None'.

        :param args: Arguments to check for 'None'.
        :return: True or False
        """
        for arg in args:
            if arg is None:
                return True

        return False

    def _convert_per_channel_q_linear_mat_mul(self, node, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert an ONNX QLinearMatMul with per-channel quantized second input, to TFLite.

        :param t_op: TFLite operator with the same inputs and outputs as the ONNX QLinearMatMul operator,
        :return: A list of TFLite operators to be added to the TFLite model.
        """
        # BatchMatMul silently doesn't support per-channel quantization. The only option is to convert to
        # FullyConnected, which supports only 2D inputs and requires the second input to be transposed.

        # Prepare the input and output tensors.
        a, a_scale, a_zp = self._get_input_with_quant_params(node, t_op, 0)
        b, b_scale, b_zp = self._get_input_with_quant_params(node, t_op, 1)
        y, y_scale, y_zp = self._get_output_quant_params(t_op)

        max_rank = max(a.rank, b.rank, y.rank)
        if a.rank == 3 and y.rank == 3 and b.rank == 2:
            # Batched 2D
            pass
        elif max_rank != 2:
            # Only 2D QLinearMatMul can be converted because of FullyConnected limitation.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, f"ONNX QLinearMatMul with '{max_rank}' dimensions and "
                                                        f"per-channel quantization cannot be converted to TFLite! "
                                                        f"Only 2D QLinearMatMul is supported.")

        if any(t.tensor_format.is_channels_last() for t in [a, b]):
            # This should be an extremely rare case since the inputs are almost always 2D (formatless).
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX `QLinearMatMul` or quantized `MatMul` with per-channel quantization and "
                     "channels first input is not yet supported.")

        # FullyConnected silently only supports 0 zero points for int8 per-channel quantization
        if not self._per_channel_zero_points_are_supported(b.type, b_zp):
            data_type_name = tfl_types.name_for_type(b.type)
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, f"ONNX QLinearMatMul with second input zero point "
                                                        f"'{b_zp}' and type '{data_type_name}' cannot be "
                                                        f"converted to TFLite!")

        # ONNXRT: Despite the documentation, ONNX Runtime only allows per-channel quantization for the last dimension.
        # In TFLite, per-channel quantization is only supported for 2D tensors, so the last dimension has index 1.
        b_quantized_dimension = 1
        size_of_quantized_dimension = b.shape.get(b_quantized_dimension)
        if size_of_quantized_dimension != 1 and size_of_quantized_dimension != b_scale.size:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     "ONNX 2D QLinearMatMul uses per-channel quantization, but the size of "
                     "the quantization parameters doesn't match dimension 1. This is unexpected.")

        # Add the quantization parameters to the tensors
        if a.quantization is None:
            set_quantization_parameters_to_tensor(a, a_scale, a_zp)
        if b.quantization is None:
            set_quantization_parameters_to_tensor(b, b_scale, b_zp, b_quantized_dimension)
        if y.quantization is None:
            set_quantization_parameters_to_tensor(y, y_scale, y_zp)

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [a, b]
        t_op.tmp_outputs = [y]

        t_op.builtin_options = fully_connected_options.FullyConnected(keep_num_dims=True)

        ops = OpsList(middle_op=t_op)

        # Transpose and re-quantize the second input tensor
        if tensor_has_data(b):
            # Static tensor

            # Transpose the tensor statically
            t_op.tmp_inputs[1] = self.context.tflite_builder.create_transposed_tensor(b)

            # The quantized dimension of the new transposed tensor is now 0
            t_op.tmp_inputs[1].quantization.quantized_dimension = 0

            if b.type == TensorType.UINT8:
                # Re-quantize the tensor statically
                t_op.tmp_inputs[1] = re_quantize_static_tensor(self.builder, t_op.tmp_inputs[1], TensorType.INT8)

        else:
            # Dynamic tensor

            if b.type == TensorType.UINT8:
                # b must be re-quantized dynamically, but the TFLite Quantize operator silently doesn't support
                # UINT8 to INT8 per-channel re-quantization
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE, "Conversion of QLinearMatMul with a dynamic second input "
                                                            "with per-channel quantization is not possible!")

            # A Transpose operator must be added
            transpose = self.context.tflite_builder.create_transpose_operator_before(t_op, 1, [1, 0])

            # The quantized dimension of the new transposed tensor is now 0
            transpose.tmp_outputs[0].quantization.quantized_dimension = 0

            ops.add_pre(transpose)

        # TFLite FullyConnected doesn't support UINT8 -> re-quantize to INT8
        self._ensure_signed_quantized_input(a, t_op, 0, a_zp, ops)
        if y.type == TensorType.UINT8:
            new_zero_output_point = calculate_uint_to_int_re_quantization_zero_point(1, y_zp)
            if new_zero_output_point.size == 1:
                new_zero_output_point = [new_zero_output_point.item()]
            else:
                new_zero_output_point = list(new_zero_output_point)
            ops.add_post(
                self.context.tflite_builder.create_quantize_operator_after(t_op, 0, TensorType.INT8, None,
                                                                           new_zero_output_point))

        return ops.flatten()

    def _handle_tensor_formats(self, t_op: tflite_model.Operator, ops: OpsList) -> None:
        def process_input_tensor(input_index: int) -> None:
            tensor = t_op.tmp_inputs[input_index]
            if not tensor.tensor_format.is_channels_last():
                return

            permutation = list(translator.create_channels_last_to_channels_first_permutation(tensor.rank))

            if tensor_has_data(tensor):
                tensor = self.builder.duplicate_tensor(tensor)
                translator.permute_static_tensor(tensor, permutation)
                t_op.tmp_inputs[input_index] = tensor

            else:
                transpose_op = self.builder.create_transpose_operator_before(t_op, input_index, permutation)
                ops.pre_ops.append(transpose_op)

            t_op.tmp_inputs[input_index].tensor_format = TensorFormat.CHANNELS_FIRST

        process_input_tensor(0)
        process_input_tensor(1)

        y = t_op.tmp_outputs[0]
        if y.tensor_format.is_channels_last():
            to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(y.rank, True)
            post_transpose = self.context.tflite_builder.create_transpose_operator_after(t_op, 0, to_tflite_perm)
            ops.add_post(post_transpose)

    def _convert_per_tensor_q_linear_mat_mul(self, node, t_op) -> list[tflite_model.Operator]:
        """Convert the ONNX QLinearMatMul operator to TFLite BatchMatMul.

        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator
        :return: A list of TFLite operators, to add to the model.
        """
        a, a_scale, a_zp = self._get_input_with_quant_params(node, t_op, 0)
        b, b_scale, b_zp = self._get_input_with_quant_params(node, t_op, 1)
        y, y_scale, y_zp = self._get_output_quant_params(t_op)

        # Add the quantization parameters to the tensors
        if a.quantization is None:
            set_quantization_parameters_to_tensor(a, a_scale, a_zp)
        if b.quantization is None:
            set_quantization_parameters_to_tensor(b, b_scale, b_zp)
        if y.quantization is None:
            set_quantization_parameters_to_tensor(y, y_scale, y_zp)

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [a, b]
        t_op.tmp_outputs = [y]
        ops = OpsList(middle_op=t_op)

        self._handle_tensor_formats(t_op, ops)

        # `FullyConnected` requires `weights` zero point == 0 according to LiteRT documentation
        #  (https://ai.google.dev/edge/litert/models/quantization_spec). But if the weights are dynamic, any zero point
        #  is supported for some reason (experimentally verified).
        # In the case of static `weights`, either convert to `BatchMatMul` (currently implemented) or prepend a
        #  `Quantize` operator to change the zero point (which would decrease accuracy).
        zero_point_works_with_fully_connected = (
                (not tensor_has_data(b)) or
                (b.type == TensorType.INT8 and b.quantization.zero_point[0] == 0) or
                (b.type == TensorType.UINT8 and b.quantization.zero_point[0] == 128)
        )
        if a.rank in (2, 3) and b.rank == 2 and zero_point_works_with_fully_connected:
            # Convert to `FullyConnected`.
            t_op.builtin_options = fully_connected_options.FullyConnected(keep_num_dims=True)

            # The second input has to be transposed for `FullyConnected`.
            if tensor_has_data(b):
                b = self.builder.create_transposed_tensor(b)
                t_op.tmp_inputs[1] = b
            else:
                ops.add_pre(self.context.tflite_builder.create_transpose_operator_before(t_op, 1, [1, 0]))
                b = t_op.tmp_inputs[1]

        else:
            # Convert to `BatchMatMul`.
            t_op.builtin_options = batch_mat_mul_options.BatchMatMul(False, False, False)

        # LiteRT `BatchMatMul` doesn't support `uint8` and `FullyConnected` can sometimes produce incorrect output if it
        #  runs in `uint8`.
        # Re-quantize the inputs and outputs to `int8`.
        self._ensure_signed_quantized_input(a, t_op, 0, a_zp, ops)
        self._ensure_signed_quantized_input(b, t_op, 1, b_zp, ops)

        if y.type == TensorType.UINT8:
            new_zero_output_point = calculate_uint_to_int_re_quantization_zero_point(1, y_zp)
            if new_zero_output_point.size == 1:
                new_zero_output_point = [new_zero_output_point.item()]
            else:
                new_zero_output_point = list(new_zero_output_point)
            ops.add_post(
                self.context.tflite_builder.create_quantize_operator_after(t_op, 0, TensorType.INT8, None,
                                                                           new_zero_output_point))

        return ops.flatten()

    def _get_input_with_quant_params(self, node: onnx_model.NodeProto, t_op, input_idx
                                     ) -> tuple[tflite_model.Tensor, np.ndarray, np.ndarray]:
        """Get input tensor and corresponding q-params from other inputs (QLinearMatMul) or
        directly from tensor's quantization property.

        :param t_op: QLinearMatMul or (U)INT8 MatMul NodeProto.
        :param node: TFLite operator with inputs and outputs corresponding to the ONNX operator
        :param input_idx: Input tensor index - 0 or 1.
        :return: Tuple with input tensor, quant. scale and quant. zero point
        """
        input_mapping = {
            "QLinearMatMul": [0, 3],
            "MatMul": [0, 1],
        }

        tensor_idx = input_mapping[node.op_type][input_idx]
        x = t_op.tmp_inputs[tensor_idx]

        if x.quantization is None:
            x_scale = t_op.tmp_inputs[tensor_idx + 1].tmp_buffer.data
            x_zero_point = t_op.tmp_inputs[tensor_idx + 2].tmp_buffer.data
        else:
            x_scale = np.array(x.quantization.scale.vector)
            x_zero_point = np.array(x.quantization.zero_point.vector)

        return x, x_scale, x_zero_point

    def _get_output_quant_params(self, t_op) -> tuple[tflite_model.Tensor, np.ndarray, np.ndarray]:
        """Get output tensor and corresponding q-params from other inputs (QLinearMatMul) or
        directly from tensor's quantization property.

        :param t_op: QLinearMatMul or (U)INT8 MatMul NodeProto.
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

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX QLinearMatMul or (U)INT8 MatMul operator to TFLite FullyConnected|BatchMatMul.

        :param node: QLinearMatMul or (U)INT8 MatMul NodeProto.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator
        :return: A list of TFLite operators, to add to the model.
        """
        # Prepare the input and output tensors.
        a, a_scale, a_zp = self._get_input_with_quant_params(node, t_op, 0)
        b, b_scale, b_zp = self._get_input_with_quant_params(node, t_op, 1)
        y, y_scale, y_zp = self._get_output_quant_params(t_op)

        # Quantization parameters must be static
        if self._at_least_1_is_none(a_scale, a_zp, b_scale, b_zp, y_scale, y_zp):
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, "Conversion of ONNX QLinearMatMul with dynamic quantization "
                                                        "parameters is not possible!")

        # ONNX only supports 8bit inputs and output
        if (self._is_unsupported_type(a.type) or
                self._is_unsupported_type(b.type) or
                self._is_unsupported_type(y.type)):
            logger.e(logger.Code.INVALID_TYPE, "ONNX QLinearMatMul only supports INT8 and UINT8 datatypes!")

        # 1D MatMul
        if a.rank == b.rank == 1:
            # TODO Surround by Reshape ops if this use case is realistic
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX QLinearMatMul with 1D inputs is not yet implemented.")

        # ONNXRT: ONNX Runtime only supports per-channel quantization for b
        # (onnxruntime/core/providers/cpu/quantization/quantize_linear_matmul.cc Line ~55)
        if b_scale.size == 1:
            return self._convert_per_tensor_q_linear_mat_mul(node, t_op)
        if b_scale.size > 1:
            # Per channel quantization
            return self._convert_per_channel_q_linear_mat_mul(node, t_op)
        logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX QLinearMatMul has no quantization parameters!")
