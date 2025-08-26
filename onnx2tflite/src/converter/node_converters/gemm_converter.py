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
from onnx2tflite.src.converter.conversion.common import OpsList, try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import (
    get_symmetric_zero_point_for_type,
    quantization_params_to_lists,
    quantize_static_float_tensor,
    re_quantize_static_tensor,
)
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import gemm_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import add_options, fully_connected_options, mul_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class GemmConverter(NodeConverter):
    node = "Gemm"

    onnx_supported_types = FLOATS + [TensorType.INT32, TensorType.INT64, TensorType.UINT32, TensorType.UINT64]
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/fully_connected.cc#L171-L196
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT8, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX Gemm operator to TFLite FullyConnected.

        :param node: ONNX Gemm operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator
        :return: A list of TFLite operators, to add to the model.
        """
        if not (2 <= len(t_op.tmp_inputs) <= 3):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"ONNX Gemm has unexpected number of inputs ({len(t_op.tmp_inputs)}), instead of 2 - 3.")

        ops = OpsList(middle_op=t_op)

        a = t_op.tmp_inputs[0]
        b = t_op.tmp_inputs[1]

        if a.type != b.type:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX Gemm has mismatched input types.")

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(a.type)

        o_gemm = cast(gemm_attributes.Gemm, node.attributes)

        if o_gemm.transA:
            # Transpose A
            if transpose := self._transpose_input(t_op, 0):
                ops.add_pre(transpose)

        if not o_gemm.transB:
            # Transpose B
            if transpose := self._transpose_input(t_op, 1):
                ops.add_pre(transpose)

        self._handle_alpha_attribute(o_gemm.alpha, t_op, ops)

        self._handle_c_tensor(o_gemm, t_op, ops)

        w_quant = t_op.tmp_inputs[1].quantization
        if w_quant is not None and w_quant.is_per_channel():
            # ONNX `Gemm` always has the `N` dimension quantized, which will always be the dimension `0` of the TFLite
            #  `FullyConnected` weights. Set the weights `quantized_dimension` to `0` to reflect this (it could have
            #  been `1` in the ONNX model).
            w_quant.quantized_dimension = 0

        t_op.builtin_options = fully_connected_options.FullyConnected(keep_num_dims=True)

        return ops.flatten()

    def _handle_alpha_attribute(self, alpha: float, t_op: tflite_model.Operator, ops: OpsList) -> None:
        """Handle the conversion of the 'alpha' attribute of the ONNX Gemm operator.

            Gemm carries out the following operation:
                Y = alpha * A * B + C * beta

            A, B and C are tensors. alpha and beta are float scalars. In order to convert the operator to
             FullyConnected, one of A or B must be multiplied by alpha.

        :param t_op: TFLite operator representing the corresponding FullyConnected operator.
        """
        if alpha == 1.:
            return

        a = t_op.tmp_inputs[0]
        b = t_op.tmp_inputs[1]

        if t_op.tmp_inputs[0].type != TensorType.FLOAT32:
            if alpha < 0:
                # Rare scenario -> For negative alpha, a Mul op must be added, or dequantize the static data,
                # multiply by alpha and quantize it back.
                logger.e(logger.Code.NOT_IMPLEMENTED, "Quantized Gemm with negative alpha is currently not supported.")

            if tensor_has_data(a):
                a.quantization.scale = tflite_model.Scale([val * alpha for val in a.quantization.scale.vector])
            elif tensor_has_data(b):
                b.quantization.scale = tflite_model.Scale([val * alpha for val in b.quantization.scale.vector])
            else:
                # Insert a Mul operator because inputs are dynamic
                # Create alpha tensor and quantize it based on alpha value -> it will result in scale value
                # being quantized to "1", and no loss of information.
                alpha_data = np.array([alpha], np.float32)
                alpha_tensor = self.context.tflite_builder.create_tensor_for_data(alpha_data, "alpha")
                zp = [get_symmetric_zero_point_for_type(a.type)]
                alpha_tensor = quantize_static_float_tensor(self.builder, alpha_tensor, a.type, [alpha], zp)

                mul_output = self.context.tflite_builder.duplicate_tensor(a, name_suffix="_scaled")
                # Mul output tensor is quantized as 'alpha * scale' because we are effectively changing the range
                # (stretching or extending) we are quantizing.
                mul_output.quantization.scale = tflite_model.Scale(
                    [alpha * val for val in mul_output.quantization.scale.vector])

                mul_op = tflite_model.Operator(builtin_options=mul_options.Mul())
                mul_op.tmp_inputs = [a, alpha_tensor]
                mul_op.tmp_outputs = [mul_output]

                t_op.tmp_inputs[0] = mul_output

                ops.add_pre(mul_op)

            return

        # -- float Gemm --

        if tensor_has_data(a):
            a.tmp_buffer.data = alpha * a.tmp_buffer.data

        elif tensor_has_data(b):
            b.tmp_buffer.data = alpha * b.tmp_buffer.data

        else:
            # Insert a mul operator
            alpha_tensor = self.context.tflite_builder.create_tensor_for_data(np.array([alpha], np.float32), "alpha")
            mul_output = self.context.tflite_builder.duplicate_tensor(a, name_suffix="_scaled")

            mul_op = tflite_model.Operator(builtin_options=mul_options.Mul())
            mul_op.tmp_inputs = [a, alpha_tensor]
            mul_op.tmp_outputs = [mul_output]

            t_op.tmp_inputs[0] = mul_output

            ops.add_pre(mul_op)

    def _handle_c_tensor(self, o_gemm: gemm_attributes.Gemm, t_op: tflite_model.Operator, ops: OpsList) -> None:
        if (c := try_get_input(t_op, 2)) is None:
            return

        if o_gemm.beta == 0.0:
            t_op.tmp_inputs[2:] = []  # Remove the C from the FullyConnected.
            return

        if tensor_has_data(c) and all(el == 0 for el in c.tmp_buffer.data.flatten()):
            # The C is just adding 0 -> ignore it.
            t_op.tmp_inputs[2:] = []  # Remove the C from the FullyConnected.
            return

        if o_gemm.beta != 1.0:
            # Handle the Beta attribute -> multiply the bias by beta.

            if tensor_has_data(c):
                # Multiply the tensor statically.
                c.tmp_buffer.data = o_gemm.beta * c.tmp_buffer.data

            else:
                # Prepend a Mul operator.
                beta_tensor = self.context.tflite_builder.create_tensor_for_data(np.array([o_gemm.beta], np.float32),
                                                                                 "beta")
                mul_output = self.context.tflite_builder.duplicate_tensor(c, name_suffix="_scaled")

                mul_op = tflite_model.Operator(builtin_options=mul_options.Mul())
                mul_op.tmp_inputs = [c, beta_tensor]
                mul_op.tmp_outputs = [mul_output]

                t_op.tmp_inputs[2] = mul_output

                ops.add_pre(mul_op)

                c = mul_output  # Following code should work with the scaled c, instead of the original one.

        if c.quantization is not None:
            # If bias is quantized it must be static
            if not tensor_has_data(c):
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Bias tensor of quantized Gemm operator is not static. Behavior of QDQ Quantizer has changed.")

            # TFLite expects scale of bias to be scale = a_scale * b_scale. ONNX quantizer computes it
            # as scale=a_scale * b_scale * beta. Re-quantize bias tensor if alpha wasn't 1.0 and thus
            # not correspond to TFLite.
            x_quant = t_op.tmp_inputs[0].quantization
            w_quant = t_op.tmp_inputs[1].quantization
            bias_scale = np.array(w_quant.scale.vector, dtype=np.float32) * x_quant.scale[0]
            bias_zp = np.zeros(len(w_quant.zero_point.vector), dtype=np.int32)

            bias_scale_list, bias_zp_list = quantization_params_to_lists(bias_scale, bias_zp)

            if not np.allclose(bias_scale.astype(np.float32), np.array(c.quantization.scale.vector, dtype=np.float32)):
                logger.w("Re-quantizing bias tensor of Gemm operator to match TFLite's scale requirements. "
                         "This can introduce small inaccuracies.")
                c = re_quantize_static_tensor(self.builder, c, TensorType.INT32, bias_scale_list, bias_zp_list)
                t_op.tmp_inputs[2] = c

        # In ONNX, C must be broadcastable to the output (shape [M, N]). In TFLite however, C must have shape [N] or
        #  [1, N]. In case C has a different shape, it must be added via an Add operator.
        n = t_op.tmp_outputs[0].shape.get(1)
        if c.shape.vector == [n] or c.shape.vector == [1, n]:
            return

        # -- Add the C via a separate Add operator. --

        add_op = tflite_model.Operator(builtin_options=add_options.Add())

        old_output = t_op.tmp_outputs[0]
        new_output = self.context.tflite_builder.duplicate_tensor(old_output)

        t_op.tmp_inputs[2:] = []  # Remove the C from the FullyConnected.
        t_op.tmp_outputs = [new_output]

        add_op.tmp_inputs = [new_output, c]
        add_op.tmp_outputs = [old_output]

        if c.quantization is not None:
            # Do not try to add bias tensor via additional Add operator. This brings two problems:
            #  1. There is need to re-quantize bias tensor from INT32 to (U)INT8. This introduces some error.
            #  2. Addition of bias is not done within single accumulator of FullyConnected op. This introduces
            #     another error.
            # Combination of errors mentioned above leads to large degradation in accuracy.
            # Bias in shape different from [N] or [1, N] is also rare, so we let user improve the model.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "ONNX operator Gemm (QDQ quantized) has bias input ('C') with shape that is not "
                     "supported by TFLite. Make sure bias tensor has shape [N] or [1, N].")

        ops.add_post(add_op)

    def _transpose_input(self, t_op: tflite_model.Operator, input_idx: int) -> tflite_model.Operator | None:
        t = t_op.tmp_inputs[input_idx]

        if t.rank != 2:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"ONNX Gemm has a main intput with {t.rank} dimensions instead of 2.")

        if tensor_has_data(t):
            # Transpose statically.
            t = self.builder.duplicate_tensor(t)  # Duplicate in case the tensor is shared among operators.
            t_op.tmp_inputs[input_idx] = t

            t.tmp_buffer.data = np.transpose(t.tmp_buffer.data)
            t.shape = tflite_model.Shape(list(t.tmp_buffer.data.shape))

            return None
        else:
            # Dynamic tensor -> prepend a Transpose op.
            return self.context.tflite_builder.create_transpose_operator_before(t_op, input_idx, [1, 0])
