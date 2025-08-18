#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import List, cast

import numpy as np

import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.quantization_utils import get_symmetric_zero_point_for_type, \
    quantize_static_float_tensor
from onnx2tflite.src.converter.quantization_utils import set_quantization_parameters_to_tensor
from onnx2tflite.src.converter.conversion.common import try_get_input, OpsList, exactly_one_is_none
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data, all_tensors_are_static
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import q_gemm_attributes
from onnx2tflite.src.tflite_generator.builtin_options import fully_connected_options
from onnx2tflite.src.tflite_generator.builtin_options import mul_options


# noinspection PyMethodMayBeStatic
class QGemmConverter(NodeConverter):
    node = 'QGemm'

    def _parse_quantization_parameters(self, q_gemm_attrs: q_gemm_attributes.QGemm,
                                       t_op: tflite_model.Operator) -> list[np.ndarray | None]:
        """ Parse the quantization parameters of the 't_op', which represents a QGemm operator. Make sure the parameters
             are valid and convertible. Return a list containing exactly 6 values, representing the quantization
             parameters.
            The Y parameters may be None, if they were omitted in the model. Otherwise, they are all numpy arrays.

        :param q_gemm_attrs: ORT QGemm attributes.
        :param t_op: TFLite operator representing the ORT QGemm operator.
        :return: The quantization parameters a_scale , a_zero_point, b_scale , b_zero_point, y_scale , y_zero_point.
        """
        parameter_tensors = t_op.tmp_inputs[1:3] + t_op.tmp_inputs[4:6]

        output_parameter_tensors = [
            try_get_input(t_op, 7),  # y scale
            try_get_input(t_op, 8)  # y zero point
        ]

        if exactly_one_is_none(*output_parameter_tensors):
            # Exactly one output quantization parameter is specified. This goes against the documentation.
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     "ONNX Runtime QGemm has a 'y_scale' input, but not 'y_zero_point'.")

        # Remove `None` values from `parameter_tensors`.
        output_parameter_tensors = list(filter(lambda x: x is not None, output_parameter_tensors))

        # Add the output parameter tensors to the input parameter tensors.
        parameter_tensors.extend(output_parameter_tensors)

        if not all_tensors_are_static(*parameter_tensors):
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     'Conversion of ONNX Runtime QGemm with dynamic quantization parameters is not possible.')

        params = [param_tensor.tmp_buffer.data for param_tensor in parameter_tensors]
        if not all(param.size == 1 for param in params):
            # Per-channel quantization. TFLite supports this only when certain conditions are met.
            is_trans_b = q_gemm_attrs.transB
            a_is_int8 = parameter_tensors[1].type == TensorType.INT8
            b_is_int8 = parameter_tensors[3].type == TensorType.INT8
            b_zp_all_zeros = not params[3].any()

            if not (is_trans_b and a_is_int8 and b_is_int8 and b_zp_all_zeros):
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "ONNX Runtime QGemm uses per-channel quantization. Conversion to TFLite is "
                         "supported only when attribute 'transB=1', types equal to INT8 and zero points of second "
                         "input tensor equal to zero.")

        if len(params) == 4:
            # The output parameters were omitted.
            params += [None] * 2

        return params

    def _transpose_input(self, t_op: tflite_model.Operator, input_idx: int) -> tflite_model.Operator | None:
        t = t_op.tmp_inputs[input_idx]

        if t.rank != 2:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f'ONNX Runtime QGemm has a main intput with {t.rank} dimensions instead of 2.')

        if tensor_has_data(t):
            # Transpose statically.
            t_op.tmp_inputs[input_idx] = self.builder.create_transposed_tensor(t)
        else:
            # Dynamic tensor -> prepend a Transpose op.
            return self.builder.create_transpose_operator_before(t_op, input_idx, [1, 0])

    def _handle_c_tensor(self, o_q_gemm: q_gemm_attributes.QGemm, t_op: tflite_model.Operator,
                         c: tflite_model.Tensor, a_scale, b_scale):
        if c is None:
            return

        if tensor_has_data(c) and all(el == 0 for el in c.tmp_buffer.data.flatten()):
            # The C is just adding 0 -> ignore it.
            return

        # According to the documentation, the default quantization parameters of C are
        #  zero_point = 0
        #  scale = alpha / beta * a_scale * b_scale
        # But QGemm has no 'beta' attribute according to the doc an ONNX Runtime. So:
        #  scale = alpha * a_scale * b_scale
        scale = o_q_gemm.alpha * a_scale * b_scale
        zero_point = np.zeros(b_scale.size).astype(np.int16)
        set_quantization_parameters_to_tensor(c, scale, zero_point)

        # In ONNX, C must be broadcastable to the output (shape [M, N]). In TFLite however, C must have shape [N] or
        #  [1, N]. In case C has a different shape, it must be added via an Add operator.
        n = t_op.tmp_outputs[0].shape.get(1)
        if c.shape.vector == [n] or c.shape.vector == [1, n]:
            # C has shape [N]
            t_op.tmp_inputs.append(c)
            return

        # Do not try to add bias tensor via additional Add operator. This brings two problems:
        #  1. There is need to re-quantize bias tensor from INT32 to (U)INT8. This introduces some error.
        #  2. Addition of bias is not done within single accumulator of FullyConnected op. This introduces
        #     another error.
        # Combination of errors mentioned above leads to large degradation in accuracy.
        # Bias in shape different from [N] or [1, N] is also rare, so we let user improve the model.
        logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                 "ONNX Runtime operator QGemm has bias input ('C') with shape that is not supported by TFLite. "
                 "Make sure bias tensor has shape [N] or [1, N].")

    def _handle_alpha_attribute(self, alpha: float, t_op: tflite_model.Operator, ops: OpsList):
        """ Handle the conversion of the 'alpha' attribute of the ORT QGemm operator.

            QGemm carries out the following operation:
                Y = alpha * A * B + C

            A, B and C are quantized tensors. alpha is a float scalar.

            If at least one of A/B is static, it suffices to multiply its scale by `alpha`, effectively multiplying
             its data by `alpha`. This however, only works if alpha is not negative.
            Otherwise, a `Mul` operator must be added before the `FullyConnected`.

        :param alpha: The `alpha` attribute of the ORT `QGemm`.
        :param t_op: TFLite operator representing the corresponding FullyConnected operator.
        :param ops: OpsList with TFLite operators to add to the model.
        """
        if alpha == 1.0:
            return

        if alpha < 0:
            # Rare scenario -> For negative alpha, a Mul op must be added, or dequantize the static data,
            #  multiply by alpha and quantize it back.
            logger.e(logger.Code.NOT_IMPLEMENTED, 'Conversion of QGemm with negative alpha is not supported.')

        # Find if the QGemm has a static input.
        static_input = None
        if tensor_has_data(t_op.tmp_inputs[0]):
            static_input = t_op.tmp_inputs[0]
        elif tensor_has_data(t_op.tmp_inputs[1]):
            static_input = t_op.tmp_inputs[1]

        if static_input is not None:
            # It suffices to multiply the `scale` quantization parameter of the `static_input` by `alpha`.
            static_input.quantization.scale = tflite_model.Scale(
                [alpha * s for s in static_input.quantization.scale.vector])

        else:
            # Prepend a `Mul` operator to multiply one input by `alpha`.

            x = t_op.tmp_inputs[0]

            # Create an alpha tensor and quantize it based on alpha value -> it will result in scale value
            #  being quantized to 1, and no loss of information.
            alpha_tensor = self.context.tflite_builder.create_tensor_for_data(np.array([alpha], np.float32), 'alpha')
            zp = [get_symmetric_zero_point_for_type(x.type)]
            alpha_tensor = quantize_static_float_tensor(self.builder, alpha_tensor, x.type, [alpha], zp)

            mul_output = self.context.tflite_builder.duplicate_tensor(x, name_suffix='_scaled')

            # Mul output tensor is quantized as 'alpha * input_scale'. This means that the literal 8bit values
            #  at the output of `Mul` will be exactly the same as the input values. Just their corresponding
            #  de-quantized float values will have been multiplied by `alpha`. Therefore, there is no information loss.
            mul_output.quantization.scale = tflite_model.Scale(
                [alpha * val for val in mul_output.quantization.scale.vector])

            mul_op = tflite_model.Operator(builtin_options=mul_options.Mul())
            mul_op.tmp_inputs = [x, alpha_tensor]
            mul_op.tmp_outputs = [mul_output]

            t_op.tmp_inputs[0] = mul_output

            ops.add_pre(mul_op)

    def convert(self, q_gemm_node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ORT QGemm operator to TFLite FullyConnected.

        :param q_gemm_node: ORT QGemm operator node.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator
        :return: A list of TFLite operators, to add to the model.
        """
        o_q_gemm = cast(q_gemm_attributes.QGemm, q_gemm_node.attributes)

        if not (6 <= len(t_op.tmp_inputs) <= 9):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f'ONNX QGemm has unexpected number of inputs ({len(t_op.tmp_inputs)}), instead of 6 - 9.')

        ops = OpsList(middle_op=t_op)

        a = t_op.tmp_inputs[0]
        b = t_op.tmp_inputs[3]
        c = try_get_input(t_op, 6)
        y = t_op.tmp_outputs[0]

        if a.type != b.type:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX Runtime QGemm has mismatched input types.')

        # Quantization parameters (numpy arrays or None). y_s and y_zp may be None (together), which means that the
        #  output of the QGemm should be float32.
        a_s, a_zp, b_s, b_zp, y_s, y_zp = self._parse_quantization_parameters(o_q_gemm, t_op)

        # Add the quantization parameters to the tensors
        set_quantization_parameters_to_tensor(a, a_s, a_zp)
        set_quantization_parameters_to_tensor(b, b_s, b_zp)
        if y_s is not None:
            set_quantization_parameters_to_tensor(y, y_s, y_zp)

        else:
            if y.type != TensorType.FLOAT32:
                logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                         'ONNX Runtime QGemm has no output quantization parameters, but its output is not float32.')

            # The output is FLOAT32. I believe ONNX Runtime does not do the computation in 8bits. I somehow couldn't
            #  find where this is handled within ONNX Runtime. In my tests, the ONNX output had a resolution of
            #  0.0020000935 (smallest difference of output values) and range of 64.192 (maximum difference of output
            #  values). That gives 64.192 / 0.0020000935 = 32094.49958214453 possible values. log_2(32094.49958214453) =
            #  14.9700384461955 bits would be the minimum required for this precision.

            # Probably convert into float FullyConnected if necessary.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     'Conversion of ONNX Runtime QGemm with a float32 output is not yet supported.')

        # Assign the operator its TFLite inputs.
        t_op.tmp_inputs = [a, b]

        self._handle_c_tensor(o_q_gemm, t_op, c, a_s, b_s)

        self._handle_alpha_attribute(o_q_gemm.alpha, t_op, ops)

        t_op.builtin_options = fully_connected_options.FullyConnected(keep_num_dims=True)

        if o_q_gemm.transA:
            # Transpose A
            if transpose := self._transpose_input(t_op, 0):
                ops.add_pre(transpose)

        if not o_q_gemm.transB:
            # Transpose B
            if transpose := self._transpose_input(t_op, 1):
                ops.add_pre(transpose)

        return ops.flatten()
