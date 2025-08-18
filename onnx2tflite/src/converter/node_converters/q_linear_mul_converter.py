#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np

import onnx2tflite.src.tflite_generator.builtin_options.mul_options as tflite_mul_options
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.quantization_utils import set_quantization_parameters_to_tensor
from onnx2tflite.src.converter.conversion.common import try_get_input
from onnx2tflite.src.converter.conversion.translator import tf_lite_type_to_numpy
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model


class QLinearMulConverter(NodeConverter):
    node = 'QLinearMul'

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX Runtime 'QLinearMul' operator to TFLite 'Mul'.

            :param node: ONNX Runtime `QLinearMul` operator.
            :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator
            :return: A list of TFLite operators, to add to the model.
        """

        t_op.builtin_options = tflite_mul_options.Mul()

        # Prepare the input and output tensors.
        y = t_op.tmp_outputs[0]
        a = t_op.tmp_inputs[0]
        a_scale_tensor = t_op.tmp_inputs[1]
        a_zp_tensor = try_get_input(t_op, 2)
        b = t_op.tmp_inputs[3]
        b_scale_tensor = t_op.tmp_inputs[4]
        b_zp_tensor = try_get_input(t_op, 5)
        y_scale_tensor = t_op.tmp_inputs[6]
        y_zp_tensor = try_get_input(t_op, 7)

        # Input and output types must be the same
        if not (a.type == b.type == y.type):
            logger.e(logger.Code.INVALID_TYPE, "ONNX QLinearMul input and output tensors must have the same data type!")

        # ONNX only supports INT8 and UINT8
        if y.type not in {TensorType.INT8, TensorType.UINT8}:
            logger.e(logger.Code.INVALID_TYPE,
                     f"ONNX QLinearMul supports only INT8 and UINT8 data types. Got '{y.type}'.")

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [a, b]
        t_op.tmp_outputs = [y]

        # Check for dynamic quantization parameters.
        for param_tensor in [a_scale_tensor, b_scale_tensor, y_scale_tensor, a_zp_tensor, b_zp_tensor, y_zp_tensor]:
            if param_tensor is not None and not tensor_has_data(param_tensor):
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         'Conversion of ONNX QLinearMul with dynamic quantization parameters is not possible.')

        numpy_zp_type = tf_lite_type_to_numpy(a.type)

        # Add the quantization parameters to the tensors
        a_scale = a_scale_tensor.tmp_buffer.data
        a_zp = a_zp_tensor.tmp_buffer.data if a_zp_tensor is not None else np.zeros(a_scale.shape, numpy_zp_type)
        set_quantization_parameters_to_tensor(a, a_scale, a_zp)

        b_scale = b_scale_tensor.tmp_buffer.data
        b_zp = b_zp_tensor.tmp_buffer.data if b_zp_tensor is not None else np.zeros(b_scale.shape, numpy_zp_type)
        set_quantization_parameters_to_tensor(b, b_scale, b_zp)

        y_scale = y_scale_tensor.tmp_buffer.data
        y_zp = y_zp_tensor.tmp_buffer.data if y_zp_tensor is not None else np.zeros(y_scale.shape, numpy_zp_type)
        set_quantization_parameters_to_tensor(y, y_scale, y_zp)

        # ORT QLinearMul supports shape broadcasting.
        additional_ops = self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0])

        return additional_ops + [t_op]
