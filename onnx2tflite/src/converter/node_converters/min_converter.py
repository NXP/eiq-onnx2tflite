#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.quantization_utils import re_quantize_static_tensor
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.minimum_options import Minimum
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


class MinConverter(NodeConverter):
    node = 'Min'

    onnx_supported_types = FLOATS + INTS + UINTS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/maximum_minimum.cc#L176-L252
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert ONNX `Min` operator into TFLite `Minimum`.

        :param node: ONNX NodeProto representing the `Min` operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators to add to the model.
        """

        ops = OpsList(middle_op=t_op)

        if len(t_op.tmp_inputs) == 1:
            # Operator has no effect.
            if self.builder.operator_can_be_skipped(t_op, self.inspector):
                self.builder.redirect_tensor(t_op.tmp_outputs[0], t_op.tmp_inputs[0])
                return []

            else:
                self.builder.turn_operator_to_identity(t_op)

        elif len(t_op.tmp_inputs) == 2:
            # Convert to TFLite `Minimum`.
            t_op.builtin_options = Minimum()

            x1 = t_op.tmp_inputs[0]
            x2 = t_op.tmp_inputs[1]
            y = t_op.tmp_outputs[0]
            if not (x1.type == x2.type == y.type):
                logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `Min` uses multiple data types.')

            if not t_op.is_qdq_quantized():
                self.assert_type_allowed(y.type)
            else:
                scale = y.quantization.scale.vector
                zp = y.quantization.zero_point.vector

                # Re-quantize input tensors if they don't match with output quantization.
                for idx, input_tensor in enumerate(t_op.tmp_inputs):
                    if input_tensor.quantization != y.quantization:
                        logger.w(f"Re-quantizing tensor '{input_tensor.name}' to match output tensor's "
                                 f"q-params of Minimum op.")
                        if tensor_has_data(input_tensor):
                            t_op.tmp_inputs[idx] = re_quantize_static_tensor(self.builder, input_tensor, y.type, scale,
                                                                             zp)
                        else:
                            ops.add_pre(self.builder.create_quantize_operator_before(t_op, idx, y.type, scale, zp))

            # ONNX `Min` supports broadcasting since v8.
            ops.add_pre(self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0]))

        else:
            # TFLite `Minimum` only supports 2 inputs.
            # Add a cascade of `Minimum` operators, which gradually compute the result.
            #  The number of added operators would be linear to the number of inputs.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     'Conversion of ONNX `Min` with more than 2 inputs is not yet implemented.')

        return ops.flatten()
