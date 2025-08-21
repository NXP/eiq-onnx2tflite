#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import uses_multiple_input_types, uses_shape_broadcasting
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import add_n_options, add_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class SumConverter(NodeConverter):
    node = "Sum"

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/add_n.cc#L132-L140
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT32]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX operator 'Sum' to TFLite 'Add' or 'AddN'."""
        num_inputs = len(t_op.tmp_inputs)

        if num_inputs == 0:
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX 'Sum' has no inputs!")

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(t_op.tmp_inputs[0].type)

        if num_inputs == 1:
            if self.builder.operator_can_be_skipped(t_op, self.inspector):
                # Skip the operator.
                logger.i("convert_sum.py: ONNX 'Sum' has only 1 input. The operator is skipped.")
                self.builder.redirect_tensor(t_op.tmp_outputs[0], t_op.tmp_inputs[0])
                return []

            t_op.builtin_options = add_options.Add()

            # TFLite sum must have 2 inputs -> add 0 for no effect
            input_type = translator.tf_lite_type_to_numpy(t_op.tmp_inputs[0].type)
            t_op.tmp_inputs.append(self.builder.create_tensor_for_data(np.asarray([0], input_type), "zero"))

            return [t_op]

        if uses_multiple_input_types(t_op):
            # Inputs have different data types
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX Operator 'Sum' has inputs with different data types!")

        additional_ops = []

        if num_inputs == 2:
            # Can use 'Add' operator to represent the 'Sum'. This is better than just using 'AddN', because 'Add' supports
            #  shape broadcasting and can be fused with an activation function.
            t_op.builtin_options = add_options.Add()

            additional_ops = self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0])

        else:
            # 'Sum' has many inputs. Convert to 'AddN'.
            t_op.builtin_options = add_n_options.AddN()

            if uses_shape_broadcasting(t_op):
                # TFLite 'AddN' doesn't support shape broadcasting at all. Need to implement explicit support.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "ONNX Operator 'Sum' with more than 2 inputs uses shape broadcasting. This requires explicit "
                         "support by the converter, which has not yet been implemented!")

        return additional_ops + [t_op]
