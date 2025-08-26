#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#


from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS


class DropoutConverter(NodeConverter):
    node = "Dropout"

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/transpose.cc#L147-L230
    tflite_supported_types = INTS + [TensorType.UINT8, TensorType.FLOAT32, TensorType.BOOL]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'Dropout' operator to TFLite.
             There is no direct equivalent, but Dropout does nothing during inference, so the operator can be skipped.

        :param node: ONNX Dropout operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        training_mode = 0
        if training_mode_tensor := try_get_input(t_op, 2):
            # The training_mode was passed as a tensor.
            if tensor_has_data(training_mode_tensor):
                training_mode = training_mode_tensor.tmp_buffer.data

            else:
                training_mode = self.inspector.try_get_inferred_tensor_data(training_mode_tensor.name)

            if training_mode is None:
                # Cannot determine if training mode is set or not.
                # If needed, add a flag to guarantee `training_mode` 0.
                # This is tested by `node:test_training_dropout*`.
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX `Dropout` with a dynamic `training_mode` input tensor is not supported.")

        if training_mode != 0:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX `Dropout` with `training_mode` != 0 is not possible.")

        # Remove the extra inputs of the operator.
        t_op.tmp_inputs[1:] = []

        # Check for extra outputs.
        if any(self.inspector.get_number_of_onnx_consumers_safe(output_tensor.name) != 0 for output_tensor in
               t_op.tmp_outputs[1:]):
            # The `Dropout` uses extra outputs, which are used later in the model. Conversion is not possible.
            # This is tested by `node:test_dropout_default_mask*`.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX `Dropout` with more than 1 output is not possible.")

        if self.builder.operator_can_be_skipped(t_op, self.inspector):
            self.builder.redirect_tensor(t_op.tmp_outputs[0], t_op.tmp_inputs[0])
            return []

        # The operator consumes a graph input tensor and also produces a graph output tensor.
        # We can return a Transpose op, which does nothing.
        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        self.builder.turn_operator_to_identity(t_op)
        return [t_op]
