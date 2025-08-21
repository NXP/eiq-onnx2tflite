#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.builder.model_builder import tensor_has_data
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, INTS


class IdentityConverter(NodeConverter):
    node = "Identity"

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/transpose.cc#L147-L230
    tflite_supported_types = INTS + [TensorType.UINT8, TensorType.FLOAT32, TensorType.BOOL]
    verified_types = tflite_supported_types

    def convert(self, _: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `Identity` operator into TFLite.

            If the input tensor is known, try to exclude the operator from the final model, and just assign the static
             data to the output tensor.
            If that is not possible, represent the identity via a `Transpose` operator with an identity permutation.
            The `Transpose` may potentially be removed or fused with another `Transpose` operator later, in the
             optimization stage.

        :param _: ONNX NodeProto representing the `Identity` operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators to add to the model.
        """
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if tensor_has_data(x):
            data = x.tmp_buffer.data

        elif (data := self.inspector.try_get_inferred_tensor_data(x.name)) is not None:
            logger.i(f"Using inferred data for tensor {x.name}.")

        if data is not None and not self.inspector.is_output_of_model(y.name):
            # Turn the `Identity` into a static tensor and return no operators.
            y.tmp_buffer.data = data

            return []

        # The operator cannot be omitted. Turn it into a `Transpose` that does nothing.
        self.builder.turn_operator_to_identity(t_op)

        self.assert_type_allowed(x.type)

        if t_op.is_quantized_without_qdq():
            propagate_quantization(x, y)

        return [t_op]
