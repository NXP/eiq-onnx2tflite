#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

import numpy as np

import onnx2tflite.src.tflite_generator.builtin_options.mul_options as tfl_mul_options
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


class MulConverter(NodeConverter):
    node = "Mul"

    onnx_supported_types = FLOATS + INTS + UINTS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/mul.cc#L390-L406
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.COMPLEX64, TensorType.UINT32, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32, TensorType.INT64, TensorType.INT32]

    # noinspection PyMethodMayBeStatic
    def _is_ones_tensor(self, tensor: tflite_model.Tensor) -> bool:
        """Determine if 'tensor' contains static data that is all the value 1.

        :param tensor: TFLite tensor.
        :return: True, if 'tensor' holds static data that is all 1.
        """
        if tensor_has_data(tensor):
            np_type = translator.tf_lite_type_to_numpy(tensor.type)
            one = np.array(1).astype(np_type)
            if np.all(tensor.tmp_buffer.data == one):
                return True
        return False

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX 'Mul' operator to TFLite."""
        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX 'Mul' has unexpected number of inputs. "
                                                     f"Got '{len(t_op.tmp_inputs)}', expected '2'.")

        x, y = t_op.tmp_inputs

        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX 'Mul' has mismatched input data types!")

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        x_only_ones = self._is_ones_tensor(x)
        y_only_ones = self._is_ones_tensor(y)

        if x_only_ones or y_only_ones:
            input_tensor = y if x_only_ones else x

            if (input_tensor.shape == t_op.tmp_outputs[0].shape) and \
                    self.builder.operator_can_be_skipped(t_op, self.inspector):
                # Operator can be skipped
                self.builder.redirect_tensor(t_op.tmp_outputs[0], input_tensor)
                return []

        ops_to_add = self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0])

        t_op.builtin_options = tfl_mul_options.Mul()

        return ops_to_add + [t_op]
