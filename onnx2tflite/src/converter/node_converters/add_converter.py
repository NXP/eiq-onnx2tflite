#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from typing import List

import numpy as np

import onnx2tflite.src.tflite_generator.builtin_options.add_options as tflite_add_options
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import uses_shape_broadcasting
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


class AddConverter(NodeConverter):
    node = 'Add'

    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/add.cc#L420-L432
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8] + INTS
    onnx_supported_types = FLOATS + INTS + UINTS
    verified_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]  # INT8 not supported by ORT

    # noinspection PyMethodMayBeStatic
    def _is_zeros_tensor(self, tensor) -> bool:
        if tensor_has_data(tensor):
            if np.all(tensor.tmp_buffer.data == 0):
                return True
        return False

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX 'Add' has unexpected number of inputs. "
                                                     f"Got '{len(t_op.tmp_inputs)}', expected '2'.")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_inputs[1]
        output = t_op.tmp_outputs[0]

        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX 'Add' has mismatched input data types!")

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)
        elif t_op.is_quantized_without_qdq():
            # ONNX Runtime doesn't support INT8 / UINT8.
            # https://github.com/microsoft/onnxruntime/blob/v1.17.0/onnxruntime/core/providers/cpu/math/element_wise_ops.cc#L156-L167
            # Keep this check in case support is added later.
            logger.e(logger.Code.NOT_IMPLEMENTED, 'Conversion of ONNX `Add` with quantized inputs is not supported.')

        x_only_zeros = self._is_zeros_tensor(x)
        y_only_zeros = self._is_zeros_tensor(y)

        is_broadcasting = uses_shape_broadcasting(t_op)
        is_channels_last = x.tensor_format.is_channels_last() or y.tensor_format.is_channels_last()

        if x_only_zeros or y_only_zeros:
            input_tensor = y if x_only_zeros else x

            if (input_tensor.shape == output.shape) and self.builder.operator_can_be_skipped(t_op, self.inspector):
                logger.i("Skipping operator 'Add' because one of the inputs was just zeros.")
                self.builder.redirect_tensor(output, input_tensor)
                return []

        additional_ops = []
        if is_broadcasting and is_channels_last:
            additional_ops = self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0])

        t_op.builtin_options = tflite_add_options.Add()
        return additional_ops + [t_op]
