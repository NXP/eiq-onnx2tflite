#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np

import onnx2tflite.src.tflite_generator.builtin_options.sub_options as tflite_sub_options
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import uses_shape_broadcasting
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


class SubConverter(NodeConverter):
    node = 'Sub'

    onnx_supported_types = FLOATS + INTS + UINTS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/sub.cc#L453-L456
    # Also `uint8`, `int8` and `int16` are supported but only quantized.
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]
    verified_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]

    # noinspection PyMethodMayBeStatic
    def _is_zeros_tensor(self, tensor) -> bool:
        if tensor_has_data(tensor):
            if np.all(tensor.tmp_buffer.data == 0):
                return True
        return False

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX 'Sub' has unexpected number of inputs. "
                                                     f"Got '{len(t_op.tmp_inputs)}', expected '2'.")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_inputs[1]
        output = t_op.tmp_outputs[0]

        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX 'Sub' has mismatched input data types!")

        if t_op.is_quantized_without_qdq():
            # ONNX Runtime currently doesn't support `(u)int8` inputs, so this case cannot be tested.
            # If support is added in the future, propagate the quantization parameters.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     'Conversion of ONNX `Sub` with quantized inputs is not yet supported.')

        elif not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        y_only_zeros = self._is_zeros_tensor(y)

        is_broadcasting = uses_shape_broadcasting(t_op)
        is_channels_last = x.tensor_format.is_channels_last() or y.tensor_format.is_channels_last()

        if y_only_zeros:
            if x.shape == output.shape and self.builder.operator_can_be_skipped(t_op, self.inspector):
                logger.i("Skipping operator 'Sub' because second input 'y' is just zeros.")
                self.builder.redirect_tensor(output, x)
                return []

        additional_ops = []
        if is_broadcasting and is_channels_last:
            additional_ops = self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0])

        t_op.builtin_options = tflite_sub_options.Sub()
        return additional_ops + [t_op]
