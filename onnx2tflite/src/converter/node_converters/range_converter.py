#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import range_options


class RangeConverter(NodeConverter):
    node = 'Range'

    onnx_supported_types = [TensorType.FLOAT32, TensorType.FLOAT64, TensorType.INT16, TensorType.INT32,
                            TensorType.INT64]
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/range.cc#L151-L156
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]
    verified_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]

    def _get_range_params(self, t_op: tflite_model.Operator) -> (int | float, int | float, int | float):
        def _try_get_data(input_index: int) -> np.ndarray | None:
            tensor = t_op.tmp_inputs[input_index]
            if tensor_has_data(tensor):
                return tensor.tmp_buffer.data

            if data := self.inspector.try_get_inferred_tensor_data(tensor.name):
                logger.d(f"Using inferred data for `Range` input `{tensor.name}`.")

            return data

        start = _try_get_data(0)
        limit = _try_get_data(1)
        delta = _try_get_data(2)

        if all(el is not None for el in (start, limit, delta)):
            return start.item(), limit.item(), delta.item()

        else:
            # We could add a flag to guarantee that the operands are valid, but I don't think it's worth it, as I have
            #  had some issues using dynamic scalars (shape []) while writing the tests.
            # Right now, the shape inference would have already failed anyway.

            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     'Conversion of ONNX `Range` with dynamic inputs is not supported.')

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX `Range` to TFLite `Range`. """

        if len(t_op.tmp_inputs) != 3:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `Range` has invalid number of inputs.')

        data_type = t_op.tmp_inputs[0].type
        if any(t.type != data_type for t in t_op.tmp_inputs):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `Range` has inputs with different data types.')

        self.assert_type_allowed(data_type)

        # Both ONNX and TFLite require the inputs to be scalars with shape [].
        # https://github.com/tensorflow/tensorflow/blob/8fcb611cdae1ffae5b643762901fbb72a8941315/tensorflow/lite/kernels/range.cc#L144-L146
        for input_tensor in t_op.tmp_inputs:
            # noinspection PySimplifyBooleanCheck
            if input_tensor.shape.vector != []:
                logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `Range` has non-scalar inputs.')

        # Get the operands of the `Range`, and check some edge cases.
        start, limit, delta = self._get_range_params(t_op)  # This call may raise an error.

        if delta == 0:
            # https://github.com/microsoft/onnxruntime/blob/d30c81d270894f41ccce7b102b1d4aedd9e628b1/onnxruntime/core/providers/cpu/generator/range.cc#L65
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `Range` has `delta` = 0, which is not allowed.')

        if (start >= limit and delta > 0) or (start <= limit and delta < 0):
            # ONNX allows instances such as 'start = 0, limit = 10, step = -1'. The output is an empty tensor. Whereas
            #  TFLite simply crashes in those cases.
            # Make sure TFLite will also produce empty output.
            t_op.tmp_inputs[1] = t_op.tmp_inputs[0]  # Set `limit` = `start`.

        t_op.builtin_options = range_options.Range()

        return [t_op]
