#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.translator import tf_lite_type_to_numpy
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.div_options import Div
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class ReciprocalConverter(NodeConverter):
    node = 'Reciprocal'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/div.cc#L279-L298
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX `Reciprocal` to TFLite `Div` with an extra tensor. """

        if not (len(t_op.tmp_inputs) == len(t_op.tmp_outputs) == 1):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `Reciprocal` has unexpected number of IO tensors.')

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        # Create a tensor with a single `1`, and use it as the first input of the `Div` operator.
        np_type = tf_lite_type_to_numpy(x.type)
        one_tensor = self.builder.create_tensor_for_data(np.ones([1], np_type), 'one')
        t_op.tmp_inputs.insert(0, one_tensor)

        t_op.builtin_options = Div()

        return [t_op]
