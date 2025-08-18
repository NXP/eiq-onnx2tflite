#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from typing import List, cast

import onnx2tflite.src.onnx_parser.builtin_attributes.leaky_relu_attributes as onnx_leaky_relu_attributes
import onnx2tflite.src.tflite_generator.builtin_options.leaky_relu_options as tfl_leaky_relu_options
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class LeakyReluConverter(NodeConverter):
    node = 'LeakyRelu'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/activations.cc#L1451-L1470
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT16]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX LeakyRelu to TFLite LeakyRelu. """

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX LeakyRelu has '{len(t_op.tmp_inputs)}' inputs!")

        x = t_op.tmp_inputs[0]
        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        attrs = cast(onnx_leaky_relu_attributes.LeakyRelu, node.attributes)
        t_op.builtin_options = tfl_leaky_relu_options.LeakyRelu(attrs.alpha)
        return [t_op]
