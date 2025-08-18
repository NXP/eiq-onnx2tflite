#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.sign_options import Sign
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


class SignConverter(NodeConverter):
    node = 'Sign'

    onnx_supported_types = FLOATS + INTS + UINTS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/sign.cc#L64-L83
    tflite_supported_types = [TensorType.FLOAT32, TensorType.FLOAT64, TensorType.INT32]
    verified_types = [TensorType.FLOAT32, TensorType.FLOAT64, TensorType.INT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert ONNX `Sign` operator into TFLite `Sign`. """

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `Sign` has unexpected number of inputs ({len(t_op.tmp_inputs)}).')

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        t_op.builtin_options = Sign()

        return [t_op]
