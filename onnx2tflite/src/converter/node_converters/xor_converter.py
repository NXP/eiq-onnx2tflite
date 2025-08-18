#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.not_equal_options import NotEqual


class XorConverter(NodeConverter):
    node = 'Xor'

    onnx_supported_types = [TensorType.BOOL]
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/comparisons.cc#L227-L262
    tflite_supported_types = [TensorType.BOOL, TensorType.FLOAT32, TensorType.INT8, TensorType.INT32, TensorType.INT64,
                              TensorType.UINT8, TensorType.STRING]
    verified_types = [TensorType.BOOL]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX `Xor` operator to TFLite `NotEqual`. """

        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `Xor` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `2`.')

        x1 = t_op.tmp_inputs[0]
        x2 = t_op.tmp_inputs[1]
        y = t_op.tmp_outputs[0]

        if x1.type != x2.type:
            logger.e(logger.Code.INVALID_ONNX_MODEL, 'ONNX `Xor` has inputs with different types.')

        self.assert_type_allowed(x1.type)

        t_op.builtin_options = NotEqual()

        ops = OpsList(middle_op=t_op)

        # ONNX `Xor` supports shape broadcasting.
        ops.add_pre(self.builder.ensure_correct_broadcasting(t_op, y))

        return ops.flatten()
