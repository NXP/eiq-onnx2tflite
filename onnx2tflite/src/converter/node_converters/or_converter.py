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
from onnx2tflite.src.tflite_generator.builtin_options.logical_or_options import LogicalOr


class OrConverter(NodeConverter):
    node = 'Or'

    onnx_supported_types = [TensorType.BOOL]
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/logical.cc#L70-L73
    tflite_supported_types = [TensorType.BOOL]
    verified_types = [TensorType.BOOL]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX `Or` operator to TFLite `LogicalOr`. """

        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `Or` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `2`.')

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]
        self.assert_type_allowed(x.type)

        t_op.builtin_options = LogicalOr()

        ops = OpsList(middle_op=t_op)

        # ONNX `Or` supports shape broadcasting.
        ops.add_pre(self.builder.ensure_correct_broadcasting(t_op, y))

        return ops.flatten()
