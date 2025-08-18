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
from onnx2tflite.src.tflite_generator.builtin_options.logical_not_options import LogicalNot


class NotConverter(NodeConverter):
    node = 'Not'

    onnx_supported_types = [TensorType.BOOL]
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/elementwise.cc#L257
    tflite_supported_types = [TensorType.BOOL]
    verified_types = [TensorType.BOOL]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX `Not` operator to TFLite `LogicalNot`. """

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `Not` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `1`.')

        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        t_op.builtin_options = LogicalNot()

        return [t_op]
