#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class FloorConverter(NodeConverter):
    node = 'Floor'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/floor.cc#L45
    tflite_supported_types = [TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX `Floor` operator to TFLite `Floor`. """

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `Floor` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `1`.')

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        t_op.opcode_index = self.builder.op_code_index_for_op_type(BuiltinOperator.FLOOR)

        return [t_op]
