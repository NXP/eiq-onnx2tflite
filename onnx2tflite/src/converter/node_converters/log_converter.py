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


class LogConverter(NodeConverter):
    node = 'Log'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/elementwise.cc#L327-L346
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT8, TensorType.INT16]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX `Log` operator to TFLite `Log`.

        :param node: ONNX `Log` operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `Log` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `1`.')

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        t_op.builtin_options = None
        t_op.opcode_index = self.context.tflite_builder.op_code_index_for_op_type(BuiltinOperator.LOG)

        return [t_op]
