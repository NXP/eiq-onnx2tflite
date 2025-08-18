#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List, cast

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import gelu_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import gelu_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class GeluConverter(NodeConverter):
    node = 'Gelu'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/activations.cc#L1563-L1586
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT8, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert ONNX `Gelu`` operator into TFLite `Gelu`.

        :param node: ONNX NodeProto representing the Gelu operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators to add to the model.
        """

        if len(t_op.tmp_inputs) != 1 or len(t_op.tmp_outputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `Gelu` has invalid number of input and output tensors.')

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        o_gelu = cast(gelu_attributes.Gelu, node.attributes)
        if o_gelu.approximate == 'none':
            approximate = False

        elif o_gelu.approximate == 'tanh':
            approximate = True

        else:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f'ONNX `Gelu` has invalid value `{o_gelu.approximate}` of the `approximate` attribute.')

        # noinspection PyUnboundLocalVariable
        t_op.builtin_options = gelu_options.Gelu(approximate)

        return [t_op]
