#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math
from typing import List

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class TanhConverter(NodeConverter):
    node = 'Tanh'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/activations.cc#L1023-L1112
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT16]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX `Tanh` operator to TFLite `Tanh`.

        :param node: ONNX `Tanh` operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `Tanh` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `1`.')

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_MODEL, 'ONNX `Tanh` has different input and output type.')

        ops = OpsList(middle_op=t_op)

        if t_op.is_qdq_quantized():
            # In case ONNX adds support for Tanh quantization in future and doesn't match TFLite's requirements
            if not math.isclose(y.quantization.scale.vector[0], 1 / 128.0):
                logger.w(
                    f"Re-quantizing output tensor '{y.name}' of Tanh op to satisfy TFLite's q-param requirements.")
                # Quantization not done with internal quantizer => re-quantize output
                quantize_op = self.builder.create_quantize_operator_after(t_op, 0, x.type, [1. / 128.], [0])
                ops.add_post(quantize_op)

        else:
            self.assert_type_allowed(x.type)

        t_op.builtin_options = None
        t_op.opcode_index = self.context.tflite_builder.op_code_index_for_op_type(BuiltinOperator.TANH)

        return ops.flatten()
