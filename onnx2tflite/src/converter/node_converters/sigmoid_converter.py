#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class SigmoidConverter(NodeConverter):
    node = 'Sigmoid'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/activations.cc#L1123-L1213
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT8, TensorType.INT16, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX 'Sigmoid' to TFLite 'Logistic' operator.

            Neither the ONNX nor TFLite variant has any attributes/parameters. The TFLite 'Logistic' doesn't even have a
             builtin options enum value (just like to Relu). So this module must assign the 'opcode_index' directly to
             the operator by itself.
        """
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        ops = OpsList(middle_op=t_op)

        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX Sigmoid has different input and output type.")

        if t_op.is_qdq_quantized():
            if not math.isclose(y.quantization.scale.vector[0], 1 / 256.0):
                logger.w(
                    f"Re-quantizing output tensor '{y.name}' of Sigmoid op to satisfy TFLite's q-param requirements.")
                # Quantization not done with internal quantizer => re-quantize output
                zp = [0 if (x.type == TensorType.UINT8) else -128]
                scale = [1. / 256.]
                ops.post_ops.append(self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp))

        else:
            self.assert_type_allowed(x.type)

        t_op.builtin_options = None
        t_op.opcode_index = self.builder.op_code_index_for_op_type(BuiltinOperator.LOGISTIC)

        return ops.flatten()
