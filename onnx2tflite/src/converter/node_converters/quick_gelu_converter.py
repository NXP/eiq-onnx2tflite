#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import quick_gelu_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.mul_options import Mul
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class QuickGeluConverter(NodeConverter):
    node = "QuickGelu"

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/activations.cc#L1123-L1213
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/mul.cc#L390-L406
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT8, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX Runtime `QuickGelu` operator into TFLite.

        There is no 'QuickGelu' in TFLite. It carries out the operation

            y = x * Sigmoid(alpha * x)

        This can be represented in TFLite as

           x -> Mul(alpha) -> Logistic -> Mul(x) -> y

        The first `Mul` can be omitted, if alpha == 1.0
        """
        qg_attributes = cast(quick_gelu_attributes.QuickGelu, node.attributes)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        self.assert_type_allowed(x.type)

        ops = []

        # ---- Mul(alpha) ----
        if qg_attributes.alpha == 1.0:
            # No need to add a `Mul` operator.
            mul_1_output = x
        else:
            # Create a `Mul` operator.
            mul_1_output = self.builder.duplicate_tensor(x, name_suffix="_scaled")
            alpha = self.builder.create_tensor_for_data(np.array([qg_attributes.alpha], np.float32), "alpha")

            mul_1_op = tflite_model.Operator(builtin_options=Mul())
            mul_1_op.tmp_inputs = [x, alpha]
            mul_1_op.tmp_outputs = [mul_1_output]

            ops.append(mul_1_op)

        # ---- Logistic ----
        logistic_output = self.builder.duplicate_tensor(mul_1_output)

        logistic_op = tflite_model.Operator(
            opcode_index=self.builder.op_code_index_for_op_type(BuiltinOperator.LOGISTIC),
            builtin_options=None
        )
        logistic_op.tmp_inputs = [mul_1_output]
        logistic_op.tmp_outputs = [logistic_output]

        ops.append(logistic_op)

        # ---- Mul(x) ----
        mul_2_op = tflite_model.Operator(builtin_options=Mul())
        mul_2_op.tmp_inputs = [logistic_output, x]
        mul_2_op.tmp_outputs = [y]

        ops.append(mul_2_op)

        return ops
