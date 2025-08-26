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
from onnx2tflite.src.onnx_parser.builtin_attributes import hard_sigmoid_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import add_options, minimum_options, mul_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS


class HardSigmoidConverter(NodeConverter):
    node = "HardSigmoid"

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/maximum_minimum.cc#L176-L252
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `HardSigmoid` operator into TFLite.

            The HardSigmoid function is :

                Max(0 ,Minimum(1, alpha * x + beta))

            which can be represented in TFLite as:

                x -> Mul(alpha) -> Add(beta) -> Min(1) -> Relu

                An alternative to Min(1) is ReluN1To1, if it has better support on the boards.

                !! If the conversion strategy is changed, update the optimization in
                    `combine_hard_sigmoid_and_mul_to_hard_swish.py` accordingly. !!


            In the future, it may be possible to convert the HardSigmoid into:

                x -> Mul(alpha) -> Add(beta) -> Relu0To1

             However, right now it is not supported on the boards and there is no Relu0To1 in the TFLite schema, so it
              might be a custom operator.

        :param node: ONNX NodeProto representing the `HardSigmoid` operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators to add to the model.
        """
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        self.assert_type_allowed(x.type)

        ops = []

        hs_attributes = cast(hard_sigmoid_attributes.HardSigmoid, node.attributes)

        # ---- Create a `Mul` operator ----
        if hs_attributes.alpha == 1.0:
            # Multiplying by 1. No need for a `Mul` operator.
            mul_output = x
        else:
            # Create a `Mul` operator
            mul_output = self.builder.duplicate_tensor(x)

            mul_op = tflite_model.Operator(builtin_options=mul_options.Mul())
            mul_op.tmp_inputs = [
                x,
                self.builder.create_tensor_for_data(np.array([hs_attributes.alpha], np.float32), "alpha")
            ]
            mul_op.tmp_outputs = [mul_output]

            ops.append(mul_op)

        # ---- Create ad `Add` operator ----
        if hs_attributes.beta == 0.0:
            # Adding 0. No need for an `Add` operator.
            add_output = mul_output
        else:
            # Create an `Add` operator
            add_output = self.builder.duplicate_tensor(x)

            add_op = tflite_model.Operator(builtin_options=add_options.Add())
            add_op.tmp_inputs = [
                mul_output,
                self.builder.create_tensor_for_data(np.array([hs_attributes.beta], np.float32), "beta")
            ]
            add_op.tmp_outputs = [add_output]

            ops.append(add_op)

        # ---- Create a `Minimum` operator ----
        # Alternatively we can use ReluN1To1, if it has better support by the boards.
        min_output = self.builder.duplicate_tensor(x)

        min_op = tflite_model.Operator(builtin_options=minimum_options.Minimum())
        min_op.tmp_inputs = [
            add_output,
            self.builder.create_tensor_for_data(np.array([1.0], np.float32), "one")
        ]
        min_op.tmp_outputs = [min_output]

        ops.append(min_op)

        # ---- Create a `Relu` operator ----
        relu_op = tflite_model.Operator(opcode_index=self.builder.op_code_index_for_op_type(BuiltinOperator.RELU))
        relu_op.tmp_inputs = [min_output]
        relu_op.tmp_outputs = [y]

        ops.append(relu_op)

        return ops
