#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx.helper
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from tests import executors


def test_two_skipped_ops_in_a_row(intermediate_tflite_model_provider):
    # ONNX: Mul -> Dropout -> Dropout -> Mul
    # TFLite: Mul -> Mul

    input_shape = [42, 23]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["x", "two"], ["o1"]),
                onnx.helper.make_node("Dropout", ["o1"], ["o2"]),
                onnx.helper.make_node("Dropout", ["o2"], ["o3"]),
                onnx.helper.make_node("Mul", ["o3", "two"], ["o"]),
            ],
            "test operator skipping",
            [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])],
        )
    )
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()

    # Make sure the model contains just 2 Mul operators.
    assert len(ops) == 2
    assert all(op.builtin_options.operator_type == BuiltinOperator.MUL for op in ops)


def test_all_ops_skipped(intermediate_tflite_model_provider):
    # The Mul ops multiply by 1.0, so they can be skipped. Except for the last one, which produces the model output.

    # ONNX: Mul -> Dropout -> Dropout -> Mul
    # TFLite: Mul

    input_shape = [42, 23]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["x", "one"], ["o1"]),
                onnx.helper.make_node("Dropout", ["o1"], ["o2"]),
                onnx.helper.make_node("Dropout", ["o2"], ["o3"]),
                onnx.helper.make_node("Mul", ["o3", "one"], ["o"]),
            ],
            "test operator skipping",
            [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])],
        )
    )
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()

    # Make sure the model contains just 1 Mul operator.
    assert len(ops) == 1
    assert ops[0].builtin_options.operator_type == BuiltinOperator.MUL


def test_last_op_skipped(intermediate_tflite_model_provider):
    # ONNX: Mul -> Dropout
    # TFLite: Mul

    input_shape = [42, 23]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["x", "two"], ["o1"]),
                onnx.helper.make_node("Dropout", ["o1"], ["o"]),
            ],
            "test operator skipping",
            [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])],
        )
    )
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()

    # Make sure the model contains just 1 Mul operator.
    assert len(ops) == 1
    assert ops[0].builtin_options.operator_type == BuiltinOperator.MUL


def test_single_op_not_skipped(intermediate_tflite_model_provider):
    # THe ONNX model contains a single Dropout operator. It cannot be skipped, because the input tensor would not
    #  be connected to the output tensor. Instead, it will get replaced by a Transpose operator, which does nothing.

    # ONNX: Dropout
    # TFLite: Transpose

    input_shape = [42, 23]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Dropout", ["x"], ["o"]),
            ],
            "test operator skipping",
            [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        )
    )
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()

    # Make sure the model contains just 1 Transpose operator.
    assert len(ops) == 1
    assert ops[0].builtin_options.operator_type == BuiltinOperator.TRANSPOSE
