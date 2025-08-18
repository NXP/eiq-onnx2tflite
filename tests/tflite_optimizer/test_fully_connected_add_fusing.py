#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx.helper
import pytest
from onnx import TensorProto

from onnx2tflite.lib.tflite.ActivationFunctionType import ActivationFunctionType
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from tests import executors


@pytest.mark.parametrize(
    "bias_shape",
    [
        [16], [1, 16]
    ])
def test_fully_connected_add_fusing(bias_shape: list[int], intermediate_tflite_model_provider):
    # ONNX: Gemm -> Add
    # TFLite: FullyConnected

    m1_shape = [8, 4]
    m2_shape = [16, 4]

    bias_data = np.arange(np.prod(bias_shape)).reshape(bias_shape).astype(np.float32)

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Gemm", ["m1", "m2"], ["o1"], transB=1),
                onnx.helper.make_node("Add", ["o1", "b"], ["o"]),
            ],
            "test fc + add fusing",
            [
                onnx.helper.make_tensor_value_info("m1", TensorProto.FLOAT, m1_shape),
                onnx.helper.make_tensor_value_info("m2", TensorProto.FLOAT, m2_shape),
            ],
            [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor("b", TensorProto.FLOAT, bias_shape, bias_data)],
        )
    )
    input_data = {
        0: np.arange(np.prod(m1_shape)).reshape(m1_shape).astype(np.float32),
        1: np.arange(np.prod(m2_shape)).reshape(m2_shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)
    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 1
    assert ops[0].builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED


@pytest.mark.parametrize(
    "bias_shape",
    [
        [1, 1, 16], [1, 1, 1, 16], [8, 16], [2, 1, 16]
    ])
def test_fully_connected_add_fusing_incompatible_shapes(bias_shape: list[int], intermediate_tflite_model_provider):
    # ONNX: Gemm -> Add
    # TFLite: FullyConnected -> Add

    m1_shape = [8, 4]
    m2_shape = [16, 4]

    bias_data = np.arange(np.prod(bias_shape)).reshape(bias_shape).astype(np.float32)

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Gemm", ["m1", "m2"], ["o1"], transB=1),
                onnx.helper.make_node("Add", ["o1", "b"], ["o"]),
            ],
            "test fc + add fusing",
            [
                onnx.helper.make_tensor_value_info("m1", TensorProto.FLOAT, m1_shape),
                onnx.helper.make_tensor_value_info("m2", TensorProto.FLOAT, m2_shape),
            ],
            [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor("b", TensorProto.FLOAT, bias_shape, bias_data)]
        )
    )
    input_data = {
        0: np.arange(np.prod(m1_shape)).reshape(m1_shape).astype(np.float32),
        1: np.arange(np.prod(m2_shape)).reshape(m2_shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)
    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 2
    assert ops[0].builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED
    assert ops[1].builtin_options.operator_type == BuiltinOperator.ADD


def test_fully_connected_add_fusing_with_activation_function_between(intermediate_tflite_model_provider):
    # ONNX: Gemm -> Relu -> Add
    # TFLite: FullyConnected -> Add   (cannot fuse FC with ADD).

    m1_shape = [8, 4]
    m2_shape = [16, 4]
    bias_shape = [16]

    bias_data = np.arange(np.prod(bias_shape)).reshape(bias_shape).astype(np.float32)

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Gemm", ["m1", "m2"], ["o1"], transB=1),
                onnx.helper.make_node("Relu", ['o1'], ['o2']),
                onnx.helper.make_node("Add", ["o2", "b"], ["o"]),
            ],
            "test fc + add fusing",
            [
                onnx.helper.make_tensor_value_info("m1", TensorProto.FLOAT, m1_shape),
                onnx.helper.make_tensor_value_info("m2", TensorProto.FLOAT, m2_shape),
            ],
            [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor("b", TensorProto.FLOAT, bias_shape, bias_data)],
        )
    )
    input_data = {
        0: np.arange(np.prod(m1_shape)).reshape(m1_shape).astype(np.float32) - 10,
        1: np.arange(np.prod(m2_shape)).reshape(m2_shape).astype(np.float32) - 5,
    }

    executors.convert_run_compare(onnx_model, input_data)
    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 2
    assert ops[0].builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED
    assert ops[1].builtin_options.operator_type == BuiltinOperator.ADD
    assert hasattr(ops[0].builtin_options, 'fused_activation_function')
    assert ops[0].builtin_options.fused_activation_function == ActivationFunctionType.RELU


def test_fully_connected_add_fusing_with_activation_function_after(intermediate_tflite_model_provider):
    # ONNX: Gemm -> Add -> Relu
    # TFLite: FullyConnected

    m1_shape = [8, 4]
    m2_shape = [16, 4]
    bias_shape = [16]

    bias_data = np.arange(np.prod(bias_shape)).reshape(bias_shape).astype(np.float32)

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Gemm", ["m1", "m2"], ["o1"], transB=1),
                onnx.helper.make_node("Add", ["o1", "b"], ["o2"]),
                onnx.helper.make_node("Relu", ['o2'], ['o']),
            ],
            "test fc + add fusing",
            [
                onnx.helper.make_tensor_value_info("m1", TensorProto.FLOAT, m1_shape),
                onnx.helper.make_tensor_value_info("m2", TensorProto.FLOAT, m2_shape),
            ],
            [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor("b", TensorProto.FLOAT, bias_shape, bias_data)],
        )
    )
    input_data = {
        0: np.arange(np.prod(m1_shape)).reshape(m1_shape).astype(np.float32) - 10,
        1: np.arange(np.prod(m2_shape)).reshape(m2_shape).astype(np.float32) - 5,
    }

    executors.convert_run_compare(onnx_model, input_data)
    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 1
    assert ops[0].builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED
    assert hasattr(ops[0].builtin_options, 'fused_activation_function')
    assert ops[0].builtin_options.fused_activation_function == ActivationFunctionType.RELU
