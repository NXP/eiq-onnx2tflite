#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        pytest.param([2, 4, 6, 8, 10], [2, 4, 6, 10, 8], id="5D"),
        pytest.param([2, 4, 6, 8], [2, 4, 8, 6], id="4D"),
        pytest.param([5, 10, 15], [5, 15, 5], id="3D"),
        pytest.param([10, 15], [15, 3], id="2D"),
    ])
def test_convert_mat_mul_same_input_rank(x_shape: List[int], y_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["x", "y"], ["output"])],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        pytest.param([2, 4, 6, 8, 10], [10, 8], id="5D@2D"),
        pytest.param([2, 4, 6, 8, 10], [6, 10, 7], id="5D@3D"),
        pytest.param([2, 4, 6, 8], [4, 8, 6], id="4D@3D"),
        pytest.param([2, 4, 6, 8], [8, 6], id="4D@2D"),
        pytest.param([5, 10, 15], [15, 5], id="3D@2D"),
    ])
def test_convert_mat_mul_different_input_rank(x_shape: List[int], y_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["x", "y"], ["output"])],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        pytest.param([2, 4, 6, 8, 10], [1, 10, 7], id="5D@3D"),
        pytest.param([2, 4, 6, 8, 10], [1, 1, 10, 7], id="5D@4D"),
        pytest.param([2, 4, 6, 8], [1, 8, 6], id="4D@3D"),
        pytest.param([5, 10, 15], [15, 5], id="3D@2D"),
        pytest.param([1, 8, 10], [2, 4, 6, 10, 7], id="3D@5D"),
        pytest.param([1, 1, 8, 10], [2, 4, 6, 10, 7], id="4D@5D"),
    ])
def test_convert_mat_mul_different_input_rank_with_broadcasting(x_shape: List[int], y_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["x", "y"], ["output"])],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        pytest.param([2, 4, 6, 8, 10], [10], id="5D@1D"),
        pytest.param([10], [3, 6, 10, 7], id="1D@4D"),
    ])
def test_convert_mat_mul_1D_input(x_shape: List[int], y_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["x", "y"], ["output"])],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape, z_shape",
    [
        pytest.param([2, 4, 6, 8, 10], [10], [2, 4, 6, 8], id="5D@1D"),
        pytest.param([10], [3, 6, 10, 7], [3, 6, 7], id="1D@4D"),
    ])
def test_convert_mat_mul_channel_last_1D_input(x_shape: List[int], y_shape: List[int], z_shape: List[int]):
    kernel_shape = [1] * (len(z_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "y"], ["z"]),
            onnx.helper.make_node("MaxPool", ["z"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape)
        ],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape, z_shape",
    [
        pytest.param([10], [3, 6, 10, 7], [3, 6, 7], id="1D@4D"),
    ])
def test_convert_mat_mul_channel_last_1D_input_static_x(x_shape: List[int], y_shape: List[int], z_shape: List[int]):
    kernel_shape = [1] * (len(z_shape) - 2)

    x_data = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "y"], ["z"]),
            onnx.helper.make_node("MaxPool", ["z"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape)
        ],
        'MatMul test',
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("x", TensorProto.FLOAT, x_shape, x_data)],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape, z_shape",
    [
        pytest.param([10], [3, 6, 10, 7], [3, 6, 7], id="1D@4D"),
    ])
def test_convert_mat_mul_channel_last_1D_input_static_y(x_shape: List[int], y_shape: List[int], z_shape: List[int]):
    kernel_shape = [1] * (len(z_shape) - 2)

    y_data = np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "y"], ["z"]),
            onnx.helper.make_node("MaxPool", ["z"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape)
        ],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("y", TensorProto.FLOAT, y_shape, y_data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32) + 1

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape, z_shape",
    [
        pytest.param([2, 4, 8, 10], [10, 5], [2, 4, 8, 5], id="4D@2D"),
        pytest.param([5, 10], [3, 6, 10, 7], [3, 6, 5, 7], id="2D@4D"),
    ])
def test_convert_mat_mul_channel_last(x_shape: List[int], y_shape: List[int], z_shape: List[int]):
    kernel_shape = [1] * (len(z_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "y"], ["z"]),
            onnx.helper.make_node("MaxPool", ["z"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape)
        ],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape, z_shape",
    [
        pytest.param([2, 4, 8, 10], [10, 5], [2, 4, 8, 5], id="4D@2D"),
        pytest.param([1, 5, 10], [3, 6, 10, 7], [3, 6, 5, 7], id="3D@4D"),
        pytest.param([5, 10], [3, 6, 10, 7], [3, 6, 5, 7], id="2D@4D"),
    ])
def test_convert_mat_mul_channel_last_static_x(x_shape: List[int], y_shape: List[int], z_shape: List[int]):
    kernel_shape = [1] * (len(z_shape) - 2)

    x_data = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "y"], ["z"]),
            onnx.helper.make_node("MaxPool", ["z"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape)
        ],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("x", TensorProto.FLOAT, x_shape, x_data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape, z_shape",
    [
        pytest.param([2, 4, 8, 10], [10, 5], [2, 4, 8, 5], id="4D@2D"),
        pytest.param([1, 5, 10], [3, 6, 10, 7], [3, 6, 5, 7], id="3D@4D"),
        pytest.param([5, 10], [3, 6, 10, 7], [3, 6, 5, 7], id="2D@4D"),
        pytest.param([10], [3, 6, 10, 7], [3, 6, 7], id="1D@4D"),
    ])
def test_convert_mat_mul_channel_last_static_y(x_shape: List[int], y_shape: List[int], z_shape: List[int]):
    kernel_shape = [1] * (len(z_shape) - 2)

    y_data = np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "y"], ["z"]),
            onnx.helper.make_node("MaxPool", ["z"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape)
        ],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("y", TensorProto.FLOAT, y_shape, y_data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32) + 1

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape, z_shape",
    [
        pytest.param([2, 4, 6, 8, 10], [10], [2, 4, 6, 8], id="5D@1D"),
        pytest.param([2, 4, 3, 10], [10], [2, 4, 3], id="4D@1D"),
    ])
def test_convert_mat_mul_channel_last__1D_static_weight_into_FC(
        x_shape: List[int], y_shape: List[int], z_shape: List[int], intermediate_tflite_model_provider):
    kernel_shape = [1] * (len(z_shape) - 2)

    y_data = np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "y"], ["z"]),
            onnx.helper.make_node("MaxPool", ["z"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape)
        ],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("y", TensorProto.FLOAT, y_shape, y_data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32) + 1

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert ops[0].builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        pytest.param([4, 6, 8, 10], [10, 5], id="4D@2D"),
        pytest.param([4, 6, 8, 10], [10, 1], id="4D@2D-contains one"),
        pytest.param([4, 6, 10], [10, 5], id="3D@2D"),
    ])
def test_convert_mat_mul_formatless__2D_static_weight_into_FC(
        x_shape: List[int], y_shape: List[int], intermediate_tflite_model_provider):
    y_data = np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "y"], ["z"]),
        ],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        [onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("y", TensorProto.FLOAT, y_shape, y_data)],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32) + 1

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert ops[0].builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        pytest.param([4, 6, 8, 10], [10, 5], id="4D@2D"),
        pytest.param([4, 6, 8, 10], [10, 1], id="4D@2D-contains one"),
        pytest.param([4, 6, 10], [10, 5], id="3D@2D"),
    ])
def test_convert_mat_mul_formatless__2D_dynamic_weight_into_FC(
        x_shape: List[int], y_shape: List[int], intermediate_tflite_model_provider):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "y"], ["z"]),
        ],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1
    }

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert ops[1].builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED


@pytest.mark.parametrize(
    "x_shape, y_shape, z_shape",
    [
        pytest.param([4, 6, 8, 10], [10, 5], [4, 6, 8, 5], id="4D@2D"),
        pytest.param([4, 6, 8, 10], [10, 1], [4, 6, 8, 1], id="4D@2D-contains one"),
        pytest.param([4, 8, 10], [10, 1], [4, 8, 1], id="3D@2D"),
    ])
def test_convert_mat_mul_channel_last__2D_static_weight_into_FC(
        x_shape: List[int], y_shape: List[int], z_shape: List[int], intermediate_tflite_model_provider):
    kernel_shape = [1] * (len(z_shape) - 2)

    y_data = np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "y"], ["z"]),
            onnx.helper.make_node("MaxPool", ["z"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape)
        ],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("y", TensorProto.FLOAT, y_shape, y_data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32) + 1

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert ops[0].builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED


@pytest.mark.parametrize(
    "x_shape, y_shape, z_shape",
    [
        pytest.param([4, 6, 8, 10], [10, 5], [4, 6, 8, 5], id="4D@2D"),
        pytest.param([4, 6, 8, 10], [10, 1], [4, 6, 8, 1], id="4D@2D-contains one"),
        pytest.param([4, 6, 10], [10, 5], [4, 6, 5], id="3D@2D"),
    ])
def test_convert_mat_mul_channel_last__2D_dynamic_weight_into_FC(
        x_shape: List[int], y_shape: List[int], z_shape: List[int], intermediate_tflite_model_provider):
    kernel_shape = [1] * (len(z_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "y"], ["z"]),
            onnx.helper.make_node("MaxPool", ["z"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape)
        ],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1
    }

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert ops[1].builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        pytest.param([2, 4, 6, 8, 10], [10], id="5D@1D"),
        pytest.param([2, 4, 10], [10], id="3D@1D"),
    ])
def test_convert_mat_mul_1D_weights_dynamic(
        x_shape: List[int], y_shape: List[int], intermediate_tflite_model_provider):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["x", "y"], ["output"])],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert ops[1].builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED


def test_convert_mat_mul_incorrect_input_type():
    x_shape = [2, 4, 6, 8, 10]
    y_shape = [10, 8]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["x", "y"], ["output"])],
        'MatMul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.INT32, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.INT32, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT32, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model)

    assert e.value.error_code == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_mat_mul__quantized(type_: TensorProto.DataType):
    x1_shape = [2, 4, 6, 8, 10]
    x2_shape = [10, 8]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x1', 's', 'zp'], ['x1_q']),
            onnx.helper.make_node('QuantizeLinear', ['x2', 's', 'zp'], ['x2_q']),
            onnx.helper.make_node('MatMul', ['x1_q', 'x2_q'], ['y'])
        ],
        'Quantized input test',
        [
            onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, x1_shape),
            onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, x2_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', type_, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(2) == logger.Code.INVALID_ONNX_MODEL
