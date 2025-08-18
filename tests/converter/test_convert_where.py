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

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize("shape", [[5, 2, 6, 3], [5, 2, 6], [10, 4], [10]], ids=(lambda x: f"shape={len(x)}D"))
def test_convert_where(shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Where", ["condition", "x", "y"], ["output"])],
        'Where test',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, shape),
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(shape)).reshape(shape) < 5,
        1: np.arange(np.prod(shape)).reshape(shape).astype(np.float32),
        2: np.arange(np.prod(shape)).reshape(shape).astype(np.float32) * 10,
    }

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_where_quantized():
    shape = [5, 2, 6, 3]
    scale = [1.0]
    zero_point = [0]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["x_quant"], axis=1),
            onnx.helper.make_node("QuantizeLinear", ["y", "scale", "zero_point"], ["y_quant"], axis=1),
            onnx.helper.make_node("Where", ["condition", "x_quant", "y_quant"], ["output"])
        ],
        'Where test quantized',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, shape),
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.UINT8, ())],
        [
            onnx.helper.make_tensor("scale", TensorProto.FLOAT, [len(scale)], scale),
            onnx.helper.make_tensor("zero_point", TensorProto.UINT8, [len(zero_point)], zero_point)
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(shape)).reshape(shape) < 5,
        1: np.arange(np.prod(shape)).reshape(shape).astype(np.float32),
        2: np.arange(np.prod(shape)).reshape(shape).astype(np.float32) * 10,
    }

    tflite_executor, _ = executors.convert_run_compare(onnx_model, input_data)
    output_quant_params = tflite_executor.get_output_details(0)['quantization_parameters']
    assert output_quant_params['scales'] == [1.0]
    assert output_quant_params['zero_points'] == [0]


def test_convert_where_with_y_as_initializer():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Where", ["condition", "x", "y"], ["output"])],
        'Where test with initializer',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, [3]),
            onnx.helper.make_tensor_value_info("x", TensorProto.INT64, [3]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT64, ())],
        initializer=[onnx.helper.make_tensor("y", TensorProto.INT64, [3], [1, -1, -1])],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.array([0, 1, 0], dtype=bool),
        1: np.array([10, 20, 30], dtype=np.int64),
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "cond_shape, x_shape, y_shape",
    [
        pytest.param([2, 3, 4, 5], [5], [2, 3, 4, 5], id="4D, x broadcast"),
        pytest.param([2, 3, 4, 5], [5], [5], id="4D, x/y broadcast"),
        pytest.param([3, 1, 1], [4, 1], [2, 3, 4, 5], id="4D, condition/x broadcast"),
        pytest.param([3, 1, 1], [4, 1], [2, 1, 1, 5], id="4D, all broadcast"),

        pytest.param([3, 4, 5], [5], [3, 4, 5], id="3D, x broadcast"),
        pytest.param([3, 4, 5], [5], [5], id="3D, x/y broadcast"),
        pytest.param([3, 1, 1], [4, 1], [3, 4, 5], id="3D, condition/x broadcast"),
        pytest.param([5], [4, 1], [3, 1, 1], id="3D, all broadcast"),
    ])
def test_convert_where_with_broadcasting(cond_shape: List[int], x_shape: List[int], y_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Where", ["condition", "x", "y"], ["output"])],
        'Where test',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, cond_shape),
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(cond_shape)).reshape(cond_shape) < 5,
        1: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        2: np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) * 10,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "cond_shape, x_shape, y_shape",
    [
        pytest.param([2, 3, 4, 5], [5], [2, 3, 4, 5], id="4D, x broadcast"),
        pytest.param([2, 3, 4, 5], [5], [5], id="4D, x/y broadcast"),
        pytest.param([3, 1, 1], [4, 1], [2, 3, 4, 5], id="4D, condition/x broadcast"),
        pytest.param([3, 1, 1], [4, 1], [2, 1, 1, 5], id="4D, all broadcast"),

        pytest.param([3, 4, 5], [5], [3, 4, 5], id="3D, x broadcast"),
        pytest.param([3, 4, 5], [5], [5], id="3D, x/y broadcast"),
        pytest.param([3, 1, 1], [4, 1], [3, 4, 5], id="3D, condition/x broadcast"),
        pytest.param([5], [4, 1], [3, 1, 1], id="3D, all broadcast"),
    ])
def test_convert_where_with_broadcasting_channel_last(cond_shape, x_shape, y_shape):
    output_shape = np.broadcast_shapes(cond_shape, x_shape, y_shape)
    kernel_shape = [1] * (len(output_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Where", ["condition", "x", "y"], ["where_out"]),
            onnx.helper.make_node("MaxPool", ["where_out"], ["output"], kernel_shape=kernel_shape,
                                  strides=kernel_shape),
        ],
        'Where test',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, cond_shape),
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(cond_shape)).reshape(cond_shape) < 5,
        1: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        2: np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32) * 10,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("type_", [
    TensorProto.INT32, TensorProto.INT64, TensorProto.UINT8, TensorProto.FLOAT
], ids=name_for_onnx_type)
def test_convert_where__types(type_: TensorProto.DataType):
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Where", ["condition", "x", "y"], ["output"])],
        'Where test',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, shape),
            onnx.helper.make_tensor_value_info("x", type_, shape),
            onnx.helper.make_tensor_value_info("y", type_, shape),
        ],
        [onnx.helper.make_tensor_value_info("output", type_, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = {
        0: np.arange(np.prod(shape)).reshape(shape) < 5,
        1: (np.random.random(shape) * 3).astype(to_numpy_type(type_)),
        2: (np.random.random(shape) * 3).astype(to_numpy_type(type_))
    }

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_where__invalid_type():
    type_ = TensorProto.DOUBLE

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Where", ["condition", "x", "y"], ["output"])],
        'Where test',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, shape),
            onnx.helper.make_tensor_value_info("x", type_, shape),
            onnx.helper.make_tensor_value_info("y", type_, shape),
        ],
        [onnx.helper.make_tensor_value_info("output", type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)
