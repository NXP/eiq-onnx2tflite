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


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        pytest.param([2, 4, 6, 8, 10, 12], [2, 4, 6, 8, 10, 12], id="6D"),
        pytest.param([2, 4, 6, 8, 10], [2, 4, 6, 8, 10], id="5D"),
        pytest.param([2, 4, 6, 8], [2, 4, 6, 8], id="4D"),
        pytest.param([5, 10, 15], [5, 10, 15], id="3D"),
        pytest.param([10, 15], [10, 15], id="2D"),
        pytest.param([42], [42], id="1D"),
    ])
def test_convert_sub(x_shape: List[int], y_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sub", ["x", "y"], ["output"])],
        'graph-sub',
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
        pytest.param([5, 10, 1, 5], [5, 1], id="4D-2D broadcasting"),
        pytest.param([5, 10, 15], [5, 10, 15], id="3D"),
        pytest.param([5, 10, 1], [15], id="3D-1D broadcasting"),
        pytest.param([10, 1], [15], id="2D"),
    ])
def test_convert_sub_with_zeros_input(x_shape: List[int], y_shape: List[int]):
    y_data = np.zeros(y_shape, dtype=np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Sub", ["x", "y"], ["z"]),
            onnx.helper.make_node('Flatten', ['z'], ['output'])
        ],
        'graph-sub',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("y", onnx.TensorProto.FLOAT, y_shape, y_data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [pytest.param([2, 10, 3], [2, 10, 3], id="3D")]
)
def test_convert_sub_with_y_zeros_input__is_graph_output(x_shape: List[int], y_shape: List[int]):
    y_data = np.zeros(y_shape, dtype=np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sub", ["x", "y"], ["z"])],
        'graph-sub',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("y", onnx.TensorProto.FLOAT, y_shape, y_data)],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [pytest.param([2, 10, 3], [2, 10, 3], id="3D")]
)
def test_convert_sub_with_x_zeros_input__is_graph_output(x_shape: List[int], y_shape: List[int]):
    x_data = np.zeros(x_shape, dtype=np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sub", ["x", "y"], ["z"])],
        'graph-sub',
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape)],
        [onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("x", onnx.TensorProto.FLOAT, x_shape, x_data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [pytest.param([2, 10, 3], [2, 10, 3], id="3D")]
)
def test_convert_sub_with_x_zeros_input(x_shape: List[int], y_shape: List[int]):
    x_data = np.zeros(x_shape, dtype=np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Sub", ["x", "y"], ["z"]),
            onnx.helper.make_node('Flatten', ['z'], ['output']),
        ],
        'graph-sub',
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("x", onnx.TensorProto.FLOAT, x_shape, x_data)],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(y_shape)).reshape(y_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [pytest.param(["batch", 10, 1], [15], id="3D-1D broadcasting")]
)
def test_convert_sub_with_symbolic_zeros_input(x_shape: List[int], y_shape: List[int]):
    y_data = np.zeros(y_shape, dtype=np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Sub", ["x", "y"], ["z"]),
            onnx.helper.make_node('Softmax', ['z'], ['output']),
        ],
        'graph-sub',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("y", onnx.TensorProto.FLOAT, y_shape, y_data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    x_shape = [1] + x_shape[1:]

    input_data = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [pytest.param(["batch", 10, 1], [15], id="3D")]
)
def test_convert_sub__with_symbolic_zeros_input__is_graph_output(x_shape: List[int], y_shape: List[int]):
    y_data = np.zeros(y_shape, dtype=np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sub", ["x", "y"], ["z"])],
        'graph-sub',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("y", onnx.TensorProto.FLOAT, y_shape, y_data)],
    )

    onnx_model = onnx.helper.make_model(graph)

    x_shape = [1] + x_shape[1:]

    input_data = np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        pytest.param([1, 1, 10, 12], [2, 4, 6, 8, 1, 1], id="6D"),
        pytest.param([10], [2, 4, 6, 8, 10], id="5D"),
        pytest.param([1, 6, 8], [2, 4, 1, 1], id="4D"),
        pytest.param([15], [5, 10, 15], id="3D"),
        pytest.param([15], [10, 1], id="2D"),
    ])
def test_convert_sub_with_shape_broadcasting(x_shape: List[int], y_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sub", ["x", "y"], ["output"])],
        'graph-sub',
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
    "x_shape, y_shape, output_rank",
    [
        pytest.param([1, 6, 8], [2, 4, 1, 1], 4, id="3D to 4D broadcast"),
        pytest.param([3, 6, 1, 1], [2, 4], 4, id="4D to 2D broadcast"),
        pytest.param([3, 6, 1, 1], [2], 4, id="4D to 1D broadcast"),
    ])
def test_convert_sub_with_channels_first_shape_broadcasting(x_shape: List[int], y_shape: List[int], output_rank: int):
    kernel_shape = [1] * (output_rank - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Sub", ["x", "y"], ["sub_out"]),
            onnx.helper.make_node("MaxPool", ["sub_out"], ["output"], kernel_shape=kernel_shape),
        ],
        'graph-sub',
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
    "data_type",
    [
        pytest.param(TensorProto.FLOAT, id="FLOAT"),
        pytest.param(TensorProto.INT64, id="INT64"),
        pytest.param(TensorProto.INT32, id="INT32"),
    ])
def test_convert_sub__types(data_type: int):
    shape = [4, 8, 12, 16]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sub", ["x", "y"], ["o"])],
        'Mul test',
        [
            onnx.helper.make_tensor_value_info("x", data_type, shape),
            onnx.helper.make_tensor_value_info("y", data_type, shape),
        ],
        [onnx.helper.make_tensor_value_info("o", data_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed((shape[0]))
    np_type = to_numpy_type(data_type)
    input_data = {
        0: np.random.random(np.prod(shape)).reshape(shape).astype(np_type),
        1: np.random.random(np.prod(shape)).reshape(shape).astype(np_type),
    }

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_sub__invalid_type():
    type_ = TensorProto.DOUBLE

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Sub', ['x', 'x'], ['y'])],
        'Invalid type test test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_sub__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x1', 's', 'zp'], ['x1_']),
            onnx.helper.make_node('QuantizeLinear', ['x2', 's', 'zp'], ['x2_']),
            onnx.helper.make_node('Sub', ['x1_', 'x2_'], ['x3']),
            onnx.helper.make_node('Reshape', ['x3', 'flat_shape'], ['x4']),
            onnx.helper.make_node('DequantizeLinear', ['x4', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [
            onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [1], [np.prod(shape)])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(2) == logger.Code.NOT_IMPLEMENTED
