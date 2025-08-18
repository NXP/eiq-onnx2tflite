#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.converter.convert import convert_model
from onnx2tflite.src.onnx_parser.meta import types
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize(
    "shape",
    [
        [42], [13, 37], [5, 5, 5], [4, 6, 8, 10], [3, 1, 4, 1, 5], [1, 2, 3, 4, 5, 6], [1, 1, 2, 3, 5, 8, 13]
    ], ids=lambda x: f"{len(x)}D input")
def test_convert_mul(shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["x", "y"], ["o"])],
        'Mul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed((shape[0]))
    input_data = {
        0: np.random.random(np.prod(shape)).reshape(shape).astype(np.float32),
        1: np.random.random(np.prod(shape)).reshape(shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_mul_zero_dimension_input():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["input1", "input2"], ["output"])],
        'Mul test',
        [
            onnx.helper.make_tensor_value_info("input1", TensorProto.INT64, [3]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT64, ())],
        initializer=[onnx.helper.make_tensor("input2", TensorProto.INT64, (), [-1])]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(3).reshape([3]).astype(np.int64)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "data_type",
    [
        pytest.param(TensorProto.FLOAT, id="FLOAT"),
        pytest.param(TensorProto.INT64, id="INT64"),
        pytest.param(TensorProto.INT32, id="INT32"),
    ])
def test_convert_mul_different_types(data_type: int):
    shape = [4, 8, 12, 16]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["x", "y"], ["o"])],
        'Mul test',
        [
            onnx.helper.make_tensor_value_info("x", data_type, shape),
            onnx.helper.make_tensor_value_info("y", data_type, shape),
        ],
        [onnx.helper.make_tensor_value_info("o", data_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed((shape[0]))
    np_type = types.to_numpy_type(data_type)
    input_data = {
        0: np.random.random(np.prod(shape)).reshape(shape).astype(np_type),
        1: np.random.random(np.prod(shape)).reshape(shape).astype(np_type),
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "data_type",
    [
        pytest.param(TensorProto.FLOAT, id="FLOAT"),
        pytest.param(TensorProto.INT64, id="INT64"),
        pytest.param(TensorProto.INT32, id="INT32"),
    ])
@pytest.mark.parametrize(
    "index_of_ones_input",
    [0, 1])
def test_convert_mul_with_skipping(data_type: int, index_of_ones_input: int):
    shape = [4, 8, 12, 16]

    np.random.seed((shape[0]))
    np_type = types.to_numpy_type(data_type)
    ones = np.ones(shape, np_type)

    if index_of_ones_input == 0:
        inputs = ["ones", "x"]
    else:
        inputs = ["x", "ones"]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Mul", inputs, ["y"]),
            onnx.helper.make_node("Add", ["y", "scalar"], ["o"])
        ],
        'Mul test',
        [onnx.helper.make_tensor_value_info("x", data_type, shape)],
        [onnx.helper.make_tensor_value_info("o", data_type, ())],
        [
            onnx.helper.make_tensor("scalar", data_type, [1], [1]),
            onnx.helper.make_tensor("ones", data_type, shape, ones),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    input_data = np.random.random(np.prod(shape)).reshape(shape).astype(np_type)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_mul_impossible_skip():
    # Operator cannot be skipped, because it produces the model output
    shape = [4, 8, 12, 16]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["x", "y"], ["o"])],
        'Mul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("y", TensorProto.FLOAT, shape, np.ones(shape, np.float32))]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed((shape[0]))
    input_data = np.random.random(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_mul_invalid_type():
    shape = [4, 8, 12, 16]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["x", "y"], ["o"])],
        'Mul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.DOUBLE, shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.DOUBLE, shape),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.DOUBLE, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_mul_mismatched_types():
    shape = [4, 8, 12, 16]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["x", "y"], ["o"])],
        'Mul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.INT32, shape),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.INT32, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ([2, 4, 6, 8], [8]),
        ([2, 4, 1, 8], [6, 1]),
        ([4, 1, 8], [6, 1]),
        ([4, 1, 8], [2, 4, 6, 1]),
    ])
def test_convert_mul_with_broadcasting(shape1: list[int], shape2: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["x", "y"], ["o"])],
        'Mul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape1),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, shape2),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed((shape1[0]))
    input_data = {
        0: np.random.random(np.prod(shape1)).reshape(shape1).astype(np.float32),
        1: np.random.random(np.prod(shape2)).reshape(shape2).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ([2, 4, 6, 8], [8]),
        ([2, 4, 1, 8], [6, 1]),
        ([4, 1, 8], [6, 1]),
        ([4, 1, 8], [2, 1, 6, 1]),
    ])
def test_convert_mul_with_channels_first_broadcasting(shape1: list[int], shape2: list[int]):
    output_shape = np.broadcast_shapes(shape1, shape2)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Mul", ["x", "y"], ["z"]),
            onnx.helper.make_node("MaxPool", ["z"], ["o"], kernel_shape=[1] * (len(output_shape) - 2))
        ],
        'Mul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape1),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, shape2),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed((shape1[0]))
    input_data = {
        0: np.random.random(np.prod(shape1)).reshape(shape1).astype(np.float32),
        1: np.random.random(np.prod(shape2)).reshape(shape2).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_mul__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Mul', ['x1', 'x1'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.NOT_IMPLEMENTED
