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
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape",
    [
        pytest.param([2, 4, 6, 8, 10, 12], [2, 4, 6, 8, 10, 12], id="6D"),
        pytest.param([2, 4, 6, 8, 10], [2, 4, 6, 8, 10], id="5D"),
        pytest.param([2, 4, 6, 8], [2, 4, 6, 8], id="4D"),
        pytest.param([5, 10, 15], [5, 10, 15], id="3D"),
        pytest.param([10, 15], [10, 15], id="2D"),
        pytest.param([42], [42], id="1D"),
    ])
def test_convert_div(input_1_shape: List[int], input_2_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Div", ["input1", "input2"], ["output"])],
        'Div test',
        [
            onnx.helper.make_tensor_value_info("input1", TensorProto.FLOAT, input_1_shape),
            onnx.helper.make_tensor_value_info("input2", TensorProto.FLOAT, input_2_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.float32),
        1: np.arange(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "type_",
    [
        TensorProto.FLOAT,
        TensorProto.INT32
    ], ids=name_for_onnx_type)
def test_convert_div__types(type_: TensorProto.DataType):
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Div", ["a", "b"], ["y"])],
        'Div test',
        [
            onnx.helper.make_tensor_value_info("a", type_, shape),
            onnx.helper.make_tensor_value_info("b", type_, shape),
        ],
        [onnx.helper.make_tensor_value_info("y", type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np_type = to_numpy_type(type_)
    input_data = {
        0: np.random.random(shape).astype(np_type) * 100 - 50,
        1: np.random.random(shape).astype(np_type) * 10 - 5,
    }

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_div__unsupported_type():
    type_ = TensorProto.DOUBLE

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Div", ["a", "b"], ["y"])],
        'Div test',
        [
            onnx.helper.make_tensor_value_info("a", type_, shape),
            onnx.helper.make_tensor_value_info("b", type_, shape),
        ],
        [onnx.helper.make_tensor_value_info("y", type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape",
    [
        pytest.param([1, 1, 10, 12], [2, 4, 6, 8, 1, 1], id="6D"),
        pytest.param([10], [2, 4, 6, 8, 10], id="5D"),
        pytest.param([1, 6, 8], [2, 4, 1, 1], id="4D"),
        pytest.param([15], [5, 10, 15], id="3D"),
        pytest.param([15], [10, 1], id="2D"),
    ])
def test_convert_div_with_shape_broadcasting(input_1_shape: List[int], input_2_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Div", ["input1", "input2"], ["output"])],
        'Div test',
        [
            onnx.helper.make_tensor_value_info("input1", TensorProto.FLOAT, input_1_shape),
            onnx.helper.make_tensor_value_info("input2", TensorProto.FLOAT, input_2_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.float32),
        1: np.arange(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape",
    [
        pytest.param([1, 6, 8], [2, 4, 1, 1], id="4D"),
        pytest.param([15], [5, 10, 15], id="3D"),
    ])
def test_convert_div_with_channels_first_shape_broadcasting(input_1_shape: List[int], input_2_shape: List[int]):
    kernel_shape = [1] * (len(input_2_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Div", ["input1", "input2"], ["div_out"]),
            onnx.helper.make_node("MaxPool", ["div_out"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape),
        ],
        'Div test',
        [
            onnx.helper.make_tensor_value_info("input1", TensorProto.FLOAT, input_1_shape),
            onnx.helper.make_tensor_value_info("input2", TensorProto.FLOAT, input_2_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.float32),
        1: np.arange(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data)


cast_int64_to_int32_cc = ConversionConfig({"cast_int64_to_int32": True})


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape",
    [
        pytest.param([5, 10, 15], [5, 10, 15], id="3D"),
        pytest.param([10, 15], [10, 15], id="2D"),
    ])
def test_convert_div__int64(input_1_shape: List[int], input_2_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Div", ["input1", "input2"], ["output"])],
        'Div test',
        [
            onnx.helper.make_tensor_value_info("input1", TensorProto.INT64, input_1_shape),
            onnx.helper.make_tensor_value_info("input2", TensorProto.INT64, input_2_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT64, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.int64),
        1: np.arange(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.int64) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data, conversion_config=cast_int64_to_int32_cc)


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape",
    [
        pytest.param([5, 10, 15], [5, 10, 15], id="3D"),
        pytest.param([10, 15], [10, 15], id="2D"),
        pytest.param([42], [42], id="1D"),
    ])
def test_convert_div_static_input__int64(input_1_shape: List[int], input_2_shape: List[int]):
    input_2_data = np.arange(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.int64) + 1

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Div", ["input1", "input2"], ["output"])],
        'Div test',
        [onnx.helper.make_tensor_value_info("input1", TensorProto.INT64, input_1_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT64, ())],
        [onnx.helper.make_tensor("input2", TensorProto.INT64, input_2_shape, input_2_data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.int64)

    executors.convert_run_compare(onnx_model, input_data, conversion_config=cast_int64_to_int32_cc)


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape",
    [
        pytest.param([15], [5, 10, 15], id="3D"),
        pytest.param([15], [10, 1], id="2D"),
    ])
def test_convert_div_with_shape_broadcasting__int64(input_1_shape: List[int], input_2_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Div", ["input1", "input2"], ["output"])],
        'Div test',
        [
            onnx.helper.make_tensor_value_info("input1", TensorProto.INT64, input_1_shape),
            onnx.helper.make_tensor_value_info("input2", TensorProto.INT64, input_2_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT64, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.int64),
        1: np.arange(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.int64) + 1,
    }

    executors.convert_run_compare(onnx_model, input_data, conversion_config=cast_int64_to_int32_cc)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_div__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x1', 's', 'zp'], ['x1_q']),
            onnx.helper.make_node('QuantizeLinear', ['x2', 's', 'zp'], ['x2_q']),
            onnx.helper.make_node('Div', ['x1_q', 'x2_q'], ['y'])
        ],
        'Quantized input test',
        [
            onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape)
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
    assert logger.conversion_log.get_node_error_code(2) == logger.Code.NOT_IMPLEMENTED
