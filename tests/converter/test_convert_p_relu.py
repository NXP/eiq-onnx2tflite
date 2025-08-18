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
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param([1], id="1D"),
        pytest.param([5, 768], id="2D"),
        pytest.param([5, 48, 16], id="3D"),
        pytest.param([5, 12, 4, 16], id="4D"),
        pytest.param([10, 6, 2, 8, 4], id="5D"),
        pytest.param([2, 5, 6, 2, 8, 4], id="6D"),
    ])
def test_convert_p_relu(shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('PRelu', ['x', 'slope'], ['o'])],
        'PRelu test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("slope", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(shape)).reshape(shape).astype(np.float32),
        1: np.arange(np.prod(shape)).reshape(shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, slope_shape",
    [
        pytest.param([2, 4, 8, 16], [1]),
        pytest.param([2, 4, 8, 16], [16]),
        pytest.param([2, 4, 8, 16], [8, 1]),
        pytest.param([2, 4, 8, 16], [2, 1, 8, 1]),
    ])
def test_convert_p_relu_broadcasting(input_shape: list[int], slope_shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('PRelu', ['x', 'slope'], ['o'])],
        'PRelu test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("slope", TensorProto.FLOAT, slope_shape),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.arange(np.prod(slope_shape)).reshape(slope_shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, slope_shape",
    [
        pytest.param([2, 4, 8, 16], [1]),
        pytest.param([2, 4, 8, 16], [16]),
        pytest.param([2, 4, 8, 16], [8, 1]),
        pytest.param([2, 4, 8, 16], [2, 1, 8, 1]),
    ])
def test_convert_p_relu_channels_first_broadcasting(input_shape: list[int], slope_shape: list[int]):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('PRelu', ['x', 'slope'], ['o1']),
            onnx.helper.make_node('MaxPool', ['o1'], ['o'], kernel_shape=[1, 1]),
        ],
        'PRelu test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("slope", TensorProto.FLOAT, slope_shape),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.arange(np.prod(slope_shape)).reshape(slope_shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_p_relu_with_invalid_type():
    data_type = TensorProto.DOUBLE
    shape = [3, 4]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('PRelu', ['x', 'slope'], ['o'])],
        'PRelu test',
        [
            onnx.helper.make_tensor_value_info("x", data_type, shape),
            onnx.helper.make_tensor_value_info("slope", data_type, shape),
        ],
        [onnx.helper.make_tensor_value_info("o", data_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_prelu__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('PRelu', ['x1', 'x1'], ['y'])
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
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.INVALID_ONNX_MODEL
