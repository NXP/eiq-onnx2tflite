#
# Copyright 2024 NXP
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
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    'shape',
    [
        [100], [10, 20], [5, 10, 15], [2, 4, 6, 8], [1, 2, 3, 4, 5]
    ], ids=lambda x: f'Shape = {x}.')
def test_convert_sqrt__positive_inputs(shape: list[int]):
    np.random.seed(42)
    x_data = np.random.rand(*shape).astype(np.float32) * 10  # Values ar strictly non-negative. <0, 10)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Sqrt', ['x'], ['y'])],
        'Sqrt test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    'shape',
    [
        [100], [10, 20], [5, 10, 15], [2, 4, 6, 8], [1, 2, 3, 4, 5]
    ], ids=lambda x: f'Shape = {x}.')
def test_convert_sqrt__negative_inputs(shape: list[int]):
    np.random.seed(42)
    x_data = (np.random.rand(*shape).astype(np.float32) - 0.5) * 20  # Values from -10 to 10

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Sqrt', ['x'], ['y'])],
        'Sqrt test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    'data_type',
    [
        TensorProto.FLOAT,
        # Other types are not supported right now.
    ], ids=lambda x: name_for_onnx_type(x))
def test_convert_sqrt__different_types(data_type: TensorProto.DataType):
    shape = [256]
    data = np.arange(-128, 128).astype(to_numpy_type(data_type))

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Sqrt', ['x'], ['y'])],
        'Sqrt test',
        [onnx.helper.make_tensor_value_info('x', data_type, shape)],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_sqrt__invalid_type():
    # Type is not supported by ONNX.
    data_type = TensorProto.INT8

    shape = [256]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Sqrt', ['x'], ['y'])],
        'Sqrt test',
        [onnx.helper.make_tensor_value_info('x', data_type, shape)],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL


def test_convert_sqrt__unsupported_type():
    # Type is not supported by TFLite.
    data_type = TensorProto.DOUBLE

    shape = [256]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Sqrt', ['x'], ['y'])],
        'Sqrt test',
        [onnx.helper.make_tensor_value_info('x', data_type, shape)],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_sqrt__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Sqrt', ['x1'], ['y'])
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
