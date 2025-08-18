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


@pytest.mark.parametrize('axis', [0, 1, 2, 3, -1, -2, -3, - 4])
@pytest.mark.parametrize('axis_type', [TensorProto.INT32, TensorProto.INT64])
def test_convert_cum_sum__axis(axis: int, axis_type: TensorProto.DataType):
    shape = [4, 5, 6, 7]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('CumSum', ['x', 'axis'], ['y'])],
        'CumSum test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('axis', axis_type, [], [axis])]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = (np.random.random(np.prod(shape)).reshape(shape).astype(np.float32) - 0.5) * 10  # <-5, 5)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_cum_sum__dynamic_axis_recast():
    shape = [4, 5, 6, 7]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('CumSum', ['x', 'axis'], ['y'])],
        'CumSum test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('axis', TensorProto.INT64, [1])
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize('exclusive', [0, 1], ids=lambda x: f' exclusive = {x}')
@pytest.mark.parametrize('reverse', [0, 1], ids=lambda x: f'reverse = {x} ')
def test_convert_cum_sum__exclusive__reverse(exclusive: int, reverse: int):
    shape = [4, 5, 6, 7]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('CumSum', ['x', 'axis'], ['y'], exclusive=exclusive, reverse=reverse)],
        'CumSum test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('axis', TensorProto.INT32, [], [2])]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = (np.random.random(np.prod(shape)).reshape(shape).astype(np.float32) - 0.5) * 10  # <-5, 5)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize('data_type', [
    TensorProto.INT32,
    TensorProto.INT64,
    TensorProto.FLOAT
], ids=lambda x: f'{name_for_onnx_type(x)}')
def test_convert_cum_sum__types(data_type: TensorProto.DataType):
    shape = [4, 5, 6, 7]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('CumSum', ['x', 'axis'], ['y'])],
        'CumSum test',
        [onnx.helper.make_tensor_value_info('x', data_type, shape)],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
        [onnx.helper.make_tensor('axis', TensorProto.INT32, [], [2])]
    )
    onnx_model = onnx.helper.make_model(graph)

    np_type = to_numpy_type(data_type)
    input_data = np.arange(0, np.prod(shape)).reshape(shape).astype(np_type)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_cum_sum__unsupported_type():
    data_type = TensorProto.DOUBLE

    shape = [4, 5, 6, 7]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('CumSum', ['x', 'axis'], ['y'])],
        'CumSum test',
        [onnx.helper.make_tensor_value_info('x', data_type, shape)],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
        [onnx.helper.make_tensor('axis', TensorProto.INT32, [], [2])]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize('axis', [0, 1, 2, 3, -1, -2, -3, - 4])
def test_convert_cum_sum__channels_first(axis: int):
    shape = [4, 5, 6, 7]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('CumSum', ['y1', 'axis'], ['y'])
        ],
        'CumSum test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('axis', TensorProto.INT32, [], [axis])]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(0, np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("_type", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_cum_sum__quantized(_type: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('CumSum', ['x1'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', _type, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.INVALID_ONNX_MODEL
