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
@pytest.mark.parametrize('keepdims', [0, 1])
def test_convert_arg_min__axis__keepdims(axis: int, keepdims: int):
    shape = [4, 5, 6, 7]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ArgMin', ['x'], ['y'], axis=axis, keepdims=keepdims)],
        'ArgMin test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = (np.random.random(shape) * 10. - 5.).astype(np.float32)

    executors.convert_run_compare(onnx_model, data)


@pytest.mark.parametrize('type_', [
    TensorProto.FLOAT, TensorProto.INT32
], ids=lambda x: f'{name_for_onnx_type(x)}')
def test_convert_arg_min__types(type_: TensorProto.DataType):
    shape = [4, 5, 6, 7]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ArgMin', ['x'], ['y'])],
        'ArgMin test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = (np.random.random(shape) * 100.).astype(to_numpy_type(type_))

    executors.convert_run_compare(onnx_model, data)


def test_convert_arg_min__unsupported_type():
    type_ = TensorProto.DOUBLE

    shape = [4, 5, 6, 7]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ArgMin', ['x'], ['y'])],
        'ArgMin test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize('axis', [0, 1, 2, 3, -1, -2, -3, - 4])
@pytest.mark.parametrize('keepdims', [0, 1])
def test_convert_arg_min__channels_first(axis: int, keepdims: int):
    shape = [4, 5, 6, 7]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('ArgMin', ['y1'], ['y'], axis=axis, keepdims=keepdims)
        ],
        'ArgMin test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
        [onnx.helper.make_tensor('axis', TensorProto.INT32, [], [axis])]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.arange(0, np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_arg_min__select_last_index__true():
    shape = [4, 5, 6, 7]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ArgMin', ['x'], ['y'], select_last_index=1)],
        'ArgMin test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_arg_min__select_last_index__false():
    shape = [5, 10]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ArgMin', ['x'], ['y'], select_last_index=0)],
        'ArgMin test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.INT32, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.choice([0, 1], shape).astype('int32')  # Ensures duplicates.

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("_type", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_arg_min__unsupported_types(_type: TensorProto.DataType):
    shape = [4, 5, 6, 7]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('ArgMin', ['x'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', _type, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
