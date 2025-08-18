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

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "x_shape, indices_shape, updates_shape",
    [
        pytest.param([8], [4, 1], [4], id='1D input'),
        pytest.param([10, 20], [5, 2], [5], id='2D input, scalar updates'),
        pytest.param([20, 20], [2, 5, 2], [2, 5], id='2D input, vector updates'),
        pytest.param([10, 10, 10], [5, 1, 1], [5, 1, 10, 10], id='3D input, tensor updates')
    ])
def test_convert_scatter_nd__dynamic_inputs(x_shape, indices_shape, updates_shape, intermediate_tflite_model_provider):
    indices = np.random.choice(np.arange(min(x_shape)), indices_shape, replace=False)
    updates = np.random.random(updates_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'])],
        'ScatterND test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('updates', TensorProto.FLOAT, updates_shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, indices_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = {
        0: (np.random.random(x_shape).astype(np.float32) - 0.5) * 20,  # <-10, 10)
        1: np.array(updates, np.float32),
        2: np.array(indices, np.int64)
    }
    config = ConversionConfig()
    config.non_negative_indices = True
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.CAST,
        BuiltinOperator.SCATTER_ND,
        BuiltinOperator.SCATTER_ND,
        BuiltinOperator.SELECT_V2
    ])


def test_convert_scatter_nd__channels_first__main_input():
    x_shape = [2, 4, 6, 8]
    indices_shape = [3, 3]
    updates_shape = [indices_shape[-1]] + list(x_shape[indices_shape[-1]:])
    indices = np.array([[1, 3, 5], [0, 2, 4], [1, 3, 2]], 'int64')
    updates = np.random.random(updates_shape)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('ScatterND', ['x1', 'indices', 'updates'], ['y'])
        ],
        'ScatterND test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('updates', TensorProto.FLOAT, updates_shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, indices_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = {
        0: (np.random.random(x_shape).astype(np.float32) - 0.5) * 20,  # <-10, 10)
        1: np.array(updates, np.float32),
        2: np.array(indices, np.int64)
    }
    config = ConversionConfig()
    config.non_negative_indices = True
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


def test_convert_scatter_nd__channels_first__updates():
    x_shape = [2, 4, 6, 8]
    indices_shape = [2, 1]
    updates_shape = indices_shape[:-1] + list(x_shape[indices_shape[-1]:])
    assert updates_shape == [2, 4, 6, 8]

    indices = np.array([[1], [0]], 'int64')
    updates = np.random.random(updates_shape)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['updates'], ['updates1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates1'], ['y'])
        ],
        'ScatterND test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('updates', TensorProto.FLOAT, updates_shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, indices_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = {
        0: (np.random.random(x_shape).astype(np.float32) - 0.5) * 20,  # <-10, 10)
        1: np.array(updates, np.float32),
        2: np.array(indices, np.int64)
    }
    config = ConversionConfig()
    config.non_negative_indices = True
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


def test_convert_scatter_nd__channels_first__indices():
    x_shape = [2, 4]
    indices_shape = [2, 2, 2, 2]
    updates_shape = indices_shape[:-1] + list(x_shape[indices_shape[-1]:])

    indices = np.array([[[[1, 3], [0, 2]], [[1, 0], [0, 0]]], [[[0, 1], [1, 2]], [[1, 1], [0, 3]]]], 'float32')
    updates = np.random.random(updates_shape)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['indices'], ['indices1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Cast', ['indices1'], ['indices2'], to=TensorProto.INT64),
            onnx.helper.make_node('ScatterND', ['x', 'indices2', 'updates'], ['y'])
        ],
        'ScatterND test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('updates', TensorProto.FLOAT, updates_shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.FLOAT, indices_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = {
        0: (np.random.random(x_shape).astype(np.float32) - 0.5) * 20,  # <-10, 10)
        1: np.array(updates, np.float32),
        2: indices
    }
    config = ConversionConfig()
    config.non_negative_indices = True
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


def test_convert_scatter_nd__static_indices__dynamic_updates(intermediate_tflite_model_provider):
    x_shape = [8]
    indices_shape = [4, 1]
    updates_shape = [4]

    indices = [4, 3, 1, 7]
    updates = [9., 10., 11., 12.]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'])],
        'ScatterND test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('updates', TensorProto.FLOAT, updates_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices_shape, indices)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = {
        0: (np.random.random(x_shape).astype(np.float32) - 0.5) * 20,  # <-10, 10)
        1: np.array(updates, np.float32)
    }

    executors.convert_run_compare(onnx_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.SCATTER_ND,
        BuiltinOperator.SCATTER_ND,
        BuiltinOperator.SELECT_V2
    ])


@pytest.mark.parametrize(
    "x_shape, indices_shape, updates_shape",
    [
        pytest.param([8], [4, 1], [4], id='1D input'),
        pytest.param([10, 20], [5, 2], [5], id='2D input, scalar updates'),
        pytest.param([20, 20], [2, 5, 2], [2, 5], id='2D input, vector updates'),
        pytest.param([10, 10, 10], [5, 1, 1], [5, 1, 10, 10], id='3D input, tensor updates'),
    ])
def test_convert_scatter_nd__static_inputs(x_shape, indices_shape, updates_shape, intermediate_tflite_model_provider):
    indices = np.random.choice(np.arange(min(x_shape)), indices_shape, replace=False)
    updates = np.random.random(updates_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'])],
        'ScatterND test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices_shape, indices),
            onnx.helper.make_tensor('updates', TensorProto.FLOAT, updates_shape, updates)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = (np.random.random(x_shape).astype(np.float32) - 0.5) * 20  # <-10, 10)

    executors.convert_run_compare(onnx_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.SCATTER_ND,
        BuiltinOperator.SCATTER_ND,
        BuiltinOperator.SELECT_V2
    ])


@pytest.mark.parametrize(
    "type_",
    [
        TensorProto.FLOAT,
        TensorProto.INT32, TensorProto.INT64,
        TensorProto.BOOL
    ], ids=name_for_onnx_type)
def test_convert_scatter_nd__types(type_: TensorProto.DataType):
    x_shape = [8]
    indices_shape = [4, 1]
    updates_shape = [4]

    indices = [[4], [3], [1], [7]]
    updates = [9., 10., 11., 12.]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'])],
        'ScatterND test',
        [
            onnx.helper.make_tensor_value_info('x', type_, x_shape),
            onnx.helper.make_tensor_value_info('updates', type_, updates_shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, indices_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', type_, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    np_type = to_numpy_type(type_)
    input_data = {
        0: (np.random.random(x_shape) * 20).astype(np_type),  # <0, 20)
        1: np.array(updates, np_type),
        2: np.array(indices, np.int64)
    }

    config = ConversionConfig()
    config.non_negative_indices = True
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


def test_convert_scatter_nd__invalid_type():
    type_ = TensorProto.DOUBLE

    x_shape = [8]
    indices_shape = [4, 1]
    updates_shape = [4]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'])],
        'ScatterND test',
        [
            onnx.helper.make_tensor_value_info('x', type_, x_shape),
            onnx.helper.make_tensor_value_info('updates', type_, updates_shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, indices_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', type_, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig()
    config.non_negative_indices = True
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=config)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


def test_convert_scatter_nd__dynamic_indices():
    x_shape = [8]
    indices_shape = [4, 1]
    updates_shape = [4]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'])],
        'ScatterND test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('updates', TensorProto.FLOAT, updates_shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, indices_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert '--guarantee-non-negative-indices' in logger.conversion_log.get_node_error_message(0)


def test_convert_scatter_nd__negative_indices():
    x_shape = [8]
    indices_shape = [4, 1]
    updates_shape = [4]

    indices = [-5, -4, -1, -2]
    updates = [9., 10., 11., 12.]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'])],
        'ScatterND test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('updates', TensorProto.FLOAT, updates_shape, updates),
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices_shape, indices)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = (np.random.random(x_shape).astype(np.float32) - 0.5) * 20  # <-10, 10)
    executors.convert_run_compare(onnx_model, input_data)


def test_convert_scatter_nd__reduction():
    x_shape = [8]
    indices_shape = [4, 1]
    updates_shape = [4]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'], reduction='add')],
        'ScatterND test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('updates', TensorProto.FLOAT, updates_shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, indices_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'reduction=add' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_scatter_nd__quantized__dynamic_inputs(type_: TensorProto.DataType):
    input_shape = [12]
    indices_shape = [4, 1]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['input', 's', 'zp'], ['x_quant']),
            onnx.helper.make_node('Split', ['x_quant', 'split'], ['x', 'updates']),
            onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'])
        ],
        'Quantized input test',
        [
            onnx.helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, indices_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('s2', TensorProto.FLOAT, [1], [0.123]),
            onnx.helper.make_tensor('zp2', type_, [1], [42]),
            onnx.helper.make_tensor('split', TensorProto.INT64, [2], [8, 4])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = {
        0: (np.random.random(input_shape) * 100).astype(np.float32),
        1: np.array([[4], [3], [1], [7]], np.int64)
    }

    config = ConversionConfig()
    config.non_negative_indices = True
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_scatter_nd__quantized__static_inputs(type_: TensorProto.DataType):
    x_shape = [8]
    indices_shape = [4, 1]
    updates_shape = [4]

    indices = [4, 3, 1, 7]
    updates = [9, 10, 11, 12]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x_q']),
            onnx.helper.make_node('ScatterND', ['x_q', 'indices', 'updates'], ['y'])
        ],
        'Quantized input test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('updates', type_, updates_shape, updates),
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices_shape, indices)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = {
        0: (np.random.random(x_shape) * 100).astype(np.float32)
    }

    config = ConversionConfig()
    config.non_negative_indices = True
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)
