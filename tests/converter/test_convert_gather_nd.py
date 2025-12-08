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
from onnx2tflite.src.conversion_config import SkipShapeInferenceConfig, ConversionConfig
from onnx2tflite.src.converter.convert import convert_model
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize("indices", [
    [[1], [0]],
    [2, 0, 1],
    [[2, 1], [0, 2]],
    [[2, 1], [0, 2], [1, 0], [2, 2]],
    [[[2], [1]], [[0], [2]]]
])
def test_convert_gather_nd(indices: list[int]):
    shape = [3, 4, 5]

    indices = np.asarray(indices)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('GatherND', ['x', 'indices'], ['y'])
        ],
        'GatherND test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = np.random.random(shape).astype('float32')

    executors.convert_run_compare(onnx_model, data)


@pytest.mark.parametrize("indices", [
    [[1], [0]],
    [2, 0, 1],
    [[-2, 1, 0, -5], [2, 3, -5, -6]],
    [[2, 1], [0, 2]],
    [[2, 1], [0, 2], [1, 0], [2, 2]],
    [[[2], [1]], [[0], [2]]]
])
def test_convert_gather_nd__channels_first__input(indices: list[int]):
    shape = [3, 4, 5, 6]

    indices = np.asarray(indices, 'int64')
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('GatherND', ['x1', 'indices'], ['y'])
        ],
        'GatherND test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices.flat)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = np.random.random(shape).astype('float32')

    executors.convert_run_compare(onnx_model, data)


def test_convert_gather_nd__channels_first__output():
    shape = [3, 4, 5, 6]
    indices = [[1], [0]]  # This will result in a 4D output.

    indices = np.asarray(indices, 'int64')
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('GatherND', ['x', 'indices'], ['x1']),
            onnx.helper.make_node('MaxPool', ['x1'], ['y'], kernel_shape=[1, 1]),
        ],
        'GatherND test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices.flat)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = np.random.random(shape).astype('float32')

    executors.convert_run_compare(onnx_model, data)


def test_convert_gather_nd__channels_first__indices():
    shape = [3, 4, 5, 6]
    indices = [[[[2, 1], [0, 1]]]]  # Shape is [1,1,2,2]

    indices = np.asarray(indices, 'float32')
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['indices'], ['indices1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Cast', ['indices1'], ['indices2'], to=TensorProto.INT64),
            onnx.helper.make_node('GatherND', ['x', 'indices2'], ['y']),
        ],
        'GatherND test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.FLOAT, indices.shape, indices.flat)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = np.random.random(shape).astype('float32')

    config = ConversionConfig()
    config.guarantee_non_negative_indices = True
    executors.convert_run_compare(onnx_model, data, conversion_config=config)


@pytest.mark.parametrize("indices", [
    [[-1], [0]],
    [-2, 0, -1],
    [[-2, -1, -3, -4], [-1, -2, -1, -1]],
    [-1, -2, -1, -1],
    [[-2, -1], [0, -2]],
    [[-2, -1], [0, -2], [-1, 0], [2, -2]],
    [[[-2], [-1]], [[0], [2]]]
])
def test_convert_gather_nd__negative_indices(indices: list[int]):
    shape = [3, 3, 4, 5]

    indices = np.asarray(indices, np.int64)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('GatherND', ['x', 'indices'], ['y'])
        ],
        'GatherND test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices.flat)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = np.random.random(shape).astype('float32')

    executors.convert_run_compare(onnx_model, data)


def test_convert_gather_nd__dynamic_indices__no_flag():
    shape = [3, 3, 4, 5]

    indices = np.asarray([0, -1])
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('GatherND', ['x', 'indices'], ['y'])
        ],
        'GatherND test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, [len(indices)])
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape[len(indices):])],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert_model(onnx_model, conversion_config=SkipShapeInferenceConfig())
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert '--guarantee-non-negative-indices' in logger.conversion_log.get_node_error_message(0)


def test_convert_gather_nd__dynamic_indices__flag():
    shape = [2, 2, 2]

    indices = np.asarray([[1], [0]], np.int64)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('GatherND', ['x', 'indices'], ['y'])
        ],
        'GatherND test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, indices.shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 2])],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = {
        0: np.arange(np.prod(shape)).reshape(shape).astype('float32'),
        1: indices
    }

    config = SkipShapeInferenceConfig()
    config.guarantee_non_negative_indices = True
    executors.convert_run_compare(onnx_model, data, conversion_config=config)


@pytest.mark.parametrize("type_", [
    TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
    TensorProto.UINT8, TensorProto.FLOAT, TensorProto.STRING, TensorProto.BOOL
], ids=name_for_onnx_type)
def test_convert_gather_nd__types(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    indices = [0, 1]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(to_numpy_type(type_))

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('GatherND', ['x', 'indices'], ['y'])],
        'GatherND test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('indices', TensorProto.INT64, [len(indices)], indices)]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_gather_nd__invalid_type():
    type_ = TensorProto.DOUBLE

    shape = [3, 14, 15]
    indices = [0, 1]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('GatherND', ['x', 'indices'], ['y'])],
        'GatherND test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('indices', TensorProto.INT64, [len(indices)], indices)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_gather_nd__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]
    indices = [0, 1]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('GatherND', ['x1', 'indices'], ['x2']),
            onnx.helper.make_node('Reshape', ['x2', 'flat_shape'], ['x3']),
            onnx.helper.make_node('DequantizeLinear', ['x3', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [2], [3, 5]),
            onnx.helper.make_tensor('indices', TensorProto.INT64, [len(indices)], indices)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_gather_nd__batch_dims():
    shape = [3, 3, 4, 5]

    indices = np.asarray([0, 1])
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('GatherND', ['x', 'indices'], ['y'], batch_dims=1)
        ],
        'GatherND test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape[len(indices):])],
        [onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'batch_dims' in logger.conversion_log.get_node_error_message(0)
