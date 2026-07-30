#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import SkipShapeInferenceConfig
from onnx2tflite.src.converter.convert import convert_model
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("indices", [
    [[[0]]],
    [[[1, 0, 2], [2, 1, 0]]],
    [[[0, 1], [1, 1]], [[2, 0], [0, 2]]],
])
def test_convert_gather_elements(axis: int, indices: list):
    shape = [3, 3, 3]

    indices = np.asarray(indices, np.int64)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('GatherElements', ['x', 'indices'], ['y'], axis=axis)
        ],
        'GatherElements test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices.flat)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    data = np.random.random(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, data)


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("indices", [
    [[-1, 0], [0, -2]],
    [[-2, -1], [-1, -2]],
])
def test_convert_gather_elements__negative_indices(axis: int, indices: list):
    shape = [2, 2]

    indices = np.asarray(indices, np.int64)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('GatherElements', ['x', 'indices'], ['y'], axis=axis)
        ],
        'GatherElements test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices.flat)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    data = np.random.random(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, data)


def test_convert_gather_elements__channels_first__input():
    shape = [3, 4, 5, 6]
    indices = np.asarray([[[[1, 0, 2], [2, 1, 0]]]], np.int64)  # Shape [1, 1, 2, 3]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('GatherElements', ['x1', 'indices'], ['y'], axis=1)
        ],
        'GatherElements test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices.flat)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    data = np.random.random(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, data)


def test_convert_gather_elements__channels_first__output():
    shape = [3, 4, 5, 6]
    indices = np.random.choice([0, 1, 2], size=np.prod(shape)).reshape(shape).astype(np.int64)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('GatherElements', ['x', 'indices'], ['x1'], axis=1),
            onnx.helper.make_node('MaxPool', ['x1'], ['y'], kernel_shape=[1, 1]),
        ],
        'GatherElements test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices.flat)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    data = np.random.random(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, data)


def test_convert_gather_elements__dynamic_indices__no_flag():
    shape = [2, 2]

    indices = np.asarray([[0, 1], [1, 0]], np.int64)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('GatherElements', ['x', 'indices'], ['y'], axis=1)
        ],
        'GatherElements test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, indices.shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, indices.shape)],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert_model(onnx_model, conversion_config=SkipShapeInferenceConfig())
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert '--guarantee-non-negative-indices' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("axis", [0, 1])
def test_convert_gather_elements__dynamic_indices__flag(axis: int):
    shape = [2, 2]

    indices = np.asarray([[0, 1], [1, 0]], np.int64)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('GatherElements', ['x', 'indices'], ['y'], axis=axis)],
        'GatherElements test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, indices.shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, indices.shape)],
    )
    onnx_model = onnx.helper.make_model(graph)

    data = {
        0: np.arange(np.prod(shape)).reshape(shape).astype(np.float32),
        1: indices
    }

    config = SkipShapeInferenceConfig()
    config.guarantee_non_negative_indices = True
    executors.convert_run_compare(onnx_model, data, conversion_config=config)


@pytest.mark.parametrize("type_", [
    TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
    TensorProto.UINT8, TensorProto.FLOAT, TensorProto.STRING, TensorProto.BOOL
], ids=name_for_onnx_type)
def test_convert_gather_elements__types(type_: TensorProto.DataType):
    shape = [3, 4]

    indices = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]]

    data = (np.random.random(shape) * 100).astype(to_numpy_type(type_))

    indices = np.asarray(indices, np.int64)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('GatherElements', ['x', 'indices'], ['y'], axis=0)],
        'GatherElements test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices.flat)]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_gather_elements__invalid_type():
    type_ = TensorProto.DOUBLE

    shape = [3, 4]
    indices = np.asarray([[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]], np.int64)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('GatherElements', ['x', 'indices'], ['y'], axis=0)],
        'GatherElements test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices.flat)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_gather_elements__quantized(type_: TensorProto.DataType):
    shape = [3, 4]
    indices = np.asarray([[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]], np.int64)

    data = (np.random.random(shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('GatherElements', ['x1', 'indices'], ['x2'], axis=0),
            onnx.helper.make_node('DequantizeLinear', ['x2', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices.shape, indices.flat)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
