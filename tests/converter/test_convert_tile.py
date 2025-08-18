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
from onnx2tflite.src.conversion_config import SkipShapeInferenceConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta import types
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize(
    "data_type",
    [
        TensorProto.INT8, TensorProto.INT32, TensorProto.INT64, TensorProto.UINT8, TensorProto.FLOAT, TensorProto.BOOL,
        TensorProto.STRING
    ], ids=name_for_onnx_type)
def test_convert_tile__input_type(data_type: TensorProto.DataType):
    shape = [2, 3, 4]

    np_type = types.to_numpy_type(data_type)
    x_data = (np.arange(np.prod(shape)) - 100).reshape(shape).astype(np_type)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Tile', ['x', 'repeats'], ['y'])],
        'Tile test',
        [onnx.helper.make_tensor_value_info('x', data_type, shape)],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
        [onnx.helper.make_tensor('repeats', TensorProto.INT64, [3], [3, 2, 4])]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


def test_convert_tile__unsupported_type():
    data_type = TensorProto.DOUBLE

    shape = [256]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Tile', ['x', 'repeats'], ['y'])],
        'Tile test',
        [onnx.helper.make_tensor_value_info('x', data_type, shape)],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
        [onnx.helper.make_tensor('repeats', TensorProto.INT64, [len(shape)], list(range(len(shape))))]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize(
    "repeats",
    [
        [1, 1, 1],
        [42, 13, 37],
        [4, 2, 0]
    ])
def test_convert_tile__repeats(repeats: list[int]):
    shape = [2, 3, 4]

    x_data = (np.arange(np.prod(shape)) - 100).reshape(shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Tile', ['x', 'repeats'], ['y'])],
        'Tile test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('repeats', TensorProto.INT64, [len(repeats)], repeats)]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


def test_convert_tile__channels_first__static_repeats():
    repeats = [1, 2, 3, 4]
    shape = [2, 3, 4, 5]

    x_data = (np.arange(np.prod(shape)) - 100).reshape(shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Tile', ['y1', 'repeats'], ['y'])
        ],
        'Tile test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('repeats', TensorProto.INT64, [len(repeats)], repeats)]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


def test_convert_tile__channels_first__dynamic_repeats():
    repeats = [1, 2, 3, 4]
    shape = [2, 3, 4, 5]

    output_shape = [s * r for s, r in zip(shape, repeats)]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Tile', ['y1', 'repeats'], ['y'])
        ],
        'Tile test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('repeats', TensorProto.INT64, [len(repeats)])
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, output_shape)],
        value_info=[onnx.helper.make_tensor_value_info('y1', TensorProto.FLOAT, shape)]

    )
    onnx_model = onnx.helper.make_model(graph)

    data = {
        0: (np.arange(np.prod(shape)) - 100).reshape(shape).astype(np.float32),
        1: np.array(repeats, np.int64)
    }

    # Skip shape inference, because it is not possible to infer the output shape of `Tile` with dynamic `repeats`.
    executors.convert_run_compare(onnx_model, data, conversion_config=SkipShapeInferenceConfig())


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_tile__quantized(type_: TensorProto.DataType):
    shape = [42]
    repeats = [5]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Tile', ['x1', 'repeats'], ['x2']),
            onnx.helper.make_node('Reshape', ['x2', 'flat_shape'], ['x3']),
            onnx.helper.make_node('DequantizeLinear', ['x3', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [1], [42 * 5]),
            onnx.helper.make_tensor('repeats', TensorProto.INT64, [len(repeats)], repeats)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
