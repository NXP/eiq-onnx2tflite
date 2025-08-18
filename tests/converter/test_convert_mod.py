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


@pytest.mark.parametrize("shape", [
    [42],
    [13, 37],
    [2, 4, 6],
    [2, 4, 6, 8],
    [2, 4, 6, 8, 10],
    [3, 1, 4, 1, 5, 9]
], ids=lambda x: f'{len(x)}D')
@pytest.mark.parametrize("_type", [
    TensorProto.INT32, TensorProto.INT64
], ids=name_for_onnx_type)
def test_convert_mod(shape: [int], _type: TensorProto.DataType):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Mod', ['a', 'b'], ['y'])],
        'Mod test',
        [
            onnx.helper.make_tensor_value_info('a', _type, shape),
            onnx.helper.make_tensor_value_info('b', _type, shape)
        ],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(shape[0])
    np_type = to_numpy_type(_type)
    a_data = (np.random.random(shape) * 1000 - 500).astype(np_type)
    b_data = (np.random.random(shape) * 1000 - 500).astype(np_type)
    b_data[b_data == 0] = 2  # Avoid dividing by 0.
    executors.convert_run_compare(onnx_model, {
        0: a_data,
        1: b_data
    })


def test_convert_mod__invalid_type():
    _type = TensorProto.FLOAT

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Mod', ['a', 'b'], ['y'])],
        'Mod test',
        [
            onnx.helper.make_tensor_value_info('a', _type, shape),
            onnx.helper.make_tensor_value_info('b', _type, shape)
        ],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL


@pytest.mark.parametrize("a_shape, b_shape", [
    ([2, 4, 6, 8], [8]),
    ([2, 4, 6, 8], [6, 1]),
    ([4, 1, 8], [2, 1, 6, 1]),
])
def test_convert_mod__broadcasting(a_shape: [int], b_shape: [int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Mod', ['a', 'b'], ['y'])],
        'Mod test',
        [
            onnx.helper.make_tensor_value_info('a', TensorProto.INT64, a_shape),
            onnx.helper.make_tensor_value_info('b', TensorProto.INT64, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(a_shape[0])
    a_data = (np.random.random(a_shape) * 1000 - 500).astype(np.int64)
    b_data = (np.random.random(b_shape) * 1000 - 500).astype(np.int64)
    b_data[b_data == 0] = 2  # Avoid dividing by 0.
    executors.convert_run_compare(onnx_model, {
        0: a_data,
        1: b_data
    })


def test_convert_mod_fmod():
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Mod', ['a', 'b'], ['y'], fmod=1)],
        'Mod test',
        [
            onnx.helper.make_tensor_value_info('a', TensorProto.INT64, shape),
            onnx.helper.make_tensor_value_info('b', TensorProto.INT64, shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'fmod=1' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_mod__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's1', 'zp1'], ['x1']),
            onnx.helper.make_node('QuantizeLinear', ['x', 's2', 'zp2'], ['x2']),
            onnx.helper.make_node('Mod', ['x1', 'x2'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s1', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp1', type_, [1], [12]),
            onnx.helper.make_tensor('s2', TensorProto.FLOAT, [1], [0.42]),
            onnx.helper.make_tensor('zp2', type_, [1], [42])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    error_code = logger.Code.NOT_IMPLEMENTED if type_ == TensorProto.INT8 else logger.Code.CONVERSION_IMPOSSIBLE
    assert logger.conversion_log.get_node_error_code(2) == error_code
