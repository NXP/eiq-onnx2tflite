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
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type, name_for_onnx_type
from tests import executors


@pytest.mark.parametrize(
    "shape",
    [
        [256],
        [10, 20],
        [6, 8, 10],
        [4, 6, 8, 10],
        [2, 4, 6, 8, 10]
    ], ids=lambda x: f'{len(x)}D')
def test_convert_max__1_input_skipped(shape: list[int], intermediate_tflite_model_provider):
    np.random.seed(42)
    data = np.random.random(shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Max', ['x'], ['x1']),
            onnx.helper.make_node('Mul', ['x1', 'two'], ['y'])
        ],
        'Max test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('two', TensorProto.FLOAT, [1], [2.])]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.MUL])


@pytest.mark.parametrize(
    "shape",
    [
        [256],
        [10, 20],
        [6, 8, 10],
        [4, 6, 8, 10],
        [2, 4, 6, 8, 10]
    ], ids=lambda x: f'{len(x)}D')
def test_convert_max__1_input_not_skipped(shape: list[int], intermediate_tflite_model_provider):
    np.random.seed(42)
    data = np.random.random(shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Max', ['x'], ['y'])],
        'Max test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.TRANSPOSE])


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        pytest.param([256], [256], id='1D, same shape'),
        pytest.param([10, 20], [10, 20], id='2D, same shape'),
        pytest.param([6, 8, 10], [6, 8, 10], id='3D, same shape'),
        pytest.param([4, 6, 8, 10], [4, 6, 8, 10], id='4D, same shape'),
        pytest.param([2, 4, 6, 8, 10], [2, 4, 6, 8, 10], id='5D, same shape'),

        pytest.param([1], [2, 4, 6], id='broadcasting left'),
        pytest.param([2, 4, 6, 8], [8], id='broadcasting right'),
        pytest.param([1, 3, 1, 4], [2, 37, 1, 42, 1], id='broadcasting both'),
    ])
def test_convert_max__2_inputs(shape1: list[int], shape2: list[int]):
    np.random.seed(42)
    data = {
        0: np.random.random(shape1).astype(np.float32),
        1: np.random.random(shape2).astype(np.float32)
    }

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Max', ['x1', 'x2'], ['y'])],
        'Max test',
        [
            onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape1),
            onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape2)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_max__3_inputs():
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Max', ['x1', 'x2', 'x3'], ['y'])],
        'Max test',
        [
            onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x3', TensorProto.FLOAT, shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize(
    "data_type",
    [
        TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64
    ], ids=lambda x: name_for_onnx_type(x))
def test_convert_max__types(data_type: TensorProto.DataType):
    shape = [42]

    np_type = to_numpy_type(data_type)

    np.random.seed(42)
    data = {
        0: (np.random.random(shape) * 10 - 5).astype(np_type),
        1: (np.random.random(shape) * 10 - 5).astype(np_type)
    }

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Max', ['x1', 'x2'], ['y'])],
        'Max test',
        [
            onnx.helper.make_tensor_value_info('x1', data_type, shape),
            onnx.helper.make_tensor_value_info('x2', data_type, shape)
        ],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_max__unsupported_type():
    data_type = TensorProto.DOUBLE

    shape = [42]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Max', ['x1', 'x2'], ['y'])],
        'Max test',
        [
            onnx.helper.make_tensor_value_info('x1', data_type, shape),
            onnx.helper.make_tensor_value_info('x2', data_type, shape)
        ],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_max__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Max', ['x1', 'x1'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', type_, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.NOT_IMPLEMENTED
