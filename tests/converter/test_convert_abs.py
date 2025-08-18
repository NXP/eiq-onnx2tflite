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
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type, name_for_onnx_type
from tests import executors


@pytest.mark.parametrize(
    "_type",
    [
        TensorProto.INT16, TensorProto.INT32, TensorProto.FLOAT
    ], ids=name_for_onnx_type)
def test_convert_abs__types(_type: TensorProto.DataType):
    shape = [3, 14, 15]

    np_type = to_numpy_type(_type)
    data = (np.random.random(shape) * 100 - 50).astype(np_type)  # [-50, 50)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Abs', ['x'], ['y'])],
        'Abs test',
        [onnx.helper.make_tensor_value_info('x', _type, shape)],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_abs__quantized():
    _type = TensorProto.INT8

    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Abs', ['x1'], ['x2']),
            onnx.helper.make_node('DequantizeLinear', ['x2', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', _type, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.NOT_IMPLEMENTED


def test_convert_abs__unsupported_type():
    _type = TensorProto.DOUBLE

    shape = [256]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Abs', ['x'], ['y'])],
        'Abs test',
        [onnx.helper.make_tensor_value_info('x', _type, shape)],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_abs__int64__dynamic_input(intermediate_tflite_model_provider):
    shape = [3, 14, 15]

    data = (np.random.random(shape) * 100 - 50).astype(np.int64)  # [-50, 50)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Abs', ['x'], ['y'])],
        'Abs test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.INT64, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig({'cast_int64_to_int32': True})

    executors.convert_run_compare(onnx_model, data, conversion_config=config)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.CAST, BuiltinOperator.ABS, BuiltinOperator.CAST
    ])


def test_convert_abs__int64__static_input(intermediate_tflite_model_provider):
    shape = [3, 14, 15]

    data = (np.random.random(shape) * 100 - 50).astype(np.int64)  # [-50, 50)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Abs', ['x'], ['y'])],
        'Abs test',
        [],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
        [onnx.helper.make_tensor('x', TensorProto.INT64, shape, data)]
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig({'cast_int64_to_int32': True})

    executors.convert_run_compare(onnx_model, {}, conversion_config=config)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.ABS, BuiltinOperator.CAST
    ])


def test_convert_abs__int64__no_flag():
    shape = [256]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Abs', ['x'], ['y'])],
        'Abs test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.INT64, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert '--cast-int64-to-int32' in logger.conversion_log.get_node_error_message(0)
