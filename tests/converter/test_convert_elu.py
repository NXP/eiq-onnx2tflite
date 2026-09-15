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

from onnx2tflite.src.converter import convert
from tests import executors
from onnx2tflite.src import errors
from onnx2tflite.src.errors import ConversionError


@pytest.mark.parametrize(
    "shape",
    [
        [42],
        [5, 10],
        [4, 6, 8],
        [2, 4, 6, 8],
        [2, 3, 4, 5, 6]
    ], ids=lambda x: f'{len(x)}D')
def test_convert_elu__shapes(shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Elu', ['x'], ['y'])],
        'Elu test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = (np.random.random(shape) * 10. - 5.).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_elu__unsupported_type():
    type_ = TensorProto.DOUBLE

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Elu', ['x'], ['y'])],
        'Elu test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)
    assert exc_info.value.code == errors.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in str(exc_info.value)


def test_convert_elu__invalid_alpha():
    shape = [100]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Elu', ['x'], ['y'], alpha=0.5)],
        'Elu test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)
    assert exc_info.value.code == errors.Code.NOT_IMPLEMENTED
    assert 'alpha=0.5' in str(exc_info.value)


def test_convert_elu__quantized_int8():
    shape = [42]

    data = (np.random.random(shape) * 10. - 5.).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('DequantizeLinear', ['x1', 's', 'zp'], ['x2']),
            onnx.helper.make_node('Elu', ['x2'], ['x3']),
            onnx.helper.make_node('QuantizeLinear', ['x3', 's', 'zp'], ['x4']),
            onnx.helper.make_node('DequantizeLinear', ['x4', 's', 'zp'], ['y'])
        ],
        'Quantized INT8 Elu test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_elu__quantized_uint8_unsupported():
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('DequantizeLinear', ['x1', 's', 'zp'], ['x2']),
            onnx.helper.make_node('Elu', ['x2'], ['x3']),
            onnx.helper.make_node('QuantizeLinear', ['x3', 's', 'zp'], ['x4']),
            onnx.helper.make_node('DequantizeLinear', ['x4', 's', 'zp'], ['y'])
        ],
        'Quantized UINT8 Elu test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', TensorProto.UINT8, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)
    assert exc_info.value.code == errors.Code.CONVERSION_IMPOSSIBLE
    assert 'INT8' in str(exc_info.value)
