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


@pytest.mark.parametrize(
    "shape",
    [
        [42],
        [5, 10],
        [6, 8, 10],
        [4, 6, 8, 10],
        [2, 4, 6, 8, 10]
    ], ids=lambda x: f'{len(x)}D')
def test_convert_sign__shapes(shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Sign', ['x'], ['y'])],
        'Sign test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.random(shape).astype(np.float32) - 0.5

    # Replace 20% of the data with `0`, to test correct behavior for `0` inputs.
    indices = np.random.choice(np.arange(input_data.size), replace=False, size=int(input_data.size * 0.2))
    assert indices.size != 0
    input_data.flat[indices] = 0

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("type_", [
    TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT32
], ids=name_for_onnx_type)
def test_convert_sign__types(type_: TensorProto.DataType):
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Sign', ['x'], ['y'])],
        'Sign test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = (np.random.random(shape) * 10. - 5.).astype(to_numpy_type(type_))

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_sign__unsupported_type():
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Sign', ['x'], ['y'])],
        'Sign test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT16, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT16, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT16' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_sign__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Sign', ['x1'], ['y'])
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
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.CONVERSION_IMPOSSIBLE
