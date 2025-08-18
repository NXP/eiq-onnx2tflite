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
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


def test_convert_pow__float32():
    shape = [100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Pow', ['base', 'pow'], ['o'])],
        'Pow test',
        [
            onnx.helper.make_tensor_value_info('base', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('pow', TensorProto.FLOAT, shape)
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    np.random.seed(42)

    # Use a combination of negative and positive numbers.
    input_data = {
        0: (np.random.random(100) * 10. - 5.).astype(np.float32),
        1: (np.random.random(100) * 10. - 5.).astype(np.float32)
    }

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_pow__int32():
    shape = [100]

    # TFLite doesn't support negative int32 powers, so just use positive values for that.
    np.random.seed(42)
    power = (np.random.random(100) * 5.).astype(np.int32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Pow', ['base', 'pow'], ['o'])],
        'Pow test',
        [onnx.helper.make_tensor_value_info('base', TensorProto.INT32, shape)],
        [onnx.helper.make_tensor_value_info('o', TensorProto.INT32, ())],
        [onnx.helper.make_tensor('pow', TensorProto.INT32, shape, power)]
    )

    onnx_model = onnx.helper.make_model(graph)

    # Use a combination of negative and positive numbers for the base.
    input_data = (np.random.random(100) * 10. - 5.).astype(np.int32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_pow__int32_dynamic_power():
    shape = [100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Pow', ['base', 'pow'], ['o'])],
        'Pow test',
        [
            onnx.helper.make_tensor_value_info('base', TensorProto.INT32, shape),
            onnx.helper.make_tensor_value_info('pow', TensorProto.INT32, shape)
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.INT32, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_pow__non_matching_types():
    shape = [100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Pow', ['base', 'pow'], ['o'])],
        'Pow test',
        [
            onnx.helper.make_tensor_value_info('base', TensorProto.INT32, shape),
            onnx.helper.make_tensor_value_info('pow', TensorProto.FLOAT, shape)
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.INT32, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


def test_convert_pow__unsupported_type():
    shape = [100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Pow', ['base', 'pow'], ['o'])],
        'Pow test',
        [
            onnx.helper.make_tensor_value_info('base', TensorProto.DOUBLE, shape),
            onnx.helper.make_tensor_value_info('pow', TensorProto.DOUBLE, shape)
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.DOUBLE, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize(
    "base_shape, pow_shape",
    [
        ([4, 10, 2], [1]),
        ([4, 10, 2], [2]),
        ([4, 10, 2], [10, 1]),

        ([5, 1], [10, 12, 1, 7]),
        ([1], [3, 4, 5, 6]),
    ])
def test_convert_pow__shape_broadcasting(base_shape: list[int], pow_shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Pow', ['base', 'pow'], ['o'])],
        'Pow test',
        [
            onnx.helper.make_tensor_value_info('base', TensorProto.FLOAT, base_shape),
            onnx.helper.make_tensor_value_info('pow', TensorProto.FLOAT, pow_shape)
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    np.random.seed(42)

    input_data = {
        0: (np.random.random(np.prod(base_shape)) * 10).reshape(base_shape).astype(np.float32),
        1: (np.random.random(np.prod(pow_shape)) * 10).reshape(pow_shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "base_shape, pow_shape",
    [
        ([4, 10, 3, 2], [1]),
        ([4, 10, 3, 2], [1, 1, 2]),
        ([4, 3, 10, 2], [10, 1]),
    ])
def test_convert_pow__shape_broadcasting_channels_first(base_shape: list[int], pow_shape: list[int]):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['base'], ['base1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Pow', ['base1', 'pow'], ['o'])
        ],
        'Pow test',
        [
            onnx.helper.make_tensor_value_info('base', TensorProto.FLOAT, base_shape),
            onnx.helper.make_tensor_value_info('pow', TensorProto.FLOAT, pow_shape)
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    np.random.seed(42)

    input_data = {
        0: (np.random.random(np.prod(base_shape)) * 10).reshape(base_shape).astype(np.float32),
        1: (np.random.random(np.prod(pow_shape)) * 10).reshape(pow_shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_pow__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Pow', ['x1', 'x1'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.INVALID_ONNX_MODEL
