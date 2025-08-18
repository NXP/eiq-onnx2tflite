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
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "h, w, block_size",
    [  # `h` and `w` must be divisible by `block_size`.
        (16, 8, 2),
        (8, 16, 4),
        (4, 4, 4),
        (2, 6, 2),
        (12, 18, 3),
        (10, 20, 5),
    ])
def test_convert_space_to_depth(h: int, w: int, block_size: int):
    shape = [2, 3, h, w]

    np.random.seed(42)
    data = np.random.random(shape).astype('float32')

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('SpaceToDepth', ['x'], ['y'], blocksize=block_size)],
        'SpaceToDepth test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


@pytest.mark.parametrize(
    "type_",
    [
        TensorProto.FLOAT
    ], ids=name_for_onnx_type)
def test_convert_space_to_depth__types(type_: TensorProto.DataType):
    shape = [2, 3, 4, 6]

    np.random.seed(42)
    data = np.random.random(shape).astype(to_numpy_type(type_))

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('SpaceToDepth', ['x'], ['y'], blocksize=2)],
        'SpaceToDepth test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_space_to_depth__unsupported_type():
    type_ = TensorProto.INT32

    shape = [2, 3, 4, 6]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('SpaceToDepth', ['x'], ['y'], blocksize=2)],
        'SpaceToDepth test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'INT32' in logger.conversion_log.get_node_error_message(0)


def test_convert_space_to_depth__format_handling(intermediate_tflite_model_provider):
    # Make sure that the IO tensors are marked as `channels_first` by the internal tensor format inference.

    shape = [2, 3, 4, 6]

    np.random.seed(42)
    data = np.random.random(shape).astype('float32')

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('SpaceToDepth', ['x'], ['y'], blocksize=2)],
        'SpaceToDepth test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.SPACE_TO_DEPTH, BuiltinOperator.TRANSPOSE
    ])


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_space_to_depth__quantized(type_: TensorProto.DataType):
    shape = [2, 3, 4, 6]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('SpaceToDepth', ['x1'], ['y'], blocksize=2)
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
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.NOT_IMPLEMENTED
