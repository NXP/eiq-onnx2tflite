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
    "channels, block_size",
    [
        (16, 2),
        (16, 4),
        (36, 2),
        (18, 3),
        (50, 5),
    ])
def test_convert_depth_to_space(channels: int, block_size: int):
    shape = [2, channels, 3, 4]

    np.random.seed(42)
    data = np.random.random(shape).astype('float32')

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('DepthToSpace', ['x'], ['y'], blocksize=block_size)],
        'DepthToSpace test',
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
def test_convert_depth_to_space__types(type_: TensorProto.DataType):
    shape = [2, 16, 3, 4]

    np.random.seed(42)
    data = np.random.random(shape).astype(to_numpy_type(type_))

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('DepthToSpace', ['x'], ['y'], blocksize=2)],
        'DepthToSpace test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_depth_to_space__unsupported_type():
    type_ = TensorProto.INT32

    shape = [2, 16, 3, 4]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('DepthToSpace', ['x'], ['y'], blocksize=2)],
        'DepthToSpace test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'INT32' in logger.conversion_log.get_node_error_message(0)


def test_convert_depth_to_space__format_handling(intermediate_tflite_model_provider):
    # Make sure that the IO tensors are marked as `channels_first` by the internal tensor format inference.

    shape = [2, 16, 3, 4]

    np.random.seed(42)
    data = np.random.random(shape).astype('float32')

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('DepthToSpace', ['x'], ['y'], blocksize=2)],
        'DepthToSpace test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.DEPTH_TO_SPACE, BuiltinOperator.TRANSPOSE
    ])


def test_convert_depth_to_space__unsupported_mode():
    mode = 'CRD'

    shape = [2, 16, 3, 4]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('DepthToSpace', ['x'], ['y'], blocksize=2, mode=mode)],
        'DepthToSpace test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'CRD' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_depth_to_space__quantized(type_: TensorProto.DataType):
    shape = [2, 16, 3, 4]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('DepthToSpace', ['x1'], ['y'], blocksize=2)
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
