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
        [10, 20],
        [4, 6, 8],
        [2, 4, 6, 8],
        [2, 3, 4, 5, 6]
    ], ids=lambda x: f'{len(x)}D')
def test_convert_reverse_sequence__shapes(shape: list[int]):
    np.random.seed(42)
    sequence_lens = np.random.randint(low=1, high=shape[0], size=[shape[1]]).astype('int64')

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReverseSequence', ['x', 'sl'], ['y'])],
        'ReverseSequence test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sl', TensorProto.INT64, [len(sequence_lens)], sequence_lens)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.arange(np.prod(shape)).reshape(shape).astype('float32')

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_reverse_sequence__zero_sequence_len():
    shape = [4, 20]

    np.random.seed(42)
    sequence_lens = np.random.randint(low=0, high=shape[0], size=[shape[1]]).astype('int64')

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReverseSequence', ['x', 'sl'], ['y'])],
        'ReverseSequence test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sl', TensorProto.INT64, [len(sequence_lens)], sequence_lens)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'contains the value `0`' in logger.conversion_log.get_node_error_message(0)


def test_convert_reverse_sequence__dynamic_sequence_len():
    shape = [4, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReverseSequence', ['x', 'sl'], ['y'])],
        'ReverseSequence test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('sl', TensorProto.INT64, [shape[1]])
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'dynamic `sequence_lens`' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.FLOAT, TensorProto.UINT8, TensorProto.INT16, TensorProto.INT32,
                                   TensorProto.INT64], ids=name_for_onnx_type)
def test_convert_reverse_sequence__types(type_: TensorProto.DataType):
    shape = [4, 20]

    np.random.seed(42)
    sequence_lens = np.random.randint(low=1, high=shape[0], size=[shape[1]]).astype('int64')

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReverseSequence', ['x', 'sl'], ['y'])],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('sl', TensorProto.INT64, [len(sequence_lens)], sequence_lens)]
    )
    onnx_model = onnx.helper.make_model(graph)

    data = np.arange(np.prod(shape)).reshape(shape).astype(to_numpy_type(type_))

    executors.convert_run_compare(onnx_model, data)


def test_convert_reverse_sequence__unsupported_type():
    type_ = TensorProto.DOUBLE

    shape = [4, 10]
    np.random.seed(42)
    sequence_lens = np.random.randint(low=1, high=shape[0], size=[shape[1]]).astype('int64')

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReverseSequence', ['x', 'sl'], ['y'])],
        'ReverseSequence test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('sl', TensorProto.INT64, [len(sequence_lens)], sequence_lens)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


def test_convert_reverse_sequence__quantized():
    type_ = TensorProto.UINT8

    shape = [4, 10]
    np.random.seed(42)
    sequence_lens = np.random.randint(low=1, high=shape[0], size=[shape[1]]).astype('int64')
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('ReverseSequence', ['x1', 'sl'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('sl', TensorProto.INT64, [len(sequence_lens)], sequence_lens)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    data = np.arange(np.prod(shape)).reshape(shape).astype('float32')

    executors.convert_run_compare(onnx_model, data)


@pytest.mark.parametrize(
    "batch_axis, time_axis",
    [
        (0, 1),
        (1, 0)
    ])
def test_convert_reverse_sequence__specific_axes(batch_axis: int, time_axis: int):
    shape = [8, 6, 4, 2]

    np.random.seed(42)
    sequence_lens = np.random.randint(low=1, high=shape[time_axis], size=[shape[batch_axis]]).astype('int64')

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReverseSequence', ['x', 'sl'], ['y'], batch_axis=batch_axis, time_axis=time_axis)],
        'ReverseSequence test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sl', TensorProto.INT64, [len(sequence_lens)], sequence_lens)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.arange(np.prod(shape)).reshape(shape).astype('float32')

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "batch_axis, time_axis",
    [
        (0, 1),
        (1, 0)
    ])
def test_convert_reverse_sequence__channels_first(batch_axis: int, time_axis: int):
    shape = [8, 6, 4, 2]

    np.random.seed(42)
    sequence_lens = np.random.randint(low=1, high=shape[time_axis], size=[shape[batch_axis]]).astype('int64')

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('ReverseSequence', ['x1', 'sl'], ['y'], batch_axis=batch_axis, time_axis=time_axis)
        ],
        'ReverseSequence test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sl', TensorProto.INT64, [len(sequence_lens)], sequence_lens)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.arange(np.prod(shape)).reshape(shape).astype('float32')

    executors.convert_run_compare(onnx_model, input_data)
