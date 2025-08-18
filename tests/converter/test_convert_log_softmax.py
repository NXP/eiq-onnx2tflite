#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize("input_shape,axis", [
    pytest.param((10,), 0, id="1D,axis=0"),
    pytest.param((10,), -1, id="1D,axis=-1"),
    pytest.param((2, 3), -1, id="2D,axis=-1"),
    pytest.param((2, 3), 0, id="2D,axis=0"),
    pytest.param((2, 3), 1, id="2D,axis=1"),
    pytest.param((2, 3, 4), 0, id="3D,axis=0"),
    pytest.param((2, 3, 4), 1, id="3D,axis=1"),
    pytest.param((2, 3, 4), 2, id="3D,axis=2"),
    pytest.param((2, 3, 4, 5), -4, id="4D,axis=-4"),
    pytest.param((2, 3, 4, 5), -1, id="4D,axis=-1"),
    pytest.param((2, 3, 4, 5), 0, id="4D,axis=0"),
    pytest.param((2, 3, 4, 5), 1, id="4D,axis=1"),
    pytest.param((2, 3, 4, 5), 2, id="4D,axis=2"),
    pytest.param((2, 3, 4, 5), 3, id="4D,axis=3"),
    pytest.param((2, 3, 4, 5, 6), -2, id="5D,axis=-2"),
    pytest.param((2, 3, 4, 5, 6), 0, id="5D,axis=0"),
    pytest.param((2, 3, 4, 5, 6), 3, id="5D,axis=3"),
])
@pytest.mark.parametrize("opset", [11, 13], ids=(lambda x: f"opset={x}"))
def test_convert_logsoftmax(input_shape, axis, opset):
    node = onnx.helper.make_node("LogSoftmax", ["x"], ["y"], axis=axis)

    graph = onnx.helper.make_graph(
        [node],
        "graph-softmax",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ())],
    )

    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-softmax",
        opset_imports=[onnx.helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)

    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("axis", list(range(-4, 4)), ids=(lambda x: f"axis={x}"))
@pytest.mark.parametrize("opset", [11, 13], ids=(lambda x: f"opset={x}"))
def test_convert_logsoftmax_with_channel_last(axis: int, opset: int):
    input_shape = (5, 3, 16, 16)
    kernel_shape = [1] * (len(input_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y'], kernel_shape=kernel_shape),
            onnx.helper.make_node('LogSoftmax', ['y'], ['z'], axis=axis)
        ],
        'maxpool+softmax',
        inputs=[onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())]
    )

    original_model = onnx.helper.make_model(graph, producer_name="onnx-maxpool-logsoftmax")
    onnx.checker.check_model(original_model)

    input_data = np.linspace(0., 1., math.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(original_model, input_data)


def test_convert_log_softmax__invalid_type():
    type_ = TensorProto.DOUBLE

    shape = [10, 20]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LogSoftmax', ['x'], ['y'])],
        'Invalid type test test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_log_softmax__quantized(type_: TensorProto.DataType):
    shape = [5, 10]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('LogSoftmax', ['x1'], ['y'])
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
