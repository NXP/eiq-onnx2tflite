#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param([1], id="1D"),
        pytest.param([5, 768], id="2D"),
        pytest.param([5, 48, 16], id="3D"),
        pytest.param([5, 12, 4, 16], id="4D"),
        pytest.param([10, 6, 2, 8, 4], id="5D"),
    ])
def test_convert_exp(input_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Exp', ['input'], ['output'])],
        'Exp test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param([5, 48, 16], id="3D"),
        pytest.param([5, 12, 4, 16], id="4D"),
    ])
def test_convert_exp_with_channels_first(input_shape: List[int]):
    kernel_shape = [1] * (len(input_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Exp", ["x"], ["exp_out"]),
            onnx.helper.make_node("MaxPool", ["exp_out"], ["output"], kernel_shape=kernel_shape),
        ],
        'Exp test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_exp_unsupported_input_type():
    input_shape = [5, 48, 16]
    input_type = TensorProto.DOUBLE

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Exp', ['input'], ['output'])],
        'Exp test',
        [onnx.helper.make_tensor_value_info("input", input_type, input_shape)],
        [onnx.helper.make_tensor_value_info("output", input_type, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model)

    assert e.value.error_code == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_exp__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Exp', ['x1'], ['y']),
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
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.INVALID_ONNX_MODEL
