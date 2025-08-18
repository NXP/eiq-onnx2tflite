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
from tests import executors


@pytest.mark.parametrize(
    "input_shape, alpha, beta, bias, size",
    [
        pytest.param([2, 4, 6, 8], 1e-5, 0.1, 0.2, 3, id="basic"),
        pytest.param([2, 4, 6, 8], 1.0, 1.0, -5.5, 99, id="large attributes"),
        pytest.param([2, 4, 6, 8], 1e-10, 1e-10, 1e-10, 1, id="tiny attributes"),
    ])
def test_convert_lrn(input_shape: List[int], alpha: float, beta: float, bias: float, size: int):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("LRN", ["input"], ["output"], alpha=alpha, beta=beta, bias=bias, size=size)],
        'LRN test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "size",
    [1, 3, 5, 7])
def test_convert_lrn_with_default_attributes(size: int):
    input_shape = [2, 4, 6, 8]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("LRN", ["input"], ["output"], size=size)],
        'LRN test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "alpha, beta, bias, size",
    [
        pytest.param(0.1, 0.2, 0.3, 4, id="even size"),
        pytest.param(0.1, 0.2, 0.3, 0, id="zero size"),
        pytest.param(0.1, 0.2, 0.3, -3, id="negative size"),

        pytest.param(0, 0.2, 0.3, 3, id="zero alpha"),
        pytest.param(-0.1, 0.2, 0.3, 3, id="negative alpha"),

        pytest.param(0.1, 0, 0.3, 3, id="zero beta"),
        pytest.param(0.1, -0.2, 0.3, 3, id="negative beta"),
    ])
def test_convert_lrn_with_invalid_attributes(alpha: float, beta: float, bias: float, size: int):
    input_shape = [2, 4, 6, 8]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("LRN", ["input"], ["output"], alpha=alpha, beta=beta, bias=bias, size=size)],
        'LRN test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE


def test_convert_lrn__invalid_type():
    type_ = TensorProto.DOUBLE

    shape = [2, 4, 6, 8]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LRN', ['x'], ['y'], size=3)],
        'Invalid type test test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)
