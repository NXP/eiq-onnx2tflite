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
    "input_shape, alpha",
    [
        pytest.param([42], -1.5, id="1D, alpha = -1.5"),
        pytest.param([13, 37], 4.2, id="2D, alpha = 4.2"),
        pytest.param([5, 10, 15], -0.1, id="3D, alpha = -0.1"),
        pytest.param([4, 8, 12, 16], 0.003, id="4D, alpha = 0.003"),
        pytest.param([2, 4, 8, 16, 32], 0.5, id="5D, alpha = 0.5"),
        pytest.param([3, 1, 4, 1, 5, 9], 1.0, id="6D, alpha = 1.0"),
    ])
def test_convert_leaky_relu(input_shape: List[int], alpha: float):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("LeakyRelu", ["x"], ["o"], alpha=alpha)],
        'LeakyRelu test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(input_shape[0])
    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)  # Range (0, 1)
    input_data = (5 - 10 * input_data)  # Range (-5, 5)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_leaky_relu_with_default_alpha():
    input_shape = [4, 8, 12, 16]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("LeakyRelu", ["x"], ["o"])],
        'LeakyRelu test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(input_shape[0])
    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)  # Range (0, 1)
    input_data = (5 - 10 * input_data)  # Range (-5, 5)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_leaky_relu_with_unsupported_type():
    # Test with int8

    input_shape = [4, 8, 12, 16]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("LeakyRelu", ["x"], ["o"])],
        'LeakyRelu test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.INT8, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL
    assert 'INT8' in logger.conversion_log.get_node_error_message(0)
