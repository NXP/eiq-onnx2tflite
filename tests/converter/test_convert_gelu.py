#
# Copyright 2024-2025 NXP
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
def test_convert_gelu__shapes(shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Gelu', ['x'], ['y'])],
        'Gelu test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = (np.random.random(np.prod(shape)).reshape(shape).astype(np.float32) - 0.5) * 10  # <-5, 5)

    executors.convert_run_compare(onnx_model, input_data, atol=1.e-6)  # There is always some error for some reason.


@pytest.mark.parametrize(
    "approximate ",
    [
        'none',
        'tanh'
    ])
def test_convert_gelu__approximate(approximate: str):
    shape = [100]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Gelu', ['x'], ['y'], approximate=approximate)],
        'Gelu test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = (np.random.random(np.prod(shape)).reshape(shape).astype(np.float32) - 0.5) * 10  # <-5, 5)

    executors.convert_run_compare(onnx_model, input_data, atol=6.8e-7)


def test_convert_gelu__unsupported_type():
    shape = [100]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Gelu', ['x'], ['y'])],
        'Gelu test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT16, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT16, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT16' in logger.conversion_log.get_node_error_message(0)


def test_convert_gelu__invalid_approximate():
    shape = [100]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Gelu', ['x'], ['y'], approximate='sigmoid')],
        'Gelu test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE
