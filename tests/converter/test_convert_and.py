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
from tests import executors


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ([42], [2, 42]),
        ([2, 4, 1, 8], [1, 6, 1])
    ])
def test_convert_and__shape_broadcasting__formatless(a_shape: list[int], b_shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('And', ['a', 'b'], ['y'])],
        'And test',
        [
            onnx.helper.make_tensor_value_info('a', TensorProto.BOOL, a_shape),
            onnx.helper.make_tensor_value_info('b', TensorProto.BOOL, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.BOOL, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = {
        0: np.random.choice([True, False], a_shape).astype('bool'),
        1: np.random.choice([True, False], b_shape).astype('bool')
    }

    executors.convert_run_compare(onnx_model, data)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ([4, 6, 1], [2, 1, 6, 8]),
        ([2, 4, 1, 8], [1, 6, 1])
    ])
def test_convert_and__shape_broadcasting__channels_first(a_shape: list[int], b_shape: list[int]):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('And', ['a', 'b'], ['x1']),
            onnx.helper.make_node('Cast', ['x1'], ['x2'], to=TensorProto.FLOAT),
            onnx.helper.make_node('MaxPool', ['x2'], ['y'], kernel_shape=[1, 1])
        ],
        'And test',
        [
            onnx.helper.make_tensor_value_info('a', TensorProto.BOOL, a_shape),
            onnx.helper.make_tensor_value_info('b', TensorProto.BOOL, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = {
        0: np.random.choice([True, False], a_shape).astype('bool'),
        1: np.random.choice([True, False], b_shape).astype('bool')
    }

    executors.convert_run_compare(onnx_model, data)


def test_convert_and__unsupported_type():
    type_ = TensorProto.DOUBLE

    shape = [256]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('And', ['x', 'x'], ['y'])],
        'And test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)
