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
    "epsilon",
    [
        0.001, 0.123, 4.2, 420.
    ])
def test_convert_instance_normalization__epsilon(epsilon: float):
    x_shape = [2, 4, 6, 8]
    scale_shape = b_shape = [x_shape[1]]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('InstanceNormalization', ['x', 'scale', 'b'], ['y'], epsilon=epsilon)],
        'InstanceNormalization test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('scale', TensorProto.FLOAT, scale_shape),
            onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    np.random.seed(42)

    input_data = {
        0: (np.random.random(x_shape) * 10 - 5).astype(np.float32),
        1: (np.random.random(scale_shape) * 10 - 5).astype(np.float32),
        2: (np.random.random(b_shape) * 10 - 5).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1.e-6)


def test_convert_instance_normalization__default_epsilon():
    x_shape = [2, 4, 6, 8]
    scale_shape = b_shape = [x_shape[1]]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('InstanceNormalization', ['x', 'scale', 'b'], ['y'])],
        'InstanceNormalization test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('scale', TensorProto.FLOAT, scale_shape),
            onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    np.random.seed(42)

    input_data = {
        0: (np.random.random(x_shape) * 10 - 5).astype(np.float32),
        1: (np.random.random(scale_shape) * 10 - 5).astype(np.float32),
        2: (np.random.random(b_shape) * 10 - 5).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1.e-6)


@pytest.mark.parametrize(
    "x_shape",
    [
        [2, 4, 6],
        [2, 4, 6, 8],
        [2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7]
    ], ids=lambda x: f'{len(x)}D')
def test_convert_instance_normalization__different_ranks(x_shape: list[int]):
    scale_shape = b_shape = [x_shape[1]]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('InstanceNormalization', ['x', 'scale', 'b'], ['y'])],
        'InstanceNormalization test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('scale', TensorProto.FLOAT, scale_shape),
            onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    np.random.seed(42)

    input_data = {
        0: (np.random.random(x_shape) * 10 - 5).astype(np.float32),
        1: (np.random.random(scale_shape) * 10 - 5).astype(np.float32),
        2: (np.random.random(b_shape) * 10 - 5).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1.e-6)


def test_convert_instance_normalization__unsupported_type():
    _type = TensorProto.DOUBLE

    x_shape = [2, 4, 6, 8]
    scale_shape = b_shape = [x_shape[1]]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('InstanceNormalization', ['x', 'scale', 'b'], ['y'])],
        'InstanceNormalization test',
        [
            onnx.helper.make_tensor_value_info('x', _type, x_shape),
            onnx.helper.make_tensor_value_info('scale', _type, scale_shape),
            onnx.helper.make_tensor_value_info('b', _type, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
