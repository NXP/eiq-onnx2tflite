#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx
from onnx import TensorProto

from onnx2tflite.src.conversion_config import ConversionConfig
from tests import executors
from onnx2tflite.src.tflite_optimizer.optimizer import Optimization


def _get_test_model(input_shape: list[int]) -> onnx.ModelProto:
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Add', ['x', 'x'], ['x1']),
            onnx.helper.make_node('Add', ['x1', 'x1'], ['x2']),
            onnx.helper.make_node('Add', ['x2', 'x2'], ['y']),
        ],
        'Keep one empty buffer test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)],
    )

    return onnx.helper.make_model(graph)


def test__disabled_optimization(intermediate_tflite_model_provider):
    input_shape = [42]

    onnx_model = _get_test_model(input_shape)

    config = ConversionConfig()
    config.optimization_whitelist = []

    input_data = np.random.random(input_shape).astype('float32')
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    buffers = intermediate_tflite_model_provider.get_buffers()
    assert len(buffers) == 4  # 1 for every tensor (x, x1, x2, y).


def test__enabled_optimization(intermediate_tflite_model_provider):
    input_shape = [42]

    onnx_model = _get_test_model(input_shape)

    config = ConversionConfig()
    config.optimization_whitelist = [Optimization.KEEP_ONE_EMPTY_BUFFER, Optimization.REMOVE_UNUSED_TENSORS]

    input_data = np.random.random(input_shape).astype('float32')
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    buffers = intermediate_tflite_model_provider.get_buffers()
    assert len(buffers) == 1  # Just 1 empty buffer left.
