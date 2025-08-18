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
    "input_shape, scale, rtol",
    [
        pytest.param([101, 2], [0.3, 0.4], 1.e-4, id="2D"),
        pytest.param([15, 10, 5], [0.1, 0.2, 0.3, 0.4, 0.5], 1.e-4, id="3D"),
        pytest.param([12, 5, 10, 3], [0.1, 0.5, 0.42], 1.e-4, id="4D"),
        pytest.param([3, 5, 2, 7, 3], [0.1, 0.5, 0.42], 1.e-4, id="5D"),
    ])
def test_convert_layer_normalization_with_default_values(input_shape: List[int], scale: List[float], rtol: float):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("LayerNormalization", ["input", "scale"], ["output"]),
        ],
        'LayerNormalization test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("scale", TensorProto.FLOAT, [len(scale)], scale)]
    )

    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(input_shape[0])
    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, rtol=rtol)


@pytest.mark.parametrize(
    "input_shape, scale, bias, axis, epsilon, rtol",
    [
        pytest.param([42, 2, 3], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.15, 0.25, 0.35, 0.45, 0.55, 0.65], 1, 0.001,
                     1.e-4, id="3D, axis=1"),
        pytest.param([2, 1, 2, 2], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                     [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85], 0, 0.001,
                     1.e-4, id="4D, axis=0"),
        pytest.param([30, 2, 2, 2], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                     [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85], 1, 0.01,
                     1.e-4, id="4D, axis=1"),
        pytest.param([11, 17, 2, 4], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                     [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85], 2, 0.001,
                     2.e-4, id="4D, axis=2"),
    ])
def test_convert_layer_normalization(input_shape: List[int], scale: List[float], bias: List[float], axis: int,
                                     epsilon: float, rtol: float):
    scale_shape = input_shape[axis:]
    if type(scale_shape) != list:
        scale_shape = [scale_shape]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("LayerNormalization", ["input", "scale", "bias"], ["output"], axis=axis,
                                  epsilon=epsilon),
        ],
        'LayerNormalization test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("scale", TensorProto.FLOAT, scale_shape, scale),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, scale_shape, bias),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(input_shape[0])
    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, rtol=rtol)


@pytest.mark.parametrize(
    "input_shape, scale, bias, axis, epsilon",
    [
        pytest.param([30, 2, 3], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.15, 0.25, 0.35, 0.45, 0.55, 0.65], 1, 0.001,
                     id="3D, axis=1"),
        pytest.param([30, 2, 2, 2], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                     [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85], 1, 0.01,
                     id="4D, axis=1"),
        pytest.param([30, 17, 2, 4], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                     [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85], 2, 0.001,
                     id="4D, axis=2"),
    ])
def test_convert_layer_normalization_with_channels_first_tensors(input_shape: List[int], scale: List[float],
                                                                 bias: List[float], axis: int, epsilon: float):
    scale_shape = input_shape[axis:]
    kernel_shape = [1] * (len(input_shape) - 2)

    if type(scale_shape) != list:
        scale_shape = [scale_shape]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("LayerNormalization", ["input", "scale", "bias"], ["tmp"], axis=axis,
                                  epsilon=epsilon),
            onnx.helper.make_node("MaxPool", ["tmp"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape)

        ],
        'LayerNormalization test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("scale", TensorProto.FLOAT, scale_shape, scale),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, scale_shape, bias),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize("input_shape, scale, mean_shape, rtol", [
    pytest.param([101, 2], [0.3, 0.4], [101, 1], 1.e-4, id="2D"),
    pytest.param([30, 16, 42, 8], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                 [30, 16, 42, 1], 0.06, id="4D"),
])
def test_convert_layer_normalization_with_mean_output(input_shape: List[int], scale: List[float], mean_shape,
                                                      rtol: float):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("LayerNormalization", ["input", "scale"], ["output", "mean"]),
        ],
        'LayerNormalization test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [
            onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ()),
            onnx.helper.make_tensor_value_info("mean", TensorProto.FLOAT, mean_shape),
        ],
        [onnx.helper.make_tensor("scale", TensorProto.FLOAT, [len(scale)], scale)]
    )

    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(input_shape[0])
    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, rtol=rtol)


@pytest.mark.parametrize("input_shape, scale, mean_shape, rtol", [
    pytest.param([101, 2], [0.3, 0.4], [101, 1], 1.e-4, id="2D"),
    pytest.param([30, 16, 42, 8], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                 [30, 16, 42, 1], 0.06, id="4D"),
])
def test_convert_layer_normalization_with_three_outputs(input_shape: List[int], scale: List[float], mean_shape,
                                                        rtol: float):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("LayerNormalization", ["input", "scale"], ["output", "mean", "inv_std_dev"]),
        ],
        'LayerNormalization test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [
            onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ()),
            onnx.helper.make_tensor_value_info("mean", TensorProto.FLOAT, mean_shape),
            onnx.helper.make_tensor_value_info("inv_std_dev", TensorProto.FLOAT, mean_shape),
        ],
        [onnx.helper.make_tensor("scale", TensorProto.FLOAT, [len(scale)], scale)]
    )

    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(input_shape[0])
    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    logger.MIN_OUTPUT_IMPORTANCE = logger.MessageImportance.DEBUG
    executors.convert_run_compare(onnx_model, input_data, rtol=rtol)


def test_convert_layer_normalization__default_bias():
    # ONNX Runtime crashes badly (Windows fatal exception: access violation) if the bias is passed with name ''.
    # Therefore, this test only uses 2 inputs, instead of 3 with the last one being ''.

    shape = [15, 10, 5]
    scale = [0.1, 0.2, 0.3, 0.4, 0.5]
    rtol = 1.e-4

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('LayerNormalization', ['x', 's'], ['y']),
        ],
        'LayerNormalization test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('s', TensorProto.FLOAT, [len(scale)], scale)]
    )

    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(shape[0])
    input_data = np.random.random(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, rtol=rtol)


def test_convert_layer_normalization__unsupported_type():
    _type = TensorProto.INT8

    x_shape = [2, 4, 6, 8]
    scale_shape = b_shape = [x_shape[1]]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LayerNormalization', ['x', 's', 'b'], ['y'])],
        'LayerNormalization test',
        [
            onnx.helper.make_tensor_value_info('x', _type, x_shape),
            onnx.helper.make_tensor_value_info('s', _type, scale_shape),
            onnx.helper.make_tensor_value_info('b', _type, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL
    assert 'INT8' in logger.conversion_log.get_node_error_message(0)
