#
# Copyright 2023-2024,2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import List

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.tflite_generator.builtin_options import transpose_options, reshape_options
from tests import executors


@pytest.mark.parametrize(
    "shape_1, shape_2",
    [
        # Resulting shape must be 3D or 4D, for the MaxPool!

        pytest.param([8], [2, 4, 6, 8]),
        pytest.param([2, 4, 6, 1], [8]),
        pytest.param([6, 1], [2, 4, 6, 8]),
        pytest.param([2, 4, 6, 8], [6, 8]),
        pytest.param([2, 4, 1, 8], [6, 1]),
        pytest.param([4, 1, 8], [2, 4, 6, 8]),
        pytest.param([4, 1, 1], [2, 4, 6, 8]),
        pytest.param([2, 1, 6, 8], [4, 1, 1]),
        pytest.param([4, 1, 8], [2, 1, 6, 1]),
        pytest.param([2, 1, 1, 8], [4, 6, 1]),
        pytest.param([2, 1, 1, 1], [4, 6, 8]),
        pytest.param([1, 6, 1], [2, 4, 1, 8]),
        pytest.param([4, 6, 8], [2, 1, 1, 1]),
        pytest.param([2, 1, 6, 8], [2, 4, 1, 1]),
        pytest.param([2, 1, 6, 8], [1, 4, 1, 1]),
        pytest.param([2, 1, 6, 1], [1, 4, 1, 8]),
        pytest.param([2, 1, 1, 1], [1, 4, 6, 8]),

        pytest.param([15], [5, 10, 15]),
        pytest.param([15], [5, 10, 1]),
        pytest.param([10, 1], [5, 1, 15]),
        pytest.param([10, 1], [5, 10, 15]),
        pytest.param([10, 15], [5, 1, 1]),
        pytest.param([10, 15], [5, 10, 1]),
        pytest.param([1, 15], [5, 10, 1]),
        pytest.param([10, 15], [5, 1, 15]),
        pytest.param([10, 1], [5, 1, 15]),
        pytest.param([1, 10, 1], [5, 1, 15]),
        pytest.param([5, 10, 1], [5, 1, 15]),
        pytest.param([5, 1, 1], [1, 1, 15]),

    ])
def test_static_channels_first_broadcasting(shape_1: List[int], shape_2: List[int]):
    output_shape = list(np.broadcast_shapes(shape_1, shape_2))
    kernel_shape = [1] * (len(output_shape) - 2)
    np.random.seed(1)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Mul", ["input_1", "input_2"], ["mul_out"]),
            onnx.helper.make_node("MaxPool", ["mul_out"], ["output"], kernel_shape=kernel_shape,
                                  strides=kernel_shape),
        ],
        'Broadcasting test',
        [],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("input_1", TensorProto.FLOAT, shape_1,
                                    np.random.randint(0, 10, shape_1).astype(np.float32)),
            onnx.helper.make_tensor("input_2", TensorProto.FLOAT, shape_2,
                                    np.random.randint(0, 10, shape_2).astype(np.float32)),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})


@pytest.mark.parametrize(
    "shape_1, shape_2",
    [
        # Resulting shape must be 3D or 4D, for the MaxPool!

        pytest.param([8], [2, 4, 6, 8]),
        pytest.param([2, 4, 6, 1], [8]),
        pytest.param([6, 1], [2, 4, 6, 8]),
        pytest.param([2, 4, 6, 8], [6, 8]),
        pytest.param([2, 4, 1, 8], [6, 1]),
        pytest.param([4, 1, 8], [2, 4, 6, 8]),
        pytest.param([4, 1, 1], [2, 4, 6, 8]),
        pytest.param([2, 1, 6, 8], [4, 1, 1]),
        pytest.param([4, 1, 8], [2, 1, 6, 1]),
        pytest.param([2, 1, 1, 8], [4, 6, 1]),
        pytest.param([2, 1, 1, 1], [4, 6, 8]),
        pytest.param([1, 6, 1], [2, 4, 1, 8]),
        pytest.param([4, 6, 8], [2, 1, 1, 1]),
        pytest.param([2, 1, 6, 8], [2, 4, 1, 1]),
        pytest.param([2, 1, 6, 8], [1, 4, 1, 1]),
        pytest.param([2, 1, 6, 1], [1, 4, 1, 8]),
        pytest.param([2, 1, 1, 1], [1, 4, 6, 8]),

        pytest.param([15], [5, 10, 15]),
        pytest.param([15], [5, 10, 1]),
        pytest.param([10, 1], [5, 1, 15]),
        pytest.param([10, 1], [5, 10, 15]),
        pytest.param([10, 15], [5, 1, 1]),
        pytest.param([10, 15], [5, 10, 1]),
        pytest.param([1, 15], [5, 10, 1]),
        pytest.param([10, 15], [5, 1, 15]),
        pytest.param([10, 1], [5, 1, 15]),
        pytest.param([1, 10, 1], [5, 1, 15]),
        pytest.param([5, 10, 1], [5, 1, 15]),
        pytest.param([5, 1, 1], [1, 1, 15]),

    ])
def test_dynamic_channels_first_broadcasting(shape_1: List[int], shape_2: List[int]):
    output_shape = list(np.broadcast_shapes(shape_1, shape_2))
    kernel_shape = [1] * (len(output_shape) - 2)
    np.random.seed(1)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Mul", ["input_1", "input_2"], ["mul_out"]),
            onnx.helper.make_node("MaxPool", ["mul_out"], ["output"], kernel_shape=kernel_shape,
                                  strides=kernel_shape),
        ],
        'Broadcasting test',
        [
            onnx.helper.make_tensor_value_info("input_1", TensorProto.FLOAT, shape_1),
            onnx.helper.make_tensor_value_info("input_2", TensorProto.FLOAT, shape_2),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.random.randint(0, 10, shape_1).astype(np.float32),
        1: np.random.randint(0, 10, shape_2).astype(np.float32)
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("quantized_shape", [
    pytest.param([], id="0D quantized input"),
    pytest.param([1], id="1D quantized input")
])
def test_broadcasting__unitary_tensor__no_broadcasting_ops_added(
        quantized_shape: list[int], intermediate_tflite_model_provider
):
    shape = [2, 4, 6, 8]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('DequantizeLinear', ['static', 'scale', 'zp'], ['dequant_out']),
            onnx.helper.make_node('MaxPool', ['x'], ['maxpool_out'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Div', ['dequant_out', 'maxpool_out'], ['y_1']),
        ],
        name='Broadcasting test',
        inputs=[
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info('y_1', TensorProto.FLOAT, shape),
        ],
        initializer=[
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [1], [0.0042]),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [1], [12]),
            onnx.helper.make_tensor('static', TensorProto.INT8, quantized_shape, [10]),
        ]
    )

    model = onnx.helper.make_model(graph)

    cc = ConversionConfig()
    cc.optimization_whitelist = []  # Disable all optimizations to preserve original ops

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(model, input_data, conversion_config=cc)

    # 3 IO Transposes + 0 broadcasting Transposes
    assert intermediate_tflite_model_provider.get_op_count(transpose_options.Transpose) == 2
    assert intermediate_tflite_model_provider.get_op_count(reshape_options.Reshape) == 0


@pytest.mark.parametrize("quantized_shape", [
    pytest.param([], id="0D quantized input"),
    pytest.param([1], id="1D quantized input")
])
def test_broadcasting__unitary_tensor__no_broadcasting_ops_added__shared_dequant(
        quantized_shape: list[int], intermediate_tflite_model_provider
):
    shape = [2, 4, 6, 8]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('DequantizeLinear', ['static', 'scale', 'zp'], ['dequant_out']),
            onnx.helper.make_node('MaxPool', ['x'], ['maxpool_out'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Div', ['dequant_out', 'maxpool_out'], ['y_1']),
            onnx.helper.make_node('Div', ['dequant_out', 'maxpool_out'], ['y_2']),
        ],
        name='Broadcasting test',
        inputs=[
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info('y_1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('y_2', TensorProto.FLOAT, shape),
        ],
        initializer=[
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [1], [0.0042]),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [1], [12]),
            onnx.helper.make_tensor('static', TensorProto.INT8, quantized_shape, [10]),
        ]
    )

    model = onnx.helper.make_model(graph)

    cc = ConversionConfig()
    cc.optimization_whitelist = []  # Disable all optimizations to preserve original ops

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(model, input_data, conversion_config=cc)

    # 3 IO Transposes + 0 broadcasting Transposes
    assert intermediate_tflite_model_provider.get_op_count(transpose_options.Transpose) == 3
    assert intermediate_tflite_model_provider.get_op_count(reshape_options.Reshape) == 0


@pytest.mark.parametrize("quantized_shape", [[2], [6, 2]])
def test_broadcasting__dynamic_tensor__broadcasting_ops_added(
        quantized_shape: list[int], intermediate_tflite_model_provider
):
    shape = [2, 4, 6, 2]
    quantized_vals = np.arange(np.prod(quantized_shape)).reshape(quantized_shape).astype(np.int8)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('DequantizeLinear', ['static', 'scale', 'zp'], ['dequant_out']),
            onnx.helper.make_node('MaxPool', ['x'], ['maxpool_out'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Div', ['dequant_out', 'maxpool_out'], ['y_1']),
            onnx.helper.make_node('Div', ['dequant_out', 'maxpool_out'], ['y_2']),
        ],
        name='Broadcasting test',
        inputs=[
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info('y_1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('y_2', TensorProto.FLOAT, shape),
        ],
        initializer=[
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [1], [0.0042]),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [1], [12]),
            onnx.helper.make_tensor('static', TensorProto.INT8, quantized_shape, quantized_vals),
        ]
    )

    model = onnx.helper.make_model(graph)

    cc = ConversionConfig()
    cc.optimization_whitelist = []  # Disable all optimizations to preserve original ops

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(model, input_data, conversion_config=cc)

    # 3 IO Transposes + 2 broadcasting Transposes
    assert intermediate_tflite_model_provider.get_op_count(transpose_options.Transpose) == 5
    assert intermediate_tflite_model_provider.get_op_count(reshape_options.Reshape) == 2
