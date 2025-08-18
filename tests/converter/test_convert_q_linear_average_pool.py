#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List, Optional

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from tests import executors


@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([3], None, [1, 1], [1], id="kernel_shape = [3], pads = [1, 1], strides = [1]."),
    pytest.param([4], None, [1, 2], [1], id="kernel_shape = [4], pads = [1, 2], strides = [1]."),
    pytest.param([6], None, [2, 2], [2], id="kernel_shape = [6], pads = [2, 2], strides = [2]."),

    pytest.param([3], "SAME_LOWER", None, [1], id="kernel_shape = [3], auto_pad = SAME_LOWER, strides = [1]."),
    pytest.param([7], "SAME_UPPER", None, [3], id="kernel_shape = [7], auto_pad = SAME_UPPER, strides = [3]."),
])
def test_convert_1d_q_linear_average_pool(kernel_shape: List[int], auto_pad: Optional[str], pads: Optional[List[int]],
                                          strides: Optional[List[int]]):
    input_shape = [10, 20, 30]
    scale = [0.1]
    zp = [0]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"], auto_pad=auto_pad,
                               kernel_shape=kernel_shape, pads=pads, strides=strides, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], scale),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [], zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.int8)

    # 'atol'=1, because of different rounding methods of TFLite and ONNX.
    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([3, 3], None, [1, 1, 1, 1], [1, 1], id="kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1,1]."),
    pytest.param([3, 5], None, [1, 1, 1, 2], [2, 2], id="kernel_shape = [3, 5], pads = [1, 1, 1, 2], strides = [2,2]."),

    pytest.param([3, 4], "SAME_UPPER", None, [1, 3], id="kernel_shape=[3, 4], auto_pad=SAME_UPPER, strides=[1, 3]."),
    pytest.param([7, 2], "SAME_UPPER", None, [3, 2], id="kernel_shape=[7, 2], auto_pad=SAME_UPPER, strides=[3, 2]."),

    pytest.param([4, 9], "VALID", None, [3, 2], id="kernel_shape=[4, 9], auto_pad=VALID, strides=[3, 2]."),
    pytest.param([2, 10], "VALID", None, [4, 1], id="kernel_shape=[2, 10], auto_pad=VALID, strides=[4, 1]."),
])
def test_convert_2d_q_linear_average_pool(kernel_shape: List[int], auto_pad: Optional[str], pads: Optional[List[int]],
                                          strides: Optional[List[int]]):
    input_shape = [5, 10, 15, 20]
    scale = [0.1]
    zp = [0]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"],
                               kernel_shape=kernel_shape,
                               auto_pad=auto_pad, pads=pads, strides=strides, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], scale),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [], zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.int8)

    # 'atol'=1, because of different rounding methods of TFLite and ONNX.
    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([3, 3], None, [1, 1, 1, 1], [1, 1], id="kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1,1]."),
    pytest.param([3, 5], None, [1, 1, 1, 2], [2, 2], id="kernel_shape = [3, 5], pads = [1, 1, 1, 2], strides = [2,2]."),

    pytest.param([3, 4], "SAME_UPPER", None, [1, 3], id="kernel_shape=[3, 4], auto_pad=SAME_UPPER, strides=[1, 3]."),
    pytest.param([7, 2], "SAME_UPPER", None, [3, 2], id="kernel_shape=[7, 2], auto_pad=SAME_UPPER, strides=[3, 2]."),

    pytest.param([4, 9], "VALID", None, [3, 2], id="kernel_shape=[4, 9], auto_pad=VALID, strides=[3, 2]."),
    pytest.param([2, 10], "VALID", None, [4, 1], id="kernel_shape=[2, 10], auto_pad=VALID, strides=[4, 1]."),
])
@pytest.mark.parametrize("scale, zp", [
    pytest.param([0.1], [128], id="scale = 0.1, zero point = 128, "),
    pytest.param([2.3], [42], id="scale = 2.3, zero point = 42, "),
])
def test_convert_q_linear_average_pool_with_uint8(kernel_shape: List[int], auto_pad: Optional[str],
                                                  pads: Optional[List[int]], strides: Optional[List[int]],
                                                  scale: list[float], zp: list[int]):
    input_shape = [5, 10, 15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"],
                               kernel_shape=kernel_shape,
                               auto_pad=auto_pad, pads=pads, strides=strides, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.UINT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.UINT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], scale),
            onnx.helper.make_tensor('zp', TensorProto.UINT8, [], zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.uint8)

    # 'atol'=1, because of different rounding methods of TFLite and ONNX.
    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize("x_scale, x_zp, y_scale, y_zp", [
    pytest.param([0.1], [10], [1.2], [-23], id="scale: 0.1 -> 1.2, zero point: 10 -> -23"),
    pytest.param([0.5], [42], [1.4], [-80], id="scale: 0.5 -> 1.4, zero point: 42 -> -80"),
])
def test_convert_q_linear_average_pool_with_different_input_and_output_quantization(x_scale: list[float],
                                                                                    x_zp: list[int],
                                                                                    y_scale: list[float],
                                                                                    y_zp: list[int]):
    input_shape = [5, 10, 15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'x_scale', 'x_zp', 'y_scale', 'y_zp'], ["output"],
                               kernel_shape=[3, 3], auto_pad="SAME_UPPER", domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('x_scale', TensorProto.FLOAT, [], x_scale),
            onnx.helper.make_tensor('x_zp', TensorProto.INT8, [], x_zp),
            onnx.helper.make_tensor('y_scale', TensorProto.FLOAT, [], y_scale),
            onnx.helper.make_tensor('y_zp', TensorProto.INT8, [], y_zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.int8)

    # 'atol'=1, because of different rounding methods of TFLite and ONNX.
    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize("x_scale, x_zp, y_scale, y_zp", [
    pytest.param([0.1], [180], [1.2], [120], id="scale: 0.1 -> 1.2, zero point: 180 -> 120"),
    pytest.param([1.4], [10], [0.4], [140], id="scale: 1.4 -> 0.4, zero point: 10 -> 140"),
])
def test_convert_q_linear_average_pool_with_different_uint8_input_and_output_quantization(x_scale: list[float],
                                                                                          x_zp: list[int],
                                                                                          y_scale: list[float],
                                                                                          y_zp: list[int]):
    input_shape = [5, 10, 15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'x_scale', 'x_zp', 'y_scale', 'y_zp'], ["output"],
                               kernel_shape=[3, 3], auto_pad="SAME_UPPER", domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.UINT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.UINT8, ())],
        [
            onnx.helper.make_tensor('x_scale', TensorProto.FLOAT, [], x_scale),
            onnx.helper.make_tensor('x_zp', TensorProto.UINT8, [], x_zp),
            onnx.helper.make_tensor('y_scale', TensorProto.FLOAT, [], y_scale),
            onnx.helper.make_tensor('y_zp', TensorProto.UINT8, [], y_zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.uint8)

    # 'atol'=2, because of different rounding methods of TFLite and ONNX and the converted TFLite model contains 2
    #  Quantize operators, which seems to introduce a larger error than 1.
    executors.convert_run_compare(onnx_model, input_data, atol=2)


@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([3, 3], None, [1, 2, 1, 0], [1, 1], id="kernel_shape = [3, 3], pads = [1, 2, 1, 0], strides = [1,1]."),
    pytest.param([3, 5], None, [1, 3, 1, 2], [2, 2], id="kernel_shape = [3, 5], pads = [1, 3, 1, 2], strides = [2,2]."),

    pytest.param([3, 5], "SAME_LOWER", None, [3, 2], id="kernel_shape=[3, 5], auto_pad=SAME_LOWER, strides=[3, 2]."),
    pytest.param([7, 3], "SAME_LOWER", None, [4, 2], id="kernel_shape=[7, 3], auto_pad=SAME_LOWER, strides=[4, 2]."),
])
def test_convert_2d_q_linear_average_pool_with_explicit_padding(kernel_shape: List[int], auto_pad: Optional[str],
                                                                pads: Optional[List[int]],
                                                                strides: Optional[List[int]]):
    input_shape = [5, 10, 15, 20]
    scale = [0.1]
    zp = [42]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"],
                               kernel_shape=kernel_shape, count_include_pad=1,
                               auto_pad=auto_pad, pads=pads, strides=strides, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], scale),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [], zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.int8)

    # 'atol'=1, because of different rounding methods of TFLite and ONNX.
    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([3, 3], None, [1, 2, 1, 0], [1, 1], id="kernel_shape = [3, 3], pads = [1, 2, 1, 0], strides = [1,1]."),
    pytest.param([3, 5], None, [1, 3, 1, 2], [2, 2], id="kernel_shape = [3, 5], pads = [1, 3, 1, 2], strides = [2,2]."),

    pytest.param([3, 5], "SAME_LOWER", None, [3, 2], id="kernel_shape=[3, 5], auto_pad=SAME_LOWER, strides=[3, 2]."),
    pytest.param([7, 3], "SAME_LOWER", None, [4, 2], id="kernel_shape=[7, 3], auto_pad=SAME_LOWER, strides=[4, 2]."),
])
def test_convert_2d_q_linear_average_pool_with_unusable_explicit_padding(kernel_shape: List[int],
                                                                         auto_pad: Optional[str],
                                                                         pads: Optional[List[int]],
                                                                         strides: Optional[List[int]]):
    # Explicit padding must be added, but 'count_include_pad=0'!

    input_shape = [5, 10, 15, 20]
    scale = [0.1]
    zp = [42]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"],
                               kernel_shape=kernel_shape, count_include_pad=0,
                               auto_pad=auto_pad, pads=pads, strides=strides, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], scale),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [], zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([3, 3], None, [1, 1, 1, 1], [1, 1], id="kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1,1]."),
    pytest.param([3, 5], None, [1, 1, 1, 2], [2, 2], id="kernel_shape = [3, 5], pads = [1, 1, 1, 2], strides = [2,2]."),

    pytest.param([3, 4], "SAME_UPPER", None, [1, 3], id="kernel_shape=[3, 4], auto_pad=SAME_UPPER, strides=[1, 3]."),
    pytest.param([7, 2], "SAME_UPPER", None, [3, 2], id="kernel_shape=[7, 2], auto_pad=SAME_UPPER, strides=[3, 2]."),
])
def test_convert_2d_q_linear_average_pool_with_count_include_pad(kernel_shape: List[int], auto_pad: Optional[str],
                                                                 pads: Optional[List[int]],
                                                                 strides: Optional[List[int]]):
    # Padding is converted to TFLite 'SAME', but ONNX requires the 0s to be used in the computation of the mean.

    input_shape = [5, 10, 15, 20]
    scale = [0.1]
    zp = [42]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"],
                               kernel_shape=kernel_shape, count_include_pad=1,
                               auto_pad=auto_pad, pads=pads, strides=strides, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], scale),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [], zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize(
    "input_shape, error_code",
    [
        pytest.param([2, 4, 6, 8, 10], logger.Code.CONVERSION_IMPOSSIBLE, id="Impossible 5D"),
        pytest.param([2, 4, 6, 8, 10, 12], logger.Code.CONVERSION_IMPOSSIBLE, id="Impossible 6D (1)"),
        pytest.param([2, 4, 6, 1, 10, 12], logger.Code.CONVERSION_IMPOSSIBLE, id="Impossible 6D (2)"),

        pytest.param([2, 4, 1, 8, 10], logger.Code.NOT_IMPLEMENTED, id="Convertible 5D (1)"),
        pytest.param([2, 4, 6, 1, 1], logger.Code.NOT_IMPLEMENTED, id="Convertible 5D (2)"),
        pytest.param([2, 4, 1, 1, 1], logger.Code.NOT_IMPLEMENTED, id="Convertible 5D (3)"),

        pytest.param([2, 4, 6, 8, 1, 1], logger.Code.NOT_IMPLEMENTED, id="Convertible 6D (1)"),
        pytest.param([2, 4, 1, 1, 1, 12], logger.Code.NOT_IMPLEMENTED, id="Convertible 6D (2)"),
        pytest.param([2, 4, 1, 1, 1, 1], logger.Code.NOT_IMPLEMENTED, id="Convertible 6D (3)"),
    ])
def test_convert_q_linear_average_pool_with_large_rank(input_shape: List[int], error_code: logger.Code):
    kernel_shape = input_shape[2:]
    scale = [0.1]
    zp = [42]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"],
                               kernel_shape=kernel_shape, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], scale),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [], zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == error_code


def test_convert_1d_q_linear_average_pool_with_symbolic_shape():
    input_shape = ["batch", 4, 6]
    kernel_shape = [6]

    scale = [0.1]
    zp = [42]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"],
                               kernel_shape=kernel_shape, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], scale),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [], zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool

    input_data = np.arange(np.prod([1, 4, 6])).reshape([1, 4, 6]).astype(np.int8)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_q_linear_average_pool_with_channels_last():
    input_shape = [5, 10, 15, 20]
    kernel_shape = [15, 20]

    scale = [0.1]
    zp = [42]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"],
                               kernel_shape=kernel_shape, channels_last=1, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], scale),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [], zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


def test_convert_q_linear_average_pool_with_ceil_mode():
    input_shape = [5, 10, 15, 20]
    kernel_shape = [15, 20]

    scale = [0.1]
    zp = [42]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"],
                               kernel_shape=kernel_shape, ceil_mode=1, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], scale),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [], zp),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_q_linear_average_pool_with_dynamic_quantization_parameters():
    input_shape = [5, 10, 15, 20]
    kernel_shape = [15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"],
                               kernel_shape=kernel_shape, domain='com.microsoft')],
        'QLinearAveragePool test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape),
            onnx.helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
            onnx.helper.make_tensor_value_info("zp", TensorProto.INT8, []),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_q_linear_average_pool_with_per_channel_quantization():
    input_shape = [5, 10, 15, 20]
    kernel_shape = [15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QLinearAveragePool", ["x", 'scale', 'zp', 'scale', 'zp'], ["output"],
                               kernel_shape=kernel_shape, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [10], [1.0] * 10),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [10], [0] * 10)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add opset with QLinearAveragePool
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_q_linear_average_pool__default_output_zero_point():
    # Simply omit the last input, which is the output zero point.
    input_shape = [5, 10, 15, 20]
    kernel_shape = [15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('QLinearAveragePool', ['x', 'scale', 'zp', 'scale'], ['y'],
                               kernel_shape=kernel_shape, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], [0.37]),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [], [0]),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))  # Add opset with QLinearAveragePool
    onnx.checker.check_model(onnx_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.int8)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_q_linear_average_pool__default_zero_points():
    # Represent the optional zero points using name ''.
    # ONNX Runtime doesn't support this right now.
    # Once it is implemented, this test will fail. We can then update this test to make sure everything works.

    input_shape = [5, 10, 15, 20]
    kernel_shape = [15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('QLinearAveragePool', ['x', 'scale', '', 'scale', ''], ['y'],
                               kernel_shape=kernel_shape, domain='com.microsoft')],
        'QLinearAveragePool test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.INT8, input_shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.INT8, ())],
        [onnx.helper.make_tensor('scale', TensorProto.FLOAT, [], [0.37])]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))  # Add opset with QLinearAveragePool
    onnx.checker.check_model(onnx_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.int8)

    with pytest.raises(Exception) as e:
        executors.convert_run_compare(onnx_model, input_data)
    assert e.value.args[0] == '[ONNXRuntimeError] : 1 : FAIL : Node () Op (QLinearAveragePool) [TypeInferenceError] ' \
                              'Input data type does not match the expected data type'
