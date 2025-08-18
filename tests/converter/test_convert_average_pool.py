#
# Copyright 2023-2025 NXP
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
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from tests import executors


def test_convert_average_pool__quantized():
    _type = TensorProto.INT8

    shape = [2, 4, 6, 8]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('AveragePool', ['x1'], ['x2'], kernel_shape=[3, 3]),
            onnx.helper.make_node('DequantizeLinear', ['x2', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', _type, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.NOT_IMPLEMENTED


def test_convert_average_pool__unsupported_type():
    _type = TensorProto.DOUBLE

    shape = [2, 4, 6, 8]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('AveragePool', ['x'], ['y'], kernel_shape=[3, 3])],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', _type, shape)],
        [onnx.helper.make_tensor_value_info('y', _type, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


def test_convert_average_pool__not_implemented_type():
    _type = TensorProto.INT8

    input_shape = [2, 4, 6, 8]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('AveragePool', ['x'], ['y'], kernel_shape=[3, 3])],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', _type, input_shape)],
        [onnx.helper.make_tensor_value_info('y', _type, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'INT8' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([3], None, [1, 1], [1], id="kernel_shape = [3], pads = [1, 1], strides = [1]."),
    pytest.param([4], None, [1, 2], [1], id="kernel_shape = [4], pads = [1, 2], strides = [1]."),
    pytest.param([6], None, [2, 2], [2], id="kernel_shape = [6], pads = [2, 2], strides = [2]."),

    pytest.param([3], "SAME_LOWER", None, [1], id="kernel_shape = [3], auto_pad = SAME_LOWER, strides = [1]."),
    pytest.param([7], "SAME_UPPER", None, [3], id="kernel_shape = [7], auto_pad = SAME_UPPER, strides = [3]."),
])
def test_convert_1d_average_pool(kernel_shape: List[int], auto_pad: Optional[str], pads: Optional[List[int]],
                                 strides: Optional[List[int]]):
    input_shape = [10, 20, 30]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("AveragePool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad,
                               pads=pads,
                               strides=strides)],
        'AveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([3, 3], None, [1, 1, 1, 1], [1, 1], id="kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1,1]."),
    pytest.param([3, 5], None, [1, 1, 1, 2], [2, 2], id="kernel_shape = [3, 5], pads = [1, 1, 1, 2], strides = [2,2]."),

    pytest.param([3, 4], "SAME_UPPER", None, [1, 3], id="kernel_shape=[3, 4], auto_pad=SAME_UPPER, strides=[1, 3]."),
    pytest.param([7, 2], "SAME_UPPER", None, [3, 2], id="kernel_shape=[7, 2], auto_pad=SAME_UPPER, strides=[3, 2]."),

    pytest.param([4, 9], "VALID", None, [3, 2], id="kernel_shape=[4, 9], auto_pad=VALID, strides=[3, 2]."),
    pytest.param([2, 10], "VALID", None, [4, 1], id="kernel_shape=[2, 10], auto_pad=VALID, strides=[4, 1]."),
])
def test_convert_2d_average_pool(kernel_shape: List[int], auto_pad: Optional[str], pads: Optional[List[int]],
                                 strides: Optional[List[int]]):
    input_shape = [5, 10, 15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("AveragePool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad,
                               pads=pads,
                               strides=strides)],
        'AveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([3, 3], None, [1, 2, 1, 0], [1, 1], id="kernel_shape = [3, 3], pads = [1, 2, 1, 0], strides = [1,1]."),
    pytest.param([3, 5], None, [1, 3, 1, 2], [2, 2], id="kernel_shape = [3, 5], pads = [1, 3, 1, 2], strides = [2,2]."),

    pytest.param([3, 5], "SAME_LOWER", None, [3, 2], id="kernel_shape=[3, 5], auto_pad=SAME_LOWER, strides=[3, 2]."),
    # pytest.param([7, 3], "SAME_LOWER", None, [4, 2], id="kernel_shape=[7, 3], auto_pad=SAME_LOWER, strides=[4, 2]."),
])
def test_convert_2d_average_pool_with_explicit_padding(kernel_shape: List[int], auto_pad: Optional[str],
                                                       pads: Optional[List[int]],
                                                       strides: Optional[List[int]]):
    """ Explicit padding must be added, and 'count_include_pad=1'.  """

    input_shape = [5, 10, 15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("AveragePool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad,
                               pads=pads, strides=strides, count_include_pad=1)],
        'AveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.xfail(
    reason="AveragePool's padding is not handled correctly in some situations. #onnxruntime/issues/24681)")
@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([7, 3], "SAME_LOWER", None, [4, 2], id="kernel_shape=[7, 3], auto_pad=SAME_LOWER, strides=[4, 2]."),
])
def test_convert_2d_average_pool_with_explicit_padding__failing(kernel_shape: List[int], auto_pad: Optional[str],
                                                                pads: Optional[List[int]],
                                                                strides: Optional[List[int]]):
    """ Explicit padding must be added, and 'count_include_pad=1'.  """

    input_shape = [5, 10, 15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("AveragePool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad,
                               pads=pads, strides=strides, count_include_pad=1)],
        'AveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_operatorsetid("", 19)]
    )

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([3, 3], None, [1, 2, 1, 0], [1, 1], id="kernel_shape = [3, 3], pads = [1, 2, 1, 0], strides = [1,1]."),
    pytest.param([3, 5], None, [1, 3, 1, 2], [2, 2], id="kernel_shape = [3, 5], pads = [1, 3, 1, 2], strides = [2,2]."),

    pytest.param([3, 5], "SAME_LOWER", None, [3, 2], id="kernel_shape=[3, 5], auto_pad=SAME_LOWER, strides=[3, 2]."),
    pytest.param([7, 3], "SAME_LOWER", None, [4, 2], id="kernel_shape=[7, 3], auto_pad=SAME_LOWER, strides=[4, 2]."),
])
def test_convert_2d_average_pool_with_unusable_explicit_padding(kernel_shape: List[int], auto_pad: Optional[str],
                                                                pads: Optional[List[int]],
                                                                strides: Optional[List[int]]):
    """ Explicit padding must be added, but 'count_include_pad=0'!  """
    input_shape = [5, 10, 15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("AveragePool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad,
                               pads=pads, strides=strides)],
        'AveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([3, 3], None, [1, 1, 1, 1], [1, 1], id="kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1,1]."),
    pytest.param([3, 5], None, [1, 1, 1, 2], [2, 2], id="kernel_shape = [3, 5], pads = [1, 1, 1, 2], strides = [2,2]."),

    pytest.param([3, 4], "SAME_UPPER", None, [1, 3], id="kernel_shape=[3, 4], auto_pad=SAME_UPPER, strides=[1, 3]."),
    # pytest.param([7, 2], "SAME_UPPER", None, [3, 2], id="kernel_shape=[7, 2], auto_pad=SAME_UPPER, strides=[3, 2]."),
    pytest.param([3, 6], "SAME_UPPER", None, [1, 2], id="kernel_shape=[3, 6], auto_pad=SAME_UPPER, strides=[1, 2]."),

    pytest.param([3, 6], "SAME_LOWER", None, [1, 2], id="kernel_shape=[3, 6], auto_pad=SAME_LOWER, strides=[1, 2]."),
])
def test_convert_2d_average_pool_with_count_include_pad(kernel_shape: List[int], auto_pad: Optional[str],
                                                        pads: Optional[List[int]], strides: Optional[List[int]]):
    """ Padding is converted to TFLite 'SAME', but ONNX requires the 0s to be used in the computation of the mean. """
    input_shape = [5, 10, 15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("AveragePool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad,
                               pads=pads, strides=strides, count_include_pad=1)],
        'AveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = np.random.random(input_shape).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, data)


@pytest.mark.xfail(
    reason="AveragePool's padding is not handled correctly in some situations. #onnxruntime/issues/24681)")
@pytest.mark.parametrize("kernel_shape, auto_pad, pads, strides", [
    pytest.param([7, 2], "SAME_UPPER", None, [3, 2], id="kernel_shape=[7, 2], auto_pad=SAME_UPPER, strides=[3, 2]."),
])
def test_convert_2d_average_pool_with_count_include_pad__failing(
        kernel_shape: List[int], auto_pad: Optional[str], pads: Optional[List[int]],
        strides: Optional[List[int]]):
    """ Padding is converted to TFLite 'SAME', but ONNX requires the 0s to be used in the computation of the mean. """
    input_shape = [5, 10, 15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("AveragePool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad,
                               pads=pads, strides=strides, count_include_pad=1)],
        'AveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_operatorsetid("", 19)]
    )

    np.random.seed(42)
    data = np.random.random(input_shape).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, data)


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
def test_convert_average_pool_with_large_rank(input_shape: List[int], error_code: logger.Code):
    kernel_shape = input_shape[2:]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("AveragePool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad="SAME_UPPER")],
        'AveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == error_code


def test_convert_average_pool_with_dilations():
    input_shape = [2, 4, 6, 8]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("AveragePool", ["x"], ["output"], kernel_shape=[3, 3], auto_pad="SAME_UPPER",
                               dilations=[2, 2])],
        'AveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import[0].version = 19  # Only v19 has dilations

    # onnx.checker.check_model(onnx_model)  # onnx checker doesn't allow dilations for AveragePool for some reason.

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


def test_convert_1d_average_pool_with_dynamic_shape():
    input_shape = [-1, 4, 6]
    kernel_shape = input_shape[2:]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("AveragePool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad="SAME_UPPER")],
        'AveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig()
    config.skip_shape_inference = True
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=config)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


def test_convert_1d_average_pool_with_symbolic_shape():
    input_shape = ["batch", 4, 6]
    kernel_shape = input_shape[2:]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("AveragePool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad="SAME_UPPER")],
        'AveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod([1, 4, 6])).reshape([1, 4, 6]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)
