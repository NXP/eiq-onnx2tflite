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
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize("kernel_shape", [[3], [4]], ids=lambda x: f"kernel_shape = {x}")
@pytest.mark.parametrize("auto_pad", ["VALID", "SAME_UPPER", "SAME_LOWER", None], ids=lambda x: f"auto_pad = {x}")
@pytest.mark.parametrize("pads", [None, [0, 1], [1, 0], [1, 1], [0, 2]], ids=lambda x: f"pads = {x}")
@pytest.mark.parametrize("strides", [[2], None], ids=lambda x: f"strides = {x}")
def test_convert_1d_max_pool(kernel_shape: List[int], auto_pad: Optional[str], pads: Optional[List[int]],
                             strides: Optional[List[int]]):
    if auto_pad is not None and pads is not None:
        return

    input_shape = [10, 20, 30]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MaxPool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad, pads=pads,
                               strides=strides)],
        'MaxPool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_1d_max_pool_quantized(type_: TensorProto.DataType):
    kernel_shape = [3]
    auto_pad = None
    pads = [0, 2]
    strides = None
    input_shape = [10, 20, 30]
    scale = [0.123]
    zero_point = [42]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["y"], axis=1),
            onnx.helper.make_node("MaxPool", ["y"], ["output"], kernel_shape=kernel_shape,
                                  auto_pad=auto_pad, pads=pads, strides=strides),
        ],
        'MaxPool test quantized',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", type_, ())],
        [
            onnx.helper.make_tensor("scale", TensorProto.FLOAT, [len(scale)], scale),
            onnx.helper.make_tensor("zero_point", type_, [len(zero_point)], zero_point)
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    tflite_executor, _ = executors.convert_run_compare(onnx_model, input_data)
    output_quant_params = tflite_executor.get_output_details(0)['quantization_parameters']
    assert np.allclose(output_quant_params['scales'], scale)
    assert np.allclose(output_quant_params['zero_points'], zero_point)


@pytest.mark.parametrize("kernel_shape", [[3, 3], [4, 5]], ids=lambda x: f"kernel_shape = {x}")
@pytest.mark.parametrize("auto_pad", ["VALID", "SAME_UPPER", "SAME_LOWER", None], ids=lambda x: f"auto_pad = {x}")
@pytest.mark.parametrize("pads", [None, [0, 1, 0, 1], [1, 1, 1, 1], [2, 1, 0, 2]], ids=lambda x: f"pads = {x}")
@pytest.mark.parametrize("strides", [[1, 2], [3, 2], None], ids=lambda x: f"strides = {x}")
def test_convert_2d_max_pool(kernel_shape: List[int], auto_pad: Optional[str], pads: Optional[List[int]],
                             strides: Optional[List[int]]):
    if auto_pad is not None and pads is not None:
        return

    input_shape = [5, 10, 15, 20]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MaxPool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad, pads=pads,
                               strides=strides)],
        'MaxPool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_2d_max_pool_quantized(type_: TensorProto.DataType):
    kernel_shape = [3, 3]
    auto_pad = None
    pads = [2, 1, 0, 2]
    strides = None
    input_shape = [5, 10, 15, 20]
    scale = [0.123]
    zero_point = [42]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["y"], axis=1),
            onnx.helper.make_node("MaxPool", ["y"], ["output"], kernel_shape=kernel_shape,
                                  auto_pad=auto_pad, pads=pads, strides=strides)],
        'MaxPool test quantized',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", type_, ())],
        [
            onnx.helper.make_tensor("scale", TensorProto.FLOAT, [len(scale)], scale),
            onnx.helper.make_tensor("zero_point", type_, [len(zero_point)], zero_point)
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    tflite_executor, _ = executors.convert_run_compare(onnx_model, input_data)
    output_quant_params = tflite_executor.get_output_details(0)['quantization_parameters']
    assert np.allclose(output_quant_params['scales'], scale)
    assert np.allclose(output_quant_params['zero_points'], zero_point)


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
def test_convert_max_pool_with_large_rank(input_shape: List[int], error_code: logger.Code):
    kernel_shape = input_shape[2:]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MaxPool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad="SAME_UPPER")],
        'MaxPool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == error_code


def test_convert_max_pool_with_2_outputs():
    input_shape = [2, 4, 6, 8]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MaxPool", ["x"], ["output", "i"], kernel_shape=[3, 3], auto_pad="SAME_UPPER")],
        'MaxPool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model)
    assert e.value.error_code == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_max_pool_with_dilations():
    input_shape = [2, 4, 6, 8]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MaxPool", ["x"], ["output"], kernel_shape=[3, 3], auto_pad="SAME_UPPER",
                               dilations=[2, 2])],
        'MaxPool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model)
    assert e.value.error_code == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_1d_max_pool_with_dynamic_shape():
    input_shape = [-1, 4, 6]
    kernel_shape = input_shape[2:]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MaxPool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad="SAME_UPPER")],
        'MaxPool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig()
    config.skip_shape_inference = True
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=config)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


def test_convert_1d_max_pool_with_symbolic_shape():
    input_shape = ["batch", 4, 6]
    kernel_shape = input_shape[2:]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MaxPool", ["x"], ["output"], kernel_shape=kernel_shape, auto_pad="SAME_UPPER")],
        'MaxPool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod([1, 4, 6])).reshape([1, 4, 6]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("input_type", [
    pytest.param(TensorProto.INT8, id="input_type=INT8"),
    pytest.param(TensorProto.UINT8, id="input_type=UINT8"),
    pytest.param(TensorProto.FLOAT, id="input_type=FLOAT")
])
def test_convert_2d_max_pool_with_negative_inputs(input_type: onnx.TensorProto.DataType):
    input_shape = [1, 1, 4, 4]
    pads = [2, 2, 2, 2]
    kernel_shape = [3, 3]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MaxPool", ["x"], ["output"], kernel_shape=kernel_shape, pads=pads)],
        'MaxPool test',
        [onnx.helper.make_tensor_value_info("x", input_type, input_shape)],
        [onnx.helper.make_tensor_value_info("output", input_type, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))
    input_data = input_data - 5  # overflows for uint8

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("input_type", [
    pytest.param(TensorProto.FLOAT16, id="input_type=FLOAT16")
])
def test_convert_2d_max_pool_invalid_type(input_type: onnx.TensorProto.DataType):
    input_shape = [1, 1, 4, 4]
    pads = [2, 2, 2, 2]
    kernel_shape = [3, 3]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MaxPool", ["x"], ["output"], kernel_shape=kernel_shape, pads=pads)],
        'MaxPool test',
        [onnx.helper.make_tensor_value_info("x", input_type, input_shape)],
        [onnx.helper.make_tensor_value_info("output", input_type, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
