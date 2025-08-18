#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math
from typing import List, Optional

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


def _create_quantize_linear_model(shape: List[int], scale: List[float], zero_point: List[int],
                                  axis: Optional[int] = None,
                                  data_type: onnx.TensorProto.DataType = onnx.TensorProto.INT8) -> onnx.ModelProto:
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["output"], axis=axis)
            ],
            name="graph",
            inputs=[onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", data_type, ())],
            initializer=[
                onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, [len(scale)], scale),
                onnx.helper.make_tensor("zero_point", data_type, [len(zero_point)], zero_point)
            ]
        )
    )

    return onnx_model


@pytest.mark.parametrize("shape,scale,zero_point", [
    ([100, 1, 16, 8], [45.6789], [4]),
    ([1, 3, 20, 20, 7], [87.654], [0]),
    ([1, 30, 40, 50], [123.456789], [-17]),
])
def test_convert_quantize_linear(shape: List[int], scale: List[float], zero_point: List[int]):
    input_data = np.arange(math.prod(shape)).reshape(shape).astype(np.float32)

    # INT8
    onnx_model = _create_quantize_linear_model(shape, scale, zero_point)
    executors.convert_run_compare(onnx_model, input_data, check_model=True, atol=1)

    # UINT8
    zero_point[0] += 128  # Make sure the zero point is a valid uint8 value
    onnx_model = _create_quantize_linear_model(shape, scale, zero_point, None, onnx.TensorProto.UINT8)
    executors.convert_run_compare(onnx_model, input_data, check_model=True, atol=1)


@pytest.mark.parametrize("shape,scale,zero_point,axis", [
    ([100, 1, 16, 8], [0.12685], [0], 0),
    ([100, 1, 16, 8], [0.12685], [0], 1),
    ([100, 1, 16, 8], [0.12685], [0], 2),
    ([100, 1, 16, 8, 7], [0.12685], [0], -1),
    ([100, 1, 16, 8, 7], [0.12685], [0], -2),
])
def test_convert_quantize_linear_with_axis(shape: List[int], scale: List[float], zero_point: List[int], axis: int):
    input_data = (np.arange(math.prod(shape)) / math.prod(shape)).reshape(shape).astype(np.float32)

    # INT8
    onnx_model = _create_quantize_linear_model(shape, scale, zero_point, axis)
    executors.convert_run_compare(onnx_model, input_data, check_model=True, atol=1)

    # UINT8
    zero_point[0] += 128  # Make sure the zero point is a valid uint8 value
    onnx_model = _create_quantize_linear_model(shape, scale, zero_point, axis, onnx.TensorProto.UINT8)
    executors.convert_run_compare(onnx_model, input_data, check_model=True, atol=1)


@pytest.mark.parametrize("axis", [-10, -6, -5, 4, 10], ids=lambda axis: f"axis = '{axis}'")
def test_convert_quantize_linear_with_invalid_axis(axis: int):
    # Set Scale and Zero Point to per-channel quantization, as axis is ignored for per-tensor per ONNX specification
    # https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
    onnx_model = _create_quantize_linear_model([2, 4, 6, 8], [1., 1., 1., 1.], [0, 0, 0, 0], axis)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE


def test_convert_quantize_linear_with_dynamic_parameters():
    io_shape = [2, 3, 4, 5]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["output"])],
            "graph",
            [
                onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, io_shape),
                onnx.helper.make_tensor_value_info("scale", onnx.TensorProto.FLOAT, io_shape),
                onnx.helper.make_tensor_value_info("zero_point", onnx.TensorProto.FLOAT, io_shape)
            ],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT8, ())],
        )
    )

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model)
    assert e.value.error_code == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_quantize_linear_with_saturate():
    io_shape = [2, 3, 4, 5]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["output"], saturate=0)],
            "graph",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, io_shape)],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT8, ())],
            [
                onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, [1], [.1]),
                onnx.helper.make_tensor("zero_point", onnx.TensorProto.INT8, [1], [0])
            ]
        )
    )

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model)
    assert e.value.error_code == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("shape,scale,zero_point,axis", [
    ([1, 3, 20, 256], [0.1, 0.3, 0.2], [-10, 1, 20], 1),
    ([1, 2, 20, 4, 256], [0.007, 0.1, 0.5, 0.24], [5, -20, 0, 50], 3),
    ([1, 3, 20, 15, 2], [0.1234, 1.234], [1, -1], -1),
])
def test_convert_quantize_linear_with_per_channel_quantization(shape: List[int], scale: List[float],
                                                               zero_point: List[int], axis: int):
    input_data = (np.arange(math.prod(shape)) / math.prod(shape)).reshape(shape).astype(np.float32)

    # INT8
    onnx_model = _create_quantize_linear_model(shape, scale, zero_point, axis)
    executors.convert_run_compare(onnx_model, input_data, check_model=True, atol=1)

    # UINT8
    zero_point = np.add(zero_point, 128)  # Make sure the zero point is a valid uint8 value
    onnx_model = _create_quantize_linear_model(shape, scale, zero_point, axis, onnx.TensorProto.UINT8)
    executors.convert_run_compare(onnx_model, input_data, check_model=True, atol=1)


@pytest.mark.parametrize("shape,scale,axis", [
    ([100, 1, 16, 8], [0.12685], 0),
    ([100, 1, 16, 8], [0.12685], 1),
    ([100, 1, 16, 8], [0.12685], 2),
    ([100, 1, 16, 8, 7], [0.12685], -1),
    ([100, 1, 16, 8, 7], [0.12685], -2),
])
def test_convert_quantize_linear_with_implicit_zero_point(shape: List[int], scale: List[float], axis: int):
    # Use a tensor with name "" to represent omitted zero point.

    input_data = (np.arange(math.prod(shape)) / math.prod(shape)).reshape(shape).astype(np.float32)

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("QuantizeLinear", ["input", "scale", ""], ["output"], axis=axis)
            ],
            name="graph",
            inputs=[onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.UINT8, ())],
            initializer=[
                onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, [len(scale)], scale),
            ]
        )
    )

    executors.convert_run_compare(onnx_model, input_data, check_model=True, atol=1)


def test_convert_quantize_linear__unsupported_type():
    type_ = TensorProto.INT32

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QuantizeLinear", ['x', 's'], ['y'])],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", type_, shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.UINT8, ())],
        [onnx.helper.make_tensor('s', type_, [1], [2])]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_quantize_linear__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('QuantizeLinear', ['x1', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.INVALID_ONNX_MODEL


@pytest.mark.parametrize(
    "attr_kwargs",
    [
        pytest.param({"block_size": 1}, id="block_size"),
        pytest.param({"precision": 1}, id="precision"),
        pytest.param({"output_dtype": 1}, id="output_dtype"),
    ])
def test_convert_quantize_linear__unsupported_attributes(attr_kwargs):
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("QuantizeLinear", ['x', 's'], ['y'], **attr_kwargs)],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.UINT8, ())],
        [onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [2])]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED