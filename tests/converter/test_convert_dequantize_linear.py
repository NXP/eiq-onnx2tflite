#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math
from typing import List

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "input_shape, input_scale, input_zero_point, input_type",
    [
        pytest.param([1, 24, 56, 56], [0.01], [10], TensorProto.INT8, id="signed types"),
        pytest.param([1, 24, 56, 56], [0.01], [130], TensorProto.UINT8, id="unsigned types"),
    ])
def test_convert_dequantize_linear(input_shape: List[int], input_scale, input_zero_point,
                                   input_type: TensorProto.DataType):
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node(
                    "DequantizeLinear",
                    ["input", "input_scale", "input_zero_point"],
                    ["output"])
            ],
            name="DequantizeLinear_test",
            inputs=[onnx.helper.make_tensor_value_info("input", input_type, input_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [], input_scale),
                onnx.helper.make_tensor("input_zero_point", input_type, [], input_zero_point),
            ]
        ),
    )

    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, input_scale, input_type",
    [
        pytest.param([1, 24, 56, 56], [0.01], TensorProto.INT8, id="signed types"),
        pytest.param([1, 24, 56, 56], [0.01], TensorProto.UINT8, id="unsigned types"),
    ])
def test_convert_dequantize_linear_undefined_zero_point(input_shape: List[int], input_scale,
                                                        input_type: TensorProto.DataType):
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node(
                    "DequantizeLinear",
                    ["input", "input_scale", ''],
                    ["output"])
            ],
            name="DequantizeLinear_test",
            inputs=[onnx.helper.make_tensor_value_info("input", input_type, input_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [], input_scale),
            ]
        ),
    )

    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("input_shape,input_scale,input_zero_point,axis", [
    pytest.param([1, 3, 20, 256], [0.00824, 0.00224, 0.0124], [0, 0, 0], 1, id="input(1, 3, 20, 256), axis=1"),
    pytest.param([1, 2, 20, 4, 256], [0.007120, 0.123124, 1.214124, 0.087124], [5, 120, 0, 50], 3,
                 id="input(1, 2, 20, 4, 256), axis=3"),
    pytest.param([1, 3, 20, 15, 2], [0.1214, 1.124125], [0, 1], -1, id="input(1, 3, 20, 15, 2), axis=-1"),
])
def test_convert_dequantize_linear_with_per_channel_quantization(
        input_shape: List[int], input_scale: List[float], input_zero_point: List[int], axis: int):
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node(
                    "DequantizeLinear",
                    ["input", "input_scale", "input_zero_point"],
                    ["output"],
                    axis=axis)
            ],
            name="DequantizeLinear_test",
            inputs=[onnx.helper.make_tensor_value_info("input", onnx.TensorProto.UINT8, input_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [len(input_scale)], input_scale),
                onnx.helper.make_tensor("input_zero_point", onnx.TensorProto.UINT8, [len(input_zero_point)],
                                        input_zero_point),
            ]
        ),
    )

    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(np.uint8)

    executors.convert_run_compare(onnx_model, input_data, check_model=True)


def test_convert_dequantize_linear_with_dynamic_quantization_parameters():
    input_shape = [13, 37]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("DequantizeLinear", ['x', 's', 'zp'], ['y'])],
            "DequantizeLinear_test",
            [
                onnx.helper.make_tensor_value_info('x', TensorProto.UINT8, input_shape),
                onnx.helper.make_tensor_value_info('s', TensorProto.FLOAT, [1]),
                onnx.helper.make_tensor_value_info('zp', TensorProto.UINT8, [1]),
            ],
            [onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, ())],
        ),
    )
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize(
    "attr_kwargs",
    [
        pytest.param({"block_size": 1}, id="block_size"),
        pytest.param({"output_dtype": 1}, id="output_dtype"),
    ])
def test_convert_dequantize_linear__unsupported_attributes(attr_kwargs):
    input_shape = [1, 24, 56, 56]
    input_scale = [0.01]
    input_zero_point = [10]
    input_type = TensorProto.INT8

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node(
                    "DequantizeLinear",
                    ["input", "input_scale", "input_zero_point"],
                    ["output"], **attr_kwargs)
            ],
            name="DequantizeLinear_test",
            inputs=[onnx.helper.make_tensor_value_info("input", input_type, input_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [], input_scale),
                onnx.helper.make_tensor("input_zero_point", input_type, [], input_zero_point),
            ]
        ),
    )

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
