#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import List, Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src.conversion_config import ConversionConfig
from tests import executors


@pytest.mark.parametrize(
    "value, value_type,  output_shape",
    [
        pytest.param(1, TensorProto.BOOL, [100], id="BOOL"),
        pytest.param(-1.234, TensorProto.DOUBLE, [1, 2, 3], id="DOUBLE"),
        pytest.param(3.14159, TensorProto.FLOAT, [3, 1, 4, 1, 5, 9], id="FLOAT"),
        pytest.param(1234.5678, TensorProto.FLOAT16, [5, 4, 3, 2], id="FLOAT16"),
        pytest.param(-42, TensorProto.INT16, [5, 6, 7, 8], id="INT16"),
        pytest.param(-23, TensorProto.INT32, [2, 3, 90, 2], id="INT32"),
        pytest.param(7355608, TensorProto.INT64, [7, 3, 5, 5, 60, 8], id="INT64"),
        pytest.param(120, TensorProto.INT8, [10, 20], id="INT8"),
        pytest.param(42, TensorProto.UINT16, [5, 6, 7, 8], id="UINT16"),
        pytest.param(23, TensorProto.UINT32, [2, 3, 90, 2], id="UINT32"),
        pytest.param(2 ** 64 - 1, TensorProto.UINT64, [7, 3, 5, 5, 60, 8], id="UINT64"),
        pytest.param(255, TensorProto.UINT8, [10, 20], id="UINT8"),
    ])
def test_convert_constant_of_shape_with_static_shape(value: Any, value_type: TensorProto.DataType,
                                                     output_shape: List[int]):
    value = onnx.helper.make_tensor("value", value_type, [1], [value])

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("ConstantOfShape", ["input"], ["output"], value=value),
        ],
        'ConstantOfShape test',
        [],
        [onnx.helper.make_tensor_value_info("output", value_type, ())],
        [onnx.helper.make_tensor("input", TensorProto.INT64, [len(output_shape)], output_shape)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {}

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "value, value_type,  output_shape",
    [
        pytest.param(1, TensorProto.BOOL, [100], id="BOOL"),
        pytest.param(-1.234, TensorProto.DOUBLE, [1, 2, 3], id="DOUBLE"),
        pytest.param(3.14159, TensorProto.FLOAT, [3, 1, 4, 1, 5, 9], id="FLOAT"),
        pytest.param(1234.5678, TensorProto.FLOAT16, [5, 4, 3, 2], id="FLOAT16"),
        pytest.param(-42, TensorProto.INT16, [5, 6, 7, 8], id="INT16"),
        pytest.param(-23, TensorProto.INT32, [2, 3, 90, 2], id="INT32"),
        pytest.param(7355608, TensorProto.INT64, [7, 3, 5, 5, 60, 8], id="INT64"),
        pytest.param(120, TensorProto.INT8, [10, 20], id="INT8"),
        pytest.param(42, TensorProto.UINT16, [5, 6, 7, 8], id="UINT16"),
        pytest.param(23, TensorProto.UINT32, [2, 3, 90, 2], id="UINT32"),
        pytest.param(2 ** 64 - 1, TensorProto.UINT64, [7, 3, 5, 5, 60, 8], id="UINT64"),
        pytest.param(255, TensorProto.UINT8, [10, 20], id="UINT8"),
    ])
def test_convert_constant_of_shape_with_dynamic_shape(value: Any, value_type: TensorProto.DataType,
                                                      output_shape: list[int]):
    value = onnx.helper.make_tensor("value", value_type, [1], [value])

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("ConstantOfShape", ["input"], ["output"], value=value),
        ],
        'ConstantOfShape test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.INT64, [len(output_shape)])],
        [onnx.helper.make_tensor_value_info("output", value_type, output_shape)],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.asarray(output_shape, np.int64)

    config = ConversionConfig()
    config.skip_shape_inference = True

    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


@pytest.mark.parametrize(
    "value, value_type,  output_shape",
    [
        pytest.param(-1234.56789, TensorProto.FLOAT, [10, 5, 20, 30], id="4D"),
        pytest.param(42.23, TensorProto.FLOAT, [10, 20, 30], id="3D"),
    ])
def test_convert_constant_of_shape_with_static_shape_and_channels_first_output(value: Any,
                                                                               value_type: TensorProto.DataType,
                                                                               output_shape: List[int]):
    value = onnx.helper.make_tensor("value", value_type, [1], [value])
    kernel_shape = [1] * (len(output_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("ConstantOfShape", ["input"], ["constant_out"], value=value),
            onnx.helper.make_node("MaxPool", ["constant_out"], ["output"], kernel_shape=kernel_shape,
                                  strides=kernel_shape)
        ],
        'ConstantOfShape test',
        [],
        [onnx.helper.make_tensor_value_info("output", value_type, ())],
        [onnx.helper.make_tensor("input", TensorProto.INT64, [len(output_shape)], output_shape)]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})


@pytest.mark.parametrize(
    "output_shape",
    [
        pytest.param([256, 128], id="shape = [256, 128]"),
        pytest.param([1, 3, 12, 16], id="shape = [1, 3, 12, 16]"),
    ])
def test_convert_constant_of_shape_with_default_value(output_shape: List[int]):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("ConstantOfShape", ["input"], ["output"]),
        ],
        'ConstantOfShape test',
        [],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("input", TensorProto.INT64, [len(output_shape)], output_shape)]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})
