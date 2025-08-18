#
# Copyright 2023-2024 NXP
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

from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape ,output_scale, output_zero_point, input_1_scale,"
    "input_1_zero_point, input_2_scale, input_2_zero_point, input_type, output_type",
    [
        pytest.param([1, 24, 56, 56], [1, 24, 56, 56], [0.01], [10], [0.1], [0], [0.05], [-5],
                     TensorProto.INT8, TensorProto.INT8, id="signed types"),
        pytest.param([1, 24, 56, 56], [1, 24, 56, 56], [0.01], [130], [0.1], [100], [0.05], [128],
                     TensorProto.UINT8, TensorProto.UINT8, id="unsigned types"),
    ])
def test_convert_q_linear_mul(input_1_shape: List[int], input_2_shape: List[int],
                              output_scale, output_zero_point, input_1_scale, input_1_zero_point, input_2_scale,
                              input_2_zero_point, input_type: TensorProto.DataType, output_type: TensorProto.DataType):
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("QLinearMul",
                                      ["input_1", "input_1_scale", "input_1_zero_point", "input_2", "input_2_scale",
                                       "input_2_zero_point", "output_scale", "output_zero_point"], ["output"],
                                      domain="com.microsoft")  # Specify which opset QLinearMul is from
            ],
            name="QLinearMul_test",
            inputs=[onnx.helper.make_tensor_value_info("input_1", input_type, input_1_shape),
                    onnx.helper.make_tensor_value_info("input_2", input_type, input_2_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", output_type, ())],
            initializer=[
                onnx.helper.make_tensor("input_1_scale", onnx.TensorProto.FLOAT, [], input_1_scale),
                onnx.helper.make_tensor("input_1_zero_point", input_type, [], input_1_zero_point),
                onnx.helper.make_tensor("input_2_scale", onnx.TensorProto.FLOAT, [], input_2_scale),
                onnx.helper.make_tensor("input_2_zero_point", input_type, [], input_2_zero_point),
                onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [], output_scale),
                onnx.helper.make_tensor("output_zero_point", output_type, [], output_zero_point),
            ]
        ),
    )

    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add the opset with QLinearMul

    input_data = {
        0: np.arange(math.prod(input_1_shape)).reshape(input_1_shape).astype(to_numpy_type(input_type)),
        1: np.arange(math.prod(input_2_shape)).reshape(input_2_shape).astype(to_numpy_type(input_type)),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape ,output_scale, output_zero_point, input_1_scale,"
    "input_1_zero_point, input_2_scale, input_2_zero_point, input_type, output_type",
    [
        pytest.param([24, 1, 56], [2, 1, 56, 56], [0.01], [10], [0.1], [0], [0.05], [-5],
                     TensorProto.INT8, TensorProto.INT8),
        pytest.param([2, 24, 1, 56], [56, 1], [0.01], [130], [0.1], [100], [0.05], [128],
                     TensorProto.UINT8, TensorProto.UINT8),
        pytest.param([56, 56], [2, 10, 1, 1], [0.01], [130], [0.1], [100], [0.05], [128],
                     TensorProto.UINT8, TensorProto.UINT8),
    ])
def test_convert_q_linear_mul_with_broadcasting(input_1_shape: List[int], input_2_shape: List[int],
                                                output_scale, output_zero_point, input_1_scale, input_1_zero_point,
                                                input_2_scale,
                                                input_2_zero_point, input_type: TensorProto.DataType,
                                                output_type: TensorProto.DataType):
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("QLinearMul",
                                      ["input_1", "input_1_scale", "input_1_zero_point", "input_2", "input_2_scale",
                                       "input_2_zero_point", "output_scale", "output_zero_point"], ["output"],
                                      domain="com.microsoft")  # Specify which opset QLinearMul is from
            ],
            name="QLinearMul_test",
            inputs=[onnx.helper.make_tensor_value_info("input_1", input_type, input_1_shape),
                    onnx.helper.make_tensor_value_info("input_2", input_type, input_2_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", output_type, ())],
            initializer=[
                onnx.helper.make_tensor("input_1_scale", onnx.TensorProto.FLOAT, [], input_1_scale),
                onnx.helper.make_tensor("input_1_zero_point", input_type, [], input_1_zero_point),
                onnx.helper.make_tensor("input_2_scale", onnx.TensorProto.FLOAT, [], input_2_scale),
                onnx.helper.make_tensor("input_2_zero_point", input_type, [], input_2_zero_point),
                onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [], output_scale),
                onnx.helper.make_tensor("output_zero_point", output_type, [], output_zero_point),
            ]
        ),
    )

    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add the opset with QLinearMul

    input_data = {
        0: np.arange(math.prod(input_1_shape)).reshape(input_1_shape).astype(to_numpy_type(input_type)),
        1: np.arange(math.prod(input_2_shape)).reshape(input_2_shape).astype(to_numpy_type(input_type)),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape, output_scale, output_zero_point, input_1_scale,"
    "input_1_zero_point, input_2_scale, input_2_zero_point, input_type, output_type",
    [
        pytest.param([24, 40, 50], [2, 1, 40, 1], [0.01], [10], [0.1], [0], [0.05], [-5],
                     TensorProto.INT8, TensorProto.INT8),
        pytest.param([2, 24, 1, 50], [40, 1], [0.01], [130], [0.1], [100], [0.05], [128],
                     TensorProto.UINT8, TensorProto.UINT8),
        pytest.param([40, 50], [2, 10, 1, 1], [0.01], [130], [0.1], [100], [0.05], [128],
                     TensorProto.UINT8, TensorProto.UINT8),
    ])
def test_convert_q_linear_mul_with_channels_first_broadcasting(input_1_shape: List[int], input_2_shape: List[int],
                                                               output_scale, output_zero_point, input_1_scale,
                                                               input_1_zero_point, input_2_scale, input_2_zero_point,
                                                               input_type: TensorProto.DataType,
                                                               output_type: TensorProto.DataType):
    mul_output_shape = list(np.broadcast_shapes(input_1_shape, input_2_shape))

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("QLinearMul",
                                      ["input_1", "input_1_scale", "input_1_zero_point", "input_2", "input_2_scale",
                                       "input_2_zero_point", "output_scale", "output_zero_point"], ["mul_out"],
                                      domain="com.microsoft"),  # Specify which opset QLinearMul is from
                onnx.helper.make_node("QLinearGlobalAveragePool",
                                      ["mul_out", "output_scale", "output_zero_point", "output_scale",
                                       "output_zero_point"], ["output"], domain="com.microsoft")
            ],
            name="QLinearMul_test",
            inputs=[onnx.helper.make_tensor_value_info("input_1", input_type, input_1_shape),
                    onnx.helper.make_tensor_value_info("input_2", input_type, input_2_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", output_type, ())],
            initializer=[
                onnx.helper.make_tensor("input_1_scale", onnx.TensorProto.FLOAT, [], input_1_scale),
                onnx.helper.make_tensor("input_1_zero_point", input_type, [], input_1_zero_point),
                onnx.helper.make_tensor("input_2_scale", onnx.TensorProto.FLOAT, [], input_2_scale),
                onnx.helper.make_tensor("input_2_zero_point", input_type, [], input_2_zero_point),
                onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [], output_scale),
                onnx.helper.make_tensor("output_zero_point", output_type, [], output_zero_point),
            ]
        ),
    )

    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add the opset with QLinearMul

    input_data = {
        0: np.arange(math.prod(input_1_shape)).reshape(input_1_shape).astype(to_numpy_type(input_type)),
        1: np.arange(math.prod(input_2_shape)).reshape(input_2_shape).astype(to_numpy_type(input_type)),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize(
    'a_shape, b_shape ,y_scale, a_scale, b_scale, input_type, output_type',
    [
        pytest.param([1, 24, 56, 56], [1, 24, 56, 56], [0.01], [0.1], [0.05],
                     TensorProto.INT8, TensorProto.INT8, id="signed types"),
        pytest.param([1, 24, 56, 56], [1, 24, 56, 56], [0.01], [0.1], [0.05],
                     TensorProto.UINT8, TensorProto.UINT8, id="unsigned types"),
    ])
def test_convert_q_linear_mul__optional_zero_points(a_shape: List[int], b_shape: List[int],
                                                    y_scale, a_scale, b_scale, input_type: TensorProto.DataType,
                                                    output_type: TensorProto.DataType):
    # Represent the optional zero points using name ''.
    # ONNX Runtime doesn't support this right now.
    # Once it is implemented, this test will fail. We can then update this test to make sure everything works.

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('QLinearMul',
                                      ['a', 'a_scale', '', 'b', 'b_scale', '', 'y_scale', ''], ['y'],
                                      domain='com.microsoft')  # Specify which opset QLinearMul is from.
            ],
            'QLinearMul_test',
            [
                onnx.helper.make_tensor_value_info('a', input_type, a_shape),
                onnx.helper.make_tensor_value_info('b', input_type, b_shape)
            ],
            [onnx.helper.make_tensor_value_info('y', output_type, ())],
            [
                onnx.helper.make_tensor('a_scale', onnx.TensorProto.FLOAT, [], a_scale),
                onnx.helper.make_tensor('b_scale', onnx.TensorProto.FLOAT, [], b_scale),
                onnx.helper.make_tensor('y_scale', onnx.TensorProto.FLOAT, [], y_scale),
            ]
        ),
    )

    onnx_model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))  # Add the opset with QLinearMul

    input_data = {
        0: np.arange(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(input_type)),
        1: np.arange(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(input_type)),
    }

    with pytest.raises(Exception) as e:
        executors.convert_run_compare(onnx_model, input_data)
    assert e.value.args[0] == '[ONNXRuntimeError] : 1 : FAIL : Node () Op (QLinearMul) [TypeInferenceError] ' \
                              'Input data type does not match the expected data type'


@pytest.mark.parametrize(
    'a_shape, b_shape, y_scale, a_scale, b_scale, input_type, output_type',
    [
        pytest.param([1, 24, 56, 56], [1, 24, 56, 56], [0.01], [0.1], [0.05],
                     TensorProto.INT8, TensorProto.INT8, id="signed types"),
        pytest.param([1, 24, 56, 56], [1, 24, 56, 56], [0.01], [0.1], [0.05],
                     TensorProto.UINT8, TensorProto.UINT8, id="unsigned types"),
    ])
def test_convert_q_linear_mul__default_output_zero_point(a_shape: List[int], b_shape: List[int],
                                                         y_scale, a_scale, b_scale, input_type: TensorProto.DataType,
                                                         output_type: TensorProto.DataType):
    # Simply omit the last input, which is the output zero point.

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('QLinearMul',
                                      ['a', 'a_scale', 'zp', 'b', 'b_scale', 'zp', 'y_scale'], ['y'],
                                      domain='com.microsoft')  # Specify which opset QLinearMul is from.
            ],
            'QLinearMul_test',
            [
                onnx.helper.make_tensor_value_info('a', input_type, a_shape),
                onnx.helper.make_tensor_value_info('b', input_type, b_shape)
            ],
            [onnx.helper.make_tensor_value_info('y', output_type, ())],
            [
                onnx.helper.make_tensor('a_scale', onnx.TensorProto.FLOAT, [], a_scale),
                onnx.helper.make_tensor('b_scale', onnx.TensorProto.FLOAT, [], b_scale),
                onnx.helper.make_tensor('y_scale', onnx.TensorProto.FLOAT, [], y_scale),
                onnx.helper.make_tensor('zp', input_type, [], [0]),
            ]
        ),
    )

    onnx_model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))  # Add the opset with QLinearMul

    input_data = {
        0: np.arange(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(input_type)),
        1: np.arange(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(input_type)),
    }

    executors.convert_run_compare(onnx_model, input_data)
