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

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape , input_1_scale, input_1_zero_point, input_2_scale, input_2_zero_point,"
    "output_scale, output_zero_point, input_1_type, input_2_type, output_type",
    [
        pytest.param([56, 5], [5, 56], [0.2], [0], [0.3], [10], [0.1], [-10],
                     TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, id="2D + signed types"),
        pytest.param([16, 32], [32, 16], [0.1], [100], [0.05], [128], [0.01], [130],
                     TensorProto.UINT8, TensorProto.UINT8, TensorProto.UINT8, id="2D + unsigned types"),
        pytest.param([16, 32], [32, 16], [0.1], [100], [0.05], [127], [0.01], [130],
                     TensorProto.UINT8, TensorProto.INT8, TensorProto.UINT8, id="2D + mixed types"),

        pytest.param([2, 24, 56], [2, 56, 24], [0.1], [0], [0.05], [-5], [0.01], [10],
                     TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, id="3D"),
        pytest.param([1, 24, 5, 16], [1, 24, 16, 5], [0.1], [100], [0.05], [127], [0.01], [130],
                     TensorProto.UINT8, TensorProto.INT8, TensorProto.UINT8, id="4D"),
        pytest.param([1, 3, 9, 5, 8], [1, 3, 9, 8, 5], [0.1], [100], [0.05], [128], [0.01], [130],
                     TensorProto.UINT8, TensorProto.UINT8, TensorProto.UINT8, id="5D"),
    ])
def test_convert_q_linear_mat_mul(input_1_shape: List[int], input_2_shape: List[int],
                                  input_1_scale, input_1_zero_point, input_2_scale, input_2_zero_point, output_scale,
                                  output_zero_point, input_1_type: TensorProto.DataType,
                                  input_2_type: TensorProto.DataType,
                                  output_type: TensorProto.DataType):
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("QLinearMatMul",
                                      ["input_1", "input_1_scale", "input_1_zero_point", "input_2", "input_2_scale",
                                       "input_2_zero_point", "output_scale", "output_zero_point"], ["output"])
            ],
            name="QLinearMatMul_test",
            inputs=[onnx.helper.make_tensor_value_info("input_1", input_1_type, input_1_shape),
                    onnx.helper.make_tensor_value_info("input_2", input_2_type, input_2_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", output_type, ())],
            initializer=[
                onnx.helper.make_tensor("input_1_scale", onnx.TensorProto.FLOAT, [], input_1_scale),
                onnx.helper.make_tensor("input_1_zero_point", input_1_type, [], input_1_zero_point),
                onnx.helper.make_tensor("input_2_scale", onnx.TensorProto.FLOAT, [], input_2_scale),
                onnx.helper.make_tensor("input_2_zero_point", input_2_type, [], input_2_zero_point),
                onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [], output_scale),
                onnx.helper.make_tensor("output_zero_point", output_type, [], output_zero_point),
            ]
        ),
    )

    input_data = {
        0: np.arange(math.prod(input_1_shape)).reshape(input_1_shape).astype(to_numpy_type(input_1_type)),
        1: np.arange(math.prod(input_2_shape)).reshape(input_2_shape).astype(to_numpy_type(input_2_type)),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape, input_1_scale, input_1_zero_point, input_2_scale, input_2_zero_point,"
    "output_scale, output_zero_point, input_1_type, input_2_type, output_type",
    [
        pytest.param([1, 12], [12, 3],
                     [0.1], [3], [0.1, 0.2, 0.3], [0, 0, 0], [0.2], [10],
                     TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, id="2D signed types"),
        pytest.param([1, 12], [12, 3],
                     [0.1], [131], [0.1, 0.2, 0.3], [0, 0, 0], [0.2], [138],
                     TensorProto.UINT8, TensorProto.INT8, TensorProto.UINT8, id="2D mixed types"),
        pytest.param([3, 8, 12], [12, 3],
                     [0.1], [3], [0.1, 0.2, 0.3], [0, 0, 0], [0.2], [10],
                     TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, id="2D signed types, batched"),
    ])
def test_convert_q_linear_mat_mul_with_per_channel_quantization(input_1_shape: List[int], input_2_shape: List[int],
                                                                input_1_scale, input_1_zero_point,
                                                                input_2_scale, input_2_zero_point,
                                                                output_scale, output_zero_point,
                                                                input_1_type: TensorProto.DataType,
                                                                input_2_type: TensorProto.DataType,
                                                                output_type: TensorProto.DataType):
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("QLinearMatMul",
                                      ["input_1", "input_1_scale", "input_1_zero_point", "input_2", "input_2_scale",
                                       "input_2_zero_point", "output_scale", "output_zero_point"], ["output"])
            ],
            name="QLinearMatMul_test",
            inputs=[onnx.helper.make_tensor_value_info("input_1", input_1_type, input_1_shape),
                    onnx.helper.make_tensor_value_info("input_2", input_2_type, input_2_shape),
                    ],
            outputs=[onnx.helper.make_tensor_value_info("output", output_type, ())],
            initializer=[
                onnx.helper.make_tensor("input_1_scale", onnx.TensorProto.FLOAT, [len(input_1_scale)], input_1_scale),
                onnx.helper.make_tensor("input_1_zero_point", input_1_type, [len(input_1_zero_point)],
                                        input_1_zero_point),
                onnx.helper.make_tensor("input_2_scale", onnx.TensorProto.FLOAT, [len(input_2_scale)], input_2_scale),
                onnx.helper.make_tensor("input_2_zero_point", input_2_type, [len(input_2_zero_point)],
                                        input_2_zero_point),
                onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [len(output_scale)], output_scale),
                onnx.helper.make_tensor("output_zero_point", output_type, [len(output_zero_point)],
                                        output_zero_point),

            ]
        ),
    )

    input_data = {
        0: np.arange(math.prod(input_1_shape)).reshape(input_1_shape).astype(to_numpy_type(input_1_type)),
        1: np.arange(math.prod(input_2_shape)).reshape(input_2_shape).astype(to_numpy_type(input_2_type))
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape, input_1_scale, input_1_zero_point, input_2_scale, input_2_zero_point,"
    "output_scale, output_zero_point, input_1_type, input_2_type, output_type",
    [
        pytest.param([1, 12], [12, 3],
                     [0.1], [3], [0.1, 0.2, 0.3], [0, 0, 0], [0.2], [10],
                     TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, id="2D + signed types"),
        pytest.param([4, 10], [10, 3],
                     [0.2], [117], [0.15, 0.22, 0.28], [128, 128, 128], [0.2], [120],
                     TensorProto.UINT8, TensorProto.UINT8, TensorProto.UINT8, id="2D + unsigned types"),
        pytest.param([7, 17], [17, 4],
                     [0.2], [120], [0.4, 0.2, 0.3, 0.15], [0, 0, 0, 0], [0.3], [127],
                     TensorProto.UINT8, TensorProto.INT8, TensorProto.UINT8, id="2D + mixed types"),
    ])
def test_convert_q_linear_mat_mul_with_per_channel_quantization_static_tensor(input_1_shape: List[int],
                                                                              input_2_shape: List[int],
                                                                              input_1_scale, input_1_zero_point,
                                                                              input_2_scale, input_2_zero_point,
                                                                              output_scale, output_zero_point,
                                                                              input_1_type: TensorProto.DataType,
                                                                              input_2_type: TensorProto.DataType,
                                                                              output_type: TensorProto.DataType):
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("QLinearMatMul",
                                      ["input_1", "input_1_scale", "input_1_zero_point", "input_2", "input_2_scale",
                                       "input_2_zero_point", "output_scale", "output_zero_point"], ["output"])
            ],
            name="QLinearMatMul_test",
            inputs=[onnx.helper.make_tensor_value_info("input_1", input_1_type, input_1_shape),
                    ],
            outputs=[onnx.helper.make_tensor_value_info("output", output_type, ())],
            initializer=[
                onnx.helper.make_tensor("input_2", input_2_type, input_2_shape,
                                        np.arange(math.prod(input_2_shape)).reshape(input_2_shape).astype(
                                            to_numpy_type(input_2_type))),

                onnx.helper.make_tensor("input_1_scale", onnx.TensorProto.FLOAT, [len(input_1_scale)], input_1_scale),
                onnx.helper.make_tensor("input_1_zero_point", input_1_type, [len(input_1_zero_point)],
                                        input_1_zero_point),
                onnx.helper.make_tensor("input_2_scale", onnx.TensorProto.FLOAT, [len(input_2_scale)], input_2_scale),
                onnx.helper.make_tensor("input_2_zero_point", input_2_type, [len(input_2_zero_point)],
                                        input_2_zero_point),
                onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [len(output_scale)], output_scale),
                onnx.helper.make_tensor("output_zero_point", output_type, [len(output_zero_point)],
                                        output_zero_point),

            ]
        ),
    )

    input_data = {
        0: np.arange(math.prod(input_1_shape)).reshape(input_1_shape).astype(to_numpy_type(input_1_type)),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)


def test_convert_q_linear_mat_mul__channels_first__dynamic():
    a_shape = [2, 4, 5, 6]
    b_shape = [2, 4, 6, 7]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('QLinearMatMul', ['a', 's', 'zp', 'b', 's', 'zp', 's', 'zp'], ['y1']),
                onnx.helper.make_node('MaxPool', ['y1'], ['y'], kernel_shape=[1, 1])
            ],
            'QLinearMatMul test',
            [
                onnx.helper.make_tensor_value_info('a', TensorProto.INT8, a_shape),
                onnx.helper.make_tensor_value_info('b', TensorProto.INT8, b_shape)
            ],
            [onnx.helper.make_tensor_value_info('y', TensorProto.INT8, ())],
            [
                onnx.helper.make_tensor('s', TensorProto.FLOAT, [], [0.0123]),
                onnx.helper.make_tensor('zp', TensorProto.INT8, [], [0])
            ]
        ),
    )

    data = {
        0: np.random.randint(0, 127, math.prod(a_shape)).reshape(a_shape).astype('int8'),
        1: np.random.randint(0, 127, math.prod(b_shape)).reshape(b_shape).astype('int8')
    }

    executors.convert_run_compare(onnx_model, data, atol=1)


def test_convert_q_linear_mat_mul__channels_first__static():
    a_shape = [2, 4, 5, 6]
    b_shape = [2, 4, 6, 7]
    data = {
        0: np.arange(math.prod(a_shape)).reshape(a_shape).astype('int8'),
        1: np.arange(math.prod(b_shape)).reshape(b_shape).astype('int8'),
    }

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('QLinearMatMul', ['a', 's', 'zp', 'b', 's', 'zp', 's', 'zp'], ['y1']),
                onnx.helper.make_node('MaxPool', ['y1'], ['y'], kernel_shape=[1, 1])
            ],
            'QLinearMatMul test',
            [],
            [onnx.helper.make_tensor_value_info('y', TensorProto.INT8, ())],
            [
                onnx.helper.make_tensor('a', TensorProto.INT8, a_shape, data[0]),
                onnx.helper.make_tensor('b', TensorProto.INT8, b_shape, data[1]),
                onnx.helper.make_tensor('s', TensorProto.FLOAT, [], [0.0123]),
                onnx.helper.make_tensor('zp', TensorProto.INT8, [], [127])
            ]
        ),
    )

    executors.convert_run_compare(onnx_model, {})


@pytest.mark.parametrize('a_shape, b_shape', [
    ([56, 5], [5, 56]),
    ([10, 6], [6, 12]),
    ([1, 10, 6], [6, 12]),
    ([3, 10, 6], [6, 12])
])
@pytest.mark.parametrize('a_type, b_type, y_type', [
    (TensorProto.INT8, TensorProto.INT8, TensorProto.INT8),
])
def test_convert_q_linear_mat_mul__into_fully_connected(intermediate_tflite_model_provider, a_shape, b_shape, a_type,
                                                        b_type, y_type):
    a_s, a_zp, b_s, b_zp, y_s, y_zp = 0.4, 50, 0.6, 0, 0.8, 20  # b_zp must be 0.

    b_data = np.arange(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(b_type))

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node('QLinearMatMul', ['a', 'a_s', 'a_zp', 'b', 'b_s', 'b_zp', 'y_s', 'y_zp'], ['y'])],
            'QLinearMatMul test',
            [onnx.helper.make_tensor_value_info('a', a_type, a_shape)],
            [onnx.helper.make_tensor_value_info('y', y_type, ())],
            [
                onnx.helper.make_tensor('b', b_type, b_shape, b_data),
                onnx.helper.make_tensor('a_s', onnx.TensorProto.FLOAT, [], [a_s]),
                onnx.helper.make_tensor('a_zp', a_type, [], [a_zp]),
                onnx.helper.make_tensor('b_s', onnx.TensorProto.FLOAT, [], [b_s]),
                onnx.helper.make_tensor('b_zp', b_type, [], [b_zp]),
                onnx.helper.make_tensor('y_s', onnx.TensorProto.FLOAT, [], [y_s]),
                onnx.helper.make_tensor('y_zp', y_type, [], [y_zp]),
            ]
        ),
    )

    input_data = np.arange(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(a_type))

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.FULLY_CONNECTED
    ])


@pytest.mark.parametrize('a_shape, b_shape', [
    ([56, 5], [5, 56]),
    ([10, 6], [6, 12]),
    ([1, 10, 6], [6, 12]),
    ([3, 10, 6], [6, 12])
])
@pytest.mark.parametrize('a_type, b_type, y_type', [
    (TensorProto.UINT8, TensorProto.UINT8, TensorProto.UINT8),
])
def test_convert_q_linear_mat_mul__into_fully_connected__uint8_weights(intermediate_tflite_model_provider, a_shape,
                                                                       b_shape, a_type, b_type, y_type):
    a_s, a_zp, b_s, b_zp, y_s, y_zp = 0.4, 50, 0.6, 128, 0.8, 20  # b_zp must be 128.

    b_data = np.arange(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(b_type))

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node('QLinearMatMul', ['a', 'a_s', 'a_zp', 'b', 'b_s', 'b_zp', 'y_s', 'y_zp'], ['y'])],
            'QLinearMatMul test',
            [onnx.helper.make_tensor_value_info('a', a_type, a_shape)],
            [onnx.helper.make_tensor_value_info('y', y_type, ())],
            [
                onnx.helper.make_tensor('b', b_type, b_shape, b_data),
                onnx.helper.make_tensor('a_s', onnx.TensorProto.FLOAT, [], [a_s]),
                onnx.helper.make_tensor('a_zp', a_type, [], [a_zp]),
                onnx.helper.make_tensor('b_s', onnx.TensorProto.FLOAT, [], [b_s]),
                onnx.helper.make_tensor('b_zp', b_type, [], [b_zp]),
                onnx.helper.make_tensor('y_s', onnx.TensorProto.FLOAT, [], [y_s]),
                onnx.helper.make_tensor('y_zp', y_type, [], [y_zp]),
            ]
        ),
    )

    input_data = np.arange(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(a_type))

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE,
        BuiltinOperator.FULLY_CONNECTED,
        BuiltinOperator.QUANTIZE,
    ])


@pytest.mark.parametrize('a_shape, b_shape', [
    ([10, 6], [6, 12]),
    ([1, 10, 6], [6, 12]),
    ([3, 10, 6], [6, 12])
])
@pytest.mark.parametrize('a_type, b_type, y_type', [
    (TensorProto.UINT8, TensorProto.INT8, TensorProto.UINT8),
])
def test_convert_q_linear_mat_mul__into_fully_connected__uint8_io(intermediate_tflite_model_provider, a_shape,
                                                                  b_shape, a_type, b_type, y_type):
    a_s, a_zp, b_s, b_zp, y_s, y_zp = 0.4, 130, 0.6, 0, 0.8, 120  # b_zp must be 0.

    b_data = np.arange(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(b_type))

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node('QLinearMatMul', ['a', 'a_s', 'a_zp', 'b', 'b_s', 'b_zp', 'y_s', 'y_zp'], ['y'])],
            'QLinearMatMul test',
            [onnx.helper.make_tensor_value_info('a', a_type, a_shape)],
            [onnx.helper.make_tensor_value_info('y', y_type, ())],
            [
                onnx.helper.make_tensor('b', b_type, b_shape, b_data),
                onnx.helper.make_tensor('a_s', onnx.TensorProto.FLOAT, [], [a_s]),
                onnx.helper.make_tensor('a_zp', a_type, [], [a_zp]),
                onnx.helper.make_tensor('b_s', onnx.TensorProto.FLOAT, [], [b_s]),
                onnx.helper.make_tensor('b_zp', b_type, [], [b_zp]),
                onnx.helper.make_tensor('y_s', onnx.TensorProto.FLOAT, [], [y_s]),
                onnx.helper.make_tensor('y_zp', y_type, [], [y_zp]),
            ]
        ),
    )
    input_data = np.arange(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(a_type))

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE,
        BuiltinOperator.FULLY_CONNECTED,
        BuiltinOperator.QUANTIZE,
    ])


@pytest.mark.parametrize('a_shape, b_shape', [
    ([56, 5], [5, 56]),
    ([10, 6], [6, 12]),
    ([1, 10, 6], [6, 12]),
    ([3, 10, 6], [6, 12])
])
def test_convert_q_linear_mat_mul__into_fully_connected__dynamic_weights(intermediate_tflite_model_provider, a_shape,
                                                                         b_shape):
    a_s, a_zp, b_s, b_zp, y_s, y_zp = 0.4, 50, 0.6, 40, 0.8, 20  # b_zp can be anything.
    a_type, b_type, y_type = [TensorProto.INT8] * 3

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node('QLinearMatMul', ['a', 'a_s', 'a_zp', 'b', 'b_s', 'b_zp', 'y_s', 'y_zp'], ['y'])],
            'QLinearMatMul test',
            [
                onnx.helper.make_tensor_value_info('a', a_type, a_shape),
                onnx.helper.make_tensor_value_info('b', b_type, b_shape)
            ],
            [onnx.helper.make_tensor_value_info('y', y_type, ())],
            [
                onnx.helper.make_tensor('a_s', onnx.TensorProto.FLOAT, [], [a_s]),
                onnx.helper.make_tensor('a_zp', a_type, [], [a_zp]),
                onnx.helper.make_tensor('b_s', onnx.TensorProto.FLOAT, [], [b_s]),
                onnx.helper.make_tensor('b_zp', b_type, [], [b_zp]),
                onnx.helper.make_tensor('y_s', onnx.TensorProto.FLOAT, [], [y_s]),
                onnx.helper.make_tensor('y_zp', y_type, [], [y_zp]),
            ]
        ),
    )

    input_data = {
        0: np.arange(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(a_type)),
        1: np.arange(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(a_type))
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.FULLY_CONNECTED
    ])
