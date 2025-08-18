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

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type
from tests import executors
from onnx2tflite.src.tflite_optimizer.optimizer import Optimization


@pytest.mark.parametrize(
    "input_shape, perm",
    [
        pytest.param([2, 4, 6, 8], [0, 1, 2, 3], id="4D identity"),
        pytest.param([2, 4, 6, 8], [1, 3, 0, 2], id="4D, perm = [1, 3, 0, 2]"),
        pytest.param([2, 4, 6, 8], [3, 2, 1, 0], id="4D, perm = [3, 2, 1, 0]"),
        pytest.param([2, 4, 6, 8], [2, 3, 1, 0], id="4D, perm = [2, 3, 1, 0]"),
        pytest.param([2, 4, 6, 8], [0, 1, 2, 3], id="4D, perm = [0, 1, 2, 3]"),
        pytest.param([2, 4, 6, 8], None, id="4D, implicit perm"),

        pytest.param([2, 4, 6, 8, 10], [4, 2, 3, 0, 1], id="5D, perm = [4, 2, 3, 0, 1]"),
        pytest.param([2, 4, 6, 8, 10], None, id="5D, implicit perm"),

        pytest.param([2, 4, 6], [1, 2, 0], id="3D, perm = [1, 2, 0]"),
        pytest.param([2, 4, 6], None, id="3D, implicit perm"),

        pytest.param([2, 4], [1, 0], id="2D, perm = [1, 0]"),
        pytest.param([2, 4], None, id="2D, implicit perm"),

        pytest.param([2], [0], id="1D, perm = [0]"),
        pytest.param([2], None, id="1D, implicit perm"),
    ])
def test_convert_formatless_transpose(input_shape: List[int], perm: Optional[List[int]]):
    # Input and output tensor are both formatless

    if perm is not None:
        transpose = onnx.helper.make_node('Transpose', ['input'], ['output'], perm=perm)
    else:
        transpose = onnx.helper.make_node('Transpose', ['input'], ['output'])

    graph = onnx.helper.make_graph(
        [transpose],
        'transpose test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, perm",
    [
        pytest.param([2, 4, 6, 8, 10, 12, 2], [4, 2, 5, 3, 6, 0, 1], id="7D, explicit permutation"),
        pytest.param([2, 4, 6, 8, 10, 12, 2], None, id="7D, implicit permutation"),

        pytest.param([2, 4, 6, 8, 10, 12, 2, 3], [7, 4, 2, 5, 3, 6, 0, 1], id="8D, explicit permutation"),
        pytest.param([2, 4, 6, 8, 10, 12, 2, 3], None, id="8D, implicit permutation"),

        pytest.param([2] * 20, list(range(19, -1, -1)), id="20D, perm = explicit permutation"),
        pytest.param([2] * 20, None, id="20D, implicit permutation"),
    ])
def test_convert_transpose__flex(input_shape: List[int], perm: Optional[List[int]], intermediate_tflite_model_provider):
    if perm is not None:
        transpose = onnx.helper.make_node('Transpose', ['input'], ['output'], perm=perm)
    else:
        transpose = onnx.helper.make_node('Transpose', ['input'], ['output'])

    graph = onnx.helper.make_graph(
        [transpose],
        'transpose test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    config = ConversionConfig()
    config.allow_select_ops = True
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.CUSTOM])


def test_convert_formatless_transpose_quantized():
    # Input and output tensor are both formatless
    input_shape = [2, 4, 6, 8]
    perm = [0, 1, 2, 3]
    scale = [1.0]
    zero_point = [0]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["y"], axis=1),
            onnx.helper.make_node('Transpose', ['y'], ['output'], perm=perm)
        ],
        'transpose test quantized',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor("scale", TensorProto.FLOAT, [len(scale)], scale),
            onnx.helper.make_tensor("zero_point", TensorProto.INT8, [len(zero_point)], zero_point)
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    tflite_executor, _ = executors.convert_run_compare(onnx_model, input_data)
    output_quant_params = tflite_executor.get_output_details(0)['quantization_parameters']
    assert output_quant_params['scales'] == [1.0]
    assert output_quant_params['zero_points'] == [0]


def test_convert_transpose__flex__quantized():
    input_shape = [1, 2, 3, 4, 5, 6, 2]
    perm = [4, 5, 1, 0, 3, 6, 2]
    scale = [1.0]
    zero_point = [0]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["y"]),
            onnx.helper.make_node('Transpose', ['y'], ['output'], perm=perm)
        ],
        'transpose test quantized',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor("scale", TensorProto.FLOAT, [len(scale)], scale),
            onnx.helper.make_tensor("zero_point", TensorProto.INT8, [len(zero_point)], zero_point)
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize(
    "input_shape, perm",
    [
        pytest.param([5, 10, 15], [1, 0, 2], id="3D, switch N and C"),
        pytest.param([5, 10, 15], [0, 2, 1], id="3D, switch C and H"),
        pytest.param([5, 10, 15], [2, 1, 0], id="3D, switch N and H"),

        pytest.param([2, 4, 6, 8], [0, 1, 2, 3], id="4D, identity"),
        pytest.param([2, 4, 6, 8], [0, 1, 3, 2], id="4D, switch H and W"),
        pytest.param([2, 4, 6, 8], [0, 2, 1, 3], id="4D, switch C and H"),
        pytest.param([2, 4, 6, 8], [0, 3, 2, 1], id="4D, switch C and W"),
        pytest.param([2, 4, 6, 8], [1, 0, 2, 3], id="4D, switch N and C"),
    ])
def test_convert_transpose_between_channels_first_tensors(input_shape: List[int], perm: List[int],
                                                          intermediate_tflite_model_provider):
    # The input and output of the Transpose operator are both channels first

    kernel_shape = [1] * (len(input_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MaxPool", ["input"], ["max_pool_out"], auto_pad="SAME_UPPER",
                                  kernel_shape=kernel_shape),
            onnx.helper.make_node("Transpose", ["max_pool_out"], ["transpose_out"], perm=perm),
            onnx.helper.make_node("MaxPool", ["transpose_out"], ["output"], auto_pad="SAME_UPPER",
                                  kernel_shape=kernel_shape),
        ],
        'transpose test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    config = ConversionConfig()
    # This optimization would cause the test "4D, identity" to fail, because the `Transpose` is unnecessary and gets
    #  removed.
    config.optimization_blacklist = [Optimization.REMOVE_IDENTITY_TRANSPOSE_OPERATORS]

    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    ops = intermediate_tflite_model_provider.get_operators()

    if len(input_shape) == 4:
        expected_ops = [BuiltinOperator.TRANSPOSE, BuiltinOperator.MAX_POOL_2D, BuiltinOperator.TRANSPOSE,
                        BuiltinOperator.MAX_POOL_2D, BuiltinOperator.TRANSPOSE]
    else:  # 3D
        expected_ops = [BuiltinOperator.TRANSPOSE, BuiltinOperator.RESHAPE, BuiltinOperator.MAX_POOL_2D,
                        BuiltinOperator.RESHAPE, BuiltinOperator.TRANSPOSE, BuiltinOperator.RESHAPE,
                        BuiltinOperator.MAX_POOL_2D, BuiltinOperator.RESHAPE, BuiltinOperator.TRANSPOSE]

    assert len(ops) == len(expected_ops)
    assert all(op.builtin_options.operator_type == expected for op, expected in zip(ops, expected_ops))


@pytest.mark.parametrize(
    "input_shape, perm",
    [
        pytest.param([5, 10, 15], [1, 0, 2], id="3D, switch N and C"),
        pytest.param([5, 10, 15], [0, 2, 1], id="3D, switch C and H"),
        pytest.param([5, 10, 15], [2, 1, 0], id="3D, switch N and H"),

        pytest.param([2, 4, 6, 8], [0, 1, 2, 3], id="4D, identity"),
        pytest.param([2, 4, 6, 8], [0, 1, 3, 2], id="4D, switch H and W"),
        pytest.param([2, 4, 6, 8], [0, 2, 1, 3], id="4D, switch C and H"),
        pytest.param([2, 4, 6, 8], [0, 3, 2, 1], id="4D, switch C and W"),
        pytest.param([2, 4, 6, 8], [1, 0, 2, 3], id="4D, switch N and C"),
    ])
def test_convert_transpose_from_channels_first(input_shape: List[int], perm: List[int],
                                               intermediate_tflite_model_provider):
    kernel_shape = [1] * (len(input_shape) - 2)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MaxPool", ["x"], ["o1"], auto_pad="SAME_UPPER", kernel_shape=kernel_shape),
            onnx.helper.make_node("Transpose", ["o1"], ["o2"], perm=perm),
            onnx.helper.make_node("Add", ["o2", "one"], ["o"]),
        ],
        'transpose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('one', TensorProto.FLOAT, [1], [1.])]
    )
    onnx_model = onnx.helper.make_model(graph)
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    config = ConversionConfig()
    # This optimization would cause the test "3D, switch C and H" to fail, because there will be an unnecessary
    #  `Transpose` operator, which gets removed.
    config.optimization_blacklist = [Optimization.REMOVE_IDENTITY_TRANSPOSE_OPERATORS]

    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    ops = intermediate_tflite_model_provider.get_operators()

    if len(input_shape) == 4:
        expected_ops = [BuiltinOperator.TRANSPOSE, BuiltinOperator.MAX_POOL_2D, BuiltinOperator.TRANSPOSE,
                        BuiltinOperator.ADD]
    else:  # 3D
        expected_ops = [BuiltinOperator.TRANSPOSE, BuiltinOperator.RESHAPE, BuiltinOperator.MAX_POOL_2D,
                        BuiltinOperator.RESHAPE, BuiltinOperator.TRANSPOSE, BuiltinOperator.ADD]

    assert len(ops) == len(expected_ops)
    assert all(op.builtin_options.operator_type == expected for op, expected in zip(ops, expected_ops))


@pytest.mark.parametrize(
    "input_shape, perm",
    [
        pytest.param([5, 10, 15], [1, 0, 2], id="3D, switch N and C"),
        pytest.param([5, 10, 15], [0, 2, 1], id="3D, switch C and H"),
        pytest.param([5, 10, 15], [2, 1, 0], id="3D, switch N and H"),

        pytest.param([2, 4, 6, 8], [0, 1, 2, 3], id="4D, identity"),
        pytest.param([2, 4, 6, 8], [0, 1, 3, 2], id="4D, switch H and W"),
        pytest.param([2, 4, 6, 8], [0, 2, 1, 3], id="4D, switch C and H"),
        pytest.param([2, 4, 6, 8], [0, 3, 2, 1], id="4D, switch C and W"),
        pytest.param([2, 4, 6, 8], [1, 0, 2, 3], id="4D, switch N and C"),
    ])
def test_convert_transpose_to_channels_first(input_shape: List[int], perm: List[int],
                                             intermediate_tflite_model_provider):
    kernel_shape = [1] * (len(input_shape) - 2)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Add", ["x", "one"], ["o1"]),
            onnx.helper.make_node("Transpose", ["o1"], ["o2"], perm=perm),
            onnx.helper.make_node("MaxPool", ["o2"], ["o"], auto_pad="SAME_UPPER", kernel_shape=kernel_shape),
        ],
        'transpose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('one', TensorProto.FLOAT, [1], [1.])]
    )
    onnx_model = onnx.helper.make_model(graph)
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    config = ConversionConfig()
    # This optimization would cause the test "3D, switch C and H" to fail, because there will be an unnecessary
    #  `Transpose` operator, which gets removed.
    config.optimization_blacklist = [Optimization.REMOVE_IDENTITY_TRANSPOSE_OPERATORS]

    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    ops = intermediate_tflite_model_provider.get_operators()

    if len(input_shape) == 4:
        expected_ops = [BuiltinOperator.ADD, BuiltinOperator.TRANSPOSE, BuiltinOperator.MAX_POOL_2D,
                        BuiltinOperator.TRANSPOSE]
    else:  # 3D
        expected_ops = [BuiltinOperator.ADD, BuiltinOperator.TRANSPOSE, BuiltinOperator.RESHAPE,
                        BuiltinOperator.MAX_POOL_2D, BuiltinOperator.RESHAPE, BuiltinOperator.TRANSPOSE]

    assert len(ops) == len(expected_ops)
    assert all(op.builtin_options.operator_type == expected for op, expected in zip(ops, expected_ops))


@pytest.mark.parametrize(
    "type_",
    [
        TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64, TensorProto.UINT8, TensorProto.BOOL,
        TensorProto.FLOAT
    ])
def test_convert_transpose__types(type_: TensorProto.DataType):
    shape = [13, 37]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Transpose', ['x'], ['y'], perm=[1, 0])],
        'transpose test',
        [onnx.helper.make_tensor_value_info("x", type_, shape)],
        [onnx.helper.make_tensor_value_info("y", type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(to_numpy_type(type_))
    executors.convert_run_compare(onnx_model, input_data)


def test_convert_transpose__unsupported_type():
    type_ = TensorProto.DOUBLE

    shape = [13, 37]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Transpose', ['x'], ['y'], perm=[1, 0])],
        'transpose test',
        [onnx.helper.make_tensor_value_info("x", type_, shape)],
        [onnx.helper.make_tensor_value_info("y", type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)
