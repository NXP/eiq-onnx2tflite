#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize(
    "input_shapes",
    [
        pytest.param([[5]] * 2, id="1D, 2 inputs"),
        pytest.param([[5]] * 3, id="1D, 3 inputs"),
        pytest.param([[5]] * 4, id="1D, 4 inputs"),
        pytest.param([[5]] * 5, id="1D, 5 inputs"),

        pytest.param([[2, 3]] * 2, id="2D, 2 inputs"),
        pytest.param([[2, 3]] * 3, id="2D, 3 inputs"),
        pytest.param([[2, 3]] * 4, id="2D, 4 inputs"),
        pytest.param([[2, 3]] * 5, id="2D, 5 inputs"),

        pytest.param([[2, 3, 5]] * 2, id="3D, 2 inputs"),
        pytest.param([[2, 3, 5]] * 3, id="3D, 3 inputs"),
        pytest.param([[2, 3, 5]] * 4, id="3D, 4 inputs"),
        pytest.param([[2, 3, 5]] * 5, id="3D, 5 inputs"),

        pytest.param([[2, 3, 5, 5]] * 2, id="4D, 2 inputs"),
        pytest.param([[2, 3, 5, 5]] * 3, id="4D, 3 inputs"),
        pytest.param([[2, 3, 5, 5]] * 4, id="4D, 4 inputs"),
        pytest.param([[2, 3, 5, 5]] * 5, id="4D, 5 inputs"),

        pytest.param([[2, 3, 4, 5, 5]] * 2, id="5D, 2 inputs"),
        pytest.param([[2, 3, 4, 5, 5]] * 3, id="5D, 3 inputs"),
        pytest.param([[2, 3, 4, 5, 5]] * 4, id="5D, 4 inputs"),
        pytest.param([[2, 3, 4, 5, 5]] * 5, id="5D, 5 inputs"),

    ])
def test_convert_sum(input_shapes: List[List[int]]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sum", ["input" + str(i) for i in range(len(input_shapes))], ["output"])],
        "Sum test",
        [onnx.helper.make_tensor_value_info("input" + str(i), TensorProto.FLOAT, input_shape)
         for i, input_shape in enumerate(input_shapes)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {i: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
                  for i, input_shape in enumerate(input_shapes)}

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_skipped_sum():
    input_shape = [2, 3, 4, 5]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Sum", ["input"], ["sum_out"]),
            onnx.helper.make_node("MaxPool", ["sum_out"], ["output"], kernel_shape=[1, 1], auto_pad="SAME_UPPER")
        ],
        "Sum test",
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_single_input_sum():
    # The 'Sum' cannot be skipped, because it produces the model output

    input_shape = [2, 3, 4, 5]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Sum", ["input"], ["output"]),
        ],
        "Sum test",
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shapes",
    [
        pytest.param([[3], [2, 3]], id="2D"),
        pytest.param([[4], [2, 3, 4]], id="3D"),
        pytest.param([[2, 3, 4, 5], [4, 5]], id="4D"),
        pytest.param([[6], [2, 3, 4, 5, 6]], id="5D"),
    ])
def test_convert_sum_with_2_input_shape_broadcasting(input_shapes: List[List[int]]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sum", ["input" + str(i) for i in range(len(input_shapes))], ["output"])],
        "Sum test",
        [onnx.helper.make_tensor_value_info("input" + str(i), TensorProto.FLOAT, input_shape)
         for i, input_shape in enumerate(input_shapes)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {i: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
                  for i, input_shape in enumerate(input_shapes)}

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shapes",
    [
        pytest.param([[4], [3, 4], [2, 3, 4]], id="3D, 3 inputs"),
        pytest.param([[4, 1], [3, 4, 1], [2, 3, 4, 1]], id="4D, 3 inputs"),
    ])
def test_convert_sum_with_many_inputs_shape_broadcasting(input_shapes: List[List[int]]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sum", ["input" + str(i) for i in range(len(input_shapes))], ["output"])],
        "Sum test",
        [onnx.helper.make_tensor_value_info("input" + str(i), TensorProto.FLOAT, input_shape)
         for i, input_shape in enumerate(input_shapes)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    # TODO Rewrite this test once shape broadcasting support is implemented.

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape",
    [
        pytest.param([2, 3, 4, 5], [5]),
        pytest.param([1, 6, 8], [2, 4, 1, 1]),
        pytest.param([15], [5, 10, 15]),
    ])
def test_convert_sum_with_channels_first_shape_broadcasting_and_2_inputs(input_1_shape, input_2_shape):
    output_shape = list(np.broadcast_shapes(input_1_shape, input_2_shape))
    kernel_shape = [1] * (len(output_shape) - 2)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Sum", ["input_1", "input_2"], ["sum_out"]),
            onnx.helper.make_node("MaxPool", ["sum_out"], ["output"], kernel_shape=kernel_shape, strides=kernel_shape),
        ],
        "Sum test",
        [
            onnx.helper.make_tensor_value_info("input_1", TensorProto.FLOAT, input_1_shape),
            onnx.helper.make_tensor_value_info("input_2", TensorProto.FLOAT, input_2_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = {
        0: np.arange(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.float32),
        1: np.arange(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shapes, output_shape",
    [
        pytest.param([[3, 4, 5], [2, 3, 4, 5], [5]], [2, 3, 4, 5], id="4D, 3 inputs"),
    ])
def test_convert_sum_with_channels_first_shape_broadcasting_and_many_inputs(input_shapes: List[List[int]],
                                                                            output_shape: List[int]):
    kernel_shape = [1] * (len(output_shape) - 2)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Sum", ["input" + str(i) for i in range(len(input_shapes))], ["sum_out"]),
            onnx.helper.make_node("MaxPool", ["sum_out"], ["output"], kernel_shape=kernel_shape),
        ],
        "Sum test",
        [onnx.helper.make_tensor_value_info("input" + str(i), TensorProto.FLOAT, input_shape)
         for i, input_shape in enumerate(input_shapes)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    # TODO Rewrite this test once shape broadcasting support is implemented.

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


def test_convert_sum_with_mismatched_input_types():
    input_shape = output_shape = [1, 3, 12, 12]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sum", ["input1", "input2"], ["output"])], "Sum test",
        [
            onnx.helper.make_tensor_value_info("input1", TensorProto.INT32, input_shape),
            onnx.helper.make_tensor_value_info("input2", TensorProto.UINT32, input_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL


def test_convert_sum__invalid_type():
    type_ = TensorProto.DOUBLE

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Sum', ['x', 'x'], ['y'])],
        'Invalid type test test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert___quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x1', 's', 'zp'], ['x1_']),
            onnx.helper.make_node('QuantizeLinear', ['x2', 's', 'zp'], ['x2_']),
            onnx.helper.make_node('Sum', ['x1_', 'x2_'], ['x3']),
            onnx.helper.make_node('Reshape', ['x3', 'flat_shape'], ['x4']),
            onnx.helper.make_node('DequantizeLinear', ['x4', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [
            onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [1], [np.prod(shape)])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(2) == logger.Code.INVALID_ONNX_MODEL
