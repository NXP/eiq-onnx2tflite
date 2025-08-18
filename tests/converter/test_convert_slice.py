#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import SkipShapeInferenceConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "input_shape, starts, ends, axes",
    [
        pytest.param([5, 10, 24], [3, 5, 15], [5, 9, 24], [0, 1, 2], id="axes = [0, 1, 2]"),
        pytest.param([5, 10, 24], [3], [7], [1], id="axes = [1]"),
        pytest.param([5, 10], [3], [7], [1], id="axes = [1]"),
        pytest.param([5, 10, 24, 17], [3, 0], [7, 3], [1, 3], id="axes = [1, 3]"),
        pytest.param([5, 10, 24, 17], [3, 0, 23, 15], [5, 9, 24, 16], [0, 1, 2, 3], id="axes = [0, 1, 2, 3]"),

        pytest.param([5, 10, 24, 17], [-7, -17], [7, 3], [1, 3], id="negative starts"),
        pytest.param([5, 10, 24, 17], [3, 0], [-3, -14], [1, 3], id="negative ends"),
        pytest.param([5, 10, 24, 17], [3, 0], [7, 3], [-3, -1], id="negative axes"),

        pytest.param([5, 10, 24, 17], [11, 20], [5, 3], [1, 3], id="overflow starts"),
        pytest.param([5, 10, 24, 17], [3, 0], [10, 18], [1, 3], id="overflow ends"),
        pytest.param([5, 10, 24, 17], [-10, -18], [5, 3], [1, 3], id="underflow starts"),
        pytest.param([5, 10, 24, 17], [3, 0], [-10, -18], [1, 3], id="underflow ends"),

        pytest.param([5, 10, 24, 17], [-27, 0, 15, 4], [12, 30, -30, 2], [3, 2, 0, 1],
                     id="over/under-flow combination"),
    ])
def test_convert_slice(input_shape, starts, ends, axes):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["output"])],
        'slice test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("starts", TensorProto.INT32, [len(starts)], starts),
            onnx.helper.make_tensor("ends", TensorProto.INT32, [len(ends)], ends),
            onnx.helper.make_tensor("axes", TensorProto.INT32, [len(axes)], axes),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize('type_', [
    TensorProto.FLOAT, TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64, TensorProto.UINT8,
    TensorProto.UINT32, TensorProto.BOOL, TensorProto.STRING
], ids=name_for_onnx_type)
def test_convert_slice__types(type_: TensorProto.DataType):
    input_shape, starts, ends, axes = [5, 10, 24], [3, 5, 15], [5, 9, 24], [0, 1, 2]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["y"])],
        'slice test',
        [onnx.helper.make_tensor_value_info("x", type_, input_shape)],
        [onnx.helper.make_tensor_value_info("y", type_, ())],
        [
            onnx.helper.make_tensor("starts", TensorProto.INT32, [len(starts)], starts),
            onnx.helper.make_tensor("ends", TensorProto.INT32, [len(ends)], ends),
            onnx.helper.make_tensor("axes", TensorProto.INT32, [len(axes)], axes),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(type_))

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_slice__unsupported_type():
    type_ = TensorProto.DOUBLE

    input_shape, starts, ends, axes = [5, 10, 24], [3, 5, 15], [5, 9, 24], [0, 1, 2]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["y"])],
        'slice test',
        [onnx.helper.make_tensor_value_info("x", type_, input_shape)],
        [onnx.helper.make_tensor_value_info("y", type_, ())],
        [
            onnx.helper.make_tensor("starts", TensorProto.INT32, [len(starts)], starts),
            onnx.helper.make_tensor("ends", TensorProto.INT32, [len(ends)], ends),
            onnx.helper.make_tensor("axes", TensorProto.INT32, [len(axes)], axes),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


def test_convert_slice_quantized():
    input_shape = [5, 10, 24]
    starts = [3, 5, 15]
    ends = [5, 9, 24]
    axes = [0, 1, 2]
    scale = [1.0]
    zero_point = [0]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["y"], axis=1),
            onnx.helper.make_node("Slice", ["y", "starts", "ends", "axes"], ["output"])
        ],
        'slice test quantized',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor("starts", TensorProto.INT32, [len(starts)], starts),
            onnx.helper.make_tensor("ends", TensorProto.INT32, [len(ends)], ends),
            onnx.helper.make_tensor("axes", TensorProto.INT32, [len(axes)], axes),
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


@pytest.mark.parametrize(
    "input_shape, starts, ends,",
    [
        pytest.param([42], [17], [40], id="1D"),
        pytest.param([13, 37], [4, 2], [6, 5], id="2D"),
        pytest.param([5, 10, 15], [3, 5, 15], [5, 9, 24], id="3D"),
        pytest.param([5, 10, 24, 17], [3, 0, 1, 7], [7, 3, 15, 8], id="4D"),
        pytest.param([5, 10, 24, 17, 3], [3, 0, 1, 7, 0], [7, 3, 15, 8, 3], id="5D"),

    ])
def test_convert_slice_with_implicit_axes(input_shape, starts, ends):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Slice", ["input", "starts", "ends"], ["output"])],
        'slice test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("starts", TensorProto.INT32, [len(starts)], starts),
            onnx.helper.make_tensor("ends", TensorProto.INT32, [len(ends)], ends),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, starts, ends, axes",
    [
        pytest.param([5, 10, 24], [3, 5, 15], [5, 9, 24], [0, 1, 2], id="axes = [0, 1, 2]"),
        pytest.param([5, 10, 24], [3], [7], [1], id="axes = [1]"),
        pytest.param([5, 10, 24, 17], [3, 0], [7, 3], [1, 3], id="axes = [1, 3]"),
        pytest.param([5, 10, 24, 17], [3, 0, 23, 15], [5, 9, 24, 16], [0, 1, 2, 3],
                     id="axes = [0, 1, 2, 3]"),

        pytest.param([5, 10, 24, 17], [-7, -15], [7, 3], [1, 3], id="negative starts (smaller than dimension size)"),
        pytest.param([5, 10, 24, 17], [-10, -17], [7, 3], [1, 3], id="negative starts (exactly dimension size)"),
        pytest.param([5, 10, 24, 17], [3, 0], [-3, -14], [1, 3], id="negative ends"),
        pytest.param([5, 10, 24, 17], [3, 0], [7, 3], [-3, -1], id="negative axes"),

        pytest.param([5, 10, 24, 17], [11, 20], [5, 3], [1, 3], id="overflow starts"),
        pytest.param([5, 10, 24, 17], [3, 0], [10, 18], [1, 3], id="overflow ends"),
        pytest.param([5, 10, 24, 17], [-10, -18], [5, 3], [1, 3], id="underflow starts"),
        pytest.param([5, 10, 24, 17], [3, 0], [-10, -18], [1, 3], id="underflow ends"),

        pytest.param([5, 10, 24, 17], [-27, 0, 15, 4], [12, 30, -30, 2], [3, 2, 0, 1],
                     id="over/under-flow combination"),
    ])
def test_convert_slice_with_channels_first_tensors(input_shape, starts, ends, axes):
    kernel_shape = [1] * (len(input_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MaxPool", ["input"], ["max_pool_out"], kernel_shape=kernel_shape,
                                  strides=kernel_shape),
            onnx.helper.make_node("Slice", ["max_pool_out", "starts", "ends", "axes"], ["output"]),
        ],
        'slice test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("starts", TensorProto.INT32, [len(starts)], starts),
            onnx.helper.make_tensor("ends", TensorProto.INT32, [len(ends)], ends),
            onnx.helper.make_tensor("axes", TensorProto.INT32, [len(axes)], axes),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, axes, steps",
    [
        pytest.param([6, 8, 10], [0, 2], [1, 1], id="steps = [1, 1]"),
        pytest.param([6, 8, 10, 12], [0, 2, 3], [1, 1, 1], id="steps = [1, 1, 1]"),
        pytest.param([6, 8, 10, 12], [0, 2, 3], [1, 1, 2], id="steps = [1, 1, 2]"),
        pytest.param([6, 8, 10, 12], [0], [2], id="steps = [2]"),
    ])
def test_convert_slice_with_explicit_steps(input_shape, axes, steps):
    starts = [1] * len(axes)
    ends = [3] * len(axes)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["output"])],
        'slice test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("starts", TensorProto.INT32, [len(starts)], starts),
            onnx.helper.make_tensor("ends", TensorProto.INT32, [len(ends)], ends),
            onnx.helper.make_tensor("axes", TensorProto.INT32, [len(axes)], axes),
            onnx.helper.make_tensor("steps", TensorProto.INT32, [len(axes)], steps),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, axes",
    [
        pytest.param([6, 8, 10], [0, 2], id="axes = [0, 2]"),
        pytest.param([6, 8, 10, 12], [0, 2, 3], id="axes = [0, 2, 3]"),
    ])
def test_convert_slice_with_dynamic_inputs(input_shape, axes):
    operand_shape = [len(axes)]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["output"])],
        'slice test',
        [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("starts", TensorProto.INT32, operand_shape),
            onnx.helper.make_tensor_value_info("ends", TensorProto.INT32, operand_shape),
            onnx.helper.make_tensor_value_info("steps", TensorProto.INT32, operand_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor("axes", TensorProto.INT32, [len(axes)], axes)]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=SkipShapeInferenceConfig())
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize(
    "input_shape, starts, ends,",
    [
        pytest.param([42], [17], [40], id="1D"),
        pytest.param([13, 37], [4, 2], [6, 5], id="2D"),
        pytest.param([5, 10, 15], [3, 5, 15], [5, 9, 24], id="3D"),
        pytest.param([5, 10, 24, 17], [3, 0, 1, 7], [7, 3, 15, 8], id="4D"),
        pytest.param([5, 10, 24, 17, 3], [3, 0, 1, 7, 0], [7, 3, 15, 8, 3], id="5D"),

    ])
def test_convert_slice_v13__omitted_axes_and_steps(input_shape, starts, ends):
    # Use tensors with name '', to represent omitted `axes` and `steps`.

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Slice", ["input", "starts", "ends", '', ''], ["output"])],
        'slice test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("starts", TensorProto.INT32, [len(starts)], starts),
            onnx.helper.make_tensor("ends", TensorProto.INT32, [len(ends)], ends),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, starts, ends, axes",
    [
        pytest.param([5, 10, 24], [5, 9, 24], [3, 5, 15], [0, 1, 2], id="axes = [0, 1, 2]"),
        pytest.param([5, 10, 24], [7], [3], [1], id="axes = [1]"),
        pytest.param([5, 10], [7], [3], [1], id="axes = [1]"),
        pytest.param([5, 10, 24, 17], [7, 3], [3, 0], [1, 3], id="axes = [1, 3]"),
        pytest.param([5, 10, 24, 17], [5, 9, 24, 16], [3, 0, 23, 15], [0, 1, 2, 3], id="axes = [0, 1, 2, 3]"),

        pytest.param([5, 10, 24, 17], [7, 3], [-7, -17], [1, 3], id="negative starts"),
        pytest.param([5, 10, 24, 17], [-3, -14], [3, 0], [1, 3], id="negative ends"),
        pytest.param([5, 10, 24, 17], [7, 3], [3, 0], [-3, -1], id="negative axes"),

        pytest.param([5, 10, 24, 17], [11, 20], [5, 3], [1, 3], id="overflow starts"),
        pytest.param([5, 10, 24, 17], [3, 0], [10, 18], [1, 3], id="overflow ends"),
        pytest.param([5, 10, 24, 17], [-10, -18], [5, 3], [1, 3], id="underflow starts"),
        pytest.param([5, 10, 24, 17], [3, 0], [-10, -18], [1, 3], id="underflow ends"),

        pytest.param([5, 10, 24, 17], [-27, 0, 15, 4], [12, 30, -30, 2], [3, 2, 0, 1],
                     id="over/under-flow combination"),
    ])
@pytest.mark.parametrize("step", [-1, -2], ids=lambda x: f"step={x} ")
def test_convert_slice__all_steps_negative(input_shape, starts, ends, axes, step):
    steps = [step] * len(axes)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["output"])],
        'slice test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("starts", TensorProto.INT32, [len(starts)], starts),
            onnx.helper.make_tensor("ends", TensorProto.INT32, [len(ends)], ends),
            onnx.helper.make_tensor("axes", TensorProto.INT32, [len(axes)], axes),
            onnx.helper.make_tensor("steps", TensorProto.INT32, [len(steps)], steps),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)
