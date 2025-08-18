#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math

import numpy as np
import onnx
import onnx.shape_inference
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from onnx2tflite.src.tflite_generator.builtin_options import transpose_options
from tests import executors


@pytest.mark.parametrize("input_shape,axes", [
    pytest.param([2, 3, 4, 5], [2], id="4D-single_positive_added_axis"),
    pytest.param([2, 3, 4, 5], [-2], id="4D-single_negative_added_axis"),
    pytest.param([3, 4, 5], [4, 0], id="4D-multiple_added_axes"),
])
def test_convert_unsqueeze_static_axes(input_shape, axes):
    axes_shape = [len(axes)]
    input_type = TensorProto.INT32

    node = onnx.helper.make_node("Unsqueeze", ["data", "axes"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-unsqueeze",
        [onnx.helper.make_tensor_value_info("data", input_type, input_shape)],
        [onnx.helper.make_tensor_value_info("output", input_type, ())],
        initializer=[
            onnx.helper.make_tensor("axes", TensorProto.INT64, axes_shape, axes),
        ]
    )
    model = onnx.helper.make_model(graph)

    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))

    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("input_shape,axes", [
    pytest.param([2, 3, 4, 5], [2], id="4D-single_positive_added_axis"),
    pytest.param([2, 3, 4, 5], [-2], id="4D-single_negative_added_axis"),
    pytest.param([3, 4, 5], [4, 0], id="4D-multiple_added_axes"),
])
def test_convert_unsqueeze_axes_attribute(input_shape, axes):
    input_type = TensorProto.INT32

    node = onnx.helper.make_node("Unsqueeze", ["data"], ["output"], axes=axes)
    graph = onnx.helper.make_graph(
        [node],
        "graph-unsqueeze",
        [onnx.helper.make_tensor_value_info("data", input_type, input_shape)],
        [onnx.helper.make_tensor_value_info("output", input_type, ())],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 9)])

    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))

    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("input_shape,axes", [
    pytest.param([2, 3, 4, 5], [2], id="4D-axes=[2]"),
    pytest.param([2, 3, 4, 5], [1, 2], id="4D-axes=[1,2]"),
])
@pytest.mark.parametrize("input_type", [
    pytest.param(TensorProto.FLOAT16, id="f16"),
    pytest.param(TensorProto.INT32, id="i32"),
    pytest.param(TensorProto.INT64, id="i64"),
])
def test_convert_unsqueeze_different_input_types_static_axis(input_shape, axes, input_type: int):
    axes_shape = [len(axes)]

    node = onnx.helper.make_node("Unsqueeze", ["data", "axes"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-unsqueeze",
        [onnx.helper.make_tensor_value_info("data", input_type, input_shape)],
        [onnx.helper.make_tensor_value_info("output", input_type, ())],
        initializer=[
            onnx.helper.make_tensor("axes", TensorProto.INT64, axes_shape, axes),
        ]
    )
    model = onnx.helper.make_model(graph)

    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))

    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("input_shape,axes,output_shape", [
    pytest.param([2, 3, 4, 5], [3], [2, 3, 4, 1, 5], id="4D-axes=[3]"),
    pytest.param([2, 3], [-1], [2, 3, 1], id="2D-axes=[-1]"),
    pytest.param([2, 3], [-2], [2, 1, 3], id="2D-axes=[-2]"),
])
def test_convert_unsqueeze_single_axis_dynamic(input_shape, axes, output_shape):
    axes_shape = [len(axes)]
    input_type = TensorProto.INT32

    node = onnx.helper.make_node("Unsqueeze", ["data", "axes"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-unsqueeze",
        [
            onnx.helper.make_tensor_value_info("data", input_type, input_shape),
            onnx.helper.make_tensor_value_info("axes", TensorProto.INT64, axes_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", input_type, output_shape)],
    )
    model = onnx.helper.make_model(graph)

    input_data = {
        0: np.random.randint(0, 255, size=math.prod(input_shape)).reshape(input_shape).astype(
            to_numpy_type(input_type)),
        1: np.array(axes, dtype=np.int64),
    }

    input_data_tflite = input_data[0]

    cc = ConversionConfig()
    cc.skip_shape_inference = True
    cc.allow_inputs_stripping = True

    executors.convert_run_compare(model, input_data, input_data_tflite=input_data_tflite, conversion_config=cc)


@pytest.mark.parametrize("input_shape,axes,output_shape", [
    pytest.param([2, 3, 4], [0, 4], [1, 2, 3, 4, 1], id="3D-axes=[0,4]"),
    pytest.param([2, 3, 4], [2, -1], [2, 3, 1, 4, 1], id="3D-axes=[2,-1]"),
    pytest.param([2, 3], [0, 1], [1, 1, 2, 3], id="2D-axes=[0,1]"),
])
def test_convert_unsqueeze_multi_axes_dynamic(input_shape, axes, output_shape):
    axes_shape = [len(axes)]
    input_type = TensorProto.INT32

    node = onnx.helper.make_node("Unsqueeze", ["data", "axes"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-unsqueeze",
        [
            onnx.helper.make_tensor_value_info("data", input_type, input_shape),
            onnx.helper.make_tensor_value_info("axes", TensorProto.INT64, axes_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", input_type, output_shape)],
    )
    model = onnx.helper.make_model(graph)

    input_data = {
        0: np.random.random_integers(0, 255, size=math.prod(input_shape))
        .reshape(input_shape).astype(to_numpy_type(input_type)),
        1: np.array(axes, dtype=np.int64),
    }
    input_data_tflite = input_data[0]

    cc = ConversionConfig()
    cc.skip_shape_inference = True
    cc.allow_inputs_stripping = True

    executors.convert_run_compare(model, input_data, input_data_tflite=input_data_tflite, conversion_config=cc)


@pytest.mark.parametrize("input_shape,axes", [
    pytest.param([2, 3, 4, 5], [3], id="4D-axes=[3]"),
    pytest.param([2, 3, 4], [0, 4], id="3D-axes=[0,4]"),
])
def test_convert_unsqueeze_inputs_static_quantized(input_shape, axes):
    input_type = TensorProto.FLOAT
    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))
    scale = np.array([0.9], dtype=np.float32)
    zero_point = np.array([0])

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["y"], axis=1),
            onnx.helper.make_node("Unsqueeze", ["y", "axes"], ["output"])
        ],
        "graph-unsqueeze-quantized",
        [],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        initializer=[
            onnx.helper.make_tensor("input", input_type, input_shape, input_data),
            onnx.helper.make_tensor("axes", TensorProto.INT64, [len(axes)], axes),
            onnx.helper.make_tensor("scale", TensorProto.FLOAT, [len(scale)], scale),
            onnx.helper.make_tensor("zero_point", TensorProto.INT8, [len(zero_point)], zero_point)
        ]
    )
    model = onnx.helper.make_model(graph)

    tflite_executor, _ = executors.convert_run_compare(model, {})
    output_quant_params = tflite_executor.get_output_details(0)['quantization_parameters']

    assert output_quant_params['scales'] == scale
    assert output_quant_params['zero_points'] == zero_point


@pytest.mark.parametrize("input_shape,axes", [
    pytest.param([2, 3, 4, 5], [3], id="4D-axes=[3]"),
    pytest.param([2, 3, 4], [0, 4], id="3D-axes=[0,4]"),
])
def test_convert_unsqueeze_with_channel_last_input(input_shape, axes):
    kernel_shape = [1] * (len(input_shape) - 2)
    axes_shape = [len(axes)]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y'], kernel_shape=kernel_shape),
            onnx.helper.make_node("Unsqueeze", ["y", 'axes'], ["output"])
        ],
        'maxpool+unsqueeze',
        inputs=[onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        initializer=[
            onnx.helper.make_tensor("axes", TensorProto.INT64, axes_shape, axes),
        ]
    )

    model = onnx.helper.make_model(graph)

    input_data = np.linspace(0., 1., math.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("input_shape,axes,kernel_shape", [
    pytest.param([2, 2, 4], [0], [2, 2], id="3D-axes=[0]"),
    pytest.param([2, 2, 4], [1], [2, 2], id="3D-axes=[1]"),
    pytest.param([2, 2, 4], [2], [1, 1], id="3D-axes=[2]"),
    pytest.param([2, 2, 4], [3], [1, 1], id="3D-axes=[3]"),
])
def test_convert_unsqueeze_with_channel_last_output(input_shape, axes, kernel_shape):
    axes_shape = [len(axes)]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Unsqueeze", ["x", 'axes'], ["y"]),
            onnx.helper.make_node('MaxPool', ['y'], ['z'], kernel_shape=kernel_shape),
        ],
        'maxpool+unsqueeze',
        inputs=[onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())],
        initializer=[
            onnx.helper.make_tensor("axes", TensorProto.INT64, axes_shape, axes),
        ]
    )

    model = onnx.helper.make_model(graph)

    input_data = np.linspace(0., 1., math.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("axes,output_shape,transpose_ops_count", [
    pytest.param([0], [1, 2, 3, 4], 3, id="output_shape=[1, 2, 3, 4]-axis=[0]"),
    pytest.param([1], [2, 1, 3, 4], 3, id="output_shape=[2, 1, 3, 4]-axis=[1]"),
    pytest.param([2], [2, 3, 1, 4], 2, id="output_shape=[2, 3, 1, 4]-axis=[2]"),
    pytest.param([3], [2, 3, 4, 1], 2, id="output_shape=[2, 3, 4, 1]-axis=[3]"),
    pytest.param([-1], [2, 3, 4, 1], 2, id="output_shape=[2, 3, 4, 1]-axis=[-1]"),
])
def test_convert_unsqueeze_with_channel_last_input_output(axes, output_shape, transpose_ops_count,
                                                          intermediate_tflite_model_provider):
    input_shape = [2, 3, 4]
    kernel_shape_pre = [2] * (len(input_shape) - 2)
    kernel_shape_post = [1] * (len(output_shape) - 2)
    axes_shape = [len(axes)]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y'], kernel_shape=kernel_shape_pre),
            onnx.helper.make_node("Unsqueeze", ["y", 'axes'], ["output"]),
            onnx.helper.make_node('MaxPool', ['output'], ['z'], kernel_shape=kernel_shape_post),
        ],
        'maxpool+unsqueeze',
        inputs=[onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, output_shape)],
        initializer=[
            onnx.helper.make_tensor("axes", TensorProto.INT64, axes_shape, axes),
        ]
    )

    model = onnx.helper.make_model(graph)

    input_data = np.linspace(0., 1., math.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(model, input_data)

    assert intermediate_tflite_model_provider.get_op_count(transpose_options.Transpose) == transpose_ops_count


@pytest.mark.parametrize("type_", [
    TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
    TensorProto.UINT8, TensorProto.UINT32, TensorProto.UINT64,
    TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE,
    TensorProto.BOOL, TensorProto.STRING
], ids=name_for_onnx_type)
def test_convert_unsqueeze__types(type_: TensorProto.DataType):
    shape = [13, 37]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Unsqueeze', ['x', 'axes'], ['y'])],
        'Invalid type test test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('axes', TensorProto.INT64, [1], [1])]
    )

    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    data = (np.random.random(shape) * 100.).astype(to_numpy_type(type_))

    executors.convert_run_compare(onnx_model, data)


def test_convert_unsqueeze__invalid_type():
    type_ = TensorProto.UINT16

    shape = [13, 37]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Unsqueeze', ['x', 'axes'], ['y'])],
        'Invalid type test test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('axes', TensorProto.INT64, [1], [1])]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'UINT16' in logger.conversion_log.get_node_error_message(0)
