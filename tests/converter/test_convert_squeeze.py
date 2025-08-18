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


@pytest.mark.parametrize("input_shape", [
    pytest.param([2, 1, 1, 2, 3, 5, 1, 2, 1], id="9D"),
    pytest.param([2, 1, 1, 5, 1], id="5D"),
    pytest.param([2, 1, 1, 5], id="4D"),
    pytest.param([3, 4, 1], id="3D"),
])
def test_convert_squeeze(input_shape):
    input_type = TensorProto.INT32

    node = onnx.helper.make_node("Squeeze", ["data"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-squeeze",
        [onnx.helper.make_tensor_value_info("data", input_type, input_shape)],
        [onnx.helper.make_tensor_value_info("output", input_type, ())],
    )
    model = onnx.helper.make_model(graph)

    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))

    executors.convert_run_compare(model, input_data)


def test_convert_squeeze__omitted_axes_input():
    # Use tensor with name "" to represent omitted `axes` input tensor.
    # ORT doesn't support this right now. If support is added in the future, this test will fail. Then we can update
    #  it to make sure conversion works correctly.

    input_shape = [2, 4, 5, 5]
    input_type = TensorProto.INT32

    node = onnx.helper.make_node("Squeeze", ["data", ""], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-squeeze",
        [onnx.helper.make_tensor_value_info("data", input_type, input_shape)],
        [onnx.helper.make_tensor_value_info("output", input_type, ())],
    )
    model = onnx.helper.make_model(graph)

    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))

    with pytest.raises(Exception) as e:
        executors.convert_run_compare(model, input_data)
    assert 'axes_tensor != nullptr was false. Axes input is null' in e.value.args[0]


@pytest.mark.parametrize("input_shape,axes", [
    pytest.param([2, 3, 1, 4], [2], id="4D-axes=[2]"),
    pytest.param([2, 3, 4, 1, 5], [-2], id="5D-axes=[-2]"),
    pytest.param([1, 3, 4, 5, 1], [4, 0], id="5D-axes=[0,4]"),
])
def test_convert_squeeze_static_axes(input_shape, axes):
    axes_shape = [len(axes)]
    input_type = TensorProto.INT32

    node = onnx.helper.make_node("Squeeze", ["data", "axes"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-squeeze",
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
    pytest.param([2, 3, 1, 5], [2], id="4D-axes=[2]"),
    pytest.param([2, 3, 4, 1, 5], [-2], id="5D-axes=[-2]"),
    pytest.param([1, 3, 4, 5, 1], [4, 0], id="5D-axes=[4,0]"),
])
def test_convert_squeeze_axes_attribute(input_shape, axes):
    input_type = TensorProto.INT32

    node = onnx.helper.make_node("Squeeze", ["data"], ["output"], axes=axes)
    graph = onnx.helper.make_graph(
        [node],
        "graph-squeeze",
        [onnx.helper.make_tensor_value_info("data", input_type, input_shape)],
        [onnx.helper.make_tensor_value_info("output", input_type, ())],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 9)])

    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))

    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("input_shape,axes", [
    pytest.param([2, 3, 1, 4, 5], [2], id="4D-axes=[2]"),
    pytest.param([2, 1, 1, 4], [1, 2], id="4D-axes=[1,2]"),
])
@pytest.mark.parametrize("input_type", [
    TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT8, TensorProto.INT16, TensorProto.INT32,
    TensorProto.INT64, TensorProto.UINT8, TensorProto.UINT32, TensorProto.UINT64, TensorProto.STRING, TensorProto.BOOL
], ids=name_for_onnx_type)
def test_convert_squeeze_different_input_types_static_axis(input_shape, axes, input_type: int):
    axes_shape = [len(axes)]

    node = onnx.helper.make_node("Squeeze", ["data", "axes"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-squeeze",
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
    pytest.param([2, 3, 4, 1, 5], [3], [2, 3, 4, 5], id="5D-axes=[3]"),
    pytest.param([2, 3, 1], [-1], [2, 3], id="3D-axes=[-1]"),
    pytest.param([2, 1, 3], [-2], [2, 3], id="3D-axes=[-2]"),
])
def test_convert_squeeze_single_axis_dynamic(input_shape, axes, output_shape):
    axes_shape = [len(axes)]
    input_type = TensorProto.INT32

    node = onnx.helper.make_node("Squeeze", ["data", "axes"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-squeeze",
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
    pytest.param([1, 2, 3, 4, 1], [0, 4], [2, 3, 4], id="5D-axes=[0,4]"),
    pytest.param([2, 3, 1, 4, 1], [2, -1], [2, 3, 4], id="5D-axes=[2,-1]"),
    pytest.param([1, 1, 2, 3], [0, 1], [2, 3], id="4D-axes=[0,1]"),
])
def test_convert_squeeze_multi_axes_dynamic(input_shape, axes, output_shape):
    axes_shape = [len(axes)]
    input_type = TensorProto.INT32

    node = onnx.helper.make_node("Squeeze", ["data", "axes"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-squeeze",
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
    pytest.param([2, 3, 4, 1, 5], [3], id="5D-axes=[3]"),
    pytest.param([1, 2, 3, 4, 1], [0, 4], id="5D-axes=[0,4]"),
])
def test_convert_squeeze_inputs_static_quantized(input_shape, axes):
    input_type = TensorProto.FLOAT
    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))
    scale = np.array([0.9], dtype=np.float32)
    zero_point = np.array([0])

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["y"], axis=1),
            onnx.helper.make_node("Squeeze", ["y", "axes"], ["output"])
        ],
        "graph-squeeze-quantized",
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
    pytest.param([2, 3, 4, 1], [3], id="4D-axes=[3]"),
    pytest.param([1, 1, 3], [0, 1], id="3D-axes=[0,1]"),
])
def test_convert_squeeze_with_channel_last_input(input_shape, axes):
    kernel_shape = [1] * (len(input_shape) - 2)
    axes_shape = [len(axes)]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y'], kernel_shape=kernel_shape),
            onnx.helper.make_node("Squeeze", ["y", 'axes'], ["output"])
        ],
        'maxpool+squeeze',
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
    pytest.param([1, 2, 5, 2, 4], [0], [2, 2], id="3D-axes=[0]"),
    pytest.param([2, 1, 5, 2, 4], [1], [2, 2], id="3D-axes=[1]"),
    pytest.param([2, 2, 5, 1, 4], [3], [1, 1], id="3D-axes=[2]"),
    pytest.param([2, 2, 5, 4, 1], [4], [1, 1], id="3D-axes=[3]"),
])
def test_convert_squeeze_with_channel_last_output(input_shape, axes, kernel_shape):
    axes_shape = [len(axes)]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Squeeze", ["x", 'axes'], ["y"]),
            onnx.helper.make_node('MaxPool', ['y'], ['z'], kernel_shape=kernel_shape),
        ],
        'maxpool+squeeze',
        inputs=[onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())],
        initializer=[
            onnx.helper.make_tensor("axes", TensorProto.INT64, axes_shape, axes),
        ]
    )

    model = onnx.helper.make_model(graph)

    input_data = np.linspace(0., 1., math.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("input_shape,axes,transpose_ops_count", [
    pytest.param([1, 2, 3, 4], [0], 3, id="input_shape=[1, 2, 3, 4]-axis=[0]"),
    pytest.param([2, 1, 3, 4], [1], 3, id="input_shape=[2, 1, 3, 4]-axis=[1]"),
    pytest.param([2, 3, 1, 4], [2], 2, id="input_shape=[2, 3, 1, 4]-axis=[2]"),
    pytest.param([2, 3, 4, 1], [3], 2, id="input_shape=[2, 3, 4, 1]-axis=[3]"),
    pytest.param([2, 3, 4, 1], [-1], 2, id="input_shape=[2, 3, 4, 1]-axis=[-1]"),
])
def test_convert_squeeze_with_channel_last_input_output(input_shape, axes, transpose_ops_count,
                                                        intermediate_tflite_model_provider):
    output_shape = [2, 3, 4]
    kernel_shape_pre = [1] * (len(input_shape) - 2)
    kernel_shape_post = [2] * (len(output_shape) - 2)
    axes_shape = [len(axes)]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y'], kernel_shape=kernel_shape_pre),
            onnx.helper.make_node("Squeeze", ["y", 'axes'], ["output"]),
            onnx.helper.make_node('MaxPool', ['output'], ['z'], kernel_shape=kernel_shape_post),
        ],
        'maxpool+squeeze',
        inputs=[onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())],
        initializer=[
            onnx.helper.make_tensor("axes", TensorProto.INT64, axes_shape, axes),
        ]
    )

    model = onnx.helper.make_model(graph)

    input_data = np.linspace(0., 1., math.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(model, input_data)

    assert intermediate_tflite_model_provider.get_op_count(transpose_options.Transpose) == transpose_ops_count


def test_convert_squeeze__invalid_type():
    type_ = TensorProto.UINT16

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Squeeze', ['x'], ['y'])],
        'Invalid type test test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'UINT16' in logger.conversion_log.get_node_error_message(0)
