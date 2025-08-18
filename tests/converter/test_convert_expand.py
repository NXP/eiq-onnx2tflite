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

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import SkipShapeInferenceConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type, name_for_onnx_type
from tests import executors


@pytest.mark.parametrize("input_shape,new_shape_data,input_type", [
    pytest.param([2, 2, 10, 1], [2], TensorProto.INT32, id="input=4D-new_shape=1D-int32"),
    pytest.param([1, 2, 10, 5], [2, 1, 1, 1], TensorProto.INT32, id="input=4D-new_shape=4D-int32"),
    pytest.param([1, 1, 10], [1, 2, 1], TensorProto.INT32, id="input=3D-new_shape=3D-int32"),
    pytest.param([1, 1, 10], [1, 2, 1], TensorProto.FLOAT, id="input=3D-new_shape=3D-float"),
    pytest.param([1, 1, 10], [2, 1], TensorProto.FLOAT, id="input=3D-new_shape=2D-float"),
    pytest.param([1, 10, 1], [5], TensorProto.FLOAT, id="input=3D-new_shape=1D-float"),
    pytest.param([10], [1, 2, 1], TensorProto.FLOAT, id="input=1D-new_shape=3D-float"),
])
def test_convert_expand_inputs_static(input_shape, new_shape_data, input_type: int):
    new_shape_shape = [len(new_shape_data)]
    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))

    node = onnx.helper.make_node("Expand", ["input", "new_shape"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-expand",
        [],
        [onnx.helper.make_tensor_value_info("output", input_type, ())],
        initializer=[
            onnx.helper.make_tensor("input", input_type, input_shape, input_data),
            onnx.helper.make_tensor("new_shape", TensorProto.INT64, new_shape_shape, new_shape_data),
        ]
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)

    executors.convert_run_compare(model, {})


def test_convert_expand_inputs_static_quantized():
    input_shape = [2, 2, 10, 1]
    new_shape_data = [2]
    input_type = TensorProto.FLOAT
    new_shape_shape = [len(new_shape_data)]
    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))
    scale = [1.0]
    zero_point = [0]

    # node = onnx.helper.make_node("Expand", ["input", "new_shape"], ["output"])
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["y"], axis=1),
            onnx.helper.make_node("Expand", ["y", "new_shape"], ["output"])
        ],
        "graph-expand-quantized",
        [],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        initializer=[
            onnx.helper.make_tensor("input", input_type, input_shape, input_data),
            onnx.helper.make_tensor("new_shape", TensorProto.INT64, new_shape_shape, new_shape_data),
            onnx.helper.make_tensor("scale", TensorProto.FLOAT, [len(scale)], scale),
            onnx.helper.make_tensor("zero_point", TensorProto.INT8, [len(zero_point)], zero_point)
        ]
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)

    tflite_executor, _ = executors.convert_run_compare(model, {})
    output_quant_params = tflite_executor.get_output_details(0)['quantization_parameters']
    assert output_quant_params['scales'] == [1.0]
    assert output_quant_params['zero_points'] == [0]


@pytest.mark.parametrize("input_shape,new_shape_data,input_type", [
    pytest.param([2, 2, 10, 1], [2], TensorProto.INT32, id="input=4D-new_shape=1D-int32"),
    pytest.param([1, 2, 10, 5], [2, 1, 1, 1], TensorProto.INT32, id="input=4D-new_shape=4D-int32"),
    pytest.param([1, 1, 10], [1, 2, 1], TensorProto.INT32, id="input=3D-new_shape=3D-int32"),
    pytest.param([1, 1, 10], [1, 2, 1], TensorProto.FLOAT, id="input=3D-new_shape=3D-float"),
    pytest.param([1, 1, 10], [2, 1], TensorProto.FLOAT, id="input=3D-new_shape=2D-float"),
    pytest.param([1, 10, 1], [5], TensorProto.FLOAT, id="input=3D-new_shape=1D-float"),
    pytest.param([10], [1, 2, 1], TensorProto.FLOAT, id="input=1D-new_shape=3D-float"),
])
def test_convert_expand_new_shape_dynamic(input_shape, new_shape_data, input_type: int):
    output_shape = np.broadcast_shapes(input_shape, new_shape_data)
    new_shape_shape = [len(new_shape_data)]
    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))

    node = onnx.helper.make_node("Expand", ["input", "new_shape"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-expand",
        [
            onnx.helper.make_tensor_value_info("new_shape", TensorProto.INT64, new_shape_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", input_type, output_shape)],
        initializer=[onnx.helper.make_tensor("input", input_type, input_shape, input_data)]
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)

    input_data = np.array(new_shape_data, dtype=np.int64)

    executors.convert_run_compare(model, input_data, conversion_config=SkipShapeInferenceConfig())


@pytest.mark.parametrize("input_shape,new_shape_data,input_type", [
    pytest.param([2, 2, 10, 1], [2], TensorProto.INT32, id="input=4D-new_shape=1D-int32"),
    pytest.param([1, 2, 10, 5], [2, 1, 1, 1], TensorProto.INT32, id="input=4D-new_shape=4D-int32"),
    pytest.param([1, 1, 10], [1, 2, 1], TensorProto.INT32, id="input=3D-new_shape=3D-int32"),
    pytest.param([1, 1, 10], [1, 2, 1], TensorProto.FLOAT, id="input=3D-new_shape=3D-float"),
    pytest.param([1, 1, 10], [2, 1], TensorProto.FLOAT, id="input=3D-new_shape=2D-float"),
    pytest.param([1, 10, 1], [5], TensorProto.FLOAT, id="input=3D-new_shape=1D-float"),
    pytest.param([10], [1, 2, 1], TensorProto.FLOAT, id="input=1D-new_shape=3D-float"),
])
def test_convert_expand_both_inputs_dynamic(input_shape, new_shape_data, input_type: int):
    output_shape = np.broadcast_shapes(input_shape, new_shape_data)
    new_shape_shape = [len(new_shape_data)]

    node = onnx.helper.make_node("Expand", ["input", "new_shape"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-expand",
        [
            onnx.helper.make_tensor_value_info("new_shape", TensorProto.INT64, new_shape_shape),
            onnx.helper.make_tensor_value_info("input", input_type, input_shape)
        ],
        [onnx.helper.make_tensor_value_info("output", input_type, output_shape)],
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)

    input_data = {
        0: np.array(new_shape_data, dtype=np.int64),
        1: np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))
    }

    executors.convert_run_compare(model, input_data, conversion_config=SkipShapeInferenceConfig())


@pytest.mark.parametrize("type_", [
    TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
    TensorProto.UINT8, TensorProto.UINT32, TensorProto.UINT64,
    TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE,
    TensorProto.BOOL
], ids=name_for_onnx_type)
def test_convert_expand__types(type_: TensorProto.DataType):
    shape = [3]
    new_shape = [5, 4, 3]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(to_numpy_type(type_))  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Expand', ['x', 'new_shape'], ['y']),
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('new_shape', TensorProto.INT64, [len(new_shape)], new_shape)]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_expand__unsupported_type():
    type_ = TensorProto.STRING

    shape = [3]
    new_shape = [5, 4, 3]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Expand', ['x', 'new_shape'], ['y']),
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('new_shape', TensorProto.INT64, [len(new_shape)], new_shape)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'STRING' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("input_shape,new_shape_data", [
    pytest.param([1, 10, 2], [2], id="input=3D-new_shape=1D"),
    pytest.param([1, 10, 2], [10, 2], id="input=3D-new_shape=2D"),
    pytest.param([2, 10, 3], [1], id="input=3D-new_shape=1D-ones"),
    pytest.param([2, 10, 1], [1, 1], id="input=3D-new_shape=2D-ones"),
    pytest.param([2, 10, 1], [1, 1, 1], id="input=3D-new_shape=3D-ones"),
])
def test_convert_expand__shape_static__skipped(input_shape, new_shape_data, intermediate_tflite_model_provider):
    new_shape_shape = [len(new_shape_data)]
    input_data = np.random.random(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(TensorProto.FLOAT))

    nodes = [
        onnx.helper.make_node("Expand", ["input", "new_shape"], ["y"]),
        onnx.helper.make_node("Softmax", ["y"], ["output"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "graph-expand",
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        initializer=[
            onnx.helper.make_tensor("new_shape", TensorProto.INT64, new_shape_shape, new_shape_data),
        ]
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)

    executors.convert_run_compare(model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.SOFTMAX
    ])
