#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import Tuple

import numpy as np
import onnx.helper
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


def test_convert_flatten_default_axis_after_conv():
    """ Ensure default flatten works without calling axis """
    node_conv = onnx.helper.make_node(
        'Conv',
        inputs=['x', 'W'], outputs=['y'],
        kernel_shape=[3, 3], pads=[0, 0, 0, 0],
    )
    node_flatten = onnx.helper.make_node(
        'Flatten', inputs=['y'], outputs=['z'],
    )
    x_shape = (5, 3, 16, 16)
    w_shape = (3, 3, 3, 3)

    graph = onnx.helper.make_graph(
        [node_conv, node_flatten],
        'conv+flatten',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("W", TensorProto.FLOAT, w_shape)
        ],
        [
            onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())
        ]
    )
    original_model = onnx.helper.make_model(graph, producer_name="onnx-conv-flatten")
    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.arange(np.prod(w_shape)).reshape(w_shape).astype(np.float32)
    }

    executors.convert_run_compare(original_model, input_data)


def test_convert_flatten_default_axis_after_qlinearconv():
    """ Ensure default flatten works without calling axis """
    node_conv = onnx.helper.make_node(
        'QLinearConv',
        inputs=['x', 'x_scale', 'x_zero_point', 'W', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'],
        outputs=['y'],
        kernel_shape=[3, 3], pads=[0, 0, 0, 0],
    )
    node_flatten = onnx.helper.make_node(
        'Flatten', inputs=['y'], outputs=['z'],
    )
    x_shape = (5, 3, 16, 16)
    x_scale = [1.0]
    x_zero_point = [0]
    w_shape = (3, 3, 3, 3)
    w_scale = [1.0]
    w_zero_point = [0]
    y_scale = [1.0]
    y_zero_point = [0]

    graph = onnx.helper.make_graph(
        [node_conv, node_flatten],
        'conv+flatten',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.INT8, x_shape),
            onnx.helper.make_tensor_value_info("W", TensorProto.INT8, w_shape)
        ],
        [
            onnx.helper.make_tensor_value_info("z", TensorProto.INT8, ())
        ],
        [
            onnx.helper.make_tensor('x_scale', TensorProto.FLOAT, [len(x_scale)], x_scale),
            onnx.helper.make_tensor('x_zero_point', TensorProto.INT8, [len(x_zero_point)], x_zero_point),
            onnx.helper.make_tensor("w_scale", TensorProto.FLOAT, [len(w_scale)], w_scale),
            onnx.helper.make_tensor('w_zero_point', TensorProto.INT8, [len(w_zero_point)], w_zero_point),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [len(y_scale)], y_scale),
            onnx.helper.make_tensor('y_zero_point', TensorProto.INT8, [len(y_zero_point)], y_zero_point)
        ]
    )
    original_model = onnx.helper.make_model(graph, producer_name="onnx-qlinearconv-flatten")
    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.int8),
        1: np.arange(np.prod(w_shape)).reshape(w_shape).astype(np.int8)
    }
    tflite_executor, _ = executors.convert_run_compare(original_model, input_data)
    output_quant_params = tflite_executor.get_output_details(0)['quantization_parameters']
    assert output_quant_params['scales'] == [1.0]
    assert output_quant_params['zero_points'] == [0]


@pytest.mark.parametrize(
    "axis, output_shape",
    [
        pytest.param(0, (1, 2940), id="axis=0"),
        pytest.param(1, (5, 588), id="axis=1"),
        pytest.param(2, (15, 196), id="axis=2"),
        pytest.param(3, (210, 14), id="axis=3"),
        pytest.param(4, (2940, 1), id="axis=4"),
        pytest.param(-1, (210, 14), id="axis=-1"),
        pytest.param(-2, (15, 196), id="axis=-2"),
        pytest.param(-3, (5, 588), id="axis=-3"),
        pytest.param(-4, (1, 2940), id="axis=-4"),
    ])
def test_convert_flatten_after_conv(axis: int, output_shape: Tuple[int]):
    """ Ensure correct transpose introduction for formatted input tensors """
    node_conv = onnx.helper.make_node(
        'Conv',
        inputs=['x', 'W'], outputs=['y'],
        kernel_shape=[3, 3], pads=[0, 0, 0, 0],
    )
    node_flatten = onnx.helper.make_node(
        'Flatten', inputs=['y'], outputs=['z'], axis=axis
    )
    x_shape = (5, 3, 16, 16)
    w_shape = (3, 3, 3, 3)

    graph = onnx.helper.make_graph(
        [node_conv, node_flatten],
        'conv+flatten',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("W", TensorProto.FLOAT, w_shape)
        ],
        [
            onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())
        ]
    )
    original_model = onnx.helper.make_model(graph, producer_name="onnx-conv-flatten")
    input_data = {
        0: np.arange(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.arange(np.prod(w_shape)).reshape(w_shape).astype(np.float32)
    }

    executors.convert_run_compare(original_model, input_data)


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(5, id="axis=5"),
        pytest.param(-5, id="axis=-5"),
    ])
def test_convert_flatten_invalid_axis(axis: int):
    """ Ensure proper exception raised when invalid axis is passed """
    node_flatten = onnx.helper.make_node(
        'Flatten', inputs=['x'], outputs=['y'], axis=axis
    )
    x_shape = (1, 3, 16, 16)

    graph = onnx.helper.make_graph(
        [node_flatten],
        'conv',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ())],
    )
    original_model = onnx.helper.make_model(graph, producer_name="onnx-flatten")
    onnx.checker.check_model(original_model)
    with pytest.raises(logger.Error) as e:
        bytes(convert.convert_model(original_model))
    assert e.value.error_code == logger.Code.SHAPE_INFERENCE_ERROR


@pytest.mark.parametrize(
    "axis, input_shape",
    [
        pytest.param(1, (10,), id="axis=1 rank=1"),
        pytest.param(0, (10,), id="axis=0 rank=1"),
        pytest.param(-1, (10,), id="axis=-1 rank=1"),

        pytest.param(1, (2, 5, 10), id="axis=1 rank=3"),
        pytest.param(2, (2, 5, 10), id="axis=2 rank=3"),
        pytest.param(3, (2, 5, 10), id="axis=3 rank=3"),
        pytest.param(-2, (2, 5, 10), id="axis=-2 rank=3"),

        pytest.param(1, (2, 5, 10, 15, 20), id="axis=1 rank=5"),
        pytest.param(3, (2, 5, 10, 15, 20), id="axis=3 rank=5"),
        pytest.param(5, (2, 5, 10, 15, 20), id="axis=5 rank=5"),
        pytest.param(-3, (2, 5, 10, 15, 20), id="axis=-3 rank=5"),

        pytest.param(1, (2, 3, 4, 2, 3, 4, 2, 3, 4), id="axis=1 rank=9"),
        pytest.param(5, (2, 3, 4, 2, 3, 4, 2, 3, 4), id="axis=5 rank=9"),
        pytest.param(9, (2, 3, 4, 2, 3, 4, 2, 3, 4), id="axis=9 rank=9"),
        pytest.param(-5, (2, 3, 4, 2, 3, 4, 2, 3, 4), id="axis=-5 rank=9"),
    ])
def test_convert_flatten_multi_ranks(axis: int, input_shape: Tuple[int]):
    """ Ensure proper flattening with different ranks """
    node_flatten = onnx.helper.make_node(
        'Flatten', inputs=['x'], outputs=['y'], axis=axis
    )

    graph = onnx.helper.make_graph(
        [node_flatten],
        'conv',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ())],
    )
    original_model = onnx.helper.make_model(graph, producer_name="onnx-flatten")
    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    }

    executors.convert_run_compare(original_model, input_data)


@pytest.mark.parametrize("axis", [0, 1, 2, 3, 4, -1], ids=(lambda x: f"axis={x}"))
def test_convert_flatten_dynamic_shape(axis: int):
    node_flatten = onnx.helper.make_node(
        'Flatten', inputs=['x'], outputs=['y'], axis=axis
    )

    graph = onnx.helper.make_graph(
        [node_flatten],
        'conv',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, ("batch_size", 2, 3, 5))],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    original_model = onnx.helper.make_model(graph, producer_name="onnx-flatten")

    original_model = ModelShapeInference.infer_shapes(original_model)
    onnx.checker.check_model(original_model)

    defined_shape = (1, 2, 3, 5)
    input_data = np.arange(np.absolute(np.prod(defined_shape))).reshape(defined_shape).astype(np.float32)

    # Don't verify output shapes, because the symbolic dimension will not match.
    executors.convert_run_compare(original_model, input_data, verify_output_shape=False)


@pytest.mark.parametrize(
    "type_",
    [
        TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE,
        TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
        TensorProto.UINT8, TensorProto.UINT32, TensorProto.UINT64,
        TensorProto.BOOL, TensorProto.STRING
    ],
    ids=name_for_onnx_type
)
def test_convert_flatten__types(type_: TensorProto.DataType):
    shape = [2, 4, 6]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Flatten', ['x'], ['y'])],
        'Flatten test',
        [onnx.helper.make_tensor_value_info("x", type_, shape)],
        [onnx.helper.make_tensor_value_info("y", type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(to_numpy_type(type_))
    executors.convert_run_compare(onnx_model, input_data)
