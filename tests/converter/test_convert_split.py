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
from onnx2tflite.src.conversion_config import ConversionConfig, SkipShapeInferenceConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "input_shape, axis",
    [
        pytest.param([5, 10], 0, id="odd dimension"),
        pytest.param([5, 10], 1, id="even dimension"),

    ])
def test_convert_split_with_default_split(input_shape: List[int], axis: int):
    output_names = [f"o{i}" for i in range(input_shape[axis])]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Split", ["x"], output_names, axis=axis)],
        'Split test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, ()) for name in output_names],
    )

    # Latest version requires the 'split' or 'num_outputs' specified. Older versions don't.
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, axis",
    [
        pytest.param([5, 10], 0, id="odd dimension"),
        pytest.param([5, 10], 1, id="even dimension"),

    ])
def test_convert_split__omitted_split(input_shape: List[int], axis: int):
    # Use the name "" to represent omitted `split` input.

    output_names = [f"o{i}" for i in range(input_shape[axis])]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Split", ["x", ""], output_names, axis=axis)],
        'Split test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, ()) for name in output_names],
    )

    # Latest version requires the 'split' or 'num_outputs' specified. Older versions don't.
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_split_with_default_axis():
    input_shape = [14, 5]
    split = [1, 3, 3, 7]

    output_names = [f"o{i}" for i in range(len(split))]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Split", ["x", "split"], output_names)],
        'Split test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, ()) for name in output_names],
        [onnx.helper.make_tensor("split", TensorProto.INT64, [len(split)], split)]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_split_v2_with_attribute_split():
    input_shape = [14, 5]
    split = [1, 3, 3, 7]
    axis = -2

    output_names = [f"o{i}" for i in range(len(split))]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Split", ["x"], output_names, axis=axis, split=split)],
        'Split test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, ()) for name in output_names],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 2)])

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    config = ConversionConfig({"ignore_opset_version": True})
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


@pytest.mark.parametrize(
    "input_shape, split, axis",
    [
        pytest.param([2, 4, 6, 8], [1, 2, 3], 2, id="positive axis"),
        pytest.param([2, 4, 6, 8], [4, 1, 1, 2], -1, id="negative axis"),

    ])
def test_convert_split_with_default_split(input_shape: List[int], split: List[int], axis: int):
    output_names = [f"o{i}" for i in range(len(split))]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Split", ["x", "split"], output_names, axis=axis)],
        'Split test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, ()) for name in output_names],
        [onnx.helper.make_tensor("split", TensorProto.INT64, [len(split)], split)]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, split, axis",
    [
        pytest.param([2, 4, 6, 8], [1, 2, 3], 2, id="positive axis"),
        pytest.param([2, 4, 6, 8], [4, 1, 1, 2], -1, id="negative axis"),

    ])
def test_convert_split_with_dynamic_split(input_shape: List[int], split: List[int], axis: int):
    output_names = [f"o{i}" for i in range(len(split))]
    output_shapes = [input_shape.copy() for _ in range(len(split))]
    for shape, chunk in zip(output_shapes, split):
        shape[axis] = chunk

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Split", ["x", "split"], output_names, axis=axis)],
        'Split test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("split", TensorProto.INT64, [len(split)])
        ],
        [onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in
         zip(output_names, output_shapes)],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.asarray(split, np.int64),
    }

    # Need to turn off shape inference, because it exits if the 'split' is dynamic and output shapes cannot be inferred.
    executors.convert_run_compare(onnx_model, input_data, conversion_config=SkipShapeInferenceConfig())


@pytest.mark.parametrize(
    "input_shape, num_outputs",
    [
        pytest.param([110], 10, id="no reminder"),  # [11, 11, ..., 11, 11]

        pytest.param([119], 10, id="reminder 9"),  # [11, 11, ..., 11, 9]
        pytest.param([115], 10, id="reminder 5"),  # [11, 11, ..., 11, 5]
        pytest.param([111], 10, id="reminder 1"),  # [11, 11, ..., 11, 1]

    ])
def test_convert_split_with_num_outputs(input_shape: List[int], num_outputs: int):
    output_names = [f"o{i}" for i in range(num_outputs)]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Split", ["x"], output_names, axis=0, num_outputs=num_outputs)],
        'Split test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, ()) for name in output_names],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_split_with_invalid_num_outputs():
    input_shape = [3]
    num_outputs = 4
    output_names = [f"o{i}" for i in range(num_outputs)]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Split", ["x"], output_names, axis=0, num_outputs=num_outputs)],
        'Split test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, ()) for name in output_names],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR


@pytest.mark.parametrize(
    "type_",
    [
        TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
        TensorProto.FLOAT, TensorProto.UINT8
    ], ids=name_for_onnx_type)
def test_convert_split__types(type_: TensorProto.DataType):
    input_shape, axis = [5, 10], 0

    output_names = [f"o{i}" for i in range(input_shape[axis])]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Split", ["x"], output_names, axis=axis)],
        'Split test',
        [onnx.helper.make_tensor_value_info("x", type_, input_shape)],
        [onnx.helper.make_tensor_value_info(name, type_, ()) for name in output_names],
    )

    # Latest version requires the 'split' or 'num_outputs' specified. Older versions don't.
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(type_))

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_split__quantized(type_: TensorProto.DataType):
    input_shape, axis = [5, 10], 0

    output_names_1 = [f"o{i}" for i in range(input_shape[axis])]
    output_names_2 = [name + "_" for name in output_names_1]
    reshape_ops = [onnx.helper.make_node('Reshape', [inpt, 'flat_shape'], [outpt])
                   for inpt, outpt in zip(output_names_1, output_names_2)]

    np.random.seed(42)
    data = (np.random.random(input_shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Split', ['x1'], output_names_1, axis=axis, num_outputs=5),
        ] + reshape_ops,
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info(name, type_, ()) for name in output_names_2],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [2], [5, 2])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
