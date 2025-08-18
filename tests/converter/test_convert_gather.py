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
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "input_shape, indices",
    [
        pytest.param([12, 6, 8, 10], [0, -2], id="indices = [0, -2]"),
        pytest.param([12, 6, 8, 10], [-5, 2, -7, 10, -11], id="indices = [-5, 2, -7, 10, -11]"),
        pytest.param([12, 6, 8, 10], [-1, -3, -5, -7, -9, -2, -4, -6, -8],
                     id="indices = [-1, -3, -5, -7, -9, -2, -4, -6, -8]"),

    ])
def test_convert_gather_with_negative_indices(input_shape: List[int], indices: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gather", ["input", "indices"], ["output"])],
        'Gather test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("indices", TensorProto.INT64, [len(indices)], indices)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, indices, axis",
    [
        pytest.param([4, 6, 8, 10], [1], 0, id="axis = 0"),
        pytest.param([4, 6, 8, 10], [1, 4, -1], 1, id="axis = 1"),
        pytest.param([4, 6, 8, 10], [0, 7], 2, id="axis = 2"),
        pytest.param([4, 6, 8, 10], [0, -7, 3, 7], 3, id="axis = 3"),
        pytest.param([4, 6, 8, 10], [-3, 8, 9], -1, id="axis = -1"),
        pytest.param([4, 6, 8, 10], [0, 3, 1, -2], -3, id="axis = -3"),

    ])
def test_convert_gather(input_shape: List[int], indices: List[int], axis: int):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gather", ["input", "indices"], ["output"], axis=axis)],
        'Gather test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("indices", TensorProto.INT64, [len(indices)], indices)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, indices, axis",
    [
        pytest.param([4, 6, 8, 10], [-3], 0, id="axis = 0"),
        pytest.param([4, 6, 8, 10], [1, 4, 5], 1, id="axis = 1"),
        pytest.param([4, 6, 8, 10], [0, 7], 2, id="axis = 2"),
        pytest.param([4, 6, 8, 10], [0, -8, 3, -3], 3, id="axis = 3"),
        pytest.param([4, 6, 8, 10], [6, -10, 9], -1, id="axis = -1"),
        pytest.param([4, 6, 8, 10], [0, -3, 2, 3], -3, id="axis = -3"),
        pytest.param([4, 6, 8, 10], [0, -4, 2, 3], -4, id="axis = -4"),

    ])
def test_convert_gather_with_channels_first_tensors(input_shape: List[int], indices: List[int], axis: int):
    kernel_shape = [1] * (len(input_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MaxPool", ["input"], ["max_pool_out"], kernel_shape=kernel_shape),
            onnx.helper.make_node("Gather", ["max_pool_out", "indices"], ["output"], axis=axis),
        ],
        'Gather test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("indices", TensorProto.INT64, [len(indices)], indices)],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, indices, axis",
    [
        pytest.param([4, 6, 8, 10], [1, 4, 5], 1, id="axis = 1"),
        pytest.param([4, 6, 8, 10], [9, 0, 4, 7, 2, 5], -1, id="axis = -1"),
        pytest.param([4, 6, 8, 10], [0], 0, id="axis = 0"),

    ])
def test_convert_gather__dynamic_indices(input_shape: List[int], indices: List[int], axis: int):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gather", ["input", "indices"], ["output"], axis=axis)],
        'Gather test',
        [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("indices", TensorProto.INT64, [len(indices)]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.array(indices, np.int64)
    }

    executors.convert_run_compare(onnx_model, input_data,
                                  conversion_config=ConversionConfig({'non_negative_indices': True}))


@pytest.mark.parametrize(
    "input_shape, indices, axis",
    [
        pytest.param([4, 6, 8, 10], [1, 4, 5], 1, id="axis = 1"),
        pytest.param([4, 6, 8, 10], [9, 0, 4, 7, 2, 5], -1, id="axis = -1"),
        pytest.param([4, 6, 8, 10], [0], 0, id="axis = 0"),

    ])
def test_convert_gather__dynamic_indices_and_channels_first(input_shape: List[int], indices: List[int], axis: int):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MaxPool", ["input"], ["x1"], kernel_shape=[1, 1]),
            onnx.helper.make_node("Gather", ["x1", "indices"], ["output"], axis=axis)
        ],
        'Gather test',
        [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("indices", TensorProto.INT64, [len(indices)]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.array(indices, np.int64)
    }

    executors.convert_run_compare(onnx_model, input_data,
                                  conversion_config=ConversionConfig({'non_negative_indices': True}))


def test_convert_gather__dynamic_indices_without_required_flag():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gather", ["input", "indices"], ["output"])],
        'Gather test',
        [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4, 6, 8]),
            onnx.helper.make_tensor_value_info("indices", TensorProto.INT64, [2]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED

    # Make sure the hint to use the flag was printed.
    assert '--guarantee-non-negative-indices' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize(
    "input_shape, indices, axis",
    [
        pytest.param([4, 6, 8, 10], [6], 1, id="overflow"),
        pytest.param([4, 6, 8, 10], [-7], 1, id="underflow"),

    ])
def test_convert_gather_with_invalid_indices(input_shape: List[int], indices: List[int], axis: int):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gather", ["input", "indices"], ["output"], axis=axis)],
        'Gather test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("indices", TensorProto.INT64, [len(indices)], indices)]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model)

    assert e.type == logger.Error
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL


@pytest.mark.parametrize(
    "input_shape, indices, axis",
    [
        pytest.param([3, 2], [[0, 1], [1, 2]], 0, id="simple 2D indices"),
        pytest.param([2, 4, 6, 8], [[0, 1, 3, 4], [1, 2, 4, 5]], -2, id="2D indices"),
        pytest.param([2, 4, 6, 8],
                     [[[0, 1, 2, 4], [1, 2, -5, 5], [-6, 3, 4, 6]], [[3, 4, 5, 7], [4, 5, 6, -8], [5, 6, 7, -7]]],
                     3, id="3D indices"),
        pytest.param([2, 4, 10], [[[[-10, 1, 4, -4]], [[8, 2, 8, 2]], [[0, -9, 5, 8]]],
                                  [[[6, 1, -2, 4]], [[1, 4, 9, 2]], [[0, 1, -6, -3]]]],
                     -1, id="4D indices"),

    ])
def test_convert_gather_with_multi_dimensional_indices(input_shape, indices, axis):
    indices = np.asarray(indices)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gather", ["input", "indices"], ["output"], axis=axis)],
        'Gather test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("indices", TensorProto.INT64, indices.shape, indices)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, output_shape, indices, axis",
    [
        # The output must be 4D for the MaxPool
        pytest.param([4, 6, 8], [4, 2, 4, 8], [[0, 1, 3, 4], [1, 2, 4, 5]], -2, id="2D indices"),
        pytest.param([10, 20], [10, 2, 3, 4],
                     [[[0, 1, 2, 4], [10, 2, -5, 5], [-6, 11, 4, 17]], [[3, 4, 5, 7], [4, 5, 6, -8], [5, 6, 7, -7]]],
                     -1, id="3D indices"),
        pytest.param([42], [2, 4, 2, 3],
                     [[[[15, -25, 40], [-12, 21, 8]], [[-2, 3, -15], [-20, -30, -40]], [[10, 20, 40], [23, 41, 7]],
                       [[1, 2, 3], [0, 0, 0]]],
                      [[[-1, -1, -1], [5, 8, 12]], [[19, 2, 10], [24, 5, 41]], [[0, -42, 0], [0, 41, 38]],
                       [[39, 38, 39], [5, 4, -14]]]],
                     0, id="4D indices"),

    ])
def test_convert_gather_with_multi_dimensional_indices_and_channels_first_output(input_shape, output_shape, indices,
                                                                                 axis):
    indices = np.asarray(indices)
    kernel_shape = [1] * (len(output_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Gather", ["input", "indices"], ["gather_out"], axis=axis),
            onnx.helper.make_node("MaxPool", ["gather_out"], ["output"], kernel_shape=kernel_shape),
        ],
        'Gather test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("indices", TensorProto.INT64, indices.shape, indices)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, indices, axis",
    [
        # The input must be 4D for the MaxPool
        pytest.param([4, 6, 8, 10], [[0, 1, 3], [2, 4, 5]], -3, id="2D indices"),
        pytest.param([4, 6, 8, 10], [[[0, 1, -4, 9], [2, -5, -7, -10], [0, 5, 1, 6]],
                                     [[2, 4, 6, 8], [-1, -5, -3, -9], [0, 9, -10, -1]]],
                     3, id="3D indices"),

        pytest.param([4, 6, 8, 10], [[[[0, 1, -4, 7], [2, -5, -7, -8], [0, 5, 1, 6]]],
                                     [[[2, 4, 6, 7], [-1, -5, -3, -7], [0, 7, -8, -1]]]],
                     -2, id="4D indices"),

    ])
def test_convert_gather_with_multi_dimensional_indices_and_channels_first_input(input_shape, indices, axis):
    indices = np.asarray(indices)
    kernel_shape = [1] * (len(input_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MaxPool", ["input"], ["max_pool_out"], kernel_shape=kernel_shape),
            onnx.helper.make_node("Gather", ["max_pool_out", "indices"], ["output"], axis=axis),
        ],
        'Gather test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("indices", TensorProto.INT64, indices.shape, indices)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, output_shape, indices, axis",
    [
        # The input must be 3D and output 4D
        pytest.param([6, 8, 10], [2, 3, 8, 10], [[0, 1, 3], [2, 4, 5]], -3, id="2D positive indices"),
        pytest.param([6, 8, 10], [6, 2, 4, 10], [[0, 1, -2, -5], [2, -8, 7, 1]], 1, id="2D negative indices"),
        pytest.param([6, 8, 10], [6, 8, 4, 3], [[9, -10, 7], [-5, 0, -9], [9, 1, -2], [2, -8, 7]], -1,
                     id="2D negative indices, last dimension"),

    ])
def test_convert_gather_with_multi_dimensional_indices_and_channels_first_tensors(input_shape, output_shape, indices,
                                                                                  axis):
    # Both the input and output tensors are channels first

    indices = np.asarray(indices)
    kernel_shape_1 = [1] * (len(input_shape) - 2)
    kernel_shape_2 = [1] * (len(output_shape) - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MaxPool", ["input"], ["max_pool_1_out"], kernel_shape=kernel_shape_1,
                                  strides=kernel_shape_1),
            onnx.helper.make_node("Gather", ["max_pool_1_out", "indices"], ["gather_out"], axis=axis),
            onnx.helper.make_node("MaxPool", ["gather_out"], ["output"], kernel_shape=kernel_shape_2),
        ],
        'Gather test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("indices", TensorProto.INT64, indices.shape, indices)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_gather_quantized():
    input_shape = [4, 6, 8, 10]
    indices = [1]
    axis = 0
    scale = [1.0]
    zero_point = [0]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["y"], axis=1),
            onnx.helper.make_node("Gather", ["y", "indices"], ["output"], axis=axis)
        ],
        'Gather test quantized',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor("indices", TensorProto.INT64, [len(indices)], indices),
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
    "type_",
    [
        TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
        TensorProto.UINT8,
        TensorProto.FLOAT,
        TensorProto.BOOL, TensorProto.STRING
    ],
    ids=name_for_onnx_type
)
def test_convert_gather__types(type_: TensorProto.DataType):
    shape = [42]
    indices = [2, 6, 4, 2, 1, 40, -15]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gather", ["input", "indices"], ["output"])],
        'Gather test',
        [onnx.helper.make_tensor_value_info("input", type_, shape)],
        [onnx.helper.make_tensor_value_info("output", type_, ())],
        [onnx.helper.make_tensor("indices", TensorProto.INT64, [len(indices)], indices)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(to_numpy_type(type_))

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_gather__invalid_type():
    type_ = TensorProto.DOUBLE

    shape = [42]
    indices = [2, 6, 4, 2, 1, 40, -15]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gather", ["input", "indices"], ["output"])],
        'Gather test',
        [onnx.helper.make_tensor_value_info("input", type_, shape)],
        [onnx.helper.make_tensor_value_info("output", type_, ())],
        [onnx.helper.make_tensor("indices", TensorProto.INT64, [len(indices)], indices)]
    )

    onnx_model = onnx.helper.make_model(graph)
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)
