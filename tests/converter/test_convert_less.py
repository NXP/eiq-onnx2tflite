#
# Copyright 2024 NXP
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
from onnx2tflite.src.onnx_parser.meta import types
from tests import executors


@pytest.mark.parametrize(
    "input_type",
    [
        pytest.param(TensorProto.FLOAT, id="float32"),
        pytest.param(TensorProto.INT32, id="int32"),
        pytest.param(TensorProto.INT64, id="int64"),
        # Other types are not supported.
    ])
def test_convert_less(input_type: TensorProto.DataType):
    np_type = types.to_numpy_type(input_type)
    x_shape = [100]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Less', ['x', 'y'], ['o'])],
        'Less test',
        [
            onnx.helper.make_tensor_value_info('x', input_type, x_shape),
            onnx.helper.make_tensor_value_info('y', input_type, x_shape)
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.BOOL, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    np.random.seed(42)

    input_data = {
        0: (np.random.random(100) * 100).astype(np_type),
        1: (np.random.random(100) * 100).astype(np_type)
    }

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_less_with_unsupported_type():
    input_shape = [100]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Less', ['x', 'y'], ['o'])],
        'Less test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT16, input_shape),
            onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT16, input_shape)
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.BOOL, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize(
    "main_shape, data_type",
    [
        pytest.param([2, 4, 6, 8], TensorProto.STRING, id="STRING"),
        pytest.param([2, 4, 6, 8], TensorProto.BOOL, id="BOOL"),
    ])
def test_convert_less_with_types_unsupported_by_onnx_try_convert(main_shape: List[int], data_type: TensorProto):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Less", ["input_1", "input_2"], ["output"]),
        ],
        'Less test',
        [
            onnx.helper.make_tensor_value_info("input_1", data_type, main_shape),
            onnx.helper.make_tensor_value_info("input_2", data_type, main_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.BOOL, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import[0].version = 17
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL


def test_convert_less_with_different_input_types():
    input_shape = [100]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Less', ['x', 'y'], ['o'])],
        'Less test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.INT32, input_shape),
            onnx.helper.make_tensor_value_info('y', TensorProto.INT64, input_shape)
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.BOOL, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        ([100], [1]),
        ([5, 10, 20], [20]),
        ([5, 10, 20], [10, 1]),
        ([5, 1, 20], [2, 1, 10, 1]),
    ])
def test_convert_less_with_shape_broadcasting(x_shape, y_shape):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Less', ['x', 'y'], ['o'])],
        'Less test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, y_shape)
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.BOOL, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    np.random.seed(42)

    input_data = {
        0: np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.random.random(np.prod(y_shape)).reshape(y_shape).astype(np.float32)
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        ([5, 10, 20], [20]),
        ([5, 10, 20], [10, 1]),
        ([5, 1, 20], [2, 1, 10, 1]),
    ])
def test_convert_less_with_channels_first_shape_broadcasting(x_shape, y_shape):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['o1'], kernel_shape=[1]),
            onnx.helper.make_node('Less', ['o1', 'y'], ['o'])
        ],
        'Less test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, y_shape)
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.BOOL, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    np.random.seed(42)

    input_data = {
        0: np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.random.random(np.prod(y_shape)).reshape(y_shape).astype(np.float32)
    }

    executors.convert_run_compare(onnx_model, input_data)
