#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx.helper
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta import types
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests.executors import convert_run_compare


@pytest.mark.parametrize("input_shape", [[5], [5, 5], [5, 5, 5], [5, 5, 5, 5]], ids=lambda x: f"rank={len(x)}D")
def test_topk__input_rank(input_shape):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("TopK", ["x", "k"], ["y", "z"])],
        "TopK test",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape), ],
        [
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ()),
            onnx.helper.make_tensor_value_info("z", TensorProto.INT64, ()),
        ],
        [onnx.helper.make_tensor("k", TensorProto.INT64, [1], [3]), ],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("axis", range(-4, 4))
def test_topk__input_axis__4d(axis):
    input_shape = [5, 3, 4, 5]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("TopK", ["x", "k"], ["y", "z"], axis=axis)],
        "TopK test",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape), ],
        [
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ()),
            onnx.helper.make_tensor_value_info("z", TensorProto.INT64, ()),
        ],
        [onnx.helper.make_tensor("k", TensorProto.INT64, [1], [3]), ],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("axis", range(-3, 3))
def test_topk__input_axis__3d(axis):
    input_shape = [4, 5, 6]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("TopK", ["x", "k"], ["y", "z"], axis=axis)],
        "TopK test",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape), ],
        [
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ()),
            onnx.helper.make_tensor_value_info("z", TensorProto.INT64, ()),
        ],
        [onnx.helper.make_tensor("k", TensorProto.INT64, [1], [3]), ],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("data_type", [TensorProto.INT32, TensorProto.INT64, TensorProto.FLOAT], ids=name_for_onnx_type)
def test_convert_topk__input_type(data_type: TensorProto.DataType):
    input_shape = [2, 3, 4]
    np_type = types.to_numpy_type(data_type)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("TopK", ["x", "k"], ["y", "z"])],
        "TopK test",
        [onnx.helper.make_tensor_value_info("x", data_type, input_shape), ],
        [
            onnx.helper.make_tensor_value_info("y", data_type, ()),
            onnx.helper.make_tensor_value_info("z", TensorProto.INT64, ()),
        ],
        [onnx.helper.make_tensor("k", TensorProto.INT64, [1], [3]), ],
    )
    onnx_model = onnx.helper.make_model(graph)

    x_data = (np.arange(np.prod(input_shape)) - 100).reshape(input_shape).astype(np_type)

    convert_run_compare(onnx_model, x_data)


def test_convert_topk__unsupported_type():
    data_type = TensorProto.INT8
    input_shape = [2, 3, 4]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("TopK", ["x", "k"], ["y", "z"])],
        "TopK test",
        [onnx.helper.make_tensor_value_info("x", data_type, input_shape), ],
        [
            onnx.helper.make_tensor_value_info("y", data_type, ()),
            onnx.helper.make_tensor_value_info("z", TensorProto.INT64, ()),
        ],
        [onnx.helper.make_tensor("k", TensorProto.INT64, [1], [3]), ],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL
    assert 'INT8' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("axis", list(range(-3, 3)) + [None], ids=lambda x: f"axis={x}")
def test_topk__channels_first(axis):
    input_shape = [3, 10, 8, 8]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MaxPool", ["input"], ["x"], kernel_shape=[1, 1]),
            onnx.helper.make_node("TopK", ["x", "k"], ["y", "z"], axis=axis)
        ],
        "TopK test",
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape), ],
        [
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ()),
            onnx.helper.make_tensor_value_info("z", TensorProto.INT64, ()),
        ],
        [onnx.helper.make_tensor("k", TensorProto.INT64, (1,), [3]), ],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(onnx_model, input_data)


def test_convert_topk__not_sorted():
    input_shape = [2, 3, 4]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("TopK", ["x", "k"], ["y", "z"], sorted=0)],
        "TopK test",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape), ],
        [
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ()),
            onnx.helper.make_tensor_value_info("z", TensorProto.INT64, ()),
        ],
        [onnx.helper.make_tensor("k", TensorProto.INT64, [1], [3]), ],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert "'sorted=0' is not possible" in logger.conversion_log.get_node_error_message(0)


def test_convert_topk__not_largest():
    input_shape = [2, 3, 4]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("TopK", ["x", "k"], ["y", "z"], largest=0)],
        "TopK test",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape), ],
        [
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ()),
            onnx.helper.make_tensor_value_info("z", TensorProto.INT64, ()),
        ],
        [onnx.helper.make_tensor("k", TensorProto.INT64, [1], [3]), ],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert "'largest=0' and 'sorted=1' is not yet implemented" in logger.conversion_log.get_node_error_message(0)


def test_convert_topk__int64_consumer():
    input_shape = [1, 3, 10]
    data_type = TensorProto.INT64
    np_type = types.to_numpy_type(data_type)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("TopK", ["x", "k"], ["y", "o"]),
            onnx.helper.make_node("Expand", ["o", "shape"], ["z"])
        ],
        "TopK test",
        [onnx.helper.make_tensor_value_info("x", data_type, input_shape), ],
        [
            onnx.helper.make_tensor_value_info("y", data_type, ()),
            onnx.helper.make_tensor_value_info("z", TensorProto.INT64, ()),
        ],
        [
            onnx.helper.make_tensor("shape", TensorProto.INT64, [3], [2, 1, 1]),
            onnx.helper.make_tensor("k", TensorProto.INT64, [1], [8]),
        ],
    )
    onnx_model = onnx.helper.make_model(graph)

    x_data = (np.arange(np.prod(input_shape)) - 100).reshape(input_shape).astype(np_type)

    convert_run_compare(onnx_model, x_data)