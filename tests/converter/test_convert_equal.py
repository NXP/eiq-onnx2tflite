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
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph
from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type, name_for_onnx_type
from tests import executors
from tests.executors import OnnxExecutor


@pytest.mark.parametrize(
    "data_type",
    [
        TensorProto.BOOL,
        TensorProto.FLOAT,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.STRING
    ], ids=name_for_onnx_type)
def test_convert_equal_with_various_types(data_type: TensorProto):
    main_shape = [2, 4, 6, 8]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Equal", ["input_1", "input_2"], ["output"]),
        ],
        'Equal test',
        [
            onnx.helper.make_tensor_value_info("input_1", data_type, main_shape),
            onnx.helper.make_tensor_value_info("input_2", data_type, main_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.BOOL, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import[0].version = 17

    numpy_type = to_numpy_type(data_type)
    np.random.seed(1)
    input_data = {
        0: np.random.randint(0, 5, main_shape).astype(numpy_type),
        1: np.random.randint(0, 5, main_shape).astype(numpy_type),
    }

    executors.convert_run_compare(onnx_model, input_data, reference_onnx_evaluation=True)


@pytest.mark.parametrize(
    "main_shape, data_type, exception",
    [
        pytest.param([2, 4, 6, 8], TensorProto.STRING, InvalidGraph, id="STRING"),
    ])
def test_convert_equal_with_types_unsupported_by_onnx(main_shape: List[int], data_type: TensorProto, exception):
    # ONNXRT: Limitation of ONNX Runtime
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Equal", ["input_1", "input_2"], ["output"]),
        ],
        'Equal test',
        [
            onnx.helper.make_tensor_value_info("input_1", data_type, main_shape),
            onnx.helper.make_tensor_value_info("input_2", data_type, main_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.BOOL, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import[0].version = 17
    onnx.checker.check_model(onnx_model)

    with pytest.raises(exception) as _:
        OnnxExecutor(onnx_model.SerializeToString())


@pytest.mark.parametrize(
    "main_shape, data_type",
    [
        pytest.param([2, 4, 6, 8], TensorProto.INT8, id="INT8"),
        pytest.param([2, 4, 6, 8], TensorProto.INT16, id="INT16"),
        pytest.param([2, 4, 6, 8], TensorProto.UINT8, id="UINT8"),
    ])
def test_convert_equal_with_types_unsupported_by_onnx_try_convert(main_shape: List[int], data_type: TensorProto):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Equal", ["input_1", "input_2"], ["output"]),
        ],
        'Equal test',
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
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize(
    "main_shape, data_type",
    [
        pytest.param([2, 4, 6, 8], TensorProto.DOUBLE, id="DOUBLE"),
        pytest.param([2, 4, 6, 8], TensorProto.FLOAT16, id="FLOAT16"),
    ])
def test_convert_equal_with_types_unsupported_by_tflite(main_shape: List[int], data_type: TensorProto):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Equal", ["input_1", "input_2"], ["output"]),
        ],
        'Equal test',
        [
            onnx.helper.make_tensor_value_info("input_1", data_type, main_shape),
            onnx.helper.make_tensor_value_info("input_2", data_type, main_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.BOOL, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import[0].version = 17

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model)
    assert e.value.error_code == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize(
    "shape_1, shape_2, output_shape",
    [
        pytest.param([6, 8], [2, 4, 1, 1], [2, 4, 6, 8], id=""),
        pytest.param([2, 4, 1, 1], [8], [2, 4, 1, 8], id=""),
        pytest.param([2, 4, 1, 1], [1, 1, 6, 8], [2, 4, 1, 8], id=""),
    ])
def test_convert_equal_with_broadcasting(shape_1: List[int], shape_2: List[int], output_shape: List[int]):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Equal", ["input_1", "input_2"], ["output"]),
        ],
        'Equal test',
        [
            onnx.helper.make_tensor_value_info("input_1", TensorProto.FLOAT, shape_1),
            onnx.helper.make_tensor_value_info("input_2", TensorProto.FLOAT, shape_2),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.BOOL, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import[0].version = 17

    np.random.seed(1)
    input_data = {
        0: np.random.randint(0, 5, shape_1).astype(np.float32),
        1: np.random.randint(0, 5, shape_2).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "shape_1, shape_2",
    [
        pytest.param([4, 1, 8], [2, 4, 6, 8]),
        pytest.param([6, 8], [2, 4, 6, 8]),
        pytest.param([4, 6, 1], [2, 1, 1, 8]),
        pytest.param([6, 1], [2, 4, 6, 8]),
        pytest.param([1, 6, 1], [2, 4, 1, 8]),
    ])
def test_convert_equal_with_channels_first_broadcasting(shape_1: List[int], shape_2: List[int]):
    kernel_shape = [1] * (len(shape_2) - 2)
    np.random.seed(1)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MaxPool", ["input_2"], ["max_pool_out"], kernel_shape=kernel_shape),
            onnx.helper.make_node("Equal", ["input_1", "max_pool_out"], ["output"]),
        ],
        'Equal test',
        [
            onnx.helper.make_tensor_value_info("input_2", TensorProto.FLOAT, shape_2),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.BOOL, ())],
        [onnx.helper.make_tensor("input_1", TensorProto.FLOAT, shape_1,
                                 np.random.randint(0, 3, shape_1).astype(np.float32))]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import[0].version = 17

    input_data = np.random.randint(0, 3, shape_2).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)
