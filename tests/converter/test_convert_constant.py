#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.converter.convert import convert_model
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "value_type, value",
    [
        pytest.param(TensorProto.INT64, {"value_int": 42}, id="value_int: positive"),
        pytest.param(TensorProto.INT64, {"value_int": 0}, id="value_int: 0"),
        pytest.param(TensorProto.INT64, {"value_int": -23}, id="value_int: negative"),

        pytest.param(TensorProto.FLOAT, {"value_float": 13.37}, id="value_float: positive"),
        pytest.param(TensorProto.FLOAT, {"value_float": 0.0}, id="value_float: 0.0"),
        pytest.param(TensorProto.FLOAT, {"value_float": -3.14159}, id="value_float: negative"),

        pytest.param(TensorProto.INT64, {"value_ints": [1, 2, 3, 4]}, id="value_ints: positive"),
        pytest.param(TensorProto.INT64, {"value_ints": [-5, -9, -121, -12, -1]}, id="value_ints: negative"),
        pytest.param(TensorProto.INT64, {"value_ints": [13, 0, -2, 24, -90, 9]}, id="value_ints: mixed"),

        pytest.param(TensorProto.FLOAT, {"value_floats": [1.2, 2.3, 3.4, 4.5]}, id="value_floats: positive"),
        pytest.param(TensorProto.FLOAT, {"value_floats": [-5.0, -9.2, -1.5, -12.2, -1.3]}, id="value_floats: negative"),
        pytest.param(TensorProto.FLOAT, {"value_floats": [13.37, .0, -0.2, 4.23, -9.1, 0.9]}, id="value_floats: mixed"),
    ])
def test_convert_constant_with_specific_types(value_type: TensorProto.DataType, value: dict):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["x"], **value),
         onnx.helper.make_node("Reshape", ["x", "new_shape"], ["o"])],
        'Constant test',
        [],
        [onnx.helper.make_tensor_value_info("o", value_type, [])],
        [onnx.helper.make_tensor("new_shape", TensorProto.INT64, [1], [np.asarray(value.popitem()[1]).size])],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})


@pytest.mark.parametrize(
    "value_type",
    [TensorProto.FLOAT,
     TensorProto.INT64, TensorProto.INT32, TensorProto.INT16, TensorProto.INT8,
     TensorProto.UINT8],
    ids=name_for_onnx_type
)
@pytest.mark.parametrize(
    "shape",
    [[1], [10], [4, 5], [2, 3, 4, 5]],
    ids=lambda x: f"{len(x)}D"
)
def test_convert_constant__value_attribute__converted_to_transpose(value_type: TensorProto.DataType, shape: list[int],
                                                                   intermediate_tflite_model_provider):
    np.random.seed(42)
    np_value = np.random.random(np.prod(shape)).astype(to_numpy_type(value_type))
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["o"], value=onnx.helper.make_tensor('', value_type, shape, np_value))],
        'Constant test',
        [],
        [onnx.helper.make_tensor_value_info("o", value_type, [])],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.TRANSPOSE])


@pytest.mark.parametrize(
    "value_type",
    [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE,
     TensorProto.INT64, TensorProto.INT32, TensorProto.INT16, TensorProto.INT8,
     TensorProto.UINT64, TensorProto.UINT32, TensorProto.UINT8,
     TensorProto.BOOL
     ],
    ids=name_for_onnx_type
)
@pytest.mark.parametrize(
    "shape",
    [[1], [10], [4, 5], [2, 3, 4, 5]],
    ids=lambda x: f"{len(x)}D"
)
def test_convert_constant__value_attribute__skipped(value_type: TensorProto.DataType, shape: list[int],
                                                    intermediate_tflite_model_provider):
    np.random.seed(42)
    np_value = np.random.random(np.prod(shape)).astype(to_numpy_type(value_type))
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Constant", [], ["x"],
                                  value=onnx.helper.make_tensor('', value_type, shape, np_value)),
            onnx.helper.make_node("Reshape", ["x", "new_shape"], ["o"])
        ],
        'Constant test',
        [],
        [onnx.helper.make_tensor_value_info("o", value_type, [])],
        [onnx.helper.make_tensor("new_shape", TensorProto.INT64, [len(shape)], shape)]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.RESHAPE])


def test_convert_constant_with_complex_type():
    value = [np.complex64(1.0)]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["x"],
                               value=onnx.helper.make_tensor('', TensorProto.COMPLEX64, [1], value)),
         onnx.helper.make_node("Reshape", ["x", "new_shape"], ["o"])],
        'Constant test',
        [],
        [onnx.helper.make_tensor_value_info("o", TensorProto.COMPLEX64, [])],
        [onnx.helper.make_tensor("new_shape", TensorProto.INT64, [1], [1])]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert_model(onnx_model)
    assert logger.conversion_log.get_logs()["shape_inference"][0]["error_code"] == logger.Code.SHAPE_INFERENCE_ERROR
