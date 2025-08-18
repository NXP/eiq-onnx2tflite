#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import Optional

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.converter.conversion import common
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize("end", list(range(-5, 5, 2)), ids=lambda x: f" end='{x}'")
@pytest.mark.parametrize("start", list(range(-6, 6, 2)), ids=lambda x: f" start='{x}' ")
@pytest.mark.parametrize("input_shape", [[5, 10, 15], [4, 8, 12, 16]], ids=lambda x: f"{len(x)}D input ")
def test_convert_shape(input_shape, start: int, end: int):
    rank = len(input_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Shape", ["input"], ["output"], start=start, end=end)],
        'shape test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT64, ())]
    )

    onnx_model = onnx.helper.make_model(graph)

    # Determine when we expect the conversion to throw an Error
    if start < 0:
        start += rank
    start = common.clamp(start, 0, rank)
    if end < 0:
        end += rank
    end = common.clamp(end, 0, rank)
    if end < start:
        # Expect error
        with pytest.raises(logger.Error):
            convert.convert_model(onnx_model)
        assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE

    else:
        input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
        executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("end", list(range(-6, 6, 2)), ids=lambda x: f" end='{x}'")
@pytest.mark.parametrize("start", list(range(-5, 5, 2)), ids=lambda x: f" start='{x}' ")
@pytest.mark.parametrize("input_shape", [[11, 3, 5], [1, 3, 3, 7]], ids=lambda x: f"{len(x)}D input ")
def test_convert_shape_with_channels_first_input(input_shape, start: int, end: int):
    rank = len(input_shape)
    kernel_shape = [1] * (rank - 2)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MaxPool", ["input"], ["mp_out"], kernel_shape=kernel_shape, strides=kernel_shape),
            onnx.helper.make_node("Shape", ["mp_out"], ["output"], start=start, end=end),
        ],
        'shape test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT64, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    # Determine when we expect the conversion to throw an Error
    if start < 0:
        start += rank
    start = common.clamp(start, 0, rank)
    if end < 0:
        end += rank
    end = common.clamp(end, 0, rank)
    if end < start:
        # Expect error
        with pytest.raises(logger.Error):
            convert.convert_model(onnx_model)
        assert logger.conversion_log.get_node_error_code(1) == logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE

    else:
        input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
        executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, start, end",
    [
        pytest.param([23], None, None, id="1D, implicit 'start' and 'end'"),
        pytest.param([5, 10], None, None, id="2D, implicit 'start' and 'end'"),
        pytest.param([5, 10, 4], None, None, id="3D, implicit 'start' and 'end'"),
        pytest.param([5, 10, 4, 8], None, None, id="4D, implicit 'start' and 'end'"),
        pytest.param([5, 3, 10, 4, 8], None, None, id="5D, implicit 'start' and 'end'"),
        pytest.param([2, 5, 3, 10, 4, 8], None, None, id="6D, implicit 'start' and 'end'"),

        pytest.param([23], None, 0, id="1D, implicit 'start'"),
        pytest.param([5, 10], None, 1, id="2D, implicit 'start'"),
        pytest.param([5, 10, 4], None, 1, id="3D, implicit 'start'"),
        pytest.param([5, 10, 4, 8], None, 2, id="4D, implicit 'start'"),
        pytest.param([5, 3, 10, 4, 8], None, 4, id="5D, implicit 'start'"),
        pytest.param([2, 5, 3, 10, 4, 8], None, 6, id="6D, implicit 'start'"),

        pytest.param([23], 0, None, id="1D, implicit 'end'"),
        pytest.param([5, 10], 1, None, id="2D, implicit 'end'"),
        pytest.param([5, 10, 4], 1, None, id="3D, implicit 'end'"),
        pytest.param([5, 10, 4, 8], 2, None, id="4D, implicit 'end'"),
        pytest.param([5, 3, 10, 4, 8], 4, None, id="5D, implicit 'end'"),
        pytest.param([2, 5, 3, 10, 4, 8], 6, None, id="6D, implicit 'end'"),
    ])
def test_convert_shape_with_implicit_arguments(input_shape, start: Optional[int], end: Optional[int]):
    shape_args = dict()
    if start is not None:
        shape_args["start"] = start
    if end is not None:
        shape_args["end"] = end

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Shape", ["input"], ["output"], **shape_args),
        ],
        'shape test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT64, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("type_", [
    TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE,
    TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
    TensorProto.UINT8, TensorProto.UINT32, TensorProto.UINT64,
    TensorProto.STRING, TensorProto.BOOL
], ids=name_for_onnx_type)
def test_convert_shape(type_):
    input_shape = [2, 4, 6, 8]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Shape", ["x"], ["y"])],
        'shape test',
        [onnx.helper.make_tensor_value_info("x", type_, input_shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.INT64, ())]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(type_))
    executors.convert_run_compare(onnx_model, input_data)


def test_convert_shape__invalid_type():
    type_ = TensorProto.UINT16

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Shape', ['x'], ['y'])],
        'Invalid type test test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'UINT16' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_shape__quantized(type_: TensorProto.DataType, intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Shape', ['x1'], ['x2']),
            onnx.helper.make_node('Reshape', ['x2', 'flat_shape'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [2], [2, 2])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig()
    config.dont_skip_nodes_with_known_outputs = True
    executors.convert_run_compare(onnx_model, data, conversion_config=config)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.SHAPE, BuiltinOperator.RESHAPE
    ])
