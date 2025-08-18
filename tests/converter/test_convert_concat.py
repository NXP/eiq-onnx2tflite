#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math

import numpy as np
import onnx.helper
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize("_type", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_concat__quantized(_type: TensorProto.DataType):
    shape = [3, 14, 15]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Concat', ['x1', 'x1'], ['x2'], axis=0),
            onnx.helper.make_node('Reshape', ['x2', 'flat_shape'], ['x3']),
            onnx.helper.make_node('DequantizeLinear', ['x3', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', _type, [1], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [1], [np.prod(shape) * 2])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


@pytest.mark.parametrize("_type", [
    TensorProto.FLOAT,
    TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
    TensorProto.UINT8,
    TensorProto.BOOL
], ids=name_for_onnx_type)
def test_convert_concat__types(_type: TensorProto.DataType):
    shape = [3, 14, 15]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(to_numpy_type(_type))  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Concat', ['x', 'x'], ['y'], axis=0),
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', _type, shape)],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_concat__unsupported_type():
    _type = TensorProto.DOUBLE

    shape = [3, 14, 15]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Concat', ['x', 'x'], ['y'], axis=0),
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', _type, shape)],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


def test_convert_4d_concat_with_negative_axis():
    node = onnx.helper.make_node("Concat", ["X", "Y"], ["Z"], axis=-3)

    graph = onnx.helper.make_graph(
        [node],
        "concat",
        [
            onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5)),
            onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 4, 4, 5))
        ],
        [onnx.helper.make_tensor_value_info("Z", TensorProto.FLOAT, ())],
    )

    original_model = onnx.helper.make_model(graph, producer_name="onnx-concat")
    onnx.checker.check_model(original_model)

    input_data = {
        0: np.arange(math.prod((2, 3, 4, 5))).reshape((2, 3, 4, 5)).astype(np.float32),
        1: np.arange(math.prod((2, 4, 4, 5))).reshape((2, 4, 4, 5)).astype(np.float32),
    }

    executors.convert_run_compare(original_model, input_data)


def test_convert_4d_concat_with_invalid_axis():
    node = onnx.helper.make_node("Concat", ["X", "Y"], ["Z"], axis=-5)

    graph = onnx.helper.make_graph(
        [node],
        "concat",
        [
            onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5)),
            onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 4, 4, 5))
        ],
        [onnx.helper.make_tensor_value_info("Z", TensorProto.FLOAT, ())],
    )

    original_model = onnx.helper.make_model(graph, producer_name="onnx-concat")

    with pytest.raises(logger.Error) as e:
        convert.convert_model(original_model)
    assert e.value.error_code == logger.Code.SHAPE_INFERENCE_ERROR


def test_convert_3d_concat_with_negative_axis():
    node = onnx.helper.make_node("Concat", ["X", "Y"], ["Z"], axis=-2)

    graph = onnx.helper.make_graph(
        [node],
        "concat",
        [
            onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4)),
            onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 4, 4))
        ],
        [onnx.helper.make_tensor_value_info("Z", TensorProto.FLOAT, ())],
    )

    original_model = onnx.helper.make_model(graph, producer_name="onnx-concat")
    onnx.checker.check_model(original_model)

    input_data = {
        0: np.arange(math.prod((2, 3, 4))).reshape((2, 3, 4)).astype(np.float32),
        1: np.arange(math.prod((2, 4, 4))).reshape((2, 4, 4)).astype(np.float32),
    }

    executors.convert_run_compare(original_model, input_data)


def test_convert_3d_concat_and_relu_with_negative_axis():
    node = onnx.helper.make_node("Concat", ["A", "B"], ["C"], axis=-2)
    node_relu = onnx.helper.make_node("Relu", ["C"], ["D"])

    graph = onnx.helper.make_graph(
        [node, node_relu],
        "concat+relu",
        [
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 4)),
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, (2, 4, 4)),
        ],
        [onnx.helper.make_tensor_value_info("D", TensorProto.FLOAT, (2, 7, 4))]
    )

    original_model = onnx.helper.make_model(graph, producer_name="onnx-concat-relu")
    onnx.checker.check_model(original_model)

    input_data = {
        0: np.arange(math.prod((2, 3, 4))).reshape((2, 3, 4)).astype(np.float32),
        1: np.arange(math.prod((2, 4, 4))).reshape((2, 4, 4)).astype(np.float32),
    }

    executors.convert_run_compare(original_model, input_data)
