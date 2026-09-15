#
# Copyright 2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors
from onnx2tflite.src import errors
from onnx2tflite.src.errors import ConversionError


@pytest.mark.parametrize(
    "input_shape",
    [
        [42],
        [5, 10],
        [6, 8, 10],
        [4, 6, 8, 10],
        [2, 4, 6, 8, 10]
    ], ids=lambda x: f'{len(x)}D')
def test_convert_tan(input_shape):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Tan", inputs=["x"], outputs=["y"])],
        "tan graph",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, input_shape)],
    )
    model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = (np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32) - 0.5) * 10  # <-5, 5)

    executors.convert_run_compare(model, input_data)


def test_convert_tan__unsupported_type():
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Tan', ['x'], ['y'])],
        'Tan test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.DOUBLE, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.DOUBLE, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)
    assert exc_info.value.code == errors.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_tan__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Tan', ['x1'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)
    assert exc_info.value.code == errors.Code.INVALID_ONNX_MODEL
