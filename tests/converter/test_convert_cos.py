#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize("_type", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_cos__quantized(_type: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Cos', ['x1'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', _type, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.INVALID_ONNX_MODEL


@pytest.mark.parametrize(
    "shape",
    [
        [42],
        [5, 10],
        [6, 8, 10],
        [4, 6, 8, 10],
        [2, 4, 6, 8, 10]
    ], ids=lambda x: f'{len(x)}D')
def test_convert_cos__shapes(shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cos', ['x'], ['y'])],
        'Cos test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = (np.random.random(np.prod(shape)).reshape(shape).astype(np.float32) - 0.5) * 10  # <-5, 5)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_cos__unsupported_type():
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cos', ['x'], ['y'])],
        'Cos test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.DOUBLE, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.DOUBLE, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
