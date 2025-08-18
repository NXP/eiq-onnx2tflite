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
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize("type_", [TensorProto.FLOAT], ids=name_for_onnx_type)
def test_convert_relu__types(type_: int):
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Relu", ["x"], ["y"])],
        'Relu test',
        [onnx.helper.make_tensor_value_info("x", type_, shape)],
        [onnx.helper.make_tensor_value_info("y", type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(23)
    data = (np.random.random(shape) * 10. - 5.).astype(to_numpy_type(type_))

    executors.convert_run_compare(onnx_model, data)


def test_convert_p_relu_with_invalid_type():
    data_type = TensorProto.DOUBLE
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Relu', ['x'], ['y'])],
        'Relu test',
        [onnx.helper.make_tensor_value_info("x", data_type, shape)],
        [onnx.helper.make_tensor_value_info("y", data_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("type_", [TensorProto.INT8], ids=name_for_onnx_type)
def test_convert_relu__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Relu', ['x1'], ['y'])
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

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.NOT_IMPLEMENTED
