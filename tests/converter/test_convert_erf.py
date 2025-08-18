#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

import numpy as np
import onnx.helper
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param([5, 768], id="2D"),
        pytest.param([5, 3, 256], id="3D"),
        pytest.param([5, 12, 4, 16], id="4D"),
    ])
def test_convert_erf(input_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Erf', ['input'], ['output'])],
        'Erf test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    config = ConversionConfig()
    config.allow_select_ops = True
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


def test_convert_erf__flex_prohibited():
    input_shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Erf', ['input'], ['output'])],
        'Erf test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    config = ConversionConfig()
    config.allow_select_ops = False

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=config)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert '--allow-select-ops' in logger.conversion_log.get_node_error_message(0)


def test_convert_erf__unsupported_type():
    type_ = TensorProto.DOUBLE

    shape = [3, 14, 15]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Erf', ['x'], ['y']),
        ],
        'Erf test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize("_type", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_erf__quantized(_type: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Erf', ['x1'], ['y'])
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

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.NOT_IMPLEMENTED
