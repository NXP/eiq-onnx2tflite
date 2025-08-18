#
# Copyright 2024-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert


def test_too_low_opset():
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Abs', ['x'], ['y'])],
        'Opset test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid(domain="", version=6)]  # Opset 6.
    )

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    log = logger.conversion_log.get_logs()['operator_conversion'][0]
    assert log['error_code'] == logger.Code.NOT_IMPLEMENTED
    assert 'opset < 7 is not supported' in log['message']
    assert '--skip-opset-version-check' in log['message']


def test_too_high_opset():
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Abs', ['x'], ['y'])],
        'Abs test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid(domain="", version=23)]  # Opset 23.
    )

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    log = logger.conversion_log.get_logs()['operator_conversion'][0]
    assert log['error_code'] == logger.Code.NOT_IMPLEMENTED
    assert 'opset > 22 is not guaranteed' in log['message']
    assert '--skip-opset-version-check' in log['message']
