#
# Copyright 2024-2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src.converter import convert
from onnx2tflite.src import errors
from onnx2tflite.src.errors import ConversionError


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

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)

    assert exc_info.value.code == errors.Code.CONVERSION_REQUIRES_OVERRIDE
    assert 'opset < 7 is not supported' in str(exc_info.value)
    assert '--skip-opset-version-check' in str(exc_info.value)



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
        opset_imports=[onnx.helper.make_opsetid(domain="", version=26)]  # Opset 26.
    )

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)

    assert exc_info.value.code == errors.Code.CONVERSION_REQUIRES_OVERRIDE
    assert 'opset > 25 is not guaranteed' in str(exc_info.value)
    assert '--skip-opset-version-check' in str(exc_info.value)

