#
# Copyright 2025 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src import errors
from onnx2tflite.src.errors import ConversionError


def test_unsupported_models_logged():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LpNormalization', ['x'], ['y'])],
        'LpNormalization test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 10])
        ],
        [
            onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 10]),
        ],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model, ConversionConfig({"skip_shape_inference": True}))

    assert exc_info.value.code == errors.Code.UNSUPPORTED_OPERATOR
    assert "Model contains unsupported nodes: [LpNormalization]." in str(exc_info.value)

