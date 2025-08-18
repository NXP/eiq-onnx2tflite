#
# Copyright 2025 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert


def test_unsupported_models_logged():
    k = 3
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('TopK', ['x', 'k'], ['y', 'i'])],
        'TopK test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 10]),
            onnx.helper.make_tensor_value_info('k', TensorProto.INT64, [1])
        ],
        [
            onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, k]),
            onnx.helper.make_tensor_value_info('i', TensorProto.INT64, [1, k])
        ],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, ConversionConfig({"skip_shape_inference": True}))

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.UNSUPPORTED_OPERATOR
    parser_logs = logger.conversion_log.get_logs()["onnx_parser"]
    assert any(["Model contains unsupported nodes: [TopK]." in parser_logs[0]["message"]])
