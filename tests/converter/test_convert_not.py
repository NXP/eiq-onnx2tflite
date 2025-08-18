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
from tests import executors


def test_convert_not():
    shape = [3, 14, 15]

    data = np.random.choice([True, False], shape).astype(np.bool_)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Not', ['x'], ['y'])],
        'Not test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.BOOL, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.BOOL, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)


def test_convert_not_unsupported_type():
    shape = [256]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Not', ['x'], ['y'])],
        'Not test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.INT32, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT32, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL
