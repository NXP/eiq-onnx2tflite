#
# Copyright 2026 NXP
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


@pytest.mark.parametrize(
    'shape', [[10], [5, 3], [5, 5, 4], [4, 3, 2, 1], [1, 2, 3, 4, 5]], ids=lambda x: f'Shape = {x}'
)
def test_convert_is_nan__correct_output(shape):
    np.random.seed(42)
    x_data = np.random.rand(*shape).astype(np.float32)

    # Add some NaNs
    x_data[x_data > 0.5] = np.nan
    assert np.isnan(x_data).any()

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('IsNaN', ['x'], ['y'])],
        'IsNaN test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_data.shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.BOOL, x_data.shape)],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


def test_convert_is_nan__invalid_type():
    # Type is not supported by ONNX.
    data_type = TensorProto.INT8

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('IsNaN', ['x'], ['y'])],
        'IsNaN test',
        [onnx.helper.make_tensor_value_info('x', data_type, [5])],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL
