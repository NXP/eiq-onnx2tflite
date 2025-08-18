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


@pytest.mark.parametrize(
    "shape",
    [
        [42],
        [5, 10],
        [4, 6, 8],
        [3, 4, 5, 6],
        [2, 3, 4, 5, 6],
    ], ids=lambda x: f'{len(x)}D')
def test_convert_hard_swish(shape: list[int]):
    np.random.seed(42)
    x_data = np.random.random(np.prod(shape)).reshape(shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('HardSwish', ['x'], ['y'])],
        'HardSwish test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


def test_convert_hard_swish__unsupported_type():
    data_type = TensorProto.DOUBLE

    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('HardSwish', ['x'], ['y'])],
        'HardSwish test',
        [onnx.helper.make_tensor_value_info('x', data_type, shape)],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
