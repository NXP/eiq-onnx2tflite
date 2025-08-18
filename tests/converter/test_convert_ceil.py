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
        [6, 8, 10],
        [4, 6, 8, 10],
        [2, 4, 6, 8, 10]
    ], ids=lambda x: f'{len(x)}D')
def test_convert_ceil__shapes(shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Ceil', ['x'], ['y'])],
        'Ceil test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = (np.random.random(shape) * 20. - 10.).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_ceil__unsupported_type():
    type_ = TensorProto.DOUBLE

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Ceil', ['x'], ['y'])],
        'Ceil test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
