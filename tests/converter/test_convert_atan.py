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

from onnx2tflite.src.converter import convert
from tests import executors
from onnx2tflite.src import errors
from onnx2tflite.src.errors import ConversionError


@pytest.mark.parametrize(
    "shape",
    [
        [42],
        [5, 10],
        [6, 8, 10],
        [4, 6, 8, 10],
        [2, 4, 6, 8, 10]
    ], ids=lambda x: f'{len(x)}D')
def test_convert_atan__shapes(shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Atan', ['x'], ['y'])],
        'Atan test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = (np.random.random(np.prod(shape)).reshape(shape).astype(np.float32) - 0.5) * 10  # <-5, 5)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_atan__unsupported_type():
    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Atan', ['x'], ['y'])],
        'Atan test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.DOUBLE, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.DOUBLE, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)
    assert exc_info.value.code == errors.Code.CONVERSION_IMPOSSIBLE
