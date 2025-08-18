#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
from onnx import TensorProto

from tests import executors


def test_eliminate_single_node(intermediate_tflite_model_provider):
    input_shape = [5, 48, 16]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Exp', ['input'], ['output']),
            onnx.helper.make_node('Exp', ['input'], ['dead']),
        ],
        'Exp test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 1
