#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx.helper
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from tests import executors


def test_prune_transpose__identity_permutation(intermediate_tflite_model_provider):
    # ONNX: Add -> Transpose -> Add
    # TFLite: Add -> Add
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Add", ["input", "input"], ["sigmoid_out"]),
                onnx.helper.make_node("Transpose", ["sigmoid_out"], ["transpose1_out"], perm=[0, 1, 2, 3]),
                onnx.helper.make_node("Add", ["transpose1_out", "transpose1_out"], ["output"]),
            ],
            "transpose_prune_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.ADD,
        BuiltinOperator.ADD,
    ])


def test_prune_transpose__identity_permutation__channels_last(intermediate_tflite_model_provider):
    # ONNX: MaxPool -> Transpose -> MaxPool
    # TFLite: Transpose (IO) -> MaxPool -> MaxPool -> Transpose (IO)
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("MaxPool", ["input"], ["max_pool_out"], kernel_shape=[3, 3]),
                onnx.helper.make_node("Transpose", ["max_pool_out"], ["transpose1_out"], perm=[0, 1, 2, 3]),
                onnx.helper.make_node("MaxPool", ["transpose1_out"], ["output"], kernel_shape=[3, 3]),
            ],
            "transpose_prune_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.MAX_POOL_2D,
        BuiltinOperator.MAX_POOL_2D,
        BuiltinOperator.TRANSPOSE,
    ])
