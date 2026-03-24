#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
from onnx import helper, TensorProto, numpy_helper

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from tests import executors


def test_remove_unnecessary_ops_before_flattened_conv(intermediate_tflite_model_provider):
    np.random.seed(42)
    input_channels = 400
    output_channels = 144

    input_shape = [5, 10, 16]
    output_shape = [1, output_channels, 1, 1]
    conv_input_shape = [1, input_channels, 1, 1]

    nodes = [
        helper.make_node("Slice", inputs=["x", "starts", "ends", "axes", "steps"], outputs=["a"]),
        helper.make_node("Reshape", inputs=["a", "reshape1_shape"], outputs=["b"]),
        helper.make_node("Transpose", inputs=["b"], outputs=["c"], perm=[1, 0]),
        helper.make_node("Reshape", inputs=["c", "reshape2_shape"], outputs=["d"]),
        helper.make_node("Conv", inputs=["d", "w"], outputs=["y"], kernel_shape=[1, 1]),
    ]

    initializers = [
        numpy_helper.from_array(np.random.randn(output_channels, input_channels, 1, 1).astype(np.float32), "w"),
        numpy_helper.from_array(np.array([0, 0, 0], dtype=np.int64), "starts"),
        numpy_helper.from_array(np.array([5, 10, 8], dtype=np.int64), "ends"),
        numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), "axes"),
        numpy_helper.from_array(np.array([1, 1, 1], dtype=np.int64), "steps"),
        numpy_helper.from_array(np.array([50, 8], dtype=np.int64), "reshape1_shape"),
        numpy_helper.from_array(np.array(conv_input_shape, dtype=np.int64), "reshape2_shape")
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="Test model",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        initializer=initializers
    )

    onnx_model = helper.make_model(graph)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, input_data, atol=1.8e-5)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.SLICE,
        # BuiltinOperator.RESHAPE,  # Removed
        # BuiltinOperator.TRANSPOSE,  # Removed
        BuiltinOperator.RESHAPE,
        BuiltinOperator.CONV_2D,
        BuiltinOperator.RESHAPE,
    ])
