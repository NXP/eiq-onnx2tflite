#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import onnx.helper
from onnx import TensorProto

from onnx2tflite.src.onnx_parser import onnx_model


def test_operator_version_present_in_node_proto():
    reshape_node = onnx.helper.make_node(
        'Reshape',
        ['X', 'new_shape'],
        ['Y'])

    average_pool_node = onnx.helper.make_node(
        "QLinearGlobalAveragePool",
        ["Y", "input_scale", "input_zero_point", "output_scale",
         "output_zero_point"],
        ["Z"],
        domain="com.microsoft")

    input_shape = (5, 3, 16, 16)

    graph = onnx.helper.make_graph(
        [reshape_node, average_pool_node],
        'node version test',
        [onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("Z", TensorProto.FLOAT, ())],
        initializer=[
            onnx.helper.make_tensor("new_shape", TensorProto.INT64, [2], [15, 256]),
            onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [], [0.0421]),
            onnx.helper.make_tensor("input_zero_point", TensorProto.INT8, [], [90]),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [], [0.0421]),
            onnx.helper.make_tensor("output_zero_point", TensorProto.INT8, [], [90]),
        ]
    )

    input_onnx_model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 9),
            onnx.helper.make_opsetid("com.microsoft", 1)
        ],
    )
    onnx.checker.check_model(input_onnx_model)

    internal_onnx_model = onnx_model.ModelProto(input_onnx_model)

    assert internal_onnx_model.graph.nodes[0].domain == ""
    assert internal_onnx_model.graph.nodes[0].version == 9
    assert internal_onnx_model.graph.nodes[1].domain == "com.microsoft"
    assert internal_onnx_model.graph.nodes[1].version == 1
