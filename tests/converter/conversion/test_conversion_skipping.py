#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from tests import executors
from onnx2tflite.src.tflite_optimizer.optimizer import Optimization


def test_node_skipping(intermediate_tflite_model_provider):
    input_shape = [4, 6, 8, 10]

    np.random.seed(42)
    data = np.random.random(input_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            # Ops that will be skipped.
            onnx.helper.make_node('Shape', ['x'], ['s1']),
            onnx.helper.make_node('Gather', ['s1', 'indices'], ['s2']),

            # Ops that will remain.
            onnx.helper.make_node('Reshape', ['x', 's2'], ['y']),
        ],
        'Skipping test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('indices', TensorProto.INT64, [4], [0, 1, 3, 2])]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.RESHAPE])

    log_message = logger.conversion_log.get_logs()['operator_conversion'][0]['message']
    assert '--dont-skip-nodes-with-known-outputs' in log_message
    assert 'Shape' in log_message
    assert 'Gather' in log_message


def test_prohibited_node_skipping(intermediate_tflite_model_provider):
    input_shape = [4, 6, 8, 10]

    np.random.seed(42)
    data = np.random.random(input_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Shape', ['x'], ['s1']),
            onnx.helper.make_node('Gather', ['s1', 'indices'], ['s2']),
            onnx.helper.make_node('Reshape', ['x', 's2'], ['y']),
        ],
        'Skipping test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('indices', TensorProto.INT64, [4], [0, 1, 3, 2])]
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig()
    config.dont_skip_nodes_with_known_outputs = True
    config.optimization_blacklist = [Optimization.ELIMINATE_DEAD_BRANCHES]  # This optimization would remove the ops.
    executors.convert_run_compare(onnx_model, data, conversion_config=config)

    intermediate_tflite_model_provider.assert_converted_model_has_operators(
        [BuiltinOperator.SHAPE, BuiltinOperator.GATHER, BuiltinOperator.RESHAPE]
    )

    assert len(logger.conversion_log.get_logs()['operator_conversion']) == 0  # No message printed.


def test_partial_node_skipping(intermediate_tflite_model_provider):
    input_shape = [4, 6, 8, 10]

    np.random.seed(42)
    data = np.random.random(input_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            # Ops that will be skipped.
            onnx.helper.make_node('Shape', ['x'], ['s1']),
            onnx.helper.make_node('Gather', ['s1', 'indices'], ['s2']),
            onnx.helper.make_node('Cast', ['s2'], ['s3'], to=TensorProto.FLOAT),

            # Ops that will remain.
            onnx.helper.make_node('HardSwish', ['s3'], ['s4']),
            onnx.helper.make_node('Add', ['x', 's4'], ['y']),

        ],
        'Skipping test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('indices', TensorProto.INT64, [1], [-1])]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators(
        [BuiltinOperator.HARD_SWISH, BuiltinOperator.ADD]
    )

    log_message = logger.conversion_log.get_logs()['operator_conversion'][0]['message']
    assert '--dont-skip-nodes-with-known-outputs' in log_message
    assert 'Shape' in log_message
    assert 'Gather' in log_message
    assert 'Cast' in log_message


def test_node_skipping__intermediate_output(intermediate_tflite_model_provider):
    # The `Shape` operator cannot be removed, because its output is also a model output.

    input_shape = [4, 6, 8, 10]

    np.random.seed(42)
    data = np.random.random(input_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Shape', ['x'], ['s1']),
            onnx.helper.make_node('Gather', ['s1', 'indices'], ['s2']),
            onnx.helper.make_node('Reshape', ['x', 's2'], ['y']),
        ],
        'Skipping test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [
            onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ()),
            onnx.helper.make_tensor_value_info('s1', TensorProto.INT64, ())
        ],
        [onnx.helper.make_tensor('indices', TensorProto.INT64, [4], [0, 1, 3, 2])]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators(
        [BuiltinOperator.SHAPE, BuiltinOperator.RESHAPE])

    log_message = logger.conversion_log.get_logs()['operator_conversion'][0]['message']
    assert '--dont-skip-nodes-with-known-outputs' in log_message
    assert 'Gather' in log_message
    assert 'Shape' not in log_message


def test_forbidden_node_skipping__channels_first_output(intermediate_tflite_model_provider):
    input_shape = [1, 3, 8, 10]
    weight_shape = [2, 3, 4, 5]
    static_data_shape = [np.prod(weight_shape).item()]

    np.random.seed(42)
    static_data = np.random.random(static_data_shape).astype(np.float32)
    x_data = np.random.random(input_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Reshape', ['static_data', 'shape'], ['w']),
            onnx.helper.make_node('Conv', ['x', 'w'], ['y'], kernel_shape=[4, 5], auto_pad='SAME_UPPER'),
        ],
        'Skipping test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('static_data', TensorProto.FLOAT, static_data_shape, static_data),
            onnx.helper.make_tensor('shape', TensorProto.INT64, [len(weight_shape)], weight_shape),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=1.e-6)

    # The `Reshape` doesn't get skipped. Just `Transpose` ops are added to handle the different formats.
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.RESHAPE,
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.CONV_2D,
        BuiltinOperator.TRANSPOSE
    ])
