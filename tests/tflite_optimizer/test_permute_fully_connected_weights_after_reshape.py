#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import os
import pathlib

import numpy as np
import onnx
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.conversion_config import ConversionConfig
from tests import executors
from onnx2tflite.src.tflite_optimizer.optimizer import Optimization


def test_permute_fully_connected_weights_after_reshape__simple_model__4d(intermediate_tflite_model_provider):
    input_shape = [4, 5, 6, 7]

    spatial_size = np.prod(input_shape[1:])
    batch_size = input_shape[0]

    flat_shape = [batch_size, spatial_size]
    weight_shape = [spatial_size, 42]

    np.random.seed(42)
    weight_data = np.random.random(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Reshape', ['x1', 'flat_shape'], ['x2']),
            onnx.helper.make_node('Gemm', ['x2', 'w'], ['y']),
        ],
        'FC weight permutation test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)],
        [
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [len(flat_shape)], flat_shape),
            onnx.helper.make_tensor('w', TensorProto.FLOAT, weight_shape, weight_data)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.random.random(input_shape).astype(np.float32)

    config = ConversionConfig()
    config.optimization_whitelist = [Optimization.PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE]
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.MAX_POOL_2D,
        # No `TRANSPOSE` right here.
        BuiltinOperator.RESHAPE,
        BuiltinOperator.FULLY_CONNECTED
    ])


def test_permute_fully_connected_weights_after_reshape__impossible_optimization__shape(
        intermediate_tflite_model_provider):
    input_shape = [4, 5, 6, 7]
    flat_shape = [12, 70]  # The first dimension isn't batch size.
    weight_shape = [70, 42]

    np.random.seed(42)
    weight_data = np.random.random(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Reshape', ['x1', 'flat_shape'], ['x2']),
            onnx.helper.make_node('Gemm', ['x2', 'w'], ['y']),
        ],
        'FC weight permutation test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)],
        [
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [len(flat_shape)], flat_shape),
            onnx.helper.make_tensor('w', TensorProto.FLOAT, weight_shape, weight_data)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.random.random(input_shape).astype(np.float32)

    config = ConversionConfig()
    config.optimization_whitelist = [Optimization.PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE]
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.MAX_POOL_2D,
        BuiltinOperator.TRANSPOSE,  # The `Transpose` wasn't removed.
        BuiltinOperator.RESHAPE,
        BuiltinOperator.FULLY_CONNECTED
    ])


def test_permute_fully_connected_weights_after_reshape__impossible_optimization__permutation(
        intermediate_tflite_model_provider):
    # The `Transpose` is already in the ONNX model, and it is not added by the `ReshapeConverter`.
    # It doesn't permute its input from channels last to channels first, so the optimization cannot be performed.

    permutation = [3, 2, 1, 0]

    input_shape = [4, 5, 6, 7]

    spatial_size = np.prod(input_shape[1:])
    batch_size = input_shape[0]

    flat_shape = [batch_size, spatial_size]
    weight_shape = [spatial_size, 42]

    np.random.seed(42)
    weight_data = np.random.random(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Transpose', ['x1'], ['x2'], perm=permutation),
            onnx.helper.make_node('Reshape', ['x2', 'flat_shape'], ['x3']),
            onnx.helper.make_node('Gemm', ['x3', 'w'], ['y']),
        ],
        'FC weight permutation test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)],
        [
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [len(flat_shape)], flat_shape),
            onnx.helper.make_tensor('w', TensorProto.FLOAT, weight_shape, weight_data),
            onnx.helper.make_tensor('perm', TensorProto.INT64, [len(permutation)], permutation)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.random.random(input_shape).astype(np.float32)

    config = ConversionConfig()
    config.optimization_whitelist = [Optimization.PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE]
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.MAX_POOL_2D,
        BuiltinOperator.TRANSPOSE,  # The `Transpose` wasn't removed.
        BuiltinOperator.RESHAPE,
        BuiltinOperator.FULLY_CONNECTED
    ])


def test_permute_fully_connected_weights_after_reshape__simple_model__3d(intermediate_tflite_model_provider):
    input_shape = [4, 5, 6]

    spatial_size = np.prod(input_shape[1:])
    batch_size = input_shape[0]

    flat_shape = [batch_size, spatial_size]
    weight_shape = [spatial_size, 42]

    np.random.seed(42)
    weight_data = np.random.random(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1]),
            onnx.helper.make_node('Reshape', ['x1', 'flat_shape'], ['x2']),
            onnx.helper.make_node('Gemm', ['x2', 'w'], ['y']),
        ],
        'FC weight permutation test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)],
        [
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [len(flat_shape)], flat_shape),
            onnx.helper.make_tensor('w', TensorProto.FLOAT, weight_shape, weight_data)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.random.random(input_shape).astype(np.float32)

    config = ConversionConfig()
    config.optimization_whitelist = [
        Optimization.PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE,
        Optimization.FUSE_RESHAPE_OPERATORS
    ]
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.RESHAPE,  # Extra `Reshape` due to 1D MaxPool.
        BuiltinOperator.MAX_POOL_2D,
        # No `TRANSPOSE` right here.
        BuiltinOperator.RESHAPE,
        BuiltinOperator.FULLY_CONNECTED
    ])


def test_permute_fully_connected_weights_after_reshape__simple_model__5d(intermediate_tflite_model_provider):
    input_shape = [2, 3, 4, 5, 6]

    spatial_size = np.prod(input_shape[1:])
    batch_size = input_shape[0]

    flat_shape = [batch_size, spatial_size]
    weight_shape = [spatial_size, 42]

    channel = input_shape[1]
    conv_w_shape = [channel, channel, 3, 3, 3]

    np.random.seed(42)
    weight_data = np.random.random(weight_shape).astype(np.float32)
    conv_w_data = np.random.random(conv_w_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Conv', ['x', 'w1'], ['x1'], kernel_shape=[3, 3, 3], auto_pad='SAME_UPPER'),
            onnx.helper.make_node('Reshape', ['x1', 'flat_shape'], ['x2']),
            onnx.helper.make_node('Gemm', ['x2', 'w2'], ['y']),
        ],
        'FC weight permutation test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)],
        [
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [len(flat_shape)], flat_shape),
            onnx.helper.make_tensor('w1', TensorProto.FLOAT, conv_w_shape, conv_w_data),
            onnx.helper.make_tensor('w2', TensorProto.FLOAT, weight_shape, weight_data),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.random.random(input_shape).astype(np.float32)

    config = ConversionConfig()
    config.optimization_whitelist = [
        Optimization.PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE,
        Optimization.FUSE_RESHAPE_OPERATORS
    ]
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.CONV_3D,
        # No `TRANSPOSE` right here.
        BuiltinOperator.RESHAPE,
        BuiltinOperator.FULLY_CONNECTED
    ])


_ARTIFACTS_DIR = pathlib.Path(__file__).parent.parent.joinpath("artifacts")


def test_permute_fully_connected_weights_after_reshape__real_model(intermediate_tflite_model_provider):
    model_path = os.path.join(_ARTIFACTS_DIR, "downloaded", "bvlcalexnet-12", "model.onnx")

    input_shape = [1, 3, 224, 224]

    onnx_model = onnx.load_model(model_path)

    np.random.seed(42)
    input_data = np.random.random(input_shape).astype(np.float32)

    config = ConversionConfig()
    config.optimization_whitelist = [Optimization.PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE]
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    ops = intermediate_tflite_model_provider.get_operators()

    # The first op is a transpose (to make the input channels last).
    assert ops[0].builtin_options.operator_type == BuiltinOperator.TRANSPOSE
    transpose_index = intermediate_tflite_model_provider.get_operator_code_at_index(ops[0].opcode_index)

    # There are no more `Transpose` operators in the model. (besides the first one).
    assert all(op.opcode_index != transpose_index for op in ops[1:])
