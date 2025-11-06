#
# Copyright 2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
from onnx import TensorProto

from tests import executors


# noinspection PyPep8Naming
def test_remove_success__4D_to_3D__channels_last(intermediate_tflite_model_provider):
    input_shape = [1, 48, 14, 14]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['input'], ['a'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Reshape', ['a', 'new_shape'], ['c']),
            onnx.helper.make_node('Transpose', ['c'], ['output'], perm=[0, 2, 1]),
        ],
        'Transpose cancellation test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 196, 48])],
        [onnx.helper.make_tensor("new_shape", TensorProto.INT64, [3], [1, 48, 196])]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3


# noinspection PyPep8Naming
def test_remove_success__4D_to_3D(intermediate_tflite_model_provider):
    input_shape = [1, 2, 2, 3]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Transpose', ['input'], ['a'], perm=[0, 3, 1, 2]),
            onnx.helper.make_node('Reshape', ['a', 'new_shape'], ['c']),
            onnx.helper.make_node('Transpose', ['c'], ['output'], perm=[0, 2, 1]),
        ],
        'Transpose cancellation test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 3])],
        [onnx.helper.make_tensor("new_shape", TensorProto.INT64, [3], [1, 3, 4])]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 1


# noinspection PyPep8Naming
def test_remove_success__4D_to_4D(intermediate_tflite_model_provider):
    input_shape = [1, 4, 5, 8]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Transpose', ['input'], ['a'], perm=[0, 3, 1, 2]),
            onnx.helper.make_node('Reshape', ['a', 'new_shape'], ['c']),
            onnx.helper.make_node('Transpose', ['c'], ['output'], perm=[0, 2, 3, 1]),
        ],
        'Transpose cancellation test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 10, 8])],
        [onnx.helper.make_tensor("new_shape", TensorProto.INT64, [4], [1, 8, 2, 10])]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 1


# noinspection PyPep8Naming
def test_remove_failure__channels_not_preserved(intermediate_tflite_model_provider):
    input_shape = [1, 2, 2, 3]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Transpose', ['input'], ['a'], perm=[0, 3, 2, 1]),  # <-
            onnx.helper.make_node('Reshape', ['a', 'new_shape'], ['c']),
            onnx.helper.make_node('Transpose', ['c'], ['output'], perm=[0, 2, 1]),
        ],
        'Transpose cancellation test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 3])],
        [onnx.helper.make_tensor("new_shape", TensorProto.INT64, [3], [1, 3, 4])]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3


def test_remove_failure__channels_not_preserved__channels_last(intermediate_tflite_model_provider):
    input_shape = [1, 48, 7, 7]

    # MaxPool causes additional Transpose to be added to the graph and internal tensors are also
    # marked as channels_last. Model ops before optimization: MaxPool -> Transpose -> Reshape -> Transpose
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['input'], ['a'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Reshape', ['a', 'new_shape'], ['c']),
            onnx.helper.make_node('Transpose', ['c'], ['output'], perm=[0, 2, 1]),
        ],
        'Transpose cancellation test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 784, 3))],
        [onnx.helper.make_tensor("new_shape", TensorProto.INT64, [3], [1, 3, 784])]  # <-
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 5


def test_remove_failure__incorrect_permutation__channels_last(intermediate_tflite_model_provider):
    input_shape = [1, 48, 14, 14]

    # MaxPool causes additional Transpose to be added to the graph and internal tensors are also
    # marked as channels_last. Model ops before optimization: MaxPool -> Transpose -> Reshape -> Transpose
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['input'], ['a'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Reshape', ['a', 'new_shape'], ['c']),
            onnx.helper.make_node('Transpose', ['c'], ['output'], perm=[1, 2, 0]),
        ],
        'Transpose cancellation test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [48, 196, 1])],
        [onnx.helper.make_tensor("new_shape", TensorProto.INT64, [3], [1, 48, 196])]  # <-
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 5
