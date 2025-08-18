#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import List

import numpy as np
import onnx.helper
from onnx import TensorProto

import onnx2tflite.src.onnx_parser.onnx_model
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.converter.convert import _convert, build_conversion_context
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.tflite_generator import tflite_model
from tests import executors


def _assert_converted_model_has_operators(onnx_model, expected_operators: List[BuiltinOperator]):
    onnx.checker.check_model(onnx_model)

    # Infer the shapes of internal tensors
    parsed_onnx_model = ModelShapeInference.infer_shapes(onnx_model)

    # Initialize the internal ONNX model representation
    internal_onnx_model = onnx2tflite.src.onnx_parser.onnx_model.ModelProto(parsed_onnx_model)

    # Convert the ONNX model to TFLite
    internal_tflite_model = _convert(build_conversion_context(internal_onnx_model))

    operators: List[tflite_model.Operator] = internal_tflite_model.sub_graphs.get_last().operators.vector
    assert len(operators) == len(expected_operators)

    for operator, expected_operator in zip(operators, expected_operators):
        assert operator.builtin_options.operator_type == expected_operator


def test_channels_first_output_annihilation():
    # ONNX: MaxPool -> Reshape -> Reshape
    # TFLite: Transpose -> MaxPool2D -> Transpose
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("MaxPool", ["input"], ["max_pool_out"], kernel_shape=[3, 3],
                                      auto_pad="SAME_UPPER"),
                onnx.helper.make_node("Reshape", ["max_pool_out", "reshape1_shape"], ["reshape1_out"]),
                onnx.helper.make_node("Reshape", ["reshape1_out", "reshape2_shape"], ["output"]),
            ],
            "reshape_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("reshape1_shape", TensorProto.INT64, [2], [36, 12]),
                onnx.helper.make_tensor("reshape2_shape", TensorProto.INT64, [4], [1, 3, 12, 12]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.TRANSPOSE, BuiltinOperator.MAX_POOL_2D, BuiltinOperator.TRANSPOSE]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_output_annihilation():
    # ONNX: Mul -> Reshape -> Reshape
    # TFLite: Mul
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["input", "scalar"], ["mul_out"]),
                onnx.helper.make_node("Reshape", ["mul_out", "reshape1_shape"], ["reshape1_out"]),
                onnx.helper.make_node("Reshape", ["reshape1_out", "reshape2_shape"], ["output"]),
            ],
            "reshape_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("reshape1_shape", TensorProto.INT64, [2], [36, 12]),
                onnx.helper.make_tensor("reshape2_shape", TensorProto.INT64, [4], [1, 3, 12, 12]),
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_input_annihilation():
    # ONNX: Reshape -> Reshape -> Mul
    # TFLite: Mul
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Reshape", ["input", "reshape1_shape"], ["reshape1_out"]),
                onnx.helper.make_node("Reshape", ["reshape1_out", "reshape2_shape"], ["reshape2_out"]),
                onnx.helper.make_node("Mul", ["reshape2_out", "scalar"], ["output"]),
            ],
            "reshape_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("reshape1_shape", TensorProto.INT64, [2], [36, 12]),
                onnx.helper.make_tensor("reshape2_shape", TensorProto.INT64, [4], [1, 3, 12, 12]),
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_forked_annihilation():
    #                         / -> Reshape -> \
    # ONNX:  Mul -> Reshape -|                 | -> Mul
    #                         \ -> Reshape  ->/
    # TFLite: Mul -> Mul
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["input", "scalar"], ["mul_out"]),
                onnx.helper.make_node("Reshape", ["mul_out", "reshape1_shape"], ["reshape1_out"]),
                onnx.helper.make_node("Reshape", ["reshape1_out", "reshape2_shape"], ["output1"]),
                onnx.helper.make_node("Reshape", ["reshape1_out", "reshape2_shape"], ["output2"]),
                onnx.helper.make_node("Mul", ["output1", "output2"], ["output"]),
            ],
            "reshape_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("reshape1_shape", TensorProto.INT64, [2], [36, 12]),
                onnx.helper.make_tensor("reshape2_shape", TensorProto.INT64, [4], [1, 3, 12, 12]),
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.MUL, BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_impossible_forked_annihilation():
    #                         / -> Mul -> Reshape -> \
    # ONNX:  Mul -> Reshape -|                        | -> Mul
    #                         \ ->     Reshape     ->/
    #
    #                           / -> Mul -> Reshape -> \
    # TFLite:  Mul -> Reshape -|                        | -> Mul
    #                           \ ->     Reshape     ->/
    #
    # The Reshapes cannot be annihilated!

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["input", "scalar"], ["mul_out"]),
                onnx.helper.make_node("Reshape", ["mul_out", "reshape1_shape"], ["reshape1_out"]),
                onnx.helper.make_node("Mul", ["reshape1_out", "scalar"], ["mul2_out"]),
                onnx.helper.make_node("Reshape", ["mul2_out", "reshape2_shape"], ["output1"]),
                onnx.helper.make_node("Reshape", ["reshape1_out", "reshape2_shape"], ["output2"]),
                onnx.helper.make_node("Mul", ["output1", "output2"], ["output"]),
            ],
            "reshape_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("reshape1_shape", TensorProto.INT64, [2], [36, 12]),
                onnx.helper.make_tensor("reshape2_shape", TensorProto.INT64, [4], [1, 3, 12, 12]),
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.MUL, BuiltinOperator.RESHAPE, BuiltinOperator.MUL, BuiltinOperator.RESHAPE,
                          BuiltinOperator.RESHAPE, BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_channels_first_output_fusing():
    # ONNX: MaxPool -> Reshape -> Reshape
    # TFLite: Transpose -> MaxPool2D -> Transpose -> Reshape
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("MaxPool", ["input"], ["max_pool_out"], kernel_shape=[3, 3],
                                      auto_pad="SAME_UPPER"),
                onnx.helper.make_node("Reshape", ["max_pool_out", "reshape1_shape"], ["reshape1_out"]),
                onnx.helper.make_node("Reshape", ["reshape1_out", "reshape2_shape"], ["output"]),
            ],
            "reshape_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("reshape1_shape", TensorProto.INT64, [2], [36, 12]),
                onnx.helper.make_tensor("reshape2_shape", TensorProto.INT64, [4], [1, 3, 144, 1]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.TRANSPOSE, BuiltinOperator.MAX_POOL_2D, BuiltinOperator.TRANSPOSE,
                          BuiltinOperator.RESHAPE]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_input_fusing():
    # ONNX: Reshape -> Reshape -> Mul
    # TFLite: Reshape -> Mul
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Reshape", ["input", "reshape1_shape"], ["reshape1_out"]),
                onnx.helper.make_node("Reshape", ["reshape1_out", "reshape2_shape"], ["reshape2_out"]),
                onnx.helper.make_node("Mul", ["reshape2_out", "scalar"], ["output"]),
            ],
            "reshape_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("reshape1_shape", TensorProto.INT64, [2], [36, 12]),
                onnx.helper.make_tensor("reshape2_shape", TensorProto.INT64, [4], [1, 3, 144, 1]),
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.RESHAPE, BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_forked_fusing():
    #                         / -> Reshape -> Mul
    # ONNX:  Mul -> Reshape -|
    #                         \ -> Reshape -> Mul
    #
    #                / -> Mul
    # TFLite:  Mul -|
    #                \ -> Reshape -> Mul
    #

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["input", "scalar"], ["mul_out"]),
                onnx.helper.make_node("Reshape", ["mul_out", "reshape1_shape"], ["reshape1_out"]),
                onnx.helper.make_node("Reshape", ["reshape1_out", "reshape2_shape"], ["reshape2_out"]),
                onnx.helper.make_node("Reshape", ["reshape1_out", "reshape3_shape"], ["reshape3_out"]),
                onnx.helper.make_node("Mul", ["reshape2_out", "scalar"], ["output1"]),
                onnx.helper.make_node("Mul", ["reshape3_out", "scalar"], ["output2"]),
            ],
            "reshape_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output1", TensorProto.FLOAT, ()),
             onnx.helper.make_tensor_value_info("output2", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("reshape1_shape", TensorProto.INT64, [2], [36, 12]),
                onnx.helper.make_tensor("reshape2_shape", TensorProto.INT64, [4], [1, 3, 12, 12]),
                onnx.helper.make_tensor("reshape3_shape", TensorProto.INT64, [4], [1, 3, 144, 1]),
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.MUL, BuiltinOperator.RESHAPE, BuiltinOperator.MUL,
                          BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)
