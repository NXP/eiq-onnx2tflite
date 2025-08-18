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
from tests import executors


def _assert_converted_model_has_operators(onnx_model, expected_operators: List[BuiltinOperator]):
    onnx.checker.check_model(onnx_model)

    # Infer the shapes of internal tensors
    parsed_onnx_model = ModelShapeInference.infer_shapes(onnx_model)

    # Initialize the internal ONNX model representation
    internal_onnx_model = onnx2tflite.src.onnx_parser.onnx_model.ModelProto(parsed_onnx_model)

    # Convert the ONNX model to TFLite
    internal_tflite_model = _convert(build_conversion_context(internal_onnx_model))

    operators = internal_tflite_model.sub_graphs.get_last().operators.vector
    assert len(operators) == len(expected_operators)

    for operator, expected_operator in zip(operators, expected_operators):
        assert operator.builtin_options.operator_type == expected_operator


def test_channels_first_output_annihilation():
    # ONNX: MaxPool -> Transpose -> Transpose
    # TFLite: Transpose -> MaxPool2D -> Transpose
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("MaxPool", ["input"], ["max_pool_out"], kernel_shape=[3, 3],
                                      auto_pad="SAME_UPPER"),
                onnx.helper.make_node("Transpose", ["max_pool_out"], ["transpose1_out"], perm=[0, 2, 3, 1]),
                onnx.helper.make_node("Transpose", ["transpose1_out"], ["output"], perm=[0, 3, 1, 2]),
            ],
            "transpose_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.TRANSPOSE, BuiltinOperator.MAX_POOL_2D, BuiltinOperator.TRANSPOSE]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_output_annihilation():
    # ONNX: Mul -> Transpose -> Transpose
    # TFLite: Mul
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["input", "scalar"], ["mul_out"]),
                onnx.helper.make_node("Transpose", ["mul_out"], ["transpose1_out"], perm=[0, 2, 3, 1]),
                onnx.helper.make_node("Transpose", ["transpose1_out"], ["output"], perm=[0, 3, 1, 2]),
            ],
            "transpose_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_input_annihilation():
    # ONNX: Transpose -> Transpose -> Mul
    # TFLite: Mul
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Transpose", ["input"], ["transpose1_out"], perm=[0, 2, 3, 1]),
                onnx.helper.make_node("Transpose", ["transpose1_out"], ["transpose2_out"], perm=[0, 3, 1, 2]),
                onnx.helper.make_node("Mul", ["transpose2_out", "scalar"], ["output"]),
            ],
            "transpose_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_forked_annihilation():
    #                           / -> Transpose -> \
    # ONNX:  Mul -> Transpose -|                   | -> Mul
    #                           \ -> Transpose  ->/
    # TFLite: Mul -> Mul
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["input", "scalar"], ["mul_out"]),
                onnx.helper.make_node("Transpose", ["mul_out"], ["transpose1_out"], perm=[0, 2, 3, 1]),
                onnx.helper.make_node("Transpose", ["transpose1_out"], ["output1"], perm=[0, 3, 1, 2]),
                onnx.helper.make_node("Transpose", ["transpose1_out"], ["output2"], perm=[0, 3, 1, 2]),
                onnx.helper.make_node("Mul", ["output1", "output2"], ["output"]),
            ],
            "transpose_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.MUL, BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_impossible_forked_annihilation():
    #                           / -> Mul -> Transpose -> \
    # ONNX:  Mul -> Transpose -|                          | -> Mul
    #                           \ ->     Transpose     ->/
    #
    #                             / -> Mul -> Transpose -> \
    # TFLite:  Mul -> Transpose -|                          | -> Mul
    #                             \ ->     Transpose     ->/
    #
    # The Transposes cannot be annihilated!

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["input", "scalar"], ["mul_out"]),
                onnx.helper.make_node("Transpose", ["mul_out"], ["transpose1_out"], perm=[0, 2, 3, 1]),
                onnx.helper.make_node("Mul", ["transpose1_out", "scalar"], ["mul2_out"]),
                onnx.helper.make_node("Transpose", ["mul2_out"], ["output1"], perm=[0, 3, 1, 2]),
                onnx.helper.make_node("Transpose", ["transpose1_out"], ["output2"], perm=[0, 3, 1, 2]),
                onnx.helper.make_node("Mul", ["output1", "output2"], ["output"]),
            ],
            "transpose_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.MUL, BuiltinOperator.TRANSPOSE, BuiltinOperator.MUL,
                          BuiltinOperator.TRANSPOSE, BuiltinOperator.TRANSPOSE, BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_channels_first_output_fusing():
    # ONNX: MaxPool -> Transpose -> Transpose
    # TFLite: Transpose -> MaxPool2D -> Transpose
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("MaxPool", ["input"], ["max_pool_out"], kernel_shape=[3, 3],
                                      auto_pad="SAME_UPPER"),
                onnx.helper.make_node("Transpose", ["max_pool_out"], ["transpose1_out"], perm=[0, 2, 3, 1]),
                onnx.helper.make_node("Transpose", ["transpose1_out"], ["output"], perm=[0, 2, 3, 1]),
            ],
            "transpose_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.TRANSPOSE, BuiltinOperator.MAX_POOL_2D, BuiltinOperator.TRANSPOSE]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_input_fusing():
    # ONNX: Transpose -> Transpose -> Mul
    # TFLite: Transpose -> Mul
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Transpose", ["input"], ["transpose1_out"], perm=[1, 2, 3, 0]),
                onnx.helper.make_node("Transpose", ["transpose1_out"], ["transpose2_out"], perm=[1, 2, 3, 0]),
                onnx.helper.make_node("Mul", ["transpose2_out", "scalar"], ["output"]),
            ],
            "transpose_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.TRANSPOSE, BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)


def test_forked_fusing():
    #                           / -> Transpose -> Mul
    # ONNX:  Mul -> Transpose -|
    #                           \ -> Transpose -> Mul
    #
    #                / -> Transpose -> Mul
    # TFLite:  Mul -|
    #                \ -> Transpose -> Mul
    #

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Mul", ["input", "scalar"], ["mul_out"]),
                onnx.helper.make_node("Transpose", ["mul_out"], ["transpose1_out"], perm=[1, 2, 3, 0]),
                onnx.helper.make_node("Transpose", ["transpose1_out"], ["transpose2_out"], perm=[1, 2, 3, 0]),
                onnx.helper.make_node("Transpose", ["transpose1_out"], ["transpose3_out"], perm=[2, 3, 0, 1]),
                onnx.helper.make_node("Mul", ["transpose2_out", "scalar"], ["output1"]),
                onnx.helper.make_node("Mul", ["transpose3_out", "scalar"], ["output2"]),
            ],
            "transpose_fuse_test",
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 12, 12])],
            [onnx.helper.make_tensor_value_info("output1", TensorProto.FLOAT, ()),
             onnx.helper.make_tensor_value_info("output2", TensorProto.FLOAT, ())],
            initializer=[
                onnx.helper.make_tensor("scalar", TensorProto.FLOAT, [1], [2.0]),
            ]
        )
    )

    input_data = np.arange(np.prod([1, 3, 12, 12])).reshape([1, 3, 12, 12]).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    expected_operators = [BuiltinOperator.MUL, BuiltinOperator.TRANSPOSE, BuiltinOperator.TRANSPOSE,
                          BuiltinOperator.MUL,
                          BuiltinOperator.MUL]
    _assert_converted_model_has_operators(onnx_model, expected_operators)
