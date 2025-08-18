#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math

import numpy as np
import pytest

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.builder import model_builder
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.add_options import Add
from onnx2tflite.src.tflite_generator.builtin_options.max_pool_2d_options import MaxPool2D
from onnx2tflite.src.tflite_generator.builtin_options.quantize_options import Quantize
from tests import model_builder_utils


# ONNX QuantizeLinear doesn't support re-quantization, so testing cannot be done like for Reshape and Transpose fusing.

@pytest.mark.parametrize("quantization_1, quantization_2", [
    pytest.param(
        (TensorType.INT8, np.array(0.1, np.float32), np.array(1, np.int8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(129, np.uint8)), id="UINT8 output"),
    pytest.param(
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(55, np.uint8)),
        (TensorType.INT8, np.array(0.1, np.float32), np.array(-73, np.int8)), id="INT8 output"),
])
def test_fusion_recasting_from_float(quantization_1, quantization_2):
    # Before: MaxPool2D -> Quantize -> Quantize
    # After:  MaxPool2D -> Quantize

    expected_operators = [BuiltinOperator.MAX_POOL_2D, BuiltinOperator.QUANTIZE]

    shape = [1, 12, 12, 3]

    builder = model_builder.ModelBuilder(3, "test")

    output_tensor = model_builder_utils.create_tensor("output_tensor", builder, shape, *quantization_2)
    input_tensor = model_builder_utils.create_tensor("input_tensor", builder, shape, TensorType.FLOAT32)
    max_pool_out = model_builder_utils.create_tensor("max_pool_out", builder, shape, TensorType.FLOAT32)
    quantize1_out = model_builder_utils.create_tensor("quantize1_out", builder, shape, *quantization_1)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.append(output_tensor)

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(input_tensor)

    max_pool_2d = model_builder_utils.create_operator(MaxPool2D(), [input_tensor], [max_pool_out])
    quantize_1 = model_builder_utils.create_operator(Quantize(), [max_pool_out], [quantize1_out])
    quantize_2 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [output_tensor])

    model_builder_utils.assign_operators(builder, [max_pool_2d, quantize_1, quantize_2])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.random.random(math.prod(shape)).reshape(shape).astype(np.float32)
    model_builder_utils.run_model(model, input_data)


@pytest.mark.parametrize("quantization_1, quantization_2", [
    pytest.param(
        (TensorType.INT8, np.array(0.1, np.float32), np.array(1, np.int8)),
        (TensorType.UINT8, np.array(0.2, np.float32), np.array(129, np.uint8)), id="Non-matching scale"),
    pytest.param(
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(55, np.uint8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(60, np.uint8)), id="Incorrect types"),
])
def test_no_fusion_recasting_from_float(quantization_1, quantization_2):
    # Before: MaxPool2D -> Quantize -> Quantize
    # After:  MaxPool2D -> Quantize -> Quantize

    expected_operators = [BuiltinOperator.MAX_POOL_2D, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE]

    shape = [1, 12, 12, 3]

    builder = model_builder.ModelBuilder(3, "test")

    output_tensor = model_builder_utils.create_tensor("output_tensor", builder, shape, *quantization_2)
    input_tensor = model_builder_utils.create_tensor("input_tensor", builder, shape, TensorType.FLOAT32)
    max_pool_out = model_builder_utils.create_tensor("max_pool_out", builder, shape, TensorType.FLOAT32)
    quantize1_out = model_builder_utils.create_tensor("quantize1_out", builder, shape, *quantization_1)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.append(output_tensor)

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(input_tensor)

    max_pool_2d = model_builder_utils.create_operator(MaxPool2D(), [input_tensor], [max_pool_out])
    quantize_1 = model_builder_utils.create_operator(Quantize(), [max_pool_out], [quantize1_out])
    quantize_2 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [output_tensor])

    model_builder_utils.assign_operators(builder, [max_pool_2d, quantize_1, quantize_2])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.random.random(math.prod(shape)).reshape(shape).astype(np.float32)
    model_builder_utils.run_model(model, input_data)


def test_forked_fusion_recasting_from_float():
    #                                  / -> Quantize
    # Before:  MaxPool2D -> Quantize -|
    #                                  \ -> Quantize
    #
    #                     / -> Quantize
    # After:  MaxPool2D -|
    #                     \ -> Quantize
    #

    expected_operators = [BuiltinOperator.MAX_POOL_2D, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE]

    shape = [1, 12, 12, 3]

    builder = model_builder.ModelBuilder(3, "test")

    quantization_1 = TensorType.INT8, np.array(0.1, np.float32), np.array(-125, np.int8)
    quantization_2 = TensorType.UINT8, np.array(0.1, np.float32), np.array(3, np.uint8)
    quantization_3 = TensorType.UINT8, np.array(0.1, np.float32), np.array(3, np.uint8)

    output1_tensor = model_builder_utils.create_tensor("output1_tensor", builder, shape, *quantization_2)
    output2_tensor = model_builder_utils.create_tensor("output2_tensor", builder, shape, *quantization_3)
    input_tensor = model_builder_utils.create_tensor("input_tensor", builder, shape, TensorType.FLOAT32)
    max_pool_out = model_builder_utils.create_tensor("max_pool_out", builder, shape, TensorType.FLOAT32)
    quantize1_out = model_builder_utils.create_tensor("quantize1_out", builder, shape, *quantization_1)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.extend([output1_tensor, output2_tensor])

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(input_tensor)

    max_pool_2d = model_builder_utils.create_operator(MaxPool2D(), [input_tensor], [max_pool_out])
    quantize_1 = model_builder_utils.create_operator(Quantize(), [max_pool_out], [quantize1_out])
    quantize_2 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [output1_tensor])
    quantize_3 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [output2_tensor])

    model_builder_utils.assign_operators(builder, [max_pool_2d, quantize_1, quantize_2, quantize_3])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.random.random(math.prod(shape)).reshape(shape).astype(np.float32)
    model_builder_utils.run_model(model, input_data)


@pytest.mark.parametrize("quantization_1, quantization_2, quantization_3", [
    pytest.param(
        (TensorType.INT8, np.array(0.1, np.float32), np.array(-125, np.int8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(3, np.uint8)),
        (TensorType.UINT8, np.array(0.2, np.float32), np.array(3, np.uint8)), id="Non-matching scale"),
    pytest.param(
        (TensorType.INT8, np.array(0.1, np.float32), np.array(-125, np.int8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(3, np.uint8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(2, np.uint8)), id="Incorrect zp"),
    pytest.param(
        (TensorType.INT8, np.array(0.1, np.float32), np.array(-125, np.int8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(3, np.uint8)),
        (TensorType.INT8, np.array(0.1, np.float32), np.array(3, np.uint8)), id="Incorrect types"),
])
def test_forked_no_fusion_recasting_from_float(quantization_1, quantization_2, quantization_3):
    #                                  / -> Quantize
    # Before:  MaxPool2D -> Quantize -|
    #                                  \ -> Quantize
    #
    #                                 / -> Quantize
    # After:  MaxPool2D -> Quantize -|
    #                                 \ -> Quantize
    #

    expected_operators = [
        BuiltinOperator.MAX_POOL_2D, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE]

    shape = [1, 12, 12, 3]

    builder = model_builder.ModelBuilder(3, "test")

    output1_tensor = model_builder_utils.create_tensor("output1_tensor", builder, shape, *quantization_2)
    output2_tensor = model_builder_utils.create_tensor("output2_tensor", builder, shape, *quantization_3)
    input_tensor = model_builder_utils.create_tensor("input_tensor", builder, shape, TensorType.FLOAT32)
    max_pool_out = model_builder_utils.create_tensor("max_pool_out", builder, shape, TensorType.FLOAT32)
    quantize1_out = model_builder_utils.create_tensor("quantize1_out", builder, shape, *quantization_1)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.extend([output1_tensor, output2_tensor])

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(input_tensor)

    max_pool_2d = model_builder_utils.create_operator(MaxPool2D(), [input_tensor], [max_pool_out])
    quantize_1 = model_builder_utils.create_operator(Quantize(), [max_pool_out], [quantize1_out])
    quantize_2 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [output1_tensor])
    quantize_3 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [output2_tensor])

    model_builder_utils.assign_operators(builder, [max_pool_2d, quantize_1, quantize_2, quantize_3])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.random.random(math.prod(shape)).reshape(shape).astype(np.float32)
    model_builder_utils.run_model(model, input_data)


@pytest.mark.parametrize("quantization_1, quantization_2", [
    pytest.param(
        (TensorType.INT8, np.array(0.1, np.float32), np.array(1, np.int8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(129, np.uint8)), id="UINT8 output"),
    pytest.param(
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(55, np.uint8)),
        (TensorType.INT8, np.array(0.1, np.float32), np.array(-73, np.int8)), id="INT8 output"),
])
def test_annihilation_recasting_from_integer(quantization_1, quantization_2):
    # Before: MaxPool2D -> Quantize -> Quantize
    # After:  MaxPool2D

    expected_operators = [BuiltinOperator.MAX_POOL_2D]

    shape = [1, 12, 12, 3]

    builder = model_builder.ModelBuilder(3, "test")

    output_tensor = model_builder_utils.create_tensor("output_tensor", builder, shape, *quantization_1)
    input_tensor = model_builder_utils.create_tensor("input_tensor", builder, shape, *quantization_1)
    max_pool_out = model_builder_utils.create_tensor("max_pool_out", builder, shape, *quantization_1)
    quantize1_out = model_builder_utils.create_tensor("quantize1_out", builder, shape, *quantization_2)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.append(output_tensor)

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(input_tensor)

    max_pool_2d = model_builder_utils.create_operator(MaxPool2D(), [input_tensor], [max_pool_out])
    quantize_1 = model_builder_utils.create_operator(Quantize(), [max_pool_out], [quantize1_out])
    quantize_2 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [output_tensor])

    model_builder_utils.assign_operators(builder, [max_pool_2d, quantize_1, quantize_2])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(quantization_1[2].dtype)
    model_builder_utils.run_model(model, input_data)


@pytest.mark.parametrize("quantization_1, quantization_2", [
    pytest.param(
        (TensorType.INT8, np.array(0.1, np.float32), np.array(1, np.int8)),
        (TensorType.UINT8, np.array(0.2, np.float32), np.array(129, np.uint8)), id="Non-matching scale"),
    pytest.param(
        (TensorType.INT8, np.array(0.1, np.float32), np.array(1, np.int8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(50, np.uint8)), id="Incorrect zp"),
    pytest.param(
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(55, np.uint8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(60, np.uint8)), id="Incorrect types"),
])
def test_no_annihilation_recasting_from_integer(quantization_1, quantization_2):
    # Before: MaxPool2D -> Quantize -> Quantize
    # After:  MaxPool2D -> Quantize -> Quantize

    expected_operators = [BuiltinOperator.MAX_POOL_2D, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE]

    shape = [1, 12, 12, 3]

    builder = model_builder.ModelBuilder(3, "test")

    output_tensor = model_builder_utils.create_tensor("output_tensor", builder, shape, *quantization_1)
    input_tensor = model_builder_utils.create_tensor("input_tensor", builder, shape, *quantization_1)
    max_pool_out = model_builder_utils.create_tensor("max_pool_out", builder, shape, *quantization_1)
    quantize1_out = model_builder_utils.create_tensor("quantize1_out", builder, shape, *quantization_2)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.append(output_tensor)

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(input_tensor)

    max_pool_2d = model_builder_utils.create_operator(MaxPool2D(), [input_tensor], [max_pool_out])
    quantize_1 = model_builder_utils.create_operator(Quantize(), [max_pool_out], [quantize1_out])
    quantize_2 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [output_tensor])

    model_builder_utils.assign_operators(builder, [max_pool_2d, quantize_1, quantize_2])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(quantization_1[2].dtype)
    model_builder_utils.run_model(model, input_data)


def test_forked_annihilation_recasting_from_integer():
    #                                  / -> Quantize -> MaxPool2D
    # Before:  MaxPool2D -> Quantize -|
    #                                  \ -> Quantize -> MaxPool2D
    #
    #                     / -> MaxPool2D
    # After:  MaxPool2D -|
    #                     \ -> MaxPool2D
    #

    expected_operators = [BuiltinOperator.MAX_POOL_2D, BuiltinOperator.MAX_POOL_2D, BuiltinOperator.MAX_POOL_2D]

    shape = [1, 12, 12, 3]

    builder = model_builder.ModelBuilder(3, "test")

    quantization_1 = TensorType.INT8, np.array(0.1, np.float32), np.array(1, np.int8)
    quantization_2 = TensorType.UINT8, np.array(0.1, np.float32), np.array(129, np.uint8)

    output1_tensor = model_builder_utils.create_tensor("output1_tensor", builder, shape, *quantization_1)
    output2_tensor = model_builder_utils.create_tensor("output2_tensor", builder, shape, *quantization_1)
    input_tensor = model_builder_utils.create_tensor("input_tensor", builder, shape, *quantization_1)
    max_pool_out = model_builder_utils.create_tensor("max_pool_out", builder, shape, *quantization_1)

    quantize1_out = model_builder_utils.create_tensor("quantize1_out", builder, shape, *quantization_2)

    quantize2_out = model_builder_utils.create_tensor("quantize2_out", builder, shape, *quantization_1)
    quantize3_out = model_builder_utils.create_tensor("quantize3_out", builder, shape, *quantization_1)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.extend([output1_tensor, output2_tensor])

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(input_tensor)

    max_pool_2d_1 = model_builder_utils.create_operator(MaxPool2D(), [input_tensor], [max_pool_out])
    quantize_1 = model_builder_utils.create_operator(Quantize(), [max_pool_out], [quantize1_out])
    quantize_2 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [quantize2_out])
    quantize_3 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [quantize3_out])
    max_pool_2d_2 = model_builder_utils.create_operator(MaxPool2D(), [quantize2_out], [output1_tensor])
    max_pool_2d_3 = model_builder_utils.create_operator(MaxPool2D(), [quantize3_out], [output2_tensor])

    model_builder_utils.assign_operators(builder, [max_pool_2d_1, quantize_1, quantize_2, quantize_3, max_pool_2d_2,
                                                   max_pool_2d_3])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.int8)
    model_builder_utils.run_model(model, input_data)


@pytest.mark.parametrize("quantization_1, quantization_2, quantization_3", [
    pytest.param(
        (TensorType.INT8, np.array(0.1, np.float32), np.array(-125, np.int8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(3, np.uint8)),
        (TensorType.UINT8, np.array(0.2, np.float32), np.array(3, np.uint8)), id="Non-matching scale"),
    pytest.param(
        (TensorType.INT8, np.array(0.1, np.float32), np.array(-125, np.int8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(3, np.uint8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(2, np.uint8)), id="Incorrect zp"),
    pytest.param(
        (TensorType.INT8, np.array(0.1, np.float32), np.array(-125, np.int8)),
        (TensorType.UINT8, np.array(0.1, np.float32), np.array(3, np.uint8)),
        (TensorType.INT8, np.array(0.1, np.float32), np.array(3, np.uint8)), id="Incorrect types"),
])
def test_forked_no_annihilation_recasting_from_integer(quantization_1, quantization_2, quantization_3):
    #                                  / -> Quantize -> MaxPool2D
    # Before:  MaxPool2D -> Quantize -|
    #                                  \ -> Quantize -> MaxPool2D
    #
    #                                 / -> Quantize -> MaxPool2D
    # After:  MaxPool2D -> Quantize -|
    #                                 \ -> Quantize -> MaxPool2D
    #

    expected_operators = [BuiltinOperator.MAX_POOL_2D, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
                          BuiltinOperator.QUANTIZE, BuiltinOperator.MAX_POOL_2D, BuiltinOperator.MAX_POOL_2D]

    shape = [1, 12, 12, 3]

    builder = model_builder.ModelBuilder(3, "test")

    output1_tensor = model_builder_utils.create_tensor("output1_tensor", builder, shape, *quantization_2)
    output2_tensor = model_builder_utils.create_tensor("output2_tensor", builder, shape, *quantization_3)
    input_tensor = model_builder_utils.create_tensor("input_tensor", builder, shape, *quantization_1)
    max_pool_out = model_builder_utils.create_tensor("max_pool_out", builder, shape, *quantization_1)

    quantize1_out = model_builder_utils.create_tensor("quantize1_out", builder, shape, *quantization_2)

    quantize2_out = model_builder_utils.create_tensor("quantize2_out", builder, shape, *quantization_2)
    quantize3_out = model_builder_utils.create_tensor("quantize3_out", builder, shape, *quantization_3)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.extend([output1_tensor, output2_tensor])

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(input_tensor)

    max_pool_2d_1 = model_builder_utils.create_operator(MaxPool2D(), [input_tensor], [max_pool_out])
    quantize_1 = model_builder_utils.create_operator(Quantize(), [max_pool_out], [quantize1_out])
    quantize_2 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [quantize2_out])
    quantize_3 = model_builder_utils.create_operator(Quantize(), [quantize1_out], [quantize3_out])
    max_pool_2d_2 = model_builder_utils.create_operator(MaxPool2D(), [quantize2_out], [output1_tensor])
    max_pool_2d_3 = model_builder_utils.create_operator(MaxPool2D(), [quantize3_out], [output2_tensor])

    model_builder_utils.assign_operators(builder, [max_pool_2d_1, quantize_1, quantize_2, quantize_3, max_pool_2d_2,
                                                   max_pool_2d_3])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.int8)
    model_builder_utils.run_model(model, input_data)


def test_parallel_fusing():
    #                 / -> Quantize -> Add
    # Before:  Add  -|
    #                 \ -> Quantize -> Add
    #
    #                            / -> Add
    # Before:  Add -> Quantize -|
    #                            \ -> Add

    expected_operators = [BuiltinOperator.ADD, BuiltinOperator.QUANTIZE, BuiltinOperator.ADD, BuiltinOperator.ADD]

    shape = [42]

    builder = model_builder.ModelBuilder(3, "test")

    quantization = TensorType.INT8, np.array(0.1, np.float32), np.array(10, np.int8)

    y1 = model_builder_utils.create_tensor("y1", builder, shape, *quantization)
    y2 = model_builder_utils.create_tensor("y2", builder, shape, *quantization)
    x = model_builder_utils.create_tensor("x", builder, shape, TensorType.FLOAT32)
    x1 = model_builder_utils.create_tensor("x1", builder, shape, TensorType.FLOAT32)
    x2 = model_builder_utils.create_tensor("x2", builder, shape, *quantization)
    x3 = model_builder_utils.create_tensor("x3", builder, shape, *quantization)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.extend([y1, y2])

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(x)

    add1 = model_builder_utils.create_operator(Add(), [x, x], [x1])
    quantize1 = model_builder_utils.create_operator(Quantize(), [x1], [x2])
    quantize2 = model_builder_utils.create_operator(Quantize(), [x1], [x3])
    add2 = model_builder_utils.create_operator(Add(), [x2, x2], [y1])
    add3 = model_builder_utils.create_operator(Add(), [x3, x3], [y2])

    model_builder_utils.assign_operators(builder, [add1, quantize1, quantize2, add2, add3])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.random.random(math.prod(shape)).reshape(shape).astype(np.float32)
    model_builder_utils.run_model(model, input_data)


def test_parallel_fusing__3_in_parallel():
    #                 / -> Quantize -> Add
    # Before:  Add  -|  -> Quantize -> Add
    #                 \ -> Quantize -> Add
    #
    #                            / -> Add
    # Before:  Add -> Quantize -|  -> Add
    #                            \ -> Add

    expected_operators = [BuiltinOperator.ADD, BuiltinOperator.QUANTIZE, BuiltinOperator.ADD, BuiltinOperator.ADD,
                          BuiltinOperator.ADD]

    shape = [42]

    builder = model_builder.ModelBuilder(3, "test")

    quantization = TensorType.INT8, np.array(0.1, np.float32), np.array(10, np.int8)

    y1 = model_builder_utils.create_tensor("y1", builder, shape, *quantization)
    y2 = model_builder_utils.create_tensor("y2", builder, shape, *quantization)
    y3 = model_builder_utils.create_tensor("y3", builder, shape, *quantization)
    x = model_builder_utils.create_tensor("x", builder, shape, TensorType.FLOAT32)
    x1 = model_builder_utils.create_tensor("x1", builder, shape, TensorType.FLOAT32)
    x2 = model_builder_utils.create_tensor("x2", builder, shape, *quantization)
    x3 = model_builder_utils.create_tensor("x3", builder, shape, *quantization)
    x4 = model_builder_utils.create_tensor("x4", builder, shape, *quantization)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.extend([y1, y2, y3])

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(x)

    add1 = model_builder_utils.create_operator(Add(), [x, x], [x1])
    quantize1 = model_builder_utils.create_operator(Quantize(), [x1], [x2])
    quantize2 = model_builder_utils.create_operator(Quantize(), [x1], [x3])
    quantize3 = model_builder_utils.create_operator(Quantize(), [x1], [x4])
    add2 = model_builder_utils.create_operator(Add(), [x2, x2], [y1])
    add3 = model_builder_utils.create_operator(Add(), [x3, x3], [y2])
    add4 = model_builder_utils.create_operator(Add(), [x4, x4], [y3])

    model_builder_utils.assign_operators(builder, [add1, quantize1, quantize2, quantize3, add2, add3, add4])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.random.random(math.prod(shape)).reshape(shape).astype(np.float32)
    model_builder_utils.run_model(model, input_data)


def test_parallel_fusing__intermediate_model_output():
    #                    / -> Quantize -> -> Add
    #                   /
    # Before:    Add  -|
    #                   \                / -> Add
    #                    \ -> Quantize -|
    #                                    \ -> model output
    #
    #                                / -> Add
    #                               / -> Add
    # After:     Add  -> Quantize -|
    #                               \ -> model output

    expected_operators = [BuiltinOperator.ADD,
                          BuiltinOperator.QUANTIZE,
                          BuiltinOperator.ADD, BuiltinOperator.ADD]

    shape = [42]

    builder = model_builder.ModelBuilder(3, "test")

    quantization = TensorType.INT8, np.array(0.1, np.float32), np.array(10, np.int8)

    y1 = model_builder_utils.create_tensor("y1", builder, shape, *quantization)
    y2 = model_builder_utils.create_tensor("y2", builder, shape, *quantization)
    x = model_builder_utils.create_tensor("x", builder, shape, TensorType.FLOAT32)
    x1 = model_builder_utils.create_tensor("x1", builder, shape, TensorType.FLOAT32)
    x2 = model_builder_utils.create_tensor("x2", builder, shape, *quantization)
    x3 = model_builder_utils.create_tensor("x3", builder, shape, *quantization)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.extend([y1, y2, x3])

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(x)

    add1 = model_builder_utils.create_operator(Add(), [x, x], [x1])
    quantize1 = model_builder_utils.create_operator(Quantize(), [x1], [x2])
    quantize2 = model_builder_utils.create_operator(Quantize(), [x1], [x3])
    add2 = model_builder_utils.create_operator(Add(), [x2, x2], [y1])
    add3 = model_builder_utils.create_operator(Add(), [x3, x3], [y2])

    model_builder_utils.assign_operators(builder, [add1, quantize1, quantize2, add2, add3])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.random.random(math.prod(shape)).reshape(shape).astype(np.float32)
    model_builder_utils.run_model(model, input_data)


def test_parallel_fusing__intermediate_model_outputs():
    # No optimization can be done.
    #                            / -> Add
    #         / -> Quantize -> -|
    #        /                   \ -> model output
    #  Add  -|
    #         \                / -> Add
    #          \ -> Quantize -|
    #                          \ -> model output

    expected_operators = [BuiltinOperator.ADD,
                          BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,  # 2 `Quantize` ops.
                          BuiltinOperator.ADD, BuiltinOperator.ADD]

    shape = [42]

    builder = model_builder.ModelBuilder(3, "test")

    quantization = TensorType.INT8, np.array(0.1, np.float32), np.array(10, np.int8)

    y1 = model_builder_utils.create_tensor("y1", builder, shape, *quantization)
    y2 = model_builder_utils.create_tensor("y2", builder, shape, *quantization)
    x = model_builder_utils.create_tensor("x", builder, shape, TensorType.FLOAT32)
    x1 = model_builder_utils.create_tensor("x1", builder, shape, TensorType.FLOAT32)
    x2 = model_builder_utils.create_tensor("x2", builder, shape, *quantization)
    x3 = model_builder_utils.create_tensor("x3", builder, shape, *quantization)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.extend([y1, y2, x3, x2])

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(x)

    add1 = model_builder_utils.create_operator(Add(), [x, x], [x1])
    quantize1 = model_builder_utils.create_operator(Quantize(), [x1], [x2])
    quantize2 = model_builder_utils.create_operator(Quantize(), [x1], [x3])
    add2 = model_builder_utils.create_operator(Add(), [x2, x2], [y1])
    add3 = model_builder_utils.create_operator(Add(), [x3, x3], [y2])

    model_builder_utils.assign_operators(builder, [add1, quantize1, quantize2, add2, add3])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.random.random(math.prod(shape)).reshape(shape).astype(np.float32)
    model_builder_utils.run_model(model, input_data)


def test_parallel_fusing__different_quantization():
    #         / -> Quantize -> Add
    #  Add  -|
    #         \ -> Quantize -> Add

    expected_operators = [BuiltinOperator.ADD,
                          BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,  # 2 `Quantize` ops.
                          BuiltinOperator.ADD, BuiltinOperator.ADD]

    shape = [42]

    builder = model_builder.ModelBuilder(3, "test")

    quantization1 = TensorType.INT8, np.array(0.1, np.float32), np.array(10, np.int8)
    quantization2 = TensorType.INT8, np.array(0.01, np.float32), np.array(-10, np.int8)

    y1 = model_builder_utils.create_tensor("y1", builder, shape, *quantization1)
    y2 = model_builder_utils.create_tensor("y2", builder, shape, *quantization2)
    x = model_builder_utils.create_tensor("x", builder, shape, TensorType.FLOAT32)
    x1 = model_builder_utils.create_tensor("x1", builder, shape, TensorType.FLOAT32)
    x2 = model_builder_utils.create_tensor("x2", builder, shape, *quantization1)
    x3 = model_builder_utils.create_tensor("x3", builder, shape, *quantization2)

    builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
    builder.get_sub_graph().outputs.tmp_outputs.extend([y1, y2])

    builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
    builder.get_sub_graph().inputs.tmp_inputs.append(x)

    add1 = model_builder_utils.create_operator(Add(), [x, x], [x1])
    quantize1 = model_builder_utils.create_operator(Quantize(), [x1], [x2])
    quantize2 = model_builder_utils.create_operator(Quantize(), [x1], [x3])
    add2 = model_builder_utils.create_operator(Add(), [x2, x2], [y1])
    add3 = model_builder_utils.create_operator(Add(), [x3, x3], [y2])

    model_builder_utils.assign_operators(builder, [add1, quantize1, quantize2, add2, add3])

    model = builder.finish()

    model_builder_utils.assert_model_contains_operators(model, expected_operators)

    input_data = np.random.random(math.prod(shape)).reshape(shape).astype(np.float32)
    model_builder_utils.run_model(model, input_data)
