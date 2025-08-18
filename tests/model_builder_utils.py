#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import List, Dict, Union

import flatbuffers
import numpy as np

from onnx2tflite.src.converter.quantization_utils import set_quantization_parameters_to_tensor
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.builder import model_builder
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta import meta
from tests.executors import TFLiteExecutor


def create_tensor(name: str, builder: model_builder.ModelBuilder, shape: List[int], tensor_type: TensorType,
                  scale: np.ndarray = None, zero_point: np.ndarray = None) -> tflite_model.Tensor:
    tensor = builder.tensor_for_name(name)
    tensor.type = tensor_type
    tensor.shape = tflite_model.Shape(shape)
    if scale is not None and zero_point is not None:
        set_quantization_parameters_to_tensor(tensor, scale, zero_point)

    return tensor


def create_operator(builtin_options: meta.BuiltinOptions, inputs: List[tflite_model.Tensor],
                    outputs: List[tflite_model.Tensor]) -> tflite_model.Operator:
    operator = tflite_model.Operator(builtin_options=builtin_options)
    operator.tmp_inputs = inputs
    operator.tmp_outputs = outputs

    return operator


def assign_operators(builder: model_builder.ModelBuilder, operators: List[tflite_model.Operator]):
    for op in operators:
        op.opcode_index = builder.op_code_index_for_op_type(op.builtin_options.operator_type)

    builder.get_operators().vector.extend(operators)


def assert_model_contains_operators(tfl_model: tflite_model.Model, expected_operators: List[BuiltinOperator]):
    ops = tfl_model.sub_graphs.get_last().operators.vector

    assert len(ops) == len(expected_operators)

    for op, expected_op in zip(ops, expected_operators):
        assert op.builtin_options.operator_type == expected_op


def run_model(tfl_model: tflite_model.Model, input_data: Union[np.ndarray, Dict[str, np.ndarray]]):
    fb_builder = flatbuffers.Builder()
    tfl_model.gen_tflite(fb_builder)
    bytes_model = bytes(fb_builder.Output())

    tflite_executor = TFLiteExecutor(model_content=bytes_model)
    tflite_executor.inference(input_data)
