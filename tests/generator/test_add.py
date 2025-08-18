#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers
import numpy as np

import onnx2tflite.src.tflite_generator.builtin_options.add_options as add_options
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from tests.executors import TFLiteExecutor


def test_add_operator_returns_sum_of_4x4_tensors():
    builder = flatbuffers.Builder(3400)

    op_codes = tflite_model.OperatorCodes([
        tflite_model.OperatorCode(BuiltinOperator.ADD)
    ])

    operators = tflite_model.Operators([
        tflite_model.Operator(
            tflite_model.OperatorInputs([1, 2]), tflite_model.OperatorOutputs([0]),
            add_options.Add())
    ])

    buffers = tflite_model.Buffers([
        tflite_model.Buffer(),
        tflite_model.Buffer(),
        tflite_model.Buffer(),
    ])

    tensors = tflite_model.Tensors([
        tflite_model.Tensor(tflite_model.Shape([4, 4]), "input_1", 1),
        tflite_model.Tensor(tflite_model.Shape([4, 4]), "input_2", 2),
        tflite_model.Tensor(tflite_model.Shape([4, 4]), "output", 0),
    ])

    subgraphs = tflite_model.SubGraphs([
        tflite_model.SubGraph(tflite_model.SubGraphInputs([1, 2]), tflite_model.SubGraphOutputs([0]), tensors,
                              operators)]
    )

    model = tflite_model.Model(3, "Test Add operator", buffers, op_codes, subgraphs)
    model.gen_tflite(builder)

    executor = TFLiteExecutor(model_content=bytes(builder.Output()))
    input_data = {
        0: np.arange(16, dtype=np.float32).reshape((4, 4)),
        1: np.ones((4, 4), dtype=np.float32) + 1.
    }
    output_data = executor.inference(input_data)

    assert np.equal(output_data, input_data[0] + input_data[1]).all()
