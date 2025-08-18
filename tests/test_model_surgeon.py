#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import os
import pathlib
from typing import cast

import numpy as np
import onnx
import pytest
from onnx import TensorProto
from onnx import numpy_helper

from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type
from tests.executors import TFLiteExecutor, OnnxExecutor, convert_run_compare
from tests.model_surgeon import ONNXSurgeon

_ARTIFACTS_DIR = pathlib.Path(__file__).parent.joinpath("artifacts")


def test_intermediate_tensors_extraction():
    model_dir = os.path.join(_ARTIFACTS_DIR, "downloaded", "bvlcalexnet-12")
    model_path = os.path.join(model_dir, "model.onnx")

    model = ONNXSurgeon().intermediate_tensors_as_outputs(model_path, "conv[1-2]_2")
    converted_tflite = bytes(convert.convert_model(model))

    input_data = numpy_helper.to_array(onnx.load_tensor(os.path.join(model_dir, "test_data_set_0", "input_0.pb")))

    onnx_executor = OnnxExecutor(model.SerializeToString())
    output_onnx = onnx_executor.inference(input_data)

    tflite_executor = TFLiteExecutor(model_content=converted_tflite)
    output_tflite = tflite_executor.inference(input_data)

    assert np.allclose(output_onnx["prob_1"], output_tflite["prob_1"])
    assert np.allclose(output_onnx["conv1_2"], output_tflite["conv1_2"], atol=1e-4)
    assert np.allclose(output_onnx["conv2_2"], output_tflite["conv2_2"], atol=1e-3)


def test_pick_out_operators_with_custom_model():
    input_shape = [5, 10, 15, 20]
    weights_shape = [input_shape[1]] + [input_shape[1]] + [3, 3]
    bias_shape = [input_shape[1]]
    flat_shape = [input_shape[0] * input_shape[1], input_shape[2] * input_shape[3]]
    print(flat_shape)
    multiplier_shape = [1]
    multiplier_data = np.asarray([4.2], np.float32)

    weights_data = np.arange(np.prod(weights_shape)).reshape(weights_shape).astype(np.float32)
    bias_data = np.arange(np.prod(bias_shape)).reshape(bias_shape).astype(np.float32)

    original_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("Add", ['x1', 'x2'], ['o0']),
                onnx.helper.make_node("Conv", ['o0', 'w1', 'b1'], ['o1'], kernel_shape=[3, 3], auto_pad="SAME_UPPER"),
                onnx.helper.make_node("Relu", ['o1'], ['o2']),
                onnx.helper.make_node("Dropout", ['o2'], ['o3']),
                onnx.helper.make_node("Reshape", ['o3', 'flat_shape'], ['o4']),
                onnx.helper.make_node("Mul", ['o4', 'multiplier'], ['o5']),
                onnx.helper.make_node("Softmax", ['o5'], ['o']),
            ],
            inputs=[
                onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, input_shape),
                onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, input_shape),
                onnx.helper.make_tensor_value_info('multiplier', TensorProto.FLOAT, multiplier_shape),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, flat_shape),
                onnx.helper.make_tensor_value_info('o4', TensorProto.FLOAT, flat_shape),
            ],
            initializer=[
                onnx.helper.make_tensor('w1', TensorProto.FLOAT, weights_shape, weights_data),
                onnx.helper.make_tensor('b1', TensorProto.FLOAT, bias_shape, bias_data),
                onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [2], flat_shape),
            ],
            name='pick_out_operators() test'
        )
    )

    onnx.checker.check_model(original_model)

    reduced_model = ONNXSurgeon().pick_out_operators(original_model, slice(1, -1))

    assert (len(reduced_model.graph.node) == len(original_model.graph.node) - 2)
    assert reduced_model.graph.node[0].op_type == original_model.graph.node[1].op_type
    assert reduced_model.graph.node[-1].op_type == original_model.graph.node[-2].op_type

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: multiplier_data
    }

    convert_run_compare(reduced_model, input_data)


def generate_input_data(model: onnx.ModelProto):
    np.random.seed(42)
    inputs = [
        {
            'shape': [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim],
            'type': to_numpy_type(cast(onnx.TensorProto.DataType, input_tensor.type.tensor_type.elem_type))
        } for input_tensor in model.graph.input
    ]
    input_data = {
        i: np.random.rand(*t['shape']).astype(t['type']) for i, t in enumerate(inputs)
    }
    return input_data


@pytest.mark.parametrize("custom_slice, atol", [
    (slice(0, None), 1.e-8),  # Full model
    (slice(14, -123), 1.e-5),
    (slice(-50, -10), 1.e-5),
    (slice(42, 50, 2), 1.7e-6),  # Not sure if this kind of slice would ever be useful, but it works.
], ids=lambda x: f"slice = [{x.start}:{x.stop or ''}:{x.step or ''}]" if isinstance(x, slice) else '')
def test_pick_out_operators_with_real_model(custom_slice: slice, atol: float | None):
    # Using resnet, because it is not just simple feed forward.
    model_dir = os.path.join(_ARTIFACTS_DIR, "downloaded", "resnet50-caffe2-v1-9")
    model_path = os.path.join(model_dir, "model.onnx")

    model = ONNXSurgeon().pick_out_operators(model_path, custom_slice)

    input_data = generate_input_data(model)

    convert_run_compare(model, input_data, atol=atol)


@pytest.mark.parametrize("input_tensors,output_tensors,atol,ops_count", [
    pytest.param(["gpu_0/res_conv1_bn_2"], ["gpu_0/res2_1_branch2a_1"], 1e-5, 14, id="Single input"),
    pytest.param(["gpu_0/res2_0_branch2a_1", "gpu_0/res2_0_branch1_1"],
                 ["gpu_0/res2_1_branch2a_1"], 1e-5, 11, id="Multiple inputs"),
])
def test_extract_subgraph_resnet(input_tensors: list, output_tensors: list, atol, ops_count: int):
    model_dir = os.path.join(_ARTIFACTS_DIR, "downloaded", "resnet50-caffe2-v1-9")
    model_path = os.path.join(model_dir, "model.onnx")

    model = ONNXSurgeon().extract_subgraph(model_path, input_tensors, output_tensors)

    input_data = generate_input_data(model)

    assert len(model.graph.node) == ops_count

    convert_run_compare(model, input_data, atol=atol)


def test_extract_subgraph_mnist__default_output_tensors():
    # Extracted part of mnist contains branch with Reshape op (initializer input).
    # This operator must be preserved.
    model_dir = os.path.join(_ARTIFACTS_DIR, "downloaded", "mnist-12")
    model_path = os.path.join(model_dir, "model.onnx")

    model = ONNXSurgeon().extract_subgraph(model_path, ["ReLU114_Output_0"])

    input_data = generate_input_data(model)

    assert len(model.graph.node) == 5

    convert_run_compare(model, input_data)


def test_extract_subgraph_mnist__initializer_input():
    # Extracted part of mnist contains branch with Reshape op (initializer input).
    # This operator must be preserved.
    model_dir = os.path.join(_ARTIFACTS_DIR, "downloaded", "mnist-12")
    model_path = os.path.join(model_dir, "model.onnx")

    model = onnx.load_model(model_path)

    # Expose initializer as input (some models do this) to check
    # if initializer isn't converted to input of extracted model
    model.graph.input.append(onnx.helper.make_tensor_value_info("Parameter193", onnx.TensorProto.FLOAT, [16, 4, 4, 10]))

    extracted_model = ONNXSurgeon().extract_subgraph(model, ["ReLU114_Output_0"])
    input_data = generate_input_data(extracted_model)

    assert len(extracted_model.graph.node) == 5

    convert_run_compare(extracted_model, input_data)


def test_extract_subgraph_mnist__non_existent_input_tensor():
    model_dir = os.path.join(_ARTIFACTS_DIR, "downloaded", "mnist-12")
    model_path = os.path.join(model_dir, "model.onnx")

    with pytest.raises(AssertionError) as e:
        ONNXSurgeon().extract_subgraph(model_path, ["hello_world"])

    assert e.value.args[0] == "Zero nodes gathered during forward pass. Is input tensor present in the model?"
