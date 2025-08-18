#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options, quantize_options, split_options, \
    split_v_options
from tests import executors


@pytest.fixture(scope="module")
def qdq_model_with_split():
    input_shape = [5, 10]
    axis = 0
    output_names = [f"o{i}" for i in range(input_shape[axis])]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Split", ["x"], output_names, axis=axis)],
        'Split test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, ()) for name in output_names],
    )

    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_split_qdq(qdq_model_with_split, intermediate_tflite_model_provider):
    input_shape = [5, 10]

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(qdq_model_with_split, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 7
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, split_options.Split)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)
    assert isinstance(ops[4].builtin_options, dequantize_options.Dequantize)
    assert isinstance(ops[5].builtin_options, dequantize_options.Dequantize)
    assert isinstance(ops[6].builtin_options, dequantize_options.Dequantize)


@pytest.fixture(scope="module")
def qdq_model_with_split__explicit_splits():
    input_shape = [14, 5]
    split = [1, 3, 3, 7]
    axis = -2

    output_names = [f"o{i}" for i in range(len(split))]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Split", ["x"], output_names, axis=axis, split=split)],
        'Split test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, ()) for name in output_names],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)])

    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_split_qdq__explicit_splits(qdq_model_with_split__explicit_splits, intermediate_tflite_model_provider):
    input_shape = [14, 5]

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(qdq_model_with_split__explicit_splits, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 6
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, split_v_options.SplitV)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)
    assert isinstance(ops[4].builtin_options, dequantize_options.Dequantize)
    assert isinstance(ops[5].builtin_options, dequantize_options.Dequantize)
