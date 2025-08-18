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
from onnx2tflite.src.tflite_generator.builtin_options import less_options, quantize_options
from tests import executors


@pytest.fixture(scope="module")
def single_less_node_qdq_model():
    quantizer = QDQQuantizer()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Less", ["x", "y"], ["output"])],
        'Less test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5]),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 5]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.BOOL, [2, 5])],
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec([2, 5], np.float32),
        "y": InputSpec([2, 5], np.float32),
    })
    quantized_model = quantizer.quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


@pytest.fixture(scope="module")
def single_less_node_qdq_model__static_y():
    quantizer = QDQQuantizer()
    y_data = np.random.random(np.prod([2, 5])).reshape([2, 5]).astype(np.float32)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Less", ["x", "y"], ["output"])],
        'Less test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 5])],
        [onnx.helper.make_tensor("y", TensorProto.FLOAT, [2, 5], y_data)]
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec([2, 5], np.float32),
    })
    quantized_model = quantizer.quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_less_qdq(single_less_node_qdq_model, intermediate_tflite_model_provider):
    np.random.seed(42)

    x_shape = [2, 5]
    y_shape = [2, 5]

    input_data = {
        0: np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.random.random(np.prod(y_shape)).reshape(y_shape).astype(np.float32),
    }

    executors.convert_run_compare(single_less_node_qdq_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, less_options.Less)


def test_convert_less_qdq__static_y(single_less_node_qdq_model__static_y, intermediate_tflite_model_provider):
    x_shape = [2, 5]

    input_data = np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np.float32)

    executors.convert_run_compare(single_less_node_qdq_model__static_y, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 2
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, less_options.Less)
