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
from onnx2tflite.src.tflite_generator.builtin_options import greater_equal_options, quantize_options
from tests import executors


@pytest.fixture(scope="module")
def single_greater_or_equal_node_qdq_model():
    quantizer = QDQQuantizer()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("GreaterOrEqual", ["x", "y"], ["output"])],
        'GreaterOrEqual test',
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
def single_greater_or_equal_node_qdq_model__static_y():
    quantizer = QDQQuantizer()

    np.random.seed(42)
    y_data = np.random.random(np.prod([2, 5])).reshape([2, 5]).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("GreaterOrEqual", ["x", "y"], ["output"])],
        'GreaterOrEqual test',
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


def test_convert_greater_or_equal_qdq(single_greater_or_equal_node_qdq_model, intermediate_tflite_model_provider):
    x_shape = [2, 5]
    y_shape = [2, 5]

    np.random.seed(42)
    input_data = {
        0: np.random.random(x_shape).astype(np.float32),
        1: np.random.random(y_shape).astype(np.float32),
    }

    executors.convert_run_compare(single_greater_or_equal_node_qdq_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, greater_equal_options.GreaterEqual)


def test_convert_greater_or_equal_qdq__static_y(single_greater_or_equal_node_qdq_model__static_y,
                                                intermediate_tflite_model_provider):
    x_shape = [2, 5]

    input_data = np.random.random(x_shape).astype(np.float32)

    executors.convert_run_compare(single_greater_or_equal_node_qdq_model__static_y, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 2
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, greater_equal_options.GreaterEqual)
