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
from onnx2tflite.src.conversion_config import QDQAwareConfig
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.tflite_generator.builtin_options import add_n_options, add_options, dequantize_options, \
    quantize_options
from tests import executors


@pytest.fixture(scope="module")
def single_sum_node_qdq_model():
    quantizer = QDQQuantizer()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sum", ["x", "y"], ["output"])],
        'Sum test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5]),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 5]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 5])],
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec([2, 5], np.float32),
        "y": InputSpec([2, 5], np.float32),
    })
    quantized_model = quantizer.quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


@pytest.fixture(scope="module")
def single_sum_node_qdq_model__three_inputs():
    quantizer = QDQQuantizer()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sum", ["x", "y", "z"], ["output"])],
        'Sum test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5]),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 5]),
            onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 5]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 5])],
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec([2, 5], np.float32),
        "y": InputSpec([2, 5], np.float32),
        "z": InputSpec([2, 5], np.float32),
    })
    quantized_model = quantizer.quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


@pytest.fixture(scope="module")
def single_sum_node_qdq_model__static_y():
    quantizer = QDQQuantizer()
    y_data = np.random.random(np.prod([2, 5])).reshape([2, 5]).astype(np.float32)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sum", ["x", "y"], ["output"])],
        'Sum test',
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


def test_convert_sum_qdq(single_sum_node_qdq_model, intermediate_tflite_model_provider):
    x_shape = [2, 5]
    y_shape = [2, 5]

    input_data = {
        0: np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.random.random(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1.,
    }

    executors.convert_run_compare(single_sum_node_qdq_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 4
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, add_options.Add)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)


def test_convert_sum_qdq__three_inputs(single_sum_node_qdq_model__three_inputs, intermediate_tflite_model_provider):
    input_shape = [2, 5]

    input_data = {
        0: np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        2: np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
    }

    executors.convert_run_compare(single_sum_node_qdq_model__three_inputs, input_data,
                                  conversion_config=QDQAwareConfig())

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 9
    assert isinstance(ops[6].builtin_options, add_n_options.AddN)


def test_convert_sum_qdq__static_y(single_sum_node_qdq_model__static_y, intermediate_tflite_model_provider):
    x_shape = [2, 5]

    input_data = np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np.float32)

    executors.convert_run_compare(single_sum_node_qdq_model__static_y, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, add_options.Add)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)
