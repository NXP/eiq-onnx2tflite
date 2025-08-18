#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
from onnx import TensorProto

from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from tests import executors


def test_convert_max_qdq(intermediate_tflite_model_provider):
    np.random.seed(42)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Max", ["x", "y"], ["output"])],
        'Max test',
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
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    x_shape = [2, 5]
    y_shape = [2, 5]

    input_data = {
        0: np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.random.random(np.prod(y_shape)).reshape(y_shape).astype(np.float32),
    }

    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.MAXIMUM, BuiltinOperator.DEQUANTIZE
    ])


def test_convert_maximum_qdq__static_y(intermediate_tflite_model_provider):
    x_shape = [2, 5]
    np.random.seed(42)

    y_data = np.random.random(np.prod([2, 5])).reshape([2, 5]).astype(np.float32) * 2.
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Max", ["x", "y"], ["output"])],
        'Max test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5])],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 5])],
        [onnx.helper.make_tensor("y", TensorProto.FLOAT, [2, 5], y_data)]
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec([2, 5], np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE, BuiltinOperator.MAXIMUM,
        BuiltinOperator.DEQUANTIZE
    ])


def test_convert_max_qdq__op_removed(intermediate_tflite_model_provider):
    np.random.seed(42)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Softmax", ["x"], ["y"]),
            onnx.helper.make_node("Max", ["y"], ["z"]),
            onnx.helper.make_node("Relu", ["z"], ["output"]),
        ],
        'Max test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5])],
        [onnx.helper.make_tensor_value_info("output", TensorProto.BOOL, [2, 5])],
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec([2, 5], np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    x_shape = [2, 5]

    input_data = np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.SOFTMAX, BuiltinOperator.RELU,
        BuiltinOperator.DEQUANTIZE
    ])
