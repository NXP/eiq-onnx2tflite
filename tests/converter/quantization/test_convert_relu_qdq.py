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
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options, quantize_options
from tests import executors


def test_convert_relu_qdq__op_removed(intermediate_tflite_model_provider):
    input_shape = [4, 6]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Squeeze', ['input'], ['x']),
            onnx.helper.make_node("Relu", ["x"], ["output"])
        ],
        'Relu test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    # Quantize ONNX model
    calibration_data_reader = RandomDataCalibrationDataReader({"input": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert (intermediate_tflite_model_provider.get_operator_code_at_index(2) == BuiltinOperator.RELU)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)
