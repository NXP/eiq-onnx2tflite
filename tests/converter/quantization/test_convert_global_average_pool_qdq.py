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
from onnx2tflite.src.tflite_generator.builtin_options import average_pool_2d_options, dequantize_options, \
    quantize_options, transpose_options
from tests import executors


@pytest.fixture(scope="module")
def qdq_model_global_average_pool():
    input_shape = [5, 3, 4, 6]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("GlobalAveragePool", ["x"], ["output"])],
        'GlobalAveragePool test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    # Quantize ONNX model
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_global_average_pool_qdq(qdq_model_global_average_pool, intermediate_tflite_model_provider):
    input_shape = [5, 3, 4, 6]

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    # Allow error up to 1.0 in quantized form
    executors.convert_run_compare(qdq_model_global_average_pool, input_data, atol=0.004)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 5
    assert isinstance(ops[0].builtin_options, transpose_options.Transpose)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, average_pool_2d_options.AveragePool2D)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)
    assert isinstance(ops[4].builtin_options, transpose_options.Transpose)
