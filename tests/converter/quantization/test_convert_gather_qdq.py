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
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options, gather_options, quantize_options
from tests import executors


@pytest.fixture(scope="module")
def qdq_model_with_gather():
    input_shape = [12, 2, 3, 2]
    indices = [-5, 2, -7, 10, -11]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Gather', ['input', 'indices'], ['output'])],
        'reshape test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("indices", TensorProto.INT64, [len(indices)], indices)]
    )

    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({"input": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_gather_qdq(qdq_model_with_gather, intermediate_tflite_model_provider):
    input_shape = [12, 2, 3, 2]

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(qdq_model_with_gather, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, gather_options.Gather)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)
