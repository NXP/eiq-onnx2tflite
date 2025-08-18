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
from onnx2tflite.src.tflite_generator.builtin_options import equal_options, quantize_options
from tests import executors


@pytest.fixture(scope="module")
def qdq_model_with_equal():
    quantizer = QDQQuantizer()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Equal", ["x", "y"], ["output"])],
        'Add test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [10, 5]),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [10, 5]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.BOOL, [10, 5])],
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec([10, 5], np.float32),
        "y": InputSpec([10, 5], np.float32),
    })
    quantized_model = quantizer.quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_equal_qdq(qdq_model_with_equal, intermediate_tflite_model_provider):
    x_shape = [10, 5]
    y_shape = [10, 5]

    # Make sure some of the inputs are equal in quantized type
    input_data = {
        0: np.random.randint(0, 5, x_shape).astype(np.float32),
        1: np.random.randint(0, 5, y_shape).astype(np.float32),
    }

    executors.convert_run_compare(qdq_model_with_equal, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, equal_options.Equal)
