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
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options, quantize_options
from tests import executors


@pytest.mark.parametrize(
    "input_shape, slope_shape",
    [
        pytest.param([2, 4, 8, 16], [1], id="1D slope [scalar]"),
        pytest.param([2, 4, 8, 16], [16], id="1D slope"),
        pytest.param([2, 4, 8, 16], [8, 1], id="2D slope"),
        pytest.param([2, 4, 8, 16], [2, 1, 8, 1], id="4D slope"),
    ])
def test_convert_relu_qdq__dynamic_slope(intermediate_tflite_model_provider, input_shape: list[int],
                                         slope_shape: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('PRelu', ['x', 'slope'], ['o'])],
        'PRelu test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("slope", TensorProto.FLOAT, slope_shape),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, input_shape)],
    )
    onnx_model = onnx.helper.make_model(graph)

    # Quantize ONNX model
    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec(input_shape, np.float32),
        "slope": InputSpec(slope_shape, np.float32),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32) * 2.,
        1: np.arange(np.prod(slope_shape)).reshape(slope_shape).astype(np.float32),
    }

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 4
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert (intermediate_tflite_model_provider.get_operator_code_at_index(1) == BuiltinOperator.PRELU)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize(
    "input_shape, slope_shape",
    [
        pytest.param([2, 4, 8, 16], [1], id="1D slope [scalar]"),
        pytest.param([2, 4, 8, 16], [16], id="1D slope"),
        pytest.param([2, 4, 8, 16], [8, 1], id="2D slope"),
        pytest.param([2, 4, 8, 16], [2, 1, 8, 1], id="4D slope"),
    ])
def test_convert_relu_qdq__static_slope(intermediate_tflite_model_provider, input_shape: list[int],
                                        slope_shape: list[int]):
    slope_data = np.arange(np.prod(slope_shape)).reshape(slope_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('PRelu', ['x', 'slope'], ['o'])],
        'PRelu test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, input_shape)],
        initializer=[onnx.helper.make_tensor("slope", TensorProto.FLOAT, slope_shape, slope_data)]
    )
    onnx_model = onnx.helper.make_model(graph)

    # Quantize ONNX model
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32) * 2.

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert (intermediate_tflite_model_provider.get_operator_code_at_index(1) == BuiltinOperator.PRELU)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)
