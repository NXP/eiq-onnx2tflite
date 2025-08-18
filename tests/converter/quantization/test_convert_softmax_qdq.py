#
# Copyright 2024-2025 NXP
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
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options, quantize_options, softmax_options
from tests import executors


@pytest.fixture(scope="module")
def qdq_softmax_model():
    input_shape = [4, 6]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Softmax", ["x"], ["output"])
        ],
        'Softmax test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    # Quantize ONNX model
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_softmax_qdq(qdq_softmax_model, intermediate_tflite_model_provider):
    input_shape = [4, 6]

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(qdq_softmax_model, input_data, save_models=True)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, softmax_options.Softmax)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize("quant_type, zero_point", [
    pytest.param(TensorProto.INT8, 127, id="INT8"),
    pytest.param(TensorProto.UINT8, 0, id="UINT8"),
])
def test_convert_softmax_qdq__non_matching_q_params(quant_type, zero_point):
    assert isinstance(quant_type, int)
    shape = [5, 4, 5]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["x_dequant"], axis=1),
            onnx.helper.make_node("Softmax", ["x_dequant"], ["y"]),
            onnx.helper.make_node("QuantizeLinear", ["y", "y_scale", "y_zero_point"], ["output"])
        ],
        'Softmax test quantized',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("output", quant_type, ())],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.586173820495605]),
            onnx.helper.make_tensor("x_zero_point", quant_type, [], [126]),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [25. / 256.]),
            onnx.helper.make_tensor("y_zero_point", quant_type, [], [127]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.random.random(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, atol=1)
