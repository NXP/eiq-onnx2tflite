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
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from tests import executors


def test_convert_hard_swish_qdq(intermediate_tflite_model_provider):
    input_shape = [4, 6]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("HardSwish", ["x"], ["output"])],
        'HardSwish test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})

    # HardSwish is during quantization (preprocessing) expanded to multiple ops
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    # Small error is expected because quantization function is done multiple times in ONNX model.
    # TFLite runs purely in (u)int8
    executors.convert_run_compare(quantized_model, input_data, atol=0.003)

    # Post-conversion optimization packs multiple ops back to HardSwish
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.HARD_SWISH, BuiltinOperator.DEQUANTIZE,
    ])


@pytest.mark.parametrize("quant_type, zero_point", [
    pytest.param(TensorProto.INT8, 0, id="INT8"),
    pytest.param(TensorProto.UINT8, 128, id="UINT8"),
])
def test_convert_hard_swish_qdq__non_matching_q_params(quant_type, zero_point, intermediate_tflite_model_provider):
    # Non-expanded quantized HardSwish can be produced by non-onnxruntime quantizer
    # so check we can convert it correctly
    assert isinstance(quant_type, int)
    shape = [5, 2, 6, 3]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["x_dequant"], axis=1),
            onnx.helper.make_node("HardSwish", ["x_dequant"], ["y"]),
            onnx.helper.make_node("QuantizeLinear", ["y", "y_scale", "y_zero_point"], ["output"])
        ],
        'HardSwish test quantized',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("output", quant_type, ())],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.002786173820495605]),
            onnx.helper.make_tensor("x_zero_point", quant_type, [], [zero_point]),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.00557940769195557]),
            onnx.helper.make_tensor("y_zero_point", quant_type, [], [zero_point]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    input_data = np.random.random(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.HARD_SWISH
    ])
