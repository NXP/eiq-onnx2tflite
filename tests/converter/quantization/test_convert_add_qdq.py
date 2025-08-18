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
from onnx2tflite.src.tflite_generator.builtin_options import add_options, dequantize_options, quantize_options
from tests import executors


@pytest.fixture(scope="module")
def single_add_node_qdq_model():
    quantizer = QDQQuantizer()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "y"], ["output"])],
        'Add test',
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


def test_convert_add_qdq(single_add_node_qdq_model, intermediate_tflite_model_provider):
    x_shape = [2, 5]
    y_shape = [2, 5]

    input_data = {
        0: np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np.float32),
        1: np.random.random(np.prod(y_shape)).reshape(y_shape).astype(np.float32) + 1.,
    }

    executors.convert_run_compare(single_add_node_qdq_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 4
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, add_options.Add)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)


def test_convert_add_qdq__different_q_ops_types(intermediate_tflite_model_provider):
    shape = [5, 2, 6, 3]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"], axis=1),
            onnx.helper.make_node("QuantizeLinear", ["y", "y_scale", "y_zero_point"], ["y_quant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["x_dequant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["y_quant", "y_scale", "y_zero_point"], ["y_dequant"], axis=1),
            onnx.helper.make_node("Add", ["x_dequant", "y_dequant"], ["z"]),
            onnx.helper.make_node("QuantizeLinear", ["z", "y_scale", "y_zero_point"], ["output"])
        ],
        'Add test quantized',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.UINT8, ())],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.02786173820495605]),
            onnx.helper.make_tensor("x_zero_point", TensorProto.INT8, [], [0]),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.0757940769195557]),
            onnx.helper.make_tensor("y_zero_point", TensorProto.UINT8, [], [128]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.random.random(np.prod(shape)).reshape(shape).astype(np.float32),
        1: np.random.random(np.prod(shape)).reshape(shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    # 'Add' op in generated model must not be converted as quantized op
    intermediate_tflite_model_provider.assert_converted_model_has_operators(
        [BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
         BuiltinOperator.DEQUANTIZE, BuiltinOperator.DEQUANTIZE,
         BuiltinOperator.ADD, BuiltinOperator.QUANTIZE])
