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
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options, quantize_options, select_v2_options, \
    softmax_options
from tests import executors


def test_convert_where_qdq(intermediate_tflite_model_provider):
    input_shape = [5, 2, 6]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Where", ["condition", "x", "y"], ["output"])],
        'Where test',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, input_shape),
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, input_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
    )

    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({
        "condition": InputSpec(input_shape, np.bool_),
        "x": InputSpec(input_shape, np.float32),
        "y": InputSpec(input_shape, np.float32),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.random.random(np.prod(input_shape)).reshape(input_shape) < 0.5,
        1: np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        2: np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
    }

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 4
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, select_v2_options.SelectV2)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)


def test_convert_where_qdq__non_first_op(intermediate_tflite_model_provider):
    input_shape = [5, 2, 6]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Softmax", ["x"], ["a"]),
            onnx.helper.make_node("Relu", ["y"], ["b"]),
            onnx.helper.make_node("Where", ["condition", "a", "b"], ["output"]),
        ],
        'Where test',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, input_shape),
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, input_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
    )

    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({
        "condition": InputSpec(input_shape, np.bool_),
        "x": InputSpec(input_shape, np.float32),
        "y": InputSpec(input_shape, np.float32),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.random.random(np.prod(input_shape)).reshape(input_shape) < 0.5,
        1: np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        2: np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
    }

    executors.convert_run_compare(quantized_model, input_data, atol=0.004)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 7
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, softmax_options.Softmax)
    # Quantize between Softmax and SelectV2 must be preserved
    assert isinstance(ops[4].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[5].builtin_options, select_v2_options.SelectV2)
    assert isinstance(ops[6].builtin_options, dequantize_options.Dequantize)


def test_convert_where_qdq__static_x(intermediate_tflite_model_provider):
    input_shape = [5, 2, 6]
    x_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32) * 5. - 2

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Where", ["condition", "x", "y"], ["output"])],
        'Where test',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, input_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, input_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("x", TensorProto.FLOAT, input_shape, x_data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({
        "condition": InputSpec(input_shape, np.bool_),
        "y": InputSpec(input_shape, np.float32),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.random.random(np.prod(input_shape)).reshape(input_shape) < 0.5,
        1: np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
    }

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, select_v2_options.SelectV2)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


def test_convert_where_qdq__static_y(intermediate_tflite_model_provider):
    input_shape = [5, 2, 6]
    y_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32) * 5. - 2

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Where", ["condition", "x", "y"], ["output"])],
        'Where test',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, input_shape),
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("y", TensorProto.FLOAT, input_shape, y_data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({
        "condition": InputSpec(input_shape, np.bool_),
        "x": InputSpec(input_shape, np.float32),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.random.random(np.prod(input_shape)).reshape(input_shape) < 0.5,
        1: np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
    }

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, select_v2_options.SelectV2)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize("quant_type, zero_point", [
    pytest.param(TensorProto.INT8, 0, id="INT8"),
    pytest.param(TensorProto.UINT8, 128, id="UINT8"),
])
def test_convert_where_qdq__different_q_params(quant_type, zero_point):
    np.random.seed(42)
    assert isinstance(quant_type, int)

    shape = [5, 2, 6, 3]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"], axis=1),
            onnx.helper.make_node("QuantizeLinear", ["y", "y_scale", "y_zero_point"], ["y_quant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["x_dequant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["y_quant", "y_scale", "y_zero_point"], ["y_dequant"], axis=1),
            onnx.helper.make_node("Where", ["condition", "x_dequant", "y_dequant"], ["z"]),
            onnx.helper.make_node("QuantizeLinear", ["z", "y_scale", "y_zero_point"], ["output"])
        ],
        'Concat test quantized',
        [
            onnx.helper.make_tensor_value_info("condition", TensorProto.BOOL, shape),
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("output", quant_type, ())],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.003886173820495605]),
            onnx.helper.make_tensor("x_zero_point", quant_type, [], [zero_point]),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.00407940769195557]),
            onnx.helper.make_tensor("y_zero_point", quant_type, [], [zero_point]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.random.random(np.prod(shape)).reshape(shape) < 0.5,
        1: np.random.random(np.prod(shape)).reshape(shape).astype(np.float32),
        2: np.random.random(np.prod(shape)).reshape(shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)
