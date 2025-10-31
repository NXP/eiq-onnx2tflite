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
from onnx2tflite.src.tflite_generator.builtin_options import concatenation_options, dequantize_options, \
    quantize_options, softmax_options
from tests import executors


@pytest.fixture
def qdq_model_concat_dynamic():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Concat", ["x", "y"], ["z"], axis=-3)],
        "concat",
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, (2, 3, 4, 5)),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, (2, 4, 4, 5))
        ],
        [onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, (2, 7, 4, 5))],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec([2, 3, 4, 5], np.float32),
        "y": InputSpec([2, 4, 4, 5], np.float32, min=0.5, max=1.5),
    })

    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_concat_qdq_dynamic(qdq_model_concat_dynamic, intermediate_tflite_model_provider):
    input_data = {
        0: np.arange(np.prod((2, 3, 4, 5))).reshape((2, 3, 4, 5)).astype(np.float32),
        1: np.arange(np.prod((2, 4, 4, 5))).reshape((2, 4, 4, 5)).astype(np.float32),
    }

    executors.convert_run_compare(qdq_model_concat_dynamic, input_data, atol=0.006)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 4
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, concatenation_options.Concatenation)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)


def test_convert_concat_qdq__not_first_op_in_model(intermediate_tflite_model_provider):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Relu", ["x", ], ["a"]),
            onnx.helper.make_node("Softmax", ["y"], ["b"]),
            onnx.helper.make_node("Relu", ["z"], ["c"]),
            onnx.helper.make_node("Concat", ["a", "b", "c"], ["output"], axis=-3),
        ],
        "concat",
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, (2, 3, 4, 5)),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, (2, 4, 4, 5)),
            onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, (2, 4, 4, 5))
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 7, 4, 5))],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec([2, 3, 4, 5], np.float32, min=0, max=1.5),
        "y": InputSpec([2, 4, 4, 5], np.float32, min=-0.5, max=0.5),
        "z": InputSpec([2, 4, 4, 5], np.float32, min=0.5, max=1.5),
    })

    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.arange(np.prod((2, 3, 4, 5))).reshape((2, 3, 4, 5)).astype(np.float32),
        1: np.arange(np.prod((2, 4, 4, 5))).reshape((2, 4, 4, 5)).astype(np.float32),
        2: np.arange(np.prod((2, 4, 4, 5))).reshape((2, 4, 4, 5)).astype(np.float32),
    }

    executors.convert_run_compare(quantized_model, input_data, atol=0.006)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 9
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[4].builtin_options, softmax_options.Softmax)
    # We must preserve Quantize op after Softmax because there is conflict - Softmax enforces
    # specific output q-params and Concatenation enforces shared input q-params
    assert isinstance(ops[6].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[7].builtin_options, concatenation_options.Concatenation)
    assert isinstance(ops[8].builtin_options, dequantize_options.Dequantize)


@pytest.fixture
def qdq_model_concat_y_static():
    y_data = np.random.random(np.prod((2, 4, 4, 5))).reshape((2, 4, 4, 5)).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Concat", ["x", "y"], ["z"], axis=-3)],
        "concat",
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, (2, 3, 4, 5)),
        ],
        [onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, (2, 7, 4, 5))],
        [onnx.helper.make_tensor("y", TensorProto.FLOAT, (2, 4, 4, 5), y_data)]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_spec = {
        "x": InputSpec([2, 3, 4, 5], np.float32, min=-0.5, max=0.5)
    }
    calibration_data_reader = RandomDataCalibrationDataReader(input_spec)
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_concat_qdq_static_y(qdq_model_concat_y_static, intermediate_tflite_model_provider):
    input_data = np.random.random(np.prod((2, 3, 4, 5))).reshape((2, 3, 4, 5)).astype(np.float32)

    executors.convert_run_compare(qdq_model_concat_y_static, input_data, atol=0.006)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, concatenation_options.Concatenation)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize("quant_type, zero_point", [
    pytest.param(TensorProto.INT8, 127),
    pytest.param(TensorProto.UINT8, 0),
])
def test_convert_concat_qdq__dynamic_input__different_q_params(quant_type, zero_point):
    shape = [5, 2, 6, 3]

    assert isinstance(quant_type, int)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"], axis=1),
            onnx.helper.make_node("QuantizeLinear", ["y", "y_scale", "y_zero_point"], ["y_quant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["x_dequant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["y_quant", "y_scale", "y_zero_point"], ["y_dequant"], axis=1),
            onnx.helper.make_node("Concat", ["x_dequant", "y_dequant"], ["z"], axis=1),
            onnx.helper.make_node("QuantizeLinear", ["z", "y_scale", "y_zero_point"], ["output"])
        ],
        'Concat test quantized',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("output", quant_type, ())],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.02786173820495605]),
            onnx.helper.make_tensor("x_zero_point", quant_type, [], [zero_point]),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.0757940769195557]),
            onnx.helper.make_tensor("y_zero_point", quant_type, [], [zero_point]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.random.random(np.prod(shape)).reshape(shape).astype(np.float32),
        1: np.random.random(np.prod(shape)).reshape(shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize("quant_type, zero_point", [
    pytest.param(TensorProto.INT8, 127, id="INT8"),
    pytest.param(TensorProto.UINT8, 0, id="UINT8"),
])
def test_convert_concat_qdq__static_input__different_q_params(quant_type, zero_point):
    assert isinstance(quant_type, int)
    shape = [5, 2, 6, 3]

    y_data = np.linspace(-1., 1, np.prod(shape)).reshape(shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["x_dequant"], axis=1),
            onnx.helper.make_node("Concat", ["x_dequant", "y"], ["z"], axis=1),
            onnx.helper.make_node("QuantizeLinear", ["z", "y_scale", "y_zero_point"], ["output"])
        ],
        'Concat test quantized',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("output", quant_type, ())],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.02786173820495605]),
            onnx.helper.make_tensor("x_zero_point", quant_type, [], [zero_point]),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.0757940769195557]),
            onnx.helper.make_tensor("y_zero_point", quant_type, [], [zero_point]),
            onnx.helper.make_tensor("y", TensorProto.FLOAT, shape, y_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.linspace(-0.5, 1.5, np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, atol=1)
