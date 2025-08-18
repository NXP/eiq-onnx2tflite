#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from tests import executors
from tests.model_surgeon import ONNXSurgeon


@pytest.mark.parametrize("a_shape, b_shape, c_shape, alpha, beta, output_shape", [
    pytest.param([8, 4], [4, 16], [16], 1.0, 1.0, [8, 16], id="a=1.0 b=1.0"),
    pytest.param([8, 4], [4, 16], [16], 0.33, 1.0, [8, 16], id="a=0.33 b=1.0"),
    pytest.param([8, 4], [4, 16], [16], 1.0, 0.66, [8, 16], id="a=1.0 b=0.66"),
    pytest.param([8, 4], [4, 16], [16], 0.5, 0.5, [8, 16], id="a=0.5 b=0.5"),
    pytest.param([8, 4], [4, 16], [16], 16.75, 32.25, [8, 16], id="a=16.75 b=32.25"),
])
def test_gemm_qdq__static_fusible_bias(a_shape: list[int], b_shape: list[int], c_shape: list[int], alpha: float,
                                       beta: float, output_shape: list[int]):
    np.random.seed(30)
    c_data = np.random.random(math.prod(c_shape)).reshape(c_shape).astype(np.float32)

    node = onnx.helper.make_node("Gemm", ["A", "B", "C"], ["Y"], alpha=alpha, beta=beta)
    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
        [onnx.helper.make_tensor("C", TensorProto.FLOAT, c_shape, c_data)]
    )
    onnx_model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    calibration_data_reader = RandomDataCalibrationDataReader({
        "A": InputSpec(a_shape, np.float32),
        "B": InputSpec(b_shape, np.float32),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    # Necessary to avoid AVX optimization issue
    quantized_model = ONNXSurgeon().intermediate_tensors_as_outputs(quantized_model.SerializeToString(),
                                                                    "Y_QuantizeLinear_Output")
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.random.random(math.prod(a_shape)).reshape(a_shape).astype(np.float32),
        1: np.random.random(math.prod(b_shape)).reshape(b_shape).astype(np.float32),
    }
    executors.convert_run_compare(quantized_model, input_data)


@pytest.mark.parametrize("x_shape, w_shape, b_shape, output_shape", [
    pytest.param([8, 4], [4, 16], [16], [8, 16]),
])
def test_gemm_qdq__per_channel__non_transposed_weights(x_shape: list[int], w_shape: list[int], b_shape: list[int],
                                                       output_shape: list[int],
                                                       intermediate_tflite_model_provider):
    np.random.seed(30)
    c_data = np.random.random(math.prod(b_shape)).reshape(b_shape).astype(np.float32)
    w_data = np.random.random(math.prod(w_shape)).reshape(w_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gemm", ["x", "w", "b"], ["y"])],
        "graph-gemm",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor("b", TensorProto.FLOAT, b_shape, c_data)
        ]
    )
    onnx_model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    # Necessary to avoid AVX optimization issue
    quantized_model = ONNXSurgeon().intermediate_tensors_as_outputs(quantized_model.SerializeToString(),
                                                                    "y_QuantizeLinear_Output")

    input_data = np.random.random(x_shape).astype(np.float32)
    executors.convert_run_compare(quantized_model, input_data)


@pytest.mark.parametrize("x_shape, w_shape, b_shape, output_shape", [
    pytest.param([8, 4], [16, 4], [16], [8, 16]),
])
def test_gemm_qdq__per_channel__transposed_weights(x_shape: list[int], w_shape: list[int], b_shape: list[int],
                                                   output_shape: list[int]):
    np.random.seed(30)
    c_data = np.random.random(math.prod(b_shape)).reshape(b_shape).astype(np.float32)
    w_data = np.random.random(math.prod(w_shape)).reshape(w_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gemm", ["x", "w", "b"], ["y"], transB=1)],
        "graph-gemm",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor("b", TensorProto.FLOAT, b_shape, c_data)
        ]
    )
    onnx_model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)
    # Necessary to avoid AVX optimization issue
    quantized_model = ONNXSurgeon().intermediate_tensors_as_outputs(quantized_model.SerializeToString(),
                                                                    "y_QuantizeLinear_Output")

    input_data = np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)


@pytest.mark.parametrize("a_shape, b_shape, c_shape, alpha, beta, output_shape", [
    pytest.param([8, 4], [4, 16], [8, 16], 0.33, 1.0, [8, 16], id="a=0.33 b=1.0"),
])
def test_gemm_qdq__static_non_fusible_bias(a_shape: list[int], b_shape: list[int], c_shape: list[int], alpha: float,
                                           beta: float, output_shape: list[int]):
    c_data = np.random.random(math.prod(c_shape)).reshape(c_shape).astype(np.float32)

    node = onnx.helper.make_node("Gemm", ["A", "B", "C"], ["Y"], alpha=alpha, beta=beta)
    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
        [onnx.helper.make_tensor("C", TensorProto.FLOAT, c_shape, c_data)]
    )
    onnx_model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    calibration_data_reader = RandomDataCalibrationDataReader({
        "A": InputSpec(a_shape, np.float32),
        "B": InputSpec(b_shape, np.float32),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    with pytest.raises(logger.Error):
        convert.convert_model(quantized_model)
    assert logger.conversion_log.get_node_error_code(5) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("a_shape, b_shape, alpha, beta, output_shape", [
    pytest.param([8, 4], [4, 16], 1.0, 1.0, [8, 16], id="a=1.0 b=1.0"),
    pytest.param([6, 8], [8, 5], 0.33, 1.0, [6, 5], id="a=0.33 b=1.0"),
    pytest.param([6, 8], [8, 5], 0.1, 1.0, [6, 5], id="a=0.33 b=1.0"),
    pytest.param([8, 4], [4, 16], 1.0, 0.66, [8, 16], id="a=1.0 b=0.66"),
    pytest.param([8, 4], [4, 16], 5.0, 0.66, [8, 16], id="a=1.0 b=0.66"),
    pytest.param([8, 4], [4, 16], 5.0, 1.0, [8, 16], id="a=1.0 b=0.66"),
])
def test_gemm_qdq__unbiased(a_shape: list[int], b_shape: list[int], alpha: float,
                            beta: float, output_shape: list[int]):
    np.random.seed(36)

    node = onnx.helper.make_node("Gemm", ["A", "B"], ["Y"], alpha=alpha, beta=beta)
    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
    )
    onnx_model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    calibration_data_reader = RandomDataCalibrationDataReader({
        "A": InputSpec(a_shape, np.float32),
        "B": InputSpec(b_shape, np.float32),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.random.random(math.prod(a_shape)).reshape(a_shape).astype(np.float32),
        1: np.random.random(math.prod(b_shape)).reshape(b_shape).astype(np.float32),
    }
    executors.convert_run_compare(quantized_model, input_data)


@pytest.mark.parametrize("a_shape, b_shape, alpha, beta, output_shape", [
    pytest.param([8, 4], [4, 16], 1.0, 1.0, [8, 16], id="a=1.0 b=1.0"),
    pytest.param([6, 8], [8, 5], 0.33, 1.0, [6, 5], id="a=0.33 b=1.0"),
    pytest.param([8, 4], [4, 16], 1.0, 0.66, [8, 16], id="a=1.0 b=0.66"),
])
def test_gemm_qdq__static_a__unbiased(a_shape: list[int], b_shape: list[int], alpha: float,
                                      beta: float, output_shape: list[int]):
    np.random.seed(300)

    a_data = np.random.random(math.prod(a_shape)).reshape(a_shape).astype(np.float32)

    node = onnx.helper.make_node("Gemm", ["A", "B"], ["Y"], alpha=alpha, beta=beta)
    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
        [onnx.helper.make_tensor("A", TensorProto.FLOAT, a_shape, a_data)]
    )
    onnx_model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    calibration_data_reader = RandomDataCalibrationDataReader({"B": InputSpec(b_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(math.prod(b_shape)).reshape(b_shape).astype(np.float32)
    executors.convert_run_compare(quantized_model, input_data)


@pytest.mark.parametrize("a_shape, b_shape, alpha, beta, output_shape", [
    pytest.param([8, 4], [4, 16], 1.0, 1.0, [8, 16], id="a=1.0 b=1.0"),
    pytest.param([6, 8], [8, 5], 0.33, 1.0, [6, 5], id="a=0.33 b=1.0"),
    pytest.param([8, 4], [4, 16], 1.0, 0.66, [8, 16], id="a=1.0 b=0.66"),
])
def test_gemm_qdq__static_b__unbiased(a_shape: list[int], b_shape: list[int], alpha: float,
                                      beta: float, output_shape: list[int]):
    np.random.seed(36)

    b_data = np.random.random(math.prod(b_shape)).reshape(b_shape).astype(np.float32)

    node = onnx.helper.make_node("Gemm", ["A", "B"], ["Y"], alpha=alpha, beta=beta)
    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
        [onnx.helper.make_tensor("B", TensorProto.FLOAT, b_shape, b_data)]
    )
    onnx_model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    calibration_data_reader = RandomDataCalibrationDataReader({
        "A": InputSpec(a_shape, np.float32),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    # Necessary to avoid AVX optimization issue
    quantized_model = ONNXSurgeon().intermediate_tensors_as_outputs(quantized_model.SerializeToString(),
                                                                    "Y_QuantizeLinear_Output")
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(math.prod(a_shape)).reshape(a_shape).astype(np.float32)
    executors.convert_run_compare(quantized_model, input_data)


def test_convert_gemm__qdq__per_channel__no_bias(intermediate_tflite_model_provider):
    a_shape = [15, 7]
    b_shape = [7, 14]

    np.random.seed(42)
    a_data = np.random.random(a_shape).astype('float32')
    b_data = np.random.random(b_shape).astype('float32')

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Gemm", ["a", "b"], ["y"])],
            "QDQ MatMul test",
            [onnx.helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape)],
            [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor("b", TensorProto.FLOAT, b_shape, b_data)]
        ),
    )

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)
    # Necessary to avoid AVX optimization issue
    quantized_model = ONNXSurgeon().intermediate_tensors_as_outputs(quantized_model.SerializeToString(),
                                                                    "y_QuantizeLinear_Output")
    executors.convert_run_compare(quantized_model, a_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.FULLY_CONNECTED, BuiltinOperator.DEQUANTIZE
    ])
    quant = intermediate_tflite_model_provider.get_operators()[1].tmp_inputs[1].quantization
    assert quant.scale.len() == quant.zero_point.len() == b_shape[1]  # Verify per-channel quantization.


def test_convert_gemm__qdq__per_channel__bias(intermediate_tflite_model_provider):
    a_shape = [15, 7]
    b_shape = [7, 14]
    c_shape = [14]

    np.random.seed(42)
    a_data = np.random.random(a_shape).astype('float32')
    b_data = np.random.random(b_shape).astype('float32')
    c_data = np.random.random(c_shape).astype('float32')

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Gemm", ["a", "b", 'c'], ["y"])],
            "QDQ MatMul test",
            [onnx.helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape)],
            [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor("b", TensorProto.FLOAT, b_shape, b_data),
                onnx.helper.make_tensor("c", TensorProto.FLOAT, c_shape, c_data)
            ]
        ),
    )

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    executors.convert_run_compare(quantized_model, a_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.FULLY_CONNECTED, BuiltinOperator.DEQUANTIZE
    ])
    quant = intermediate_tflite_model_provider.get_operators()[1].tmp_inputs[1].quantization
    assert quant.scale.len() == quant.zero_point.len() == b_shape[1]  # Verify per-channel quantization.


def test_convert_gemm__qdq__per_channel__no_bias__trans_b(intermediate_tflite_model_provider):
    a_shape = [15, 7]
    b_shape = [14, 7]

    np.random.seed(42)
    a_data = np.random.random(a_shape).astype('float32')
    b_data = np.random.random(b_shape).astype('float32')

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Gemm", ["a", "b"], ["y"], transB=1)],
            "QDQ MatMul test",
            [onnx.helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape)],
            [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor("b", TensorProto.FLOAT, b_shape, b_data)]
        ),
    )

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)
    # Necessary to avoid AVX optimization issue
    quantized_model = ONNXSurgeon().intermediate_tensors_as_outputs(quantized_model.SerializeToString(),
                                                                    "y_QuantizeLinear_Output")
    executors.convert_run_compare(quantized_model, a_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.FULLY_CONNECTED, BuiltinOperator.DEQUANTIZE
    ])
    quant = intermediate_tflite_model_provider.get_operators()[1].tmp_inputs[1].quantization
    assert quant.scale.len() == quant.zero_point.len() == b_shape[0]  # Verify per-channel quantization.


def test_convert_gemm__qdq__per_channel__bias__trans_b(intermediate_tflite_model_provider):
    a_shape = [15, 7]
    b_shape = [14, 7]
    c_shape = [14]

    np.random.seed(42)
    a_data = np.random.random(a_shape).astype('float32')
    b_data = np.random.random(b_shape).astype('float32')
    c_data = np.random.random(c_shape).astype('float32')

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Gemm", ["a", "b", 'c'], ["y"], transB=1)],
            "QDQ MatMul test",
            [onnx.helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape)],
            [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor("b", TensorProto.FLOAT, b_shape, b_data),
                onnx.helper.make_tensor("c", TensorProto.FLOAT, c_shape, c_data)
            ]
        ),
    )

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    executors.convert_run_compare(quantized_model, a_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.FULLY_CONNECTED, BuiltinOperator.DEQUANTIZE
    ])
    quant = intermediate_tflite_model_provider.get_operators()[1].tmp_inputs[1].quantization
    assert quant.scale.len() == quant.zero_point.len() == b_shape[0]  # Verify per-channel quantization.
