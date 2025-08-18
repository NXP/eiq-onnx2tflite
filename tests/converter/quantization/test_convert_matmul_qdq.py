#
# Copyright 2024-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import List

import numpy as np
import onnx
import pytest
from onnx import TensorProto
from tensorflow.lite.python.interpreter import OpResolverType

from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import QDQAwareConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type
from onnx2tflite.src.tflite_generator.builtin_options import batch_mat_mul_options, dequantize_options, \
    fully_connected_options, quantize_options, transpose_options
from tests import executors


@pytest.mark.parametrize("input_1_shape, input_2_shape, output_shape", [
    pytest.param([2, 24, 56], [2, 56, 24], [2, 24, 24], id="3D"),
    pytest.param([1, 24, 5, 16], [1, 24, 16, 5], [1, 24, 5, 5], id="4D"),
    pytest.param([1, 3, 9, 5, 8], [1, 3, 9, 8, 5], [1, 3, 9, 5, 5], id="5D"),
])
def test_convert_matmul_qdq__into_batch_mat_mul(input_1_shape: List[int], input_2_shape: List[int],
                                                output_shape: List[int], intermediate_tflite_model_provider):
    np.random.seed(42)

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[onnx.helper.make_node("MatMul", ["input_1", "input_2"], ["output"])],
            name="QDQ MatMul test",
            inputs=[onnx.helper.make_tensor_value_info("input_1", TensorProto.FLOAT, input_1_shape),
                    onnx.helper.make_tensor_value_info("input_2", TensorProto.FLOAT, input_2_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        ),
    )

    calibration_data_reader = RandomDataCalibrationDataReader({
        "input_1": InputSpec(input_1_shape, np.float32),
        "input_2": InputSpec(input_2_shape, np.float32),
    })

    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.random.random(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.float32),
        1: np.random.random(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32),
    }

    executors.convert_run_compare(quantized_model, input_data, atol=0.02)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 4
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, batch_mat_mul_options.BatchMatMul)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize("input_1_shape, input_2_shape, output_shape", [
    pytest.param([56, 5], [5, 56], [56, 56], id="2D"),
])
def test_convert_matmul_qdq__into_fully_connected(input_1_shape: List[int], input_2_shape: List[int],
                                                  output_shape: List[int], intermediate_tflite_model_provider):
    np.random.seed(42)

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[onnx.helper.make_node("MatMul", ["input_1", "input_2"], ["output"])],
            name="QDQ MatMul test",
            inputs=[onnx.helper.make_tensor_value_info("input_1", TensorProto.FLOAT, input_1_shape),
                    onnx.helper.make_tensor_value_info("input_2", TensorProto.FLOAT, input_2_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        ),
    )

    calibration_data_reader = RandomDataCalibrationDataReader({
        "input_1": InputSpec(input_1_shape, np.float32),
        "input_2": InputSpec(input_2_shape, np.float32),
    })

    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.random.random(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.float32),
        1: np.random.random(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32),
    }

    executors.convert_run_compare(quantized_model, input_data, atol=0.02)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 5
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, transpose_options.Transpose)
    assert isinstance(ops[3].builtin_options, fully_connected_options.FullyConnected)
    assert isinstance(ops[4].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize("input_1_shape, input_2_shape, output_shape", [
    pytest.param([2, 24, 56], [2, 56, 24], [2, 24, 24], id="3D"),
    pytest.param([1, 24, 5, 16], [1, 24, 16, 5], [1, 24, 5, 5], id="4D"),
    pytest.param([1, 3, 9, 5, 8], [1, 3, 9, 8, 5], [1, 3, 9, 5, 5], id="5D"),
])
def test_convert_matmul_qdq__static_input__into_batch_mat_mul(input_1_shape: List[int], input_2_shape: List[int],
                                                              output_shape: List[int],
                                                              intermediate_tflite_model_provider):
    np.random.seed(42)
    input_2_data = np.random.random(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32)

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[onnx.helper.make_node("MatMul", ["input_1", "input_2"], ["output"])],
            name="QDQ MatMul test",
            inputs=[onnx.helper.make_tensor_value_info("input_1", TensorProto.FLOAT, input_1_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
            initializer=[onnx.helper.make_tensor("input_2", TensorProto.FLOAT, input_2_shape, input_2_data)]
        ),
    )

    calibration_data_reader = RandomDataCalibrationDataReader({"input_1": InputSpec(input_1_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.float32)

    # Run on reference evaluator. Otherwise, there is big error caused by specialized ops.
    executors.convert_run_compare(quantized_model, input_data,
                                  reference_onnx_evaluation=True,
                                  tflite_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                                  atol=0.03)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, batch_mat_mul_options.BatchMatMul)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize("input_1_shape, input_2_shape, output_shape", [
    pytest.param([12, 5], [5, 15], [15, 12], id="2D"),
])
def test_convert_matmul_qdq__static_input__into_fully_connected(input_1_shape: List[int], input_2_shape: List[int],
                                                                output_shape: List[int],
                                                                intermediate_tflite_model_provider):
    np.random.seed(42)
    input_2_data = np.random.random(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32)

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[onnx.helper.make_node("MatMul", ["input_1", "input_2"], ["output"])],
            name="QDQ MatMul test",
            inputs=[onnx.helper.make_tensor_value_info("input_1", TensorProto.FLOAT, input_1_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
            initializer=[onnx.helper.make_tensor("input_2", TensorProto.FLOAT, input_2_shape, input_2_data)]
        ),
    )

    calibration_data_reader = RandomDataCalibrationDataReader({"input_1": InputSpec(input_1_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.float32)

    # Run on reference evaluator. Otherwise, there is big error caused by specialized ops.
    executors.convert_run_compare(quantized_model, input_data,
                                  reference_onnx_evaluation=True,
                                  tflite_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, fully_connected_options.FullyConnected)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize("quant_type, zero_point", [
    pytest.param(TensorProto.INT8, 0, id="INT8"),
    pytest.param(TensorProto.UINT8, 128, id="UINT8"),
])
def test_convert_matmul_qdq__second_input_dynamic(quant_type, zero_point):
    np.random.seed(42)

    input_1_shape = [2, 24, 56]
    input_2_shape = [2, 56, 24]

    assert isinstance(quant_type, int)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"], axis=1),
            onnx.helper.make_node("QuantizeLinear", ["y", "y_scale", "y_zero_point"], ["y_quant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["input_1_dq"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["y_quant", "y_scale", "y_zero_point"], ["input_2_dq"], axis=1),
            onnx.helper.make_node("MatMul", ["input_1_dq", "input_2_dq"], ["z"]),
            onnx.helper.make_node("QuantizeLinear", ["z", "z_scale", "z_zero_point"], ["output"])
        ],
        'Concat test quantized',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_1_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, input_2_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", quant_type, ())],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.003920648247003555]),
            onnx.helper.make_tensor("x_zero_point", quant_type, [], [zero_point]),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.007840289734303951]),
            onnx.helper.make_tensor("y_zero_point", quant_type, [], [zero_point]),
            onnx.helper.make_tensor("z_scale", TensorProto.FLOAT, [], [0.0776829719543457]),
            onnx.helper.make_tensor("z_zero_point", quant_type, [], [zero_point]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.random.random(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.float32),
        1: np.random.random(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1,
                                  conversion_config=QDQAwareConfig())


@pytest.mark.parametrize("quant_type, zero_point", [
    pytest.param(TensorProto.INT8, 0, id="INT8"),
    pytest.param(TensorProto.UINT8, 128, id="UINT8"),
])
def test_convert_matmul_qdq__second_input_static__per_tensor_quant(quant_type: TensorProto.DataType, zero_point):
    np.random.seed(42)

    input_1_shape = [2, 24, 56]
    input_2_shape = [2, 56, 24]

    y_data = np.random.random(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32) * 127
    y_data = y_data.astype(to_numpy_type(quant_type))

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["input_1_dq"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["y_quant", "y_scale", "y_zero_point"], ["input_2_dq"], axis=1),
            onnx.helper.make_node("MatMul", ["input_1_dq", "input_2_dq"], ["z"]),
            onnx.helper.make_node("QuantizeLinear", ["z", "z_scale", "z_zero_point"], ["output"])
        ],
        'Concat test quantized',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_1_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", quant_type, ())],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.003920648247003555]),
            onnx.helper.make_tensor("x_zero_point", quant_type, [], [zero_point]),
            onnx.helper.make_tensor("y_quant", quant_type, input_2_shape, y_data),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.007840289734303951]),
            onnx.helper.make_tensor("y_zero_point", quant_type, [], [zero_point]),
            onnx.helper.make_tensor("z_scale", TensorProto.FLOAT, [], [0.726829719543457]),
            onnx.helper.make_tensor("z_zero_point", quant_type, [], [zero_point]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    input_data = np.random.random(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("quant_type, x_zp, y_zp, z_zp", [
    pytest.param(TensorProto.INT8, [3], [0, 0, 0], [1], id="INT8"),
    pytest.param(TensorProto.UINT8, [126], [128, 128, 128], [120], id="UINT8"),
])
def test_convert_matmul_qdq__second_input_static__per_channel_quant(quant_type: TensorProto.DataType, x_zp, y_zp, z_zp,
                                                                    intermediate_tflite_model_provider):
    np.random.seed(50)

    input_1_shape = [4, 10]
    input_2_shape = [10, 3]

    y_data = np.random.random(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32) * 127
    y_data = y_data.astype(to_numpy_type(quant_type))

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["input_1_dq"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["y_quant", "y_scale", "y_zero_point"], ["input_2_dq"], axis=1),
            onnx.helper.make_node("MatMul", ["input_1_dq", "input_2_dq"], ["z"]),
            onnx.helper.make_node("QuantizeLinear", ["z", "z_scale", "z_zero_point"], ["output"])
        ],
        'Concat test quantized',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_1_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", quant_type, [4, 3])],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.2]),
            onnx.helper.make_tensor("x_zero_point", quant_type, [], x_zp),
            onnx.helper.make_tensor("y_quant", quant_type, input_2_shape, y_data),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [3], [0.15, 0.22, 0.28]),
            onnx.helper.make_tensor("y_zero_point", quant_type, [3], y_zp),
            onnx.helper.make_tensor("z_scale", TensorProto.FLOAT, [], [0.74]),
            onnx.helper.make_tensor("z_zero_point", quant_type, [], z_zp),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    input_data = np.random.random(np.prod(input_1_shape)).reshape(input_1_shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, fully_connected_options.FullyConnected)


@pytest.mark.parametrize("quant_type, x_zp, y_zp, z_zp", [
    pytest.param(TensorProto.INT8, [3], [0, 15, 40], [1], id="INT8"),
    pytest.param(TensorProto.UINT8, [126], [50, 127, 200], [120], id="UINT8"),
])
def test_convert_matmul_qdq__per_channel_quant__unsupported(quant_type: TensorProto.DataType, x_zp, y_zp, z_zp,
                                                            intermediate_tflite_model_provider):
    input_1_shape = [4, 10]
    input_2_shape = [10, 3]

    y_data = np.random.random(np.prod(input_2_shape)).reshape(input_2_shape).astype(np.float32) * 127
    y_data = y_data.astype(to_numpy_type(quant_type))

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["input_1_dq"], axis=1),
            onnx.helper.make_node("DequantizeLinear", ["y_quant", "y_scale", "y_zero_point"], ["input_2_dq"], axis=1),
            onnx.helper.make_node("MatMul", ["input_1_dq", "input_2_dq"], ["z"]),
            onnx.helper.make_node("QuantizeLinear", ["z", "z_scale", "z_zero_point"], ["output"])
        ],
        'Concat test quantized',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_1_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", quant_type, [4, 3])],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.2]),
            onnx.helper.make_tensor("x_zero_point", quant_type, [], x_zp),
            onnx.helper.make_tensor("y_quant", quant_type, input_2_shape, y_data),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [3], [0.15, 0.22, 0.28]),
            onnx.helper.make_tensor("y_zero_point", quant_type, [3], y_zp),
            onnx.helper.make_tensor("z_scale", TensorProto.FLOAT, [], [0.74]),
            onnx.helper.make_tensor("z_zero_point", quant_type, [], z_zp),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(3) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_matmul__qdq__per_channel(intermediate_tflite_model_provider):
    a_shape = [15, 7]
    b_shape = [7, 14]

    np.random.seed(42)
    a_data = np.random.random(a_shape).astype('float32')
    b_data = np.random.random(b_shape).astype('float32')

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["a", "b"], ["y"])],
            "QDQ MatMul test",
            [onnx.helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape)],
            [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor("b", TensorProto.FLOAT, b_shape, b_data)]
        ),
    )

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    # Run on reference evaluator. Otherwise, there is big error caused by specialized ops.
    executors.convert_run_compare(quantized_model, a_data,
                                  reference_onnx_evaluation=True,
                                  tflite_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.FULLY_CONNECTED, BuiltinOperator.DEQUANTIZE
    ])
    quant = intermediate_tflite_model_provider.get_operators()[1].tmp_inputs[1].quantization
    assert quant.scale.len() == quant.zero_point.len() == b_shape[1]


def test_convert_matmul__qdq__per_channel__high_rank(intermediate_tflite_model_provider):
    # Should be quantized per-tensor.

    a_shape = [3, 5, 7]
    b_shape = [3, 7, 4]

    np.random.seed(42)
    a_data = np.random.random(a_shape).astype('float32')
    b_data = np.random.random(b_shape).astype('float32')

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["a", "b"], ["y"])],
            "QDQ MatMul test",
            [onnx.helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape)],
            [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor("b", TensorProto.FLOAT, b_shape, b_data)]
        ),
    )

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    # Run on reference evaluator. Otherwise, there is big error caused by specialized ops.
    executors.convert_run_compare(quantized_model, a_data,
                                  reference_onnx_evaluation=True,
                                  tflite_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                                  atol=0.013)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.BATCH_MATMUL, BuiltinOperator.DEQUANTIZE
    ])
    quant = intermediate_tflite_model_provider.get_operators()[1].tmp_inputs[1].quantization
    assert quant.scale.len() == quant.zero_point.len() == 1  # Per-tensor quantization.
