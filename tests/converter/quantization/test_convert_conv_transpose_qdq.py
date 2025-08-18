#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import os
import tempfile

import numpy as np
import onnx
import onnxruntime
import onnxruntime.quantization
from onnx import TensorProto
from onnxruntime.quantization import QuantType

from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options, quantize_options, \
    transpose_conv_options, transpose_options
from tests import executors
from tests.executors import OnnxExecutor, TFLiteExecutor


def test_convert_conv_transpose_qdq(intermediate_tflite_model_provider):
    np.random.seed(23)

    kernel_shape = [3, 3]
    input_shape = [2, 3, 4, 5]
    weight_shape = [input_shape[1], 6] + kernel_shape
    bias_shape = [weight_shape[1]]

    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape)],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    # Quantize ONNX model
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_onnx_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(quantized_onnx_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 5
    assert isinstance(ops[0].builtin_options, transpose_options.Transpose)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, transpose_conv_options.TransposeConv)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)
    assert isinstance(ops[4].builtin_options, transpose_options.Transpose)

    w_quant = ops[2].tmp_inputs[1].quantization
    assert w_quant.scale.len() == w_quant.zero_point.len() == 1  # Verify per-tensor quantization.


def test_convert_conv_transpose_qdq__per_channel__int8(intermediate_tflite_model_provider):
    kernel_shape = [3, 3]
    input_shape = [2, 3, 4, 5]
    output_shape = [2, 3, 6, 7]
    weight_shape = [input_shape[1], 3] + kernel_shape
    bias_shape = [weight_shape[1]]
    np.random.seed(23)

    weights = np.random.random(weight_shape).astype(np.float32)
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape)],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    tfl_model = convert.convert_model(ModelShapeInference.infer_shapes(quantized_model))

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    onnx_executor = OnnxExecutor(quantized_model.SerializeToString())
    onnx_output = onnx_executor.inference(input_data)

    tflite_executor = TFLiteExecutor(model_content=bytes(tfl_model))
    tflite_output = tflite_executor.inference(input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 5
    assert isinstance(ops[0].builtin_options, transpose_options.Transpose)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, transpose_conv_options.TransposeConv)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)
    assert isinstance(ops[4].builtin_options, transpose_options.Transpose)

    # Quantize output of float ONNX model and quantized TFLite model and check
    # if error is in range <-3, 3>. TransposeConv suffers from error aggregation.
    last_dq_op = ops[3].tmp_inputs[0].quantization
    scale = last_dq_op.scale.get(0)
    zp = last_dq_op.zero_point.get(0)

    onnx_output_quant = np.add(np.divide(onnx_output, scale), zp).astype(np.int8)
    tflite_output_quant = np.add(np.divide(tflite_output, scale), zp).astype(np.int8)
    assert np.all(np.abs(onnx_output_quant - tflite_output_quant) <= 3)

    w_quant = intermediate_tflite_model_provider.get_operators()[2].tmp_inputs[1].quantization
    assert w_quant.scale.len() == w_quant.zero_point.len() == bias_shape[0]  # Verify per-channel quantization.


def test_convert_conv_transpose_qdq__per_channel__mixed_types(intermediate_tflite_model_provider):
    np.random.seed(23)

    kernel_shape = [3, 3]
    input_shape = [2, 6, 4, 5]
    output_shape = [2, 6, 6, 7]
    weight_shape = [input_shape[1], 6] + kernel_shape
    bias_shape = [weight_shape[1]]

    weights = np.random.random(weight_shape).astype(np.float32)
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape)],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with tempfile.TemporaryDirectory() as temp_dir:
        calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})

        input_model_path = os.path.join(temp_dir, "model.onnx")
        output_model_path = os.path.join(temp_dir, "quantized_model.onnx")
        onnx.save_model(onnx_model, input_model_path)

        # Create per-channel quantized variant of the model
        onnxruntime.quantization.quantize_static(
            input_model_path,
            output_model_path,
            calibration_data_reader,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )

        quantized_onnx_model = onnx.load_model(output_model_path)
        input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

        executors.convert_run_compare(quantized_onnx_model, input_data)

        # This model with not be executed as optimized because TFLite doesn't support non-matching
        # quantization types for TransposeConv
        ops = intermediate_tflite_model_provider.get_operators()
        assert len(ops) == 9
