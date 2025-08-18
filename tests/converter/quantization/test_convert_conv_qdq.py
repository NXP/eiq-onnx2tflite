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
from onnx import TensorProto
from onnxruntime.quantization import QuantType

from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.converter import convert
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.tflite_generator.builtin_options import conv_2d_options, dequantize_options, quantize_options, \
    transpose_options
from tests import executors
from tests.executors import OnnxExecutor, TFLiteExecutor
from tests.model_surgeon import ONNXSurgeon


def test_convert_conv_qdq(intermediate_tflite_model_provider):
    np.random.seed(23)

    kernel_shape = [3, 3]
    weight_shape = [20, 10] + kernel_shape
    input_shape = [5, weight_shape[1], 15, 20]
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    bias_shape = [weight_shape[0]]
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape)],
        'Conv test',
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
    tfl_model = convert.convert_model(ModelShapeInference.infer_shapes(quantized_onnx_model))

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    # We are comparing outputs of float ONNX model and quantized TFLite due
    # to the high error when running quantized ONNX (AVX optimization issue)
    onnx_executor = OnnxExecutor(onnx_model.SerializeToString())
    onnx_output = onnx_executor.inference(input_data)

    tflite_executor = TFLiteExecutor(model_content=bytes(tfl_model))
    tflite_output = tflite_executor.inference(input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 5
    assert isinstance(ops[0].builtin_options, transpose_options.Transpose)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, conv_2d_options.Conv2D)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)
    assert isinstance(ops[4].builtin_options, transpose_options.Transpose)

    # Quantize output of float ONNX model and quantized TFLite model and check
    # if error is in range <-1, 1>
    last_dq_op = ops[3].tmp_inputs[0].quantization
    scale = last_dq_op.scale.get(0)
    zp = last_dq_op.zero_point.get(0)

    onnx_output_quant = np.add(np.divide(onnx_output, scale), zp).astype(np.int8)
    tflite_output_quant = np.add(np.divide(tflite_output, scale), zp).astype(np.int8)
    assert np.all(np.abs(onnx_output_quant - tflite_output_quant) <= 1)


def test_convert_conv_qdq__per_channel__mixed_types(intermediate_tflite_model_provider):
    np.random.seed(23)

    kernel_shape = [3, 3]
    input_shape = [2, 5, 9, 8]
    weight_shape = [4, input_shape[1]] + kernel_shape
    output_shape = [2, 4, 7, 6]
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    bias_shape = [weight_shape[0]]
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w"], ["output"], kernel_shape=kernel_shape)],
        'Conv test',
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
        tfl_model = convert.convert_model(ModelShapeInference.infer_shapes(quantized_onnx_model))

        # We are comparing outputs of float ONNX model and quantized TFLite due
        # to the high error when running quantized ONNX (AVX optimization issue)
        onnx_executor = OnnxExecutor(onnx_model.SerializeToString())
        onnx_output = onnx_executor.inference(input_data)

        tflite_executor = TFLiteExecutor(model_content=bytes(tfl_model))
        tflite_output = tflite_executor.inference(input_data)

        ops = intermediate_tflite_model_provider.get_operators()
        assert len(ops) == 6
        assert isinstance(ops[0].builtin_options, transpose_options.Transpose)
        assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
        assert isinstance(ops[2].builtin_options, conv_2d_options.Conv2D)
        assert isinstance(ops[3].builtin_options, quantize_options.Quantize)
        assert isinstance(ops[4].builtin_options, dequantize_options.Dequantize)
        assert isinstance(ops[5].builtin_options, transpose_options.Transpose)

        # Quantize output of float ONNX model and quantized TFLite model and check
        # if error is in range <-1, 1>
        last_dq_op = ops[4].tmp_inputs[0].quantization
        scale = last_dq_op.scale.get(0)
        zp = last_dq_op.zero_point.get(0)

        onnx_output_quant = np.add(np.divide(onnx_output, scale), zp).astype(np.int8)
        tflite_output_quant = np.add(np.divide(tflite_output, scale), zp).astype(np.int8)
        assert np.all(np.abs(onnx_output_quant - tflite_output_quant) <= 1)


def test_convert_conv__qdq__per_channel__no_bias(intermediate_tflite_model_provider):
    np.random.seed(42)

    kernel_shape = [3, 3]
    input_shape = [2, 5, 9, 8]
    weight_shape = [4, input_shape[1]] + kernel_shape
    weights = np.random.random(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w"], ["output"], kernel_shape=kernel_shape)],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )
    onnx_model = onnx.helper.make_model(graph)

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    input_data = np.random.random(input_shape).astype(np.float32)
    executors.convert_run_compare(quantized_model, input_data, reference_onnx_evaluation=True)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE, BuiltinOperator.CONV_2D,
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])

    conv = intermediate_tflite_model_provider.get_operators()[2]
    w_tensor = conv.tmp_inputs[1]
    assert w_tensor.quantization is not None
    assert w_tensor.quantization.scale.len() == weight_shape[0]
    assert w_tensor.quantization.zero_point.len() == weight_shape[0]


def test_convert_conv__qdq__per_channel__bias(intermediate_tflite_model_provider):
    np.random.seed(42)

    kernel_shape = [3, 3]
    input_shape = [2, 5, 9, 8]
    weight_shape = [4, input_shape[1]] + kernel_shape
    weights = np.random.random(weight_shape).astype(np.float32)

    bias_shape = [weight_shape[0]]
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape)],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    input_data = np.random.random(input_shape).astype(np.float32)
    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE, BuiltinOperator.CONV_2D,
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])

    conv = intermediate_tflite_model_provider.get_operators()[2]
    w_tensor = conv.tmp_inputs[1]
    assert w_tensor.quantization is not None
    assert w_tensor.quantization.scale.len() == weight_shape[0]
    assert w_tensor.quantization.zero_point.len() == weight_shape[0]

    bias_tensor = conv.tmp_inputs[1]
    assert bias_tensor.quantization is not None
    assert bias_tensor.quantization.scale.len() == weight_shape[0]
    assert bias_tensor.quantization.zero_point.len() == weight_shape[0]


def test_convert_conv__depthwise__qdq__per_channel__no_bias(intermediate_tflite_model_provider):
    np.random.seed(42)

    kernel_shape = [3, 3]
    input_shape = [2, 5, 9, 8]
    weight_shape = [input_shape[1], 1] + kernel_shape
    weights = np.random.random(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w"], ["output"], kernel_shape=kernel_shape,
                               group=input_shape[1])],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )
    onnx_model = onnx.helper.make_model(graph)

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    input_data = np.random.random(input_shape).astype(np.float32)
    executors.convert_run_compare(quantized_model, input_data, reference_onnx_evaluation=True)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE, BuiltinOperator.DEPTHWISE_CONV_2D,
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])

    conv = intermediate_tflite_model_provider.get_operators()[2]
    w_tensor = conv.tmp_inputs[1]
    assert w_tensor.quantization is not None
    assert w_tensor.quantization.scale.len() == weight_shape[0]
    assert w_tensor.quantization.zero_point.len() == weight_shape[0]


def test_convert_conv__depthwise__qdq__per_channel__bias(intermediate_tflite_model_provider):
    np.random.seed(42)

    kernel_shape = [3, 3]
    input_shape = [2, 5, 9, 8]
    weight_shape = [input_shape[1], 1] + kernel_shape
    weights = np.random.random(weight_shape).astype(np.float32)

    bias_shape = [weight_shape[0]]
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape,
                               group=input_shape[1])],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    input_data = np.random.random(input_shape).astype(np.float32)
    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE, BuiltinOperator.DEPTHWISE_CONV_2D,
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])

    conv = intermediate_tflite_model_provider.get_operators()[2]
    w_tensor = conv.tmp_inputs[1]
    assert w_tensor.quantization is not None
    assert w_tensor.quantization.scale.len() == weight_shape[0]
    assert w_tensor.quantization.zero_point.len() == weight_shape[0]

    bias_tensor = conv.tmp_inputs[1]
    assert bias_tensor.quantization is not None
    assert bias_tensor.quantization.scale.len() == weight_shape[0]
    assert bias_tensor.quantization.zero_point.len() == weight_shape[0]


def test_convert_conv__separable__qdq__per_channel__no_bias(intermediate_tflite_model_provider):
    np.random.seed(42)

    kernel_shape = [3, 3]
    input_shape = [2, 4, 9, 8]
    weight_shape = [4, 2] + kernel_shape
    group = 2

    weights = np.random.random(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=kernel_shape,
                               group=group)],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )
    onnx_model = onnx.helper.make_model(graph)

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)
    quantized_model = ONNXSurgeon().intermediate_tensors_as_outputs(quantized_model.SerializeToString(),
                                                                    "y_QuantizeLinear_Output")

    input_data = np.random.random(input_shape).astype(np.float32)
    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.SPLIT,
        BuiltinOperator.CONV_2D, BuiltinOperator.CONV_2D,
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE, BuiltinOperator.TRANSPOSE
    ])

    conv1 = intermediate_tflite_model_provider.get_operators()[3]
    assert conv1.tmp_inputs[1].quantization is not None
    assert conv1.tmp_inputs[1].quantization.scale.len() == int(weight_shape[0] / group)
    assert conv1.tmp_inputs[1].quantization.zero_point.len() == int(weight_shape[0] / group)

    conv2 = intermediate_tflite_model_provider.get_operators()[4]
    assert conv2.tmp_inputs[1].quantization is not None
    assert conv2.tmp_inputs[1].quantization.scale.len() == int(weight_shape[0] / group)
    assert conv2.tmp_inputs[1].quantization.zero_point.len() == int(weight_shape[0] / group)


def test_convert_conv__separable__qdq__per_channel__bias(intermediate_tflite_model_provider):
    np.random.seed(42)

    kernel_shape = [3, 3]
    input_shape = [2, 4, 9, 8]
    weight_shape = [4, 2] + kernel_shape
    group = 2
    bias_shape = [weight_shape[0]]

    bias = np.random.random(bias_shape).astype(np.float32)
    weights = np.random.random(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape,
                               group=group)],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {"per_channel": True})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    input_data = np.random.random(input_shape).astype(np.float32)
    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.SPLIT,
        BuiltinOperator.CONV_2D, BuiltinOperator.CONV_2D,
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])

    conv1 = intermediate_tflite_model_provider.get_operators()[3]
    assert conv1.tmp_inputs[1].quantization is not None
    assert conv1.tmp_inputs[1].quantization.scale.len() == int(weight_shape[0] / group)
    assert conv1.tmp_inputs[1].quantization.zero_point.len() == int(weight_shape[0] / group)

    conv2 = intermediate_tflite_model_provider.get_operators()[4]
    assert conv2.tmp_inputs[1].quantization is not None
    assert conv2.tmp_inputs[1].quantization.scale.len() == int(weight_shape[0] / group)
    assert conv2.tmp_inputs[1].quantization.zero_point.len() == int(weight_shape[0] / group)
