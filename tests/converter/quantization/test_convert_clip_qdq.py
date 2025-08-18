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
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options, maximum_options, minimum_options, \
    quantize_options
from tests import executors


def test_convert_clip_qdq__as_relu(intermediate_tflite_model_provider):
    input_shape = [4, 6]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Clip", ["x"], ["output"])
        ],
        'Softmax test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    # Quantize ONNX model
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert intermediate_tflite_model_provider.get_operator_code_at_index(1) == BuiltinOperator.RELU
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize(
    "min, max",
    [
        pytest.param(-.1, .1, id='<-0.1, 0.1>'),
        pytest.param(0., 0.42, id='<0, 0.42>'),
    ])
def test_convert_clip_qdq__non_relu(min: float, max: float, intermediate_tflite_model_provider):
    input_shape = [5, 10, 15]

    np.random.seed(42)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x', 'min', 'max'], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)],
        [
            onnx.helper.make_tensor('min', TensorProto.FLOAT, [1], [min]),
            onnx.helper.make_tensor('max', TensorProto.FLOAT, [1], [max])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize(
    "min, max, builtin_operator",
    [
        pytest.param(-1., 1., BuiltinOperator.RELU_N1_TO_1, id='<-1, 1> (ReluN1To1)'),
        # QDQ Clip is never converted to Relu0To1 or Relu6 because QDQQuantizer adjust input q-params
        # to match outputs'. Relu op is used instead.
    ])
def test_convert_clip_qdq__as_special_relu(min: float, max: float, builtin_operator: BuiltinOperator,
                                           intermediate_tflite_model_provider):
    input_shape = [5, 10, 15]
    np.random.seed(42)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x', 'min', 'max'], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)],
        [
            onnx.helper.make_tensor('min', TensorProto.FLOAT, [1], [min]),
            onnx.helper.make_tensor('max', TensorProto.FLOAT, [1], [max])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec(input_shape,
                       custom_data_generator=lambda shape: np.random.random(shape).astype(np.float32) * 10 - 4)
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    opcode_index = ops[1].opcode_index

    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert intermediate_tflite_model_provider.get().operator_codes.get(opcode_index).builtin_code == builtin_operator
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize(
    "max",
    [
        pytest.param(6., id='<min, 6> (Relu)'),
        pytest.param(1., id='<min, 1> (Relu)'),
    ])
def test_convert_clip_qdq__missing_min(max: float, intermediate_tflite_model_provider):
    input_shape = [5, 10, 15]
    np.random.seed(42)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x', '', 'max'], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)],
        [
            onnx.helper.make_tensor('max', TensorProto.FLOAT, [1], [max])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()

    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert intermediate_tflite_model_provider.get_operator_code_at_index(1) == BuiltinOperator.RELU
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


@pytest.mark.parametrize(
    "min, max",
    [
        pytest.param(-.1, .1, id='<-0.1, 0.1>'),
        pytest.param(0., 0.42, id='<0, 0.42>'),
    ])
def test_convert_clip_qdq__dynamic_min_max(min: float, max: float, intermediate_tflite_model_provider):
    input_shape = [5, 10, 15]

    np.random.seed(42)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x', 'min', 'max'], ['y'])],
        'Clip test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info('min', TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info('max', TensorProto.FLOAT, [1]),
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_spec = {
        "x": InputSpec(input_shape, np.float32),
        "min": InputSpec([1], np.float32),
        "max": InputSpec([1], np.float32),
    }
    calibration_data_reader = RandomDataCalibrationDataReader(input_spec, num_samples=50)
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.random.rand(*input_shape).astype(np.float32) - 0.5,
        1: np.array([min]).astype(np.float32),
        2: np.array([max]).astype(np.float32),
    }

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 6
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[3].builtin_options, maximum_options.Maximum)
    assert isinstance(ops[4].builtin_options, minimum_options.Minimum)
    assert isinstance(ops[5].builtin_options, dequantize_options.Dequantize)
