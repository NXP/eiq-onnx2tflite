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
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options, mirror_pad_options, pad_v2_options, \
    quantize_options
from tests import executors


@pytest.fixture
def qdq_model_with_pad__static_constant():
    input_shape = [10, 5, 3, 4]
    pads = [0, 2, 0, 0, 4, 1, 0, 2]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads', 'constant'], ['o'], mode='constant')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
            onnx.helper.make_tensor('constant', TensorProto.FLOAT, [1], [0.214159]),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_pad_qdq__static_constant(qdq_model_with_pad__static_constant, intermediate_tflite_model_provider):
    input_shape = [10, 5, 3, 4]

    # Make sure some of the inputs are pad in quantized type
    input_data = (np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32) * 3) - 1.

    executors.convert_run_compare(qdq_model_with_pad__static_constant, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, pad_v2_options.PadV2)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


def test_convert_pad_qdq__dynamic_constant__not_first_op_in_model(intermediate_tflite_model_provider):
    input_shape = [10, 5, 3, 4]
    constant_shape = [1]
    pads = [0, 2, 0, 0, 4, 1, 0, 2]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Relu", ['x'], ['y']),
            onnx.helper.make_node("Pad", ['y', 'pads', 'constant'], ['o'], mode='constant'),
        ],
        'Pad test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("constant", TensorProto.FLOAT, constant_shape)
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads), ]
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec(input_shape, np.float32),
        "constant": InputSpec(constant_shape, np.float32),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    # Make sure some of the inputs are pad in quantized type
    input_data = {
        0: (np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32) * 3) - 1,
        1: np.asarray([0.31415], dtype=np.float32),
    }

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 5
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[3].builtin_options, pad_v2_options.PadV2)
    assert isinstance(ops[4].builtin_options, dequantize_options.Dequantize)


@pytest.fixture
def qdq_model_with_pad__default_constant():
    input_shape = [10, 5, 3, 4]
    pads = [0, 2, 0, 0, 4, 1, 0, 2]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads'], ['o'], mode='constant')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_pad_qdq__default_constant(qdq_model_with_pad__default_constant, intermediate_tflite_model_provider):
    input_shape = [10, 5, 3, 4]

    # Make sure some of the inputs are pad in quantized type
    input_data = (np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32) * 3) - 1.

    executors.convert_run_compare(qdq_model_with_pad__default_constant, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, pad_v2_options.PadV2)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)


@pytest.fixture
def qdq_model_with_pad__dynamic_constant():
    input_shape = [10, 5, 3, 4]
    constant_shape = [1]
    pads = [0, 2, 0, 0, 4, 1, 0, 2]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads', 'constant'], ['o'], mode='constant')],
        'Pad test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("constant", TensorProto.FLOAT, constant_shape),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads)]
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec(input_shape, np.float32),
        "constant": InputSpec(constant_shape, np.float32),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_pad_qdq__dynamic_constant(qdq_model_with_pad__dynamic_constant, intermediate_tflite_model_provider):
    input_shape = [10, 5, 3, 4]

    input_data = {
        0: np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.asarray([0.31415], dtype=np.float32),
    }

    executors.convert_run_compare(qdq_model_with_pad__dynamic_constant, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 4
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[2].builtin_options, pad_v2_options.PadV2)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)


@pytest.fixture
def qdq_model_with_pad__reflect_mode():
    input_shape = [10, 20, 4]
    pads = [0, 2, 3, 1, 0, 2]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads'], ['o'], mode='reflect')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_pad_qdq__reflect_mode(qdq_model_with_pad__reflect_mode, intermediate_tflite_model_provider):
    input_shape = [10, 20, 4]

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(qdq_model_with_pad__reflect_mode, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].builtin_options, mirror_pad_options.MirrorPad)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)
