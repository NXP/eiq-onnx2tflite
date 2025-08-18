#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx.helper
from onnx import TensorProto

from onnx2quant.qdq_quantization import RandomDataCalibrationDataReader, QDQQuantizer
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from tests import executors


def test_simple_case(intermediate_tflite_model_provider):
    # This test would also detect any changes to the `HardSigmoid` converter, which have not been reflected in the
    #  optimization.

    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('HardSigmoid', ['x'], ['x1'], alpha=1. / 6.),
                onnx.helper.make_node('Mul', ['x', 'x1'], ['y'])
            ],
            'HardSwish combining test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())]
        )
    )
    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.HARD_SWISH
    ])


def test_deconstructed__different_input_order(intermediate_tflite_model_provider):
    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Mul', ['x', 'alpha'], ['x1']),
                onnx.helper.make_node('Add', ['beta', 'x1'], ['x2']),  # Swapped inputs.
                onnx.helper.make_node('Min', ['one', 'x2'], ['x3']),  # Swapped inputs.
                onnx.helper.make_node('Relu', ['x3'], ['x4']),
                onnx.helper.make_node('Mul', ['x4', 'x'], ['y']),  # Swapped inputs.
            ],
            'HardSwish combining test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('alpha', TensorProto.FLOAT, [1], [1. / 6.]),
                onnx.helper.make_tensor('beta', TensorProto.FLOAT, [1], [.5]),
                onnx.helper.make_tensor('one', TensorProto.FLOAT, [1], [1.])
            ]
        )
    )
    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.HARD_SWISH
    ])


def test_deconstructed__wrong_alpha(intermediate_tflite_model_provider):
    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Mul', ['x', 'alpha'], ['x1']),
                onnx.helper.make_node('Add', ['beta', 'x1'], ['x2']),
                onnx.helper.make_node('Min', ['one', 'x2'], ['x3']),
                onnx.helper.make_node('Relu', ['x3'], ['x4']),
                onnx.helper.make_node('Mul', ['x4', 'x'], ['y']),
            ],
            'HardSwish combining test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('alpha', TensorProto.FLOAT, [1], [0.42]),  # Wrong value.
                onnx.helper.make_tensor('beta', TensorProto.FLOAT, [1], [.5]),
                onnx.helper.make_tensor('one', TensorProto.FLOAT, [1], [1.])
            ]
        )
    )
    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.MUL,
        BuiltinOperator.ADD,
        BuiltinOperator.MINIMUM,
        BuiltinOperator.RELU,
        BuiltinOperator.MUL
    ])


def test_deconstructed__wrong_beta(intermediate_tflite_model_provider):
    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Mul', ['x', 'alpha'], ['x1']),
                onnx.helper.make_node('Add', ['beta', 'x1'], ['x2']),
                onnx.helper.make_node('Min', ['one', 'x2'], ['x3']),
                onnx.helper.make_node('Relu', ['x3'], ['x4']),
                onnx.helper.make_node('Mul', ['x4', 'x'], ['y']),
            ],
            'HardSwish combining test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('alpha', TensorProto.FLOAT, [1], [1. / 6.]),
                onnx.helper.make_tensor('beta', TensorProto.FLOAT, [1], [.42]),  # Wrong value.
                onnx.helper.make_tensor('one', TensorProto.FLOAT, [1], [1.])
            ]
        )
    )
    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.MUL,
        BuiltinOperator.ADD,
        BuiltinOperator.MINIMUM,
        BuiltinOperator.RELU,
        BuiltinOperator.MUL
    ])


def test_deconstructed__wrong_upper_clip(intermediate_tflite_model_provider):
    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Mul', ['x', 'alpha'], ['x1']),
                onnx.helper.make_node('Add', ['beta', 'x1'], ['x2']),
                onnx.helper.make_node('Min', ['one', 'x2'], ['x3']),
                onnx.helper.make_node('Relu', ['x3'], ['x4']),
                onnx.helper.make_node('Mul', ['x4', 'x'], ['y']),
            ],
            'HardSwish combining test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('alpha', TensorProto.FLOAT, [1], [1. / 6.]),
                onnx.helper.make_tensor('beta', TensorProto.FLOAT, [1], [.5]),
                onnx.helper.make_tensor('one', TensorProto.FLOAT, [1], [0.42])  # Wrong value.
            ]
        )
    )
    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.MUL,
        BuiltinOperator.ADD,
        BuiltinOperator.MINIMUM,
        BuiltinOperator.RELU,
        BuiltinOperator.MUL
    ])


def test_hard_swish__quantized(intermediate_tflite_model_provider):
    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('HardSwish', ['x'], ['y'])
            ],
            'HardSwish combining test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())]
        )
    )
    # Quantize the model.
    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    data = np.random.random(shape).astype(np.float32)

    # ONNX runs in float32, and only the input and final output are quantized.
    # TFLite runs everything quantized, so there is a small error.
    executors.convert_run_compare(onnx_model, data, atol=0.0026)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE,
        BuiltinOperator.HARD_SWISH,
        BuiltinOperator.DEQUANTIZE
    ])


def test_deconstructed__quantized(intermediate_tflite_model_provider):
    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('HardSigmoid', ['x'], ['x1'], alpha=1. / 6.),
                onnx.helper.make_node('Mul', ['x', 'x1'], ['y'])
            ],
            'HardSwish combining test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())]
        )
    )
    # Quantize the model.
    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    data = np.random.random(shape).astype(np.float32)

    # ONNX runs in float32, and only the input and final output are quantized.
    # TFLite runs everything quantized, so there is a small error.
    executors.convert_run_compare(onnx_model, data, atol=0.0026)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE,
        BuiltinOperator.HARD_SWISH,
        BuiltinOperator.DEQUANTIZE
    ])
