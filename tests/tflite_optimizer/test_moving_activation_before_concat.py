#
# Copyright 2024-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx.helper
from onnx import TensorProto

from onnx2quant.qdq_quantization import QDQQuantizer, RandomDataCalibrationDataReader
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.conversion_config import ConversionConfig
from tests import executors
from onnx2tflite.src.tflite_optimizer.optimizer import Optimization


def test__single_input(intermediate_tflite_model_provider):
    """
                │                                   │
        ┌───────▼───────┐                        ┌──▼───┐
        │ Concatenation │                        │ Relu │
        └───────┬───────┘                        └──┬───┘
                │               ─────►              │
             ┌──▼───┐                       ┌───────▼───────┐
             │ Relu │                       │ Concatenation │
             └──┬───┘                       └───────┬───────┘
    """

    shape = [2, 3, 6, 8]
    w_shape = [3, 3, 3, 3]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Conv', ['x', 'w'], ['x1'], kernel_shape=[3, 3], auto_pad='SAME_UPPER'),
                onnx.helper.make_node('Concat', ['x1'], ['x2'], axis=0),
                onnx.helper.make_node('Relu', ['x2'], ['y'])
            ],
            'Test moving activation before Concatenation',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, np.random.random(w_shape).astype(np.float32))]
        )
    )

    data = np.random.random(shape).astype(np.float32)

    config = ConversionConfig()
    config.optimization_whitelist = [Optimization.MOVE_ACTIVATION_BEFORE_CONCAT]
    executors.convert_run_compare(onnx_model, data, conversion_config=config)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.CONV_2D,

        BuiltinOperator.RELU, BuiltinOperator.CONCATENATION,  # Relu is before the concat.

        BuiltinOperator.TRANSPOSE
    ])


def test__multiple_inputs(intermediate_tflite_model_provider):
    """
             │     │                             │         │
        ┌────▼─────▼────┐                     ┌──▼───┐  ┌──▼───┐
        │ Concatenation │                     │ Relu │  │ Relu │
        └───────┬───────┘                     └──┬───┘  └──┬───┘
                │               ─────►           └────┬────┘
             ┌──▼───┐                         ┌───────▼───────┐
             │ Relu │                         │ Concatenation │
             └──┬───┘                         └───────┬───────┘
    """

    shape = [2, 3, 6, 8]
    w_shape = [3, 3, 3, 3]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Conv', ['x', 'w'], ['x1'], kernel_shape=[3, 3], auto_pad='SAME_UPPER'),
                onnx.helper.make_node('Conv', ['x', 'w'], ['x2'], kernel_shape=[3, 3], auto_pad='SAME_UPPER'),
                onnx.helper.make_node('Concat', ['x1', 'x2'], ['x3'], axis=0),
                onnx.helper.make_node('Relu', ['x3'], ['y'])
            ],
            'Test moving activation before Concatenation',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, np.random.random(w_shape).astype(np.float32))]
        )
    )
    data = np.random.random(shape).astype(np.float32)

    config = ConversionConfig()
    config.optimization_whitelist = [Optimization.MOVE_ACTIVATION_BEFORE_CONCAT]
    executors.convert_run_compare(onnx_model, data, conversion_config=config)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.CONV_2D, BuiltinOperator.CONV_2D,

        BuiltinOperator.RELU, BuiltinOperator.RELU, BuiltinOperator.CONCATENATION,  # Relus are before the concat.

        BuiltinOperator.TRANSPOSE
    ])


def test__quantized(intermediate_tflite_model_provider):
    """
             │     │                                │               │
        ┌────▼─────▼────┐                        ┌──▼───┐        ┌──▼───┐
        │ Concatenation │                        │ Relu │        │ Relu │
        └───────┬───────┘                        └──┬───┘        └──┬───┘
                │               ─────►              └───────┬───────┘
             ┌──▼───┐                               ┌───────▼───────┐
             │ Relu │                               │ Concatenation │
             └──┬───┘                               └───────┬───────┘
    """
    np.random.seed(42)

    shape = [2, 3, 6, 8]
    w_shape = [3, 3, 3, 3]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Conv', ['x', 'w1'], ['x1'], kernel_shape=[3, 3], auto_pad='SAME_UPPER'),
                onnx.helper.make_node('Conv', ['x', 'w2'], ['x2'], kernel_shape=[3, 3], auto_pad='SAME_UPPER'),
                onnx.helper.make_node('Concat', ['x1', 'x2'], ['x3'], axis=0),
                onnx.helper.make_node('Relu', ['x3'], ['y'])
            ],
            'Test moving activation before Concatenation',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('w1', TensorProto.FLOAT, w_shape, np.random.random(w_shape).astype(np.float32)),
                onnx.helper.make_tensor('w2', TensorProto.FLOAT, w_shape, np.random.random(w_shape).astype(np.float32))
            ]
        )
    )

    # Quantize the model.
    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    data = np.random.random(shape).astype(np.float32)

    config = ConversionConfig()
    config.optimization_whitelist = [
        Optimization.MOVE_ACTIVATION_BEFORE_CONCAT, Optimization.FUSE_QUANTIZE_INTO_PRECEDING_OPS
    ]
    executors.convert_run_compare(onnx_model, data, conversion_config=config, atol=0.04)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE, BuiltinOperator.CONV_2D,
        BuiltinOperator.CONV_2D,

        BuiltinOperator.RELU, BuiltinOperator.RELU, BuiltinOperator.CONCATENATION,  # Relus are before the concat.

        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])


def test__without_conv(intermediate_tflite_model_provider):
    """ There is no `Conv2D` before the `Concat`, so the optimization will not be applied."""

    shape = [2, 3, 6, 8]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Concat', ['x', 'x'], ['x1'], axis=0),
                onnx.helper.make_node('Relu', ['x1'], ['y'])
            ],
            'Test moving activation before Concatenation',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        )
    )

    data = np.random.random(shape).astype(np.float32)

    config = ConversionConfig()
    config.optimization_whitelist = [Optimization.MOVE_ACTIVATION_BEFORE_CONCAT]
    executors.convert_run_compare(onnx_model, data, conversion_config=config)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.CONCATENATION, BuiltinOperator.RELU  # There is not change.
    ])
