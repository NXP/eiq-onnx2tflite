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
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from tests import executors
from tests.executors import OnnxExecutor, TFLiteExecutor
from onnx2tflite.src.tflite_optimizer.optimizer import Optimization

logger.MIN_OUTPUT_IMPORTANCE = logger.MessageImportance.DEBUG


def test_replace_average_pool_with_sum__float32(intermediate_tflite_model_provider):
    shape = [12, 15, 5, 6]
    flat_shape = [12, 15]  # N and C must stay the same.
    w_shape = [42, 15]

    np.random.seed(42)
    w_data = np.random.random(w_shape).astype('float32')
    x_data = np.random.random(shape).astype('float32')

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('AveragePool', ['x'], ['x1'], kernel_shape=shape[2:]),
                onnx.helper.make_node('Reshape', ['x1', 'flat_shape'], ['x2']),
                onnx.helper.make_node('Gemm', ['x2', 'w'], ['y'], transB=1)
            ],
            'Test replace AveragePool before FullyConnected with Sum',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape), ],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
                onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [len(flat_shape)], flat_shape)
            ],
        )
    )

    config = ConversionConfig()
    config.optimization_whitelist = [
        Optimization.PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE,
        Optimization.REPLACE_AVERAGE_POOL_BEFORE_FULLY_CONNECTED_WITH_SUM
    ]

    executors.convert_run_compare(onnx_model, x_data, conversion_config=config)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.SUM, BuiltinOperator.FULLY_CONNECTED
    ])


def test_replace_average_pool_with_sum__wrong_shapes(intermediate_tflite_model_provider):
    shape = [12, 15, 5, 6]
    flat_shape = [9, 20]  # N and C don't stay unchanged by the `Reshape`.
    w_shape = [42, 20]

    np.random.seed(42)
    w_data = np.random.random(w_shape).astype('float32')
    x_data = np.random.random(shape).astype('float32')

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('AveragePool', ['x'], ['x1'], kernel_shape=shape[2:]),
                onnx.helper.make_node('Reshape', ['x1', 'flat_shape'], ['x2']),
                onnx.helper.make_node('Gemm', ['x2', 'w'], ['y'], transB=1)
            ],
            'Test replace AveragePool before FullyConnected with Sum',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape), ],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
                onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [len(flat_shape)], flat_shape)
            ],
        )
    )

    config = ConversionConfig()
    config.optimization_whitelist = [
        Optimization.PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE,
        Optimization.REPLACE_AVERAGE_POOL_BEFORE_FULLY_CONNECTED_WITH_SUM
    ]

    executors.convert_run_compare(onnx_model, x_data, conversion_config=config)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        # The `AveragePool2D` and `Reshape` remain.
        BuiltinOperator.AVERAGE_POOL_2D, BuiltinOperator.TRANSPOSE, BuiltinOperator.RESHAPE,
        BuiltinOperator.FULLY_CONNECTED
    ])


def _get_max_error(onnx_model: onnx.ModelProto, config: ConversionConfig, data: np.ndarray | dict[np.ndarray]) -> float:
    onnx.checker.check_model(onnx_model)

    tfl_model = convert.convert_model(onnx_model, config)

    onnx_executor = OnnxExecutor(onnx_model.SerializeToString())
    onnx_output = onnx_executor.inference(data)

    tflite_executor = TFLiteExecutor(model_content=bytes(tfl_model))
    tflite_output = tflite_executor.inference(data)

    return np.abs(onnx_output - tflite_output).max()


def test_replace_average_pool_with_sum__quantized():
    shape = [5, 6, 5, 6]
    flat_shape = [5, 6]  # N and C must stay the same.
    w_shape = [2, 6]

    np.random.seed(42)
    w_data = np.random.random(w_shape).astype('float32')
    x_data = np.random.random(shape).astype('float32')

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('AveragePool', ['x'], ['x1'], kernel_shape=shape[2:]),
                onnx.helper.make_node('Reshape', ['x1', 'flat_shape'], ['x2']),
                onnx.helper.make_node('Gemm', ['x2', 'w'], ['y'], transB=1)
            ],
            'Test replace AveragePool before FullyConnected with Sum',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape), ],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
                onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [len(flat_shape)], flat_shape)
            ],
        )
    )
    # Quantize the model.
    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    # Convert without the optimization and record the error.
    config = ConversionConfig()
    config.optimization_whitelist = []
    non_optimized_error = _get_max_error(onnx_model, config, x_data)

    # Convert with the optimization and record the error.
    config = ConversionConfig()
    config.optimization_whitelist = [
        Optimization.PERMUTE_FULLY_CONNECTED_WEIGHTS_AFTER_RESHAPE,
        Optimization.REPLACE_AVERAGE_POOL_BEFORE_FULLY_CONNECTED_WITH_SUM
    ]
    optimized_error = _get_max_error(onnx_model, config, x_data)

    # Compare the errors.
    # On Linux, they are 0.0068713427 and 0.0068712234, so the optimized version is slightly more accurate.
    assert np.allclose(non_optimized_error, optimized_error, atol=2.e-7)
