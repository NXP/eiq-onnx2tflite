#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx.helper
import pytest
from onnx import TensorProto

from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.tflite_optimizer.graph_utils import op_type_to_builtin_operator_map
from onnx2tflite.src.tflite_optimizer.optimizer import Optimization
from tests import executors


def _get_sum_model() -> (onnx.ModelProto, list[int]):
    shape = [2, 3, 4, 5]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('ReduceSum', ['x1'], ['y11']),
                onnx.helper.make_node('ReduceSum', ['x2'], ['y21']),
                onnx.helper.make_node('Reshape', ['y11', 'shape1'], ['y12']),
                onnx.helper.make_node('Reshape', ['y12', 'shape2'], ['y13']),
                onnx.helper.make_node('Reshape', ['y21', 'shape1'], ['y22']),
                onnx.helper.make_node('Reshape', ['y22', 'shape2'], ['y23']),
                onnx.helper.make_node('Concat', ['y13', 'y23'], ['y'], axis=0),
            ],
            'Test',
            [
                onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape)
            ], [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor('shape1', TensorProto.INT64, [2], [1, 1]),
                onnx.helper.make_tensor('shape2', TensorProto.INT64, [4], [1, 1, 1, 1])
            ]
        )
    )

    # Quantize the model.
    calibration_reader = RandomDataCalibrationDataReader({
        'x1': InputSpec(shape, np.float32),
        'x2': InputSpec(shape, np.float32, 1., 2.)
    })
    q_config = QuantizationConfig(calibration_reader)
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    return onnx_model, shape


def _get_reduce_prod_model() -> (onnx.ModelProto, list[int]):
    shape = [2, 3, 4, 5]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('ReduceProd', ['x1'], ['y11']),
                onnx.helper.make_node('ReduceProd', ['x2'], ['y21']),
                onnx.helper.make_node('Reshape', ['y11', 'shape1'], ['y12']),
                onnx.helper.make_node('Reshape', ['y12', 'shape2'], ['y13']),
                onnx.helper.make_node('Reshape', ['y21', 'shape1'], ['y22']),
                onnx.helper.make_node('Reshape', ['y22', 'shape2'], ['y23']),
                onnx.helper.make_node('Concat', ['y13', 'y23'], ['y'], axis=0),
            ],
            'Test',
            [
                onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape)
            ], [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor('shape1', TensorProto.INT64, [2], [1, 1]),
                onnx.helper.make_tensor('shape2', TensorProto.INT64, [4], [1, 1, 1, 1])
            ]
        )
    )

    # Quantize the model.
    calibration_reader = RandomDataCalibrationDataReader({
        'x1': InputSpec(shape, np.float32),
        'x2': InputSpec(shape, np.float32, 1., 2.)
    })
    q_config = QuantizationConfig(calibration_reader)
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    return onnx_model, shape


def _get_mean_model() -> (onnx.ModelProto, list[int]):
    shape = [2, 3, 4, 5]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('ReduceMean', ['x1'], ['y11']),
                onnx.helper.make_node('ReduceMean', ['x2'], ['y21']),
                onnx.helper.make_node('Reshape', ['y11', 'shape1'], ['y12']),
                onnx.helper.make_node('Reshape', ['y12', 'shape2'], ['y13']),
                onnx.helper.make_node('Reshape', ['y21', 'shape1'], ['y22']),
                onnx.helper.make_node('Reshape', ['y22', 'shape2'], ['y23']),
                onnx.helper.make_node('Concat', ['y13', 'y23'], ['y'], axis=0),
            ],
            'Test',
            [
                onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape)
            ], [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor('shape1', TensorProto.INT64, [2], [1, 1]),
                onnx.helper.make_tensor('shape2', TensorProto.INT64, [4], [1, 1, 1, 1])
            ]
        )
    )

    # Quantize the model.
    calibration_reader = RandomDataCalibrationDataReader({
        'x1': InputSpec(shape, np.float32),
        'x2': InputSpec(shape, np.float32, 1., 2.)
    })
    q_config = QuantizationConfig(calibration_reader)
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    return onnx_model, shape


def _get_leaky_relu_model() -> (onnx.ModelProto, list[int]):
    shape = [5, 5]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('LeakyRelu', ['x1'], ['y11']),
                onnx.helper.make_node('LeakyRelu', ['x2'], ['y21']),
                onnx.helper.make_node('Reshape', ['y11', 'shape1'], ['y12']),
                onnx.helper.make_node('Reshape', ['y12', 'shape2'], ['y13']),
                onnx.helper.make_node('Reshape', ['y21', 'shape1'], ['y22']),
                onnx.helper.make_node('Reshape', ['y22', 'shape2'], ['y23']),
                onnx.helper.make_node('Concat', ['y13', 'y23'], ['y'], axis=0),
            ],
            'Test',
            [
                onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape)
            ], [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor('shape1', TensorProto.INT64, [1], [np.prod(shape)]),
                onnx.helper.make_tensor('shape2', TensorProto.INT64, [len(shape)], shape)
            ]
        )
    )

    # Quantize the model.
    calibration_reader = RandomDataCalibrationDataReader({
        'x1': InputSpec(shape, np.float32),
        'x2': InputSpec(shape, np.float32, 1., 2.)
    })
    q_config = QuantizationConfig(calibration_reader)
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    return onnx_model, shape


def _get_hard_swish_model() -> (onnx.ModelProto, list[int]):
    shape = [5, 5]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('HardSwish', ['x1'], ['y11']),
                onnx.helper.make_node('HardSwish', ['x2'], ['y21']),
                onnx.helper.make_node('Reshape', ['y11', 'shape1'], ['y12']),
                onnx.helper.make_node('Reshape', ['y12', 'shape2'], ['y13']),
                onnx.helper.make_node('Reshape', ['y21', 'shape1'], ['y22']),
                onnx.helper.make_node('Reshape', ['y22', 'shape2'], ['y23']),
                onnx.helper.make_node('Concat', ['y13', 'y23'], ['y'], axis=0),
            ],
            'Test',
            [
                onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape)
            ], [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor('shape1', TensorProto.INT64, [1], [np.prod(shape)]),
                onnx.helper.make_tensor('shape2', TensorProto.INT64, [len(shape)], shape)
            ]
        )
    )

    # Quantize the model.
    calibration_reader = RandomDataCalibrationDataReader({
        'x1': InputSpec(shape, np.float32),
        'x2': InputSpec(shape, np.float32, 1., 2.)
    })
    q_config = QuantizationConfig(calibration_reader)
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    return onnx_model, shape


def _get_fully_connected_model() -> (onnx.ModelProto, list[int]):
    shape = [5, 5]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Gemm', ['x1', 'x1'], ['y11'], transB=1),
                onnx.helper.make_node('Gemm', ['x2', 'x2'], ['y21'], transB=1),
                onnx.helper.make_node('Reshape', ['y11', 'shape1'], ['y12']),
                onnx.helper.make_node('Reshape', ['y12', 'shape2'], ['y13']),
                onnx.helper.make_node('Reshape', ['y21', 'shape1'], ['y22']),
                onnx.helper.make_node('Reshape', ['y22', 'shape2'], ['y23']),
                onnx.helper.make_node('Concat', ['y13', 'y23'], ['y'], axis=0),
            ],
            'Test',
            [
                onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape)
            ], [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor('shape1', TensorProto.INT64, [1], [np.prod(shape)]),
                onnx.helper.make_tensor('shape2', TensorProto.INT64, [len(shape)], shape)
            ]
        )
    )

    # Quantize the model.
    calibration_reader = RandomDataCalibrationDataReader({
        'x1': InputSpec(shape, np.float32),
        'x2': InputSpec(shape, np.float32, 1., 2.)
    })
    q_config = QuantizationConfig(calibration_reader)
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    return onnx_model, shape


def _get_batch_mat_mul_model() -> (onnx.ModelProto, list[int]):
    shape = [2, 4, 4]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('MatMul', ['x1', 'x1'], ['y11']),
                onnx.helper.make_node('MatMul', ['x2', 'x2'], ['y21']),
                onnx.helper.make_node('Reshape', ['y11', 'shape1'], ['y12']),
                onnx.helper.make_node('Reshape', ['y12', 'shape2'], ['y13']),
                onnx.helper.make_node('Reshape', ['y21', 'shape1'], ['y22']),
                onnx.helper.make_node('Reshape', ['y22', 'shape2'], ['y23']),
                onnx.helper.make_node('Concat', ['y13', 'y23'], ['y'], axis=0),
            ],
            'Test',
            [
                onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape)
            ], [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor('shape1', TensorProto.INT64, [1], [np.prod(shape)]),
                onnx.helper.make_tensor('shape2', TensorProto.INT64, [len(shape)], shape)
            ]
        )
    )

    # Quantize the model.
    calibration_reader = RandomDataCalibrationDataReader({
        'x1': InputSpec(shape, np.float32),
        'x2': InputSpec(shape, np.float32, 1., 2.)
    })
    q_config = QuantizationConfig(calibration_reader)
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    return onnx_model, shape


def _get_2_input_test_model(op_type: str):
    """ The `onnx2quant` is quite smart, so it is not trivial to create a test case where there is an `Add` operator
         followed by a `Quantize` which is changing the quantization parameters. Here, extra `Reshape` operators are
         added. They force the addition of the `Quantize`, but they are removed in an optimization step.
    """
    shape = [2, 3, 4, 5]

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node(op_type, ['x1', 'x2'], ['y1']),
                onnx.helper.make_node(op_type, ['x3', 'x4'], ['y2']),

                # The `Reshape` pairs negate each other, so they will get removed by an optimization.
                onnx.helper.make_node('Reshape', ['y1', 'shape1'], ['y3']),
                onnx.helper.make_node('Reshape', ['y3', 'shape2'], ['y4']),
                onnx.helper.make_node('Reshape', ['y2', 'shape1'], ['y5']),
                onnx.helper.make_node('Reshape', ['y5', 'shape2'], ['y6']),

                onnx.helper.make_node('Concat', ['y6', 'y4'], ['y'], axis=0)
            ],
            'Test',
            [
                onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info('x3', TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info('x4', TensorProto.FLOAT, shape)
            ],
            [
                onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ()),
            ],
            [
                onnx.helper.make_tensor('shape1', TensorProto.INT64, [1], [np.prod(shape)]),
                onnx.helper.make_tensor('shape2', TensorProto.INT64, [len(shape)], shape)
            ]
        )
    )

    # Quantize the model.
    calibration_reader = RandomDataCalibrationDataReader({
        'x1': InputSpec(shape, np.float32),
        'x2': InputSpec(shape, np.float32, 1., 2.),
        'x3': InputSpec(shape, np.float32, 2., 3.),
        'x4': InputSpec(shape, np.float32, 3., 4.)
    })
    q_config = QuantizationConfig(calibration_reader)
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    return onnx_model, shape


@pytest.mark.parametrize('op_type', [
    'Add', 'Mul', 'Sub', 'PRelu'
])
def test_fuse_quantize_into_preceding_op__2_input_ops__not_optimized(op_type: str, intermediate_tflite_model_provider):
    """ This test makes sure that the following tests are valid. """
    onnx_model, shape = _get_2_input_test_model(op_type)

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
        2: (np.random.random(shape) + 2).astype(np.float32),
        3: (np.random.random(shape) + 3).astype(np.float32)
    }

    # Prohibit the optimization and make sure the extra `Quantize` operator is added.
    config = ConversionConfig()
    config.optimization_blacklist = [Optimization.FUSE_QUANTIZE_INTO_PRECEDING_OPS]
    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config, atol=0.05)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        op_type_to_builtin_operator_map[op_type],
        op_type_to_builtin_operator_map[op_type],
        BuiltinOperator.QUANTIZE,  # This is the extra `Quantize` operator which changes the quantization parameters.
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


@pytest.mark.parametrize('op_type', [
    'Add', 'Mul', 'Sub'
])
def test_fuse_quantize_into_preceding_op__2_input_ops(op_type: str, intermediate_tflite_model_provider):
    onnx_model, shape = _get_2_input_test_model(op_type)

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
        2: (np.random.random(shape) + 2).astype(np.float32),
        3: (np.random.random(shape) + 3).astype(np.float32)
    }

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, atol=0.05)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        op_type_to_builtin_operator_map[op_type],
        op_type_to_builtin_operator_map[op_type],
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__batch_mat_mul__non_optimized(intermediate_tflite_model_provider):
    onnx_model, shape = _get_batch_mat_mul_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    config = ConversionConfig()
    config.optimization_blacklist = [Optimization.FUSE_QUANTIZE_INTO_PRECEDING_OPS]

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, atol=0.05, conversion_config=config)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.BATCH_MATMUL,
        BuiltinOperator.BATCH_MATMUL,
        BuiltinOperator.QUANTIZE,  # The extra `Quantize` op.
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__batch_mat_mul(intermediate_tflite_model_provider):
    onnx_model, shape = _get_batch_mat_mul_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, atol=0.05)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.BATCH_MATMUL,
        BuiltinOperator.BATCH_MATMUL,
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__fully_connected__non_optimized(intermediate_tflite_model_provider):
    onnx_model, shape = _get_fully_connected_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    config = ConversionConfig()
    config.optimization_blacklist = [Optimization.FUSE_QUANTIZE_INTO_PRECEDING_OPS]

    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.FULLY_CONNECTED,
        BuiltinOperator.FULLY_CONNECTED,
        BuiltinOperator.QUANTIZE,  # The extra `Quantize` op.
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__fully_connected(intermediate_tflite_model_provider):
    onnx_model, shape = _get_fully_connected_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, atol=0.07)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.FULLY_CONNECTED,
        BuiltinOperator.FULLY_CONNECTED,
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__hard_swish__non_optimized(intermediate_tflite_model_provider):
    onnx_model, shape = _get_hard_swish_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    config = ConversionConfig()
    config.optimization_blacklist = [Optimization.FUSE_QUANTIZE_INTO_PRECEDING_OPS]

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config, atol=0.007)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.HARD_SWISH,
        BuiltinOperator.HARD_SWISH,
        BuiltinOperator.QUANTIZE,  # The extra `Quantize` op.
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__hard_swish(intermediate_tflite_model_provider):
    onnx_model, shape = _get_hard_swish_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, atol=0.007)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.HARD_SWISH,
        BuiltinOperator.HARD_SWISH,
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__leaky_relu__non_optimized(intermediate_tflite_model_provider):
    onnx_model, shape = _get_leaky_relu_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    config = ConversionConfig()
    config.optimization_blacklist = [Optimization.FUSE_QUANTIZE_INTO_PRECEDING_OPS]

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config, atol=0.008)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.LEAKY_RELU,
        BuiltinOperator.LEAKY_RELU,
        BuiltinOperator.QUANTIZE,  # The extra `Quantize` op.
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__leaky_relu(intermediate_tflite_model_provider):
    onnx_model, shape = _get_leaky_relu_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, atol=0.008)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.LEAKY_RELU,
        BuiltinOperator.LEAKY_RELU,
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__mean__non_optimized(intermediate_tflite_model_provider):
    onnx_model, shape = _get_mean_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    config = ConversionConfig()
    config.optimization_blacklist = [Optimization.FUSE_QUANTIZE_INTO_PRECEDING_OPS]

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.MEAN,
        BuiltinOperator.MEAN,
        BuiltinOperator.QUANTIZE,  # The extra `Quantize` op.
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__mean(intermediate_tflite_model_provider):
    onnx_model, shape = _get_mean_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.MEAN,
        BuiltinOperator.MEAN,
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__reduce_prod__non_optimized(intermediate_tflite_model_provider):
    onnx_model, shape = _get_reduce_prod_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    config = ConversionConfig()
    config.optimization_blacklist = [Optimization.FUSE_QUANTIZE_INTO_PRECEDING_OPS]

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.REDUCE_PROD,
        BuiltinOperator.REDUCE_PROD,
        BuiltinOperator.QUANTIZE,  # The extra `Quantize` op.
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__reduce_prod(intermediate_tflite_model_provider):
    onnx_model, shape = _get_reduce_prod_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.REDUCE_PROD,
        BuiltinOperator.REDUCE_PROD,
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__sum__non_optimized(intermediate_tflite_model_provider):
    onnx_model, shape = _get_sum_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    config = ConversionConfig()
    config.optimization_blacklist = [Optimization.FUSE_QUANTIZE_INTO_PRECEDING_OPS]

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.SUM,
        BuiltinOperator.SUM,
        BuiltinOperator.QUANTIZE,  # The extra `Quantize` op.
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])


def test_fuse_quantize_into_preceding_op__sum(intermediate_tflite_model_provider):
    onnx_model, shape = _get_sum_model()

    np.random.seed(42)
    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: (np.random.random(shape) + 1).astype(np.float32),
    }

    # The error is caused by the `Concatenation` having to re-quantize data into a bad range.
    executors.convert_run_compare(onnx_model, input_data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.SUM,
        BuiltinOperator.SUM,
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.DEQUANTIZE
    ])
