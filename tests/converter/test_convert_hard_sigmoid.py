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

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize(
    "shape",
    [
        [256], [10, 20], [5, 10, 10], [2, 4, 6, 8], [2, 4, 6, 8, 10], [1, 2, 3, 4, 5, 6]
    ])
def test_convert_hard_sigmoid__default_attributes(shape: list[int], intermediate_tflite_model_provider):
    np.random.seed(42)
    x_data = (np.random.rand(*shape).astype(np.float32) - 0.5) * 20  # Values from -10 to 10

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('HardSigmoid', ['x'], ['y'])],
        'HardSigmoid test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators(
        [BuiltinOperator.MUL, BuiltinOperator.ADD, BuiltinOperator.MINIMUM, BuiltinOperator.RELU])


@pytest.mark.parametrize(
    "alpha, beta",
    [
        pytest.param(2.0, 2.0, id='alpha = 2, beta = 2'),
        pytest.param(-1.0, 8.0, id='alpha = -1, beta = 8'),
        pytest.param(-0.123, -1.234, id='alpha = -0.123, beta = -1.234'),
        pytest.param(0.5, -0.2, id='alpha = 0.5, beta = -0.2'),
    ])
def test_convert_hard_sigmoid(alpha: float, beta: float, intermediate_tflite_model_provider):
    shape = [100]
    np.random.seed(42)
    x_data = (np.random.rand(*shape).astype(np.float32) - 0.5) * 20  # Values from -10 to 10

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('HardSigmoid', ['x'], ['y'], alpha=alpha, beta=beta)],
        'HardSigmoid test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators(
        [BuiltinOperator.MUL, BuiltinOperator.ADD, BuiltinOperator.MINIMUM, BuiltinOperator.RELU])


def test_convert_hard_sigmoid__neutral_alpha(intermediate_tflite_model_provider):
    shape = [100]

    alpha = 1.0  # No `Mul` operator needs to be added.

    np.random.seed(42)
    x_data = (np.random.rand(*shape).astype(np.float32) - 0.5) * 20  # Values from -10 to 10

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('HardSigmoid', ['x'], ['y'], alpha=alpha)],
        'HardSigmoid test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators(
        [BuiltinOperator.ADD, BuiltinOperator.MINIMUM, BuiltinOperator.RELU])


def test_convert_hard_sigmoid__neutral_beta(intermediate_tflite_model_provider):
    shape = [100]

    beta = 0.0  # No `Add` operator needs to be added.

    np.random.seed(42)
    x_data = (np.random.rand(*shape).astype(np.float32) - 0.5) * 20  # Values from -10 to 10

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('HardSigmoid', ['x'], ['y'], beta=beta)],
        'HardSigmoid test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators(
        [BuiltinOperator.MUL, BuiltinOperator.MINIMUM, BuiltinOperator.RELU])


def test_convert_hard_sigmoid__neutral_alpha_and_beta(intermediate_tflite_model_provider):
    shape = [100]

    alpha = 1.0  # No `Mul` operator needs to be added.
    beta = 0.0  # No `Add` operator needs to be added.

    np.random.seed(42)
    x_data = (np.random.rand(*shape).astype(np.float32) - 0.5) * 20  # Values from -10 to 10

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('HardSigmoid', ['x'], ['y'], alpha=alpha, beta=beta)],
        'HardSigmoid test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators(
        [BuiltinOperator.MINIMUM, BuiltinOperator.RELU])


def test_convert_hard_sigmoid__unsupported_type(intermediate_tflite_model_provider):
    shape = [100]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('HardSigmoid', ['x'], ['y'])],
        'HardSigmoid test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT16, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT16, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_hard_sigmoid__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('HardSigmoid', ['x1'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', type_, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.INVALID_ONNX_MODEL
