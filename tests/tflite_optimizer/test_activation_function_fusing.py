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

from onnx2quant.qdq_quantization import QDQQuantizer, RandomDataCalibrationDataReader
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.lib.tflite.ActivationFunctionType import ActivationFunctionType
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.onnx_parser.onnx_model import NodeProto
from onnx2tflite.src.tflite_optimizer.graph_utils import builtin_operator_for_op_type
from tests import executors

_activation_function_type_for_name: dict[str, BuiltinOperator] = {
    'Relu': ActivationFunctionType.RELU,
    'ReluN1To1': ActivationFunctionType.RELU_N1_TO_1,
    'Relu6': ActivationFunctionType.RELU6,
    'Tanh': ActivationFunctionType.TANH,
    'Sign': ActivationFunctionType.SIGN_BIT
}


@pytest.mark.parametrize('activation', ['Relu', 'ReluN1To1', 'Relu6'])
def test_activation_fusing__conv2d(activation: str, intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8]
    w_shape = [4, 4, 1, 1]

    np.random.seed(42)
    w_data = np.random.random(w_shape).astype('float32') * 5. - 10.
    data = np.random.random(shape).astype('float32') * 5. - 10.

    match activation:
        case 'Relu':
            activation_node = onnx.helper.make_node('Relu', ['x1'], ['y'])
        case 'ReluN1To1':
            activation_node = onnx.helper.make_node('Clip', ['x1', '-1', '1'], ['y'])  # Will get converted to ReluN1To1
        case 'Relu6':
            activation_node = onnx.helper.make_node('Clip', ['x1', '0', '6'], ['y'])  # Will get converted to Relu6
        case _:
            raise ValueError

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Conv', ['x', 'w'], ['x1'], kernel_shape=w_shape[2:]),
                activation_node
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
                onnx.helper.make_tensor('-1', TensorProto.FLOAT, [1], [-1.]),
                onnx.helper.make_tensor('0', TensorProto.FLOAT, [1], [0.]),
                onnx.helper.make_tensor('1', TensorProto.FLOAT, [1], [1.]),
                onnx.helper.make_tensor('6', TensorProto.FLOAT, [1], [6.])
            ]
        )
    )
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.CONV_2D,
        # No activation function right here.
        BuiltinOperator.TRANSPOSE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[1].builtin_options.fused_activation_function == _activation_function_type_for_name[activation]


@pytest.mark.parametrize('activation', ['Relu', 'ReluN1To1', 'Relu6'])
def test_activation_fusing__conv3d(activation: str, intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8, 10]
    w_shape = [4, 4, 1, 1, 1]

    np.random.seed(42)
    w_data = np.random.random(w_shape).astype('float32') * 5. - 10.
    data = np.random.random(shape).astype('float32') * 5. - 10.

    match activation:
        case 'Relu':
            activation_node = onnx.helper.make_node('Relu', ['x1'], ['y'])
        case 'ReluN1To1':
            activation_node = onnx.helper.make_node('Clip', ['x1', '-1', '1'], ['y'])  # Will get converted to ReluN1To1
        case 'Relu6':
            activation_node = onnx.helper.make_node('Clip', ['x1', '0', '6'], ['y'])  # Will get converted to Relu6
        case _:
            raise ValueError

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Conv', ['x', 'w'], ['x1'], kernel_shape=w_shape[2:]),
                activation_node
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
                onnx.helper.make_tensor('-1', TensorProto.FLOAT, [1], [-1.]),
                onnx.helper.make_tensor('0', TensorProto.FLOAT, [1], [0.]),
                onnx.helper.make_tensor('1', TensorProto.FLOAT, [1], [1.]),
                onnx.helper.make_tensor('6', TensorProto.FLOAT, [1], [6.])
            ]
        )
    )
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.CONV_3D,
        # No activation function right here.
        BuiltinOperator.TRANSPOSE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[1].builtin_options.fused_activation_function == _activation_function_type_for_name[activation]


@pytest.mark.parametrize('activation', ['Relu', 'ReluN1To1', 'Relu6'])
def test_activation_fusing__depthwise_conv2d(activation: str, intermediate_tflite_model_provider):
    shape = [2, 1, 6, 8]
    w_shape = [1, 1, 3, 3]

    np.random.seed(42)
    w_data = np.random.random(w_shape).astype('float32') * 5. - 10.
    data = np.random.random(shape).astype('float32') * 5. - 10.

    match activation:
        case 'Relu':
            activation_node = onnx.helper.make_node('Relu', ['x1'], ['y'])
        case 'ReluN1To1':
            activation_node = onnx.helper.make_node('Clip', ['x1', '-1', '1'], ['y'])  # Will get converted to ReluN1To1
        case 'Relu6':
            activation_node = onnx.helper.make_node('Clip', ['x1', '0', '6'], ['y'])  # Will get converted to Relu6
        case _:
            raise ValueError

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Conv', ['x', 'w'], ['x1'], kernel_shape=w_shape[2:]),
                activation_node
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
                onnx.helper.make_tensor('-1', TensorProto.FLOAT, [1], [-1.]),
                onnx.helper.make_tensor('0', TensorProto.FLOAT, [1], [0.]),
                onnx.helper.make_tensor('1', TensorProto.FLOAT, [1], [1.]),
                onnx.helper.make_tensor('6', TensorProto.FLOAT, [1], [6.])
            ]
        )
    )
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.DEPTHWISE_CONV_2D,
        # No activation function right here.
        BuiltinOperator.TRANSPOSE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[1].builtin_options.fused_activation_function == _activation_function_type_for_name[activation]


@pytest.mark.parametrize('activation', ['Relu', 'ReluN1To1', 'Relu6'])
def test_activation_fusing__transpose_conv(activation: str, intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8]
    w_shape = [4, 4, 1, 1]

    np.random.seed(42)
    w_data = np.random.random(w_shape).astype('float32') * 5. - 10.
    data = np.random.random(shape).astype('float32') * 5. - 10.

    match activation:
        case 'Relu':
            activation_node = onnx.helper.make_node('Relu', ['x1'], ['y'])
        case 'ReluN1To1':
            activation_node = onnx.helper.make_node('Clip', ['x1', '-1', '1'], ['y'])  # Will get converted to ReluN1To1
        case 'Relu6':
            activation_node = onnx.helper.make_node('Clip', ['x1', '0', '6'], ['y'])  # Will get converted to Relu6
        case _:
            raise ValueError

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('ConvTranspose', ['x', 'w'], ['x1'], kernel_shape=w_shape[2:]),
                activation_node
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
                onnx.helper.make_tensor('-1', TensorProto.FLOAT, [1], [-1.]),
                onnx.helper.make_tensor('0', TensorProto.FLOAT, [1], [0.]),
                onnx.helper.make_tensor('1', TensorProto.FLOAT, [1], [1.]),
                onnx.helper.make_tensor('6', TensorProto.FLOAT, [1], [6.])
            ]
        )
    )
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.TRANSPOSE_CONV,
        # No activation function right here.
        BuiltinOperator.TRANSPOSE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[1].builtin_options.fused_activation_function == _activation_function_type_for_name[activation]


@pytest.mark.parametrize('activation', ['Relu', 'ReluN1To1', 'Relu6'])
def test_activation_fusing__max_pool_2d(activation: str, intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8]

    np.random.seed(42)
    data = np.random.random(shape).astype('float32') * 5. - 10.

    match activation:
        case 'Relu':
            activation_node = onnx.helper.make_node('Relu', ['x1'], ['y'])
        case 'ReluN1To1':
            activation_node = onnx.helper.make_node('Clip', ['x1', '-1', '1'], ['y'])  # Will get converted to ReluN1To1
        case 'Relu6':
            activation_node = onnx.helper.make_node('Clip', ['x1', '0', '6'], ['y'])  # Will get converted to Relu6
        case _:
            raise ValueError

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[3, 3]),
                activation_node
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('-1', TensorProto.FLOAT, [1], [-1.]),
                onnx.helper.make_tensor('0', TensorProto.FLOAT, [1], [0.]),
                onnx.helper.make_tensor('1', TensorProto.FLOAT, [1], [1.]),
                onnx.helper.make_tensor('6', TensorProto.FLOAT, [1], [6.])
            ]
        )
    )
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.MAX_POOL_2D,
        # No activation function right here.
        BuiltinOperator.TRANSPOSE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[1].builtin_options.fused_activation_function == _activation_function_type_for_name[activation]


@pytest.mark.parametrize('activation', ['Relu', 'ReluN1To1', 'Relu6'])
def test_activation_fusing__average_pool_2d(activation: str, intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8]

    np.random.seed(42)
    data = np.random.random(shape).astype('float32') * 5. - 10.

    match activation:
        case 'Relu':
            activation_node = onnx.helper.make_node('Relu', ['x1'], ['y'])
        case 'ReluN1To1':
            activation_node = onnx.helper.make_node('Clip', ['x1', '-1', '1'], ['y'])  # Will get converted to ReluN1To1
        case 'Relu6':
            activation_node = onnx.helper.make_node('Clip', ['x1', '0', '6'], ['y'])  # Will get converted to Relu6
        case _:
            raise ValueError

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('AveragePool', ['x'], ['x1'], kernel_shape=[3, 3]),
                activation_node
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('-1', TensorProto.FLOAT, [1], [-1.]),
                onnx.helper.make_tensor('0', TensorProto.FLOAT, [1], [0.]),
                onnx.helper.make_tensor('1', TensorProto.FLOAT, [1], [1.]),
                onnx.helper.make_tensor('6', TensorProto.FLOAT, [1], [6.])
            ]
        )
    )
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.AVERAGE_POOL_2D,
        # No activation function right here.
        BuiltinOperator.TRANSPOSE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[1].builtin_options.fused_activation_function == _activation_function_type_for_name[activation]


@pytest.mark.parametrize('activation', ['Relu', 'ReluN1To1', 'Relu6'])
def test_activation_fusing__svdf(activation: str, intermediate_tflite_model_provider):
    # Add this test once conversion is implemented.
    # Also add tests for quantized activation fusing.
    assert 'SVDF' not in NodeProto.op_type_to_attribute_constructor_map


@pytest.mark.parametrize('activation', ['Relu', 'ReluN1To1', 'Relu6'])
def test_activation_fusing__fully_connected(activation: str, intermediate_tflite_model_provider):
    shape = [5, 10]
    w_shape = [2, 10]

    np.random.seed(42)
    data = np.random.random(shape).astype('float32') * 5. - 10.
    w_data = np.random.random(w_shape).astype('float32') * 5. - 10.

    match activation:
        case 'Relu':
            activation_node = onnx.helper.make_node('Relu', ['x1'], ['y'])
        case 'ReluN1To1':
            activation_node = onnx.helper.make_node('Clip', ['x1', '-1', '1'], ['y'])  # Will get converted to ReluN1To1
        case 'Relu6':
            activation_node = onnx.helper.make_node('Clip', ['x1', '0', '6'], ['y'])  # Will get converted to Relu6
        case _:
            raise ValueError

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Gemm', ['x', 'w'], ['x1'], transB=1),
                activation_node
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
                onnx.helper.make_tensor('-1', TensorProto.FLOAT, [1], [-1.]),
                onnx.helper.make_tensor('0', TensorProto.FLOAT, [1], [0.]),
                onnx.helper.make_tensor('1', TensorProto.FLOAT, [1], [1.]),
                onnx.helper.make_tensor('6', TensorProto.FLOAT, [1], [6.])
            ]
        )
    )
    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.FULLY_CONNECTED
        # No activation function right here.
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[0].builtin_options.fused_activation_function == _activation_function_type_for_name[activation]


@pytest.mark.parametrize('op', ['Add', 'Mul', 'Sub', 'Div'])
@pytest.mark.parametrize('activation', ['Relu', 'ReluN1To1', 'Relu6'])
def test_activation_fusing__add__mul__sub__div(activation: str, op: str, intermediate_tflite_model_provider):
    shape = [2, 3, 4, 5]

    np.random.seed(42)
    a_data = np.random.random(shape).astype('float32') * 5. - 10.
    b_data = np.random.random(shape).astype('float32') * 5. - 10.

    match activation:
        case 'Relu':
            activation_node = onnx.helper.make_node('Relu', ['x1'], ['y'])
        case 'ReluN1To1':
            activation_node = onnx.helper.make_node('Clip', ['x1', '-1', '1'], ['y'])  # Will get converted to ReluN1To1
        case 'Relu6':
            activation_node = onnx.helper.make_node('Clip', ['x1', '0', '6'], ['y'])  # Will get converted to Relu6
        case _:
            raise ValueError

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node(op, ['a', 'b'], ['x1']),
                activation_node
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('a', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('b', TensorProto.FLOAT, shape, b_data),
                onnx.helper.make_tensor('-1', TensorProto.FLOAT, [1], [-1.]),
                onnx.helper.make_tensor('0', TensorProto.FLOAT, [1], [0.]),
                onnx.helper.make_tensor('1', TensorProto.FLOAT, [1], [1.]),
                onnx.helper.make_tensor('6', TensorProto.FLOAT, [1], [6.])
            ]
        )
    )
    executors.convert_run_compare(onnx_model, a_data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        builtin_operator_for_op_type(op)
        # No activation function right here.
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[0].builtin_options.fused_activation_function == _activation_function_type_for_name[activation]


def test_activation_fusing__quantized__conv2d(intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8]
    w_shape = [4, 4, 1, 1]

    np.random.seed(42)
    w_data = np.random.random(w_shape).astype('float32') * 5. - 10.
    data = np.random.random(shape).astype('float32') * 5. - 10.

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Conv', ['x', 'w'], ['x1'], kernel_shape=w_shape[2:], auto_pad='SAME_UPPER'),
                onnx.helper.make_node('Relu', ['x1'], ['y'])
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data)]
        )
    )
    # Quantize the model.
    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.CONV_2D,
        # No activation function right here.
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[2].builtin_options.fused_activation_function == ActivationFunctionType.RELU


def test_activation_fusing__quantized__depthwise_conv2d(intermediate_tflite_model_provider):
    shape = [2, 1, 6, 8]
    w_shape = [1, 1, 3, 3]  # input_channels == output_channels == group. This will get converted to DepthwiseConv2D.

    np.random.seed(42)
    w_data = np.random.random(w_shape).astype('float32') * 5. - 10.
    data = np.random.random(shape).astype('float32') * 5. - 10.

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Conv', ['x', 'w'], ['x1'], kernel_shape=w_shape[2:], auto_pad='SAME_UPPER'),
                onnx.helper.make_node('Relu', ['x1'], ['y'])
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data)]
        )
    )
    # Quantize the model.
    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.DEPTHWISE_CONV_2D,
        # No activation function right here.
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[2].builtin_options.fused_activation_function == ActivationFunctionType.RELU


def test_activation_fusing__quantized__transpose_conv(intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8]
    w_shape = [4, 4, 2, 3]

    np.random.seed(42)
    w_data = np.random.random(w_shape).astype('float32') * 5. - 10.
    data = np.random.random(shape).astype('float32') * 5. - 10.

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('ConvTranspose', ['x', 'w'], ['x1'], kernel_shape=w_shape[2:],
                                      auto_pad='SAME_UPPER'),
                onnx.helper.make_node('Relu', ['x1'], ['y'])
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data)]
        )
    )
    # Quantize the model.
    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.TRANSPOSE_CONV,
        # No activation function right here.
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[2].builtin_options.fused_activation_function == ActivationFunctionType.RELU


def test_activation_fusing__quantized__max_pool_2d(intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8]

    np.random.seed(42)
    data = np.random.random(shape).astype('float32') * 5. - 10.

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[3, 3], auto_pad='SAME_UPPER'),
                onnx.helper.make_node('Relu', ['x1'], ['y'])
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        )
    )
    # Quantize the model.
    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.MAX_POOL_2D,
        # No activation function right here.
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[2].builtin_options.fused_activation_function == ActivationFunctionType.RELU


def test_activation_fusing__quantized__max_pool_2d__different_output_quantization(
        intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8]

    np.random.seed(42)
    data = np.random.random(shape).astype('float32') * 5. - 10.

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('QuantizeLinear', ['x', 's1', 'zp1'], ['a']),

                onnx.helper.make_node('DequantizeLinear', ['a', 's1', 'zp1'], ['b']),
                onnx.helper.make_node('MaxPool', ['b'], ['c'], kernel_shape=[3, 3],
                                      auto_pad='SAME_UPPER'),
                onnx.helper.make_node('QuantizeLinear', ['c', 's1', 'zp1'], ['d']),

                onnx.helper.make_node('DequantizeLinear', ['d', 's1', 'zp1'], ['e']),
                onnx.helper.make_node('Relu', ['e'], ['f']),
                onnx.helper.make_node('QuantizeLinear', ['f', 's2', 'zp2'], ['g']),  # Different output quantization.

                onnx.helper.make_node('DequantizeLinear', ['g', 's2', 'zp2'], ['y']),
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('s1', TensorProto.FLOAT, [1], [0.0123]),
                onnx.helper.make_tensor('zp1', TensorProto.INT8, [1], [0]),
                onnx.helper.make_tensor('s2', TensorProto.FLOAT, [1], [0.042]),
                onnx.helper.make_tensor('zp2', TensorProto.INT8, [1], [42])
            ]
        )
    )

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.MAX_POOL_2D,
        BuiltinOperator.RELU,  # Not fused.
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE

    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[2].builtin_options.fused_activation_function == ActivationFunctionType.NONE


def test_activation_fusing__quantized__average_pool_2d(intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8]

    np.random.seed(42)
    data = np.random.random(shape).astype('float32') * 5. - 10.

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('QuantizeLinear', ['x', 's1', 'zp1'], ['a']),

                onnx.helper.make_node('DequantizeLinear', ['a', 's1', 'zp1'], ['b']),
                onnx.helper.make_node('AveragePool', ['b'], ['c'], kernel_shape=[3, 3],
                                      auto_pad='SAME_UPPER'),
                onnx.helper.make_node('QuantizeLinear', ['c', 's1', 'zp1'], ['d']),

                onnx.helper.make_node('DequantizeLinear', ['d', 's1', 'zp1'], ['e']),
                onnx.helper.make_node('Relu', ['e'], ['f']),
                onnx.helper.make_node('QuantizeLinear', ['f', 's1', 'zp1'], ['g']),  # Same output quantization.

                onnx.helper.make_node('DequantizeLinear', ['g', 's1', 'zp1'], ['y']),
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('s1', TensorProto.FLOAT, [], [0.0123]),
                onnx.helper.make_tensor('zp1', TensorProto.INT8, [], [0]),
            ]
        )
    )

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.AVERAGE_POOL_2D,
        # No activation function right here.
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[2].builtin_options.fused_activation_function == ActivationFunctionType.RELU


def test_activation_fusing__quantized__average_pool_2d__different_output_quantization(
        intermediate_tflite_model_provider):
    shape = [2, 4, 6, 8]

    np.random.seed(42)
    data = np.random.random(shape).astype('float32') * 5. - 10.

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('QuantizeLinear', ['x', 's1', 'zp1'], ['a']),

                onnx.helper.make_node('DequantizeLinear', ['a', 's1', 'zp1'], ['b']),
                onnx.helper.make_node('AveragePool', ['b'], ['c'], kernel_shape=[3, 3],
                                      auto_pad='SAME_UPPER'),
                onnx.helper.make_node('QuantizeLinear', ['c', 's1', 'zp1'], ['d']),

                onnx.helper.make_node('DequantizeLinear', ['d', 's1', 'zp1'], ['e']),
                onnx.helper.make_node('Relu', ['e'], ['f']),
                onnx.helper.make_node('QuantizeLinear', ['f', 's2', 'zp2'], ['g']),  # Different output quantization.

                onnx.helper.make_node('DequantizeLinear', ['g', 's2', 'zp2'], ['y']),
            ],
            'Activation fusing test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('s1', TensorProto.FLOAT, [], [0.0123]),
                onnx.helper.make_tensor('zp1', TensorProto.INT8, [], [0]),
                onnx.helper.make_tensor('s2', TensorProto.FLOAT, [], [0.042]),
                onnx.helper.make_tensor('zp2', TensorProto.INT8, [], [42])
            ]
        )
    )

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.AVERAGE_POOL_2D,
        BuiltinOperator.RELU,  # Not fused.
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE

    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[2].builtin_options.fused_activation_function == ActivationFunctionType.NONE


def test_activation_fusing__quantized__fully_connected(intermediate_tflite_model_provider):
    a_shape = [5, 10]
    b_shape = [2, 10]

    np.random.seed(42)
    data = {
        0: np.random.random(a_shape).astype('float32') * 5. - 10.,
        1: np.random.random(b_shape).astype('float32') * 5. - 10.
    }

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Gemm', ['a', 'b'], ['x1'], transB=1),
                onnx.helper.make_node('Relu', ['x1'], ['y'])
            ],
            'Activation fusing test',
            [
                onnx.helper.make_tensor_value_info('a', TensorProto.FLOAT, a_shape),
                onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT, b_shape)
            ],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        )
    )
    # Quantize the model.
    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.FULLY_CONNECTED,
        # No activation function right here.
        BuiltinOperator.DEQUANTIZE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[2].builtin_options.fused_activation_function == ActivationFunctionType.RELU


@pytest.mark.parametrize('op', ['Add', 'Mul', 'Sub'])  # Div is missing because we don't support qdq quantized version.
def test_activation_fusing__quantized__add__mul__sub(op: str, intermediate_tflite_model_provider):
    shape = [13, 37]

    np.random.seed(42)
    data = {
        0: np.random.random(shape).astype('float32') * 5. - 10.,
        1: np.random.random(shape).astype('float32') * 5. - 10.
    }

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node(op, ['a', 'b'], ['x']),
                onnx.helper.make_node('Relu', ['x'], ['y'])
            ],
            'Activation fusing test',
            [
                onnx.helper.make_tensor_value_info('a', TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT, shape)
            ],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        )
    )
    # Quantize the model.
    q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        builtin_operator_for_op_type(op),
        # No activation function right here.
        BuiltinOperator.DEQUANTIZE
    ])
    ops = intermediate_tflite_model_provider.get_operators()
    # noinspection PyUnresolvedReferences
    assert ops[2].builtin_options.fused_activation_function == ActivationFunctionType.RELU
