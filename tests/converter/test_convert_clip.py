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
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta import types
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize("_type", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_clip__quantized__into_maximum_minimum(_type: TensorProto.DataType, intermediate_tflite_model_provider):
    shape = [3, 14, 15]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Clip', ['x1', 'min', 'max'], ['x2']),
            onnx.helper.make_node('Reshape', ['x2', 'flat_shape'], ['x3']),
            onnx.helper.make_node('DequantizeLinear', ['x3', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', _type, [1], [12]),
            onnx.helper.make_tensor('min', _type, [], [24]),
            onnx.helper.make_tensor('max', _type, [], [42]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [1], [np.prod(shape)])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE,
        BuiltinOperator.MAXIMUM,
        BuiltinOperator.MINIMUM,
        BuiltinOperator.RESHAPE,
        BuiltinOperator.DEQUANTIZE
    ])


@pytest.mark.parametrize("_type", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_clip__quantized__into_relu(_type: TensorProto.DataType, intermediate_tflite_model_provider):
    shape = [3, 14, 15]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Clip', ['x1', 'min'], ['x2']),
            onnx.helper.make_node('Reshape', ['x2', 'flat_shape'], ['x3']),
            onnx.helper.make_node('DequantizeLinear', ['x3', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', _type, [1], [12]),
            onnx.helper.make_tensor('min', _type, [], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [1], [np.prod(shape)])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE,
        BuiltinOperator.RELU,
        BuiltinOperator.RESHAPE,
        BuiltinOperator.DEQUANTIZE
    ])


@pytest.mark.parametrize(
    "min, max",
    [
        pytest.param(-.1, .1, id='<-0.1, 0.1>'),
        pytest.param(-1., 1., id='<-1, 1>'),
        pytest.param(0., 0.42, id='<0, 0.42>'),
    ])
def test_convert_clip_v6(min: float, max: float):
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x'], ['y'], min=min, max=max)],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 6)])

    config = ConversionConfig({"ignore_opset_version": True})
    executors.convert_run_compare(onnx_model, x_data, conversion_config=config)


@pytest.mark.parametrize(
    "min, max",
    [
        pytest.param(-.1, .1, id='<-0.1, 0.1>'),
        pytest.param(0., 0.42, id='<0, 0.42>'),
    ])
def test_convert_clip_v11(min: float, max: float):
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x', 'min', 'max'], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('min', TensorProto.FLOAT, [1], [min]),
            onnx.helper.make_tensor('max', TensorProto.FLOAT, [1], [max])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "min, max",
    [
        pytest.param(-.1, .1, id='<-0.1, 0.1>'),
        pytest.param(0., 0.42, id='<0, 0.42>'),
    ])
def test_convert_clip_v11_dynamic_min_max(min: float, max: float):
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = {
        0: np.random.rand(*x_shape).astype(np.float32) - 0.5,
        1: np.array([min]).astype(np.float32),
        2: np.array([max]).astype(np.float32),
    }

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x', 'min', 'max'], ['y'])],
        'Clip test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('min', TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info('max', TensorProto.FLOAT, [1]),
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "min, max, builtin_operator",
    [
        pytest.param(-1., 1., BuiltinOperator.RELU_N1_TO_1, id='<-1, 1> (ReluN1To1)'),
        pytest.param(0., 6., BuiltinOperator.RELU6, id='<0, 6> (Relu6)'),
        pytest.param(0., 1., BuiltinOperator.RELU_0_TO_1, id='<0, 1> (Relu0To1)'),
    ])
def test_convert_clip_as_relu(min: float, max: float, builtin_operator: BuiltinOperator,
                              intermediate_tflite_model_provider):
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x', 'min', 'max'], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('min', TensorProto.FLOAT, [1], [min]),
            onnx.helper.make_tensor('max', TensorProto.FLOAT, [1], [max])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 1

    opcode_index = ops[0].opcode_index

    assert intermediate_tflite_model_provider.get().operator_codes.get(opcode_index).builtin_code == builtin_operator


@pytest.mark.parametrize(
    "min, max",
    [
        pytest.param(-10, 10, id='<-10, 10>'),
        pytest.param(0, 1, id='<0, 1>'),
        pytest.param(-1, 1, id='<-1, 1>'),
        pytest.param(0, 6, id='<0, 6>'),
        pytest.param(0, 5, id='<0, 5>'),
    ])
def test_convert_per_tensor_int_quantized_clip(min: float, max: float):
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = (np.random.rand(*x_shape).astype(np.float32) * 20.) - 10.

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['y1']),
            onnx.helper.make_node('Clip', ['y1', 'min', 'max'], ['y'])
        ],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('min', TensorProto.INT8, [1], [min]),
            onnx.helper.make_tensor('max', TensorProto.INT8, [1], [max]),
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [0.6]),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [1], [1])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "min, max, scale, zp, builtin_operator",
    [
        pytest.param(0, 1, 1., 0, BuiltinOperator.RELU_0_TO_1, id='<0, 1>'),
        pytest.param(-10, -9, 1., -10, BuiltinOperator.RELU_0_TO_1, id='<-10, -9>, zp=-10'),
        pytest.param(-10, -7, 1. / 3., -10, BuiltinOperator.RELU_0_TO_1, id='<-10, -7>, scale=1/3 zp=-10'),

        pytest.param(-1, 1, 1., 0, BuiltinOperator.RELU_N1_TO_1, id='<-1, 1>'),
        pytest.param(round((-1 / 0.1234) - 42), round((1 / 0.1234) - 42), 0.1234, -42, BuiltinOperator.RELU_N1_TO_1,
                     id='scale = 0.1234, zp=-42'),

        pytest.param(0, 6, 1., 0, BuiltinOperator.RELU6, id='<0, 6>'),
        pytest.param(-23, round((6 / 3.14159) - 23), 3.14159, -23, BuiltinOperator.RELU6, id='scale = e.14159, zp=-23'),
    ])
def test_convert_per_tensor_int_quantized_clip_as_special_relu(min: float, max: float, scale: float, zp: int,
                                                               builtin_operator: BuiltinOperator,
                                                               intermediate_tflite_model_provider):
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = (np.random.rand(*x_shape).astype(np.float32) * 20.) - 10.

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['y1']),
            onnx.helper.make_node('Clip', ['y1', 'min', 'max'], ['y'])
        ],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('min', TensorProto.INT8, [1], [min]),
            onnx.helper.make_tensor('max', TensorProto.INT8, [1], [max]),
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [scale]),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [1], [zp])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 2  # The first operator is the 'Quantize'

    opcode_index = ops[1].opcode_index

    assert intermediate_tflite_model_provider.get().operator_codes.get(opcode_index).builtin_code == builtin_operator


@pytest.mark.parametrize(
    "min, max, scale, zp, builtin_operator",
    [
        pytest.param(0, 1, 1., 0, BuiltinOperator.RELU_0_TO_1, id='<0, 1>'),
        pytest.param(128, 129, 1., 128, BuiltinOperator.RELU_0_TO_1, id='<128, 129>, zp=128'),
        pytest.param(0, 5, 0.2, 0, BuiltinOperator.RELU_0_TO_1, id='<0, 5>, scale=0.2'),
        pytest.param(128, 133, 0.2, 128, BuiltinOperator.RELU_0_TO_1, id='<128, 133>, scale=0.2, zp=128'),

        pytest.param(0, 6, 1., 0, BuiltinOperator.RELU6, id='<0, 6>'),
        pytest.param(0, 3, 2., 0, BuiltinOperator.RELU6, id='<0, 3>, scale=2'),
        pytest.param(42, 54, 0.5, 42, BuiltinOperator.RELU6, id='<12, 54>, scale=0.5, zp=42'),

        pytest.param(0, 2, 1., 1, BuiltinOperator.RELU_N1_TO_1, id='<0, 2>, zp=1'),
        pytest.param(0, 20, .1, 10, BuiltinOperator.RELU_N1_TO_1, id='<0, 20>, scale = 0.1, zp=10'),
        pytest.param(118, 138, .1, 128, BuiltinOperator.RELU_N1_TO_1, id='<118, 138>, scale = 0.1, zp=128'),
    ])
def test_convert_per_tensor_uint_quantized_clip_as_special_relu(min: float, max: float, scale: float, zp: int,
                                                                builtin_operator: BuiltinOperator,
                                                                intermediate_tflite_model_provider):
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = (np.random.rand(*x_shape).astype(np.float32) * 20.) - 10.

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['y1']),
            onnx.helper.make_node('Clip', ['y1', 'min', 'max'], ['y'])
        ],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.UINT8, ())],
        [
            onnx.helper.make_tensor('min', TensorProto.UINT8, [1], [min]),
            onnx.helper.make_tensor('max', TensorProto.UINT8, [1], [max]),
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [scale]),
            onnx.helper.make_tensor('zp', TensorProto.UINT8, [1], [zp])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 2  # The first operator is the 'Quantize'

    opcode_index = ops[1].opcode_index

    assert intermediate_tflite_model_provider.get().operator_codes.get(opcode_index).builtin_code == builtin_operator


@pytest.mark.parametrize(
    "scale, zp, data_type",
    [
        pytest.param(1., 128, TensorProto.UINT8, id='UINT8: scale = 1, zp = 128'),
        pytest.param(.123, 42, TensorProto.UINT8, id='UINT8: scale = 0.123, zp = 42'),
        pytest.param(13.37, 200, TensorProto.UINT8, id='UINT8: scale = 13.37, zp = 200'),

        pytest.param(1., 0, TensorProto.INT8, id='INT8: scale = 1, zp = 0'),
        pytest.param(.123, -42, TensorProto.INT8, id='INT8: scale = 0.123, zp = -42'),
        pytest.param(13.37, 23, TensorProto.INT8, id='INT8: scale = 13.37, zp = 23'),
    ])
def test_convert_per_tensor_quantized_clip_as_relu(scale: float, zp: int, data_type: TensorProto.DataType,
                                                   intermediate_tflite_model_provider):
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = (np.random.rand(*x_shape).astype(np.float32) * 20.) - 10.

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['y1']),
            onnx.helper.make_node('Clip', ['y1', 'min'], ['y'])
        ],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
        [
            onnx.helper.make_tensor('min', data_type, [1], [zp]),
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [scale]),
            onnx.helper.make_tensor('zp', data_type, [1], [zp])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 2  # The first operator is the 'Quantize'

    opcode_index = ops[1].opcode_index

    assert intermediate_tflite_model_provider.get().operator_codes.get(
        opcode_index).builtin_code == BuiltinOperator.RELU


def test_convert_per_channel_quantized_clip():
    x_shape = [5, 10, 15]

    scale = [.1, .2, .3, .4, .5]
    zp = [0, 1, -1, 5, -5]

    np.random.seed(42)

    # x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['y1'], axis=0),
            onnx.helper.make_node('Clip', ['y1', 'min', 'max'], ['y'])
        ],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT8, ())],
        [
            onnx.helper.make_tensor('min', TensorProto.INT8, [1], [-1]),
            onnx.helper.make_tensor('max', TensorProto.INT8, [1], [1]),
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [len(scale)], scale),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [len(zp)], zp)
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    # Per channel quantization propagation is not implemented yet
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.NOT_IMPLEMENTED


def test_convert_clip_v6_default_values():
    x_shape = [5, 10, 15]

    np.random.seed(42)

    # Contains some very large numbers and some +-inf
    x_data = ((np.random.rand(*x_shape) - 0.5) * 8.0e+38).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x'], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 6)])

    config = ConversionConfig({"ignore_opset_version": True})
    executors.convert_run_compare(onnx_model, x_data, conversion_config=config)


@pytest.mark.parametrize(
    "data_type",
    [
        pytest.param(TensorProto.UINT8, id='uint8'),
        pytest.param(TensorProto.INT8, id='int8'),
        pytest.param(TensorProto.INT16, id='int16'),
        pytest.param(TensorProto.INT32, id='int32'),
        pytest.param(TensorProto.INT64, id='int64'),
        pytest.param(TensorProto.FLOAT, id='float32'),
    ])
def test_convert_clip_v11_default_values(data_type: TensorProto.DataType):
    x_shape = [2, 3]

    np.random.seed(42)

    dtype = types.to_numpy_type(data_type)

    if dtype.kind in ('i', 'u'):
        max_val = np.iinfo(dtype).max
        min_val = np.iinfo(dtype).min

    elif dtype.kind == 'f':
        max_val = np.finfo(dtype).max
        min_val = np.finfo(dtype).min

    else:
        assert False

    # Input contains some very large numbers and some +-inf
    noise = np.asarray([[1, -1, -1, 1, 0, 0]]).astype(dtype)
    x_data = (np.asarray([min_val, max_val] * 3) + noise).astype(dtype).reshape(x_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x'], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', data_type, x_shape)],
        [onnx.helper.make_tensor_value_info('y', data_type, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


def test_convert_clip_with_invalid_type():
    # Using DOUBLE
    x_shape = [2, 3]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x'], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.DOUBLE, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.DOUBLE, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


def test_convert_clip_with_reversed_min_max():
    # Test a case where min > max

    x_shape = [5, 10, 15]
    min_value = 10
    max_value = 5

    np.random.seed(42)

    x_data = (np.random.rand(*x_shape) * 15.).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x', 'min', 'max'], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('min', TensorProto.FLOAT, [1], [min_value]),
            onnx.helper.make_tensor('max', TensorProto.FLOAT, [1], [max_value])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


def test_convert_clip_with_omitted_min(intermediate_tflite_model_provider):
    x_shape = [5, 10, 15]
    max_value = 5

    np.random.seed(42)
    x_data = (np.random.rand(*x_shape) * 15.).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x', '', 'max'], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('max', TensorProto.FLOAT, [1], [max_value])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 1
    assert ops[0].builtin_options.operator_type == BuiltinOperator.MINIMUM


def test_convert_clip_with_omitted_max(intermediate_tflite_model_provider):
    x_shape = [5, 10, 15]
    min_value = 10

    np.random.seed(42)
    x_data = (np.random.rand(*x_shape) * 15.).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x', 'min', ''], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('min', TensorProto.FLOAT, [1], [min_value]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 1
    assert ops[0].builtin_options.operator_type == BuiltinOperator.MAXIMUM


def test_convert_clip_with_omitted_inputs(intermediate_tflite_model_provider):
    x_shape = [5, 10, 15]

    np.random.seed(42)
    x_data = (np.random.rand(*x_shape) * 15.).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Clip', ['x', '', ''], ['y'])],
        'Clip test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 1
    assert ops[0].builtin_options.operator_type == BuiltinOperator.TRANSPOSE  # Identity (must have at least 1 op).
