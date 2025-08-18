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

from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type, name_for_onnx_type
from tests import executors


@pytest.mark.parametrize("_type", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_cast__quantized__same_type(_type: TensorProto.DataType):
    shape = [3, 14, 15]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('QuantizeLinear', ['x', 's2', 'zp2'], ['x2']),
            onnx.helper.make_node('Cast', ['x1'], ['x3'], to=_type),

            # The QLinearAdd is to verify quantization propagation.
            onnx.helper.make_node('QLinearAdd', ['x2', 's2', 'zp2', 'x3', 's', 'zp', 's', 'zp'], ['x4'],
                                  domain='com.microsoft'),

            onnx.helper.make_node('DequantizeLinear', ['x4', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [], [1.42]),
            onnx.helper.make_tensor('zp', _type, [], [12]),
            onnx.helper.make_tensor('s2', TensorProto.FLOAT, [], [0.42]),
            onnx.helper.make_tensor('zp2', _type, [], [42])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add the opset with QLinearAdd

    executors.convert_run_compare(onnx_model, data)


def test_convert_cast__quantized__different_types():
    # Cast UINT8 to INT8
    shape = [3, 14, 15]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),  # To INT8
            onnx.helper.make_node('Cast', ['x1'], ['x3'], to=TensorProto.UINT8),  # To UINT8

            onnx.helper.make_node('QuantizeLinear', ['x', 's2', 'zp2'], ['x2']),  # To UINT8

            onnx.helper.make_node('QLinearAdd', ['x2', 's2', 'zp2', 'x3', 's3', 'zp3', 's3', 'zp3'], ['x4'],
                                  domain='com.microsoft'),  # UINT8 computation.

            onnx.helper.make_node('DequantizeLinear', ['x4', 's3', 'zp3'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [], [1.42]),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [], [12]),
            onnx.helper.make_tensor('s2', TensorProto.FLOAT, [], [0.42]),
            onnx.helper.make_tensor('zp2', TensorProto.UINT8, [], [42]),
            onnx.helper.make_tensor('s3', TensorProto.FLOAT, [], [0.123]),
            onnx.helper.make_tensor('zp3', TensorProto.UINT8, [], [140])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add the opset with QLinearAdd

    executors.convert_run_compare(onnx_model, data)


def test_convert_cast_with_quantized_input():
    x_shape = [256]
    flat_shape = [8, 2, 2, 8]

    x_data = np.random.random(256).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's1', 'zp1'], ['y1']),
            onnx.helper.make_node('Cast', ['y1'], ['y2'], to=TensorProto.UINT8),
            onnx.helper.make_node('Reshape', ['y2', 'ns'], ['y3']),
            onnx.helper.make_node('Transpose', ['y3'], ['y4'], perm=[3, 1, 0, 2]),
            onnx.helper.make_node('QLinearAdd', ['y4', 's2', 'zp2', 'one', 's2', 'zp2', 's2', 'zp2'], ['y'],
                                  domain="com.microsoft"),
        ],
        'Cast test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.UINT8, ())],
        [
            onnx.helper.make_tensor('s1', TensorProto.FLOAT, [], [0.1]),
            onnx.helper.make_tensor('zp1', TensorProto.INT8, [], [-1]),
            onnx.helper.make_tensor('s2', TensorProto.FLOAT, [], [0.05]),
            onnx.helper.make_tensor('zp2', TensorProto.UINT8, [], [130]),
            onnx.helper.make_tensor('one', TensorProto.UINT8, [1], [1]),
            onnx.helper.make_tensor('ns', TensorProto.INT64, [4], flat_shape),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add the opset with QLinearAdd

    cc = ConversionConfig()
    cc.tflite_quantization_integrity_check = False

    executors.convert_run_compare(onnx_model, x_data, conversion_config=cc)


def _get_tensor_limits(from_type: np.dtype, to_type: np.dtype):
    """
    To avoid C++ (and ONNX) undefined behaviour at the cast testing, this function returns range representable in both
    from_type and to_type.

    :param from_type:
    :param to_type:
    :return: tuple: (min_value, max_value, value_range)
    """
    if from_type.kind == 'f':
        if from_type == np.dtype(np.float16):
            min_value = -2 ** 15
            max_value = 2 ** 15
        else:
            min_value = - 2 ** 19
            max_value = 2 ** 19
    else:
        min_value = np.iinfo(from_type).min
        max_value = np.iinfo(from_type).max
    if to_type.kind not in ['f', 'b']:
        to_type_limits = np.iinfo(to_type)
        min_value = max(min_value, to_type_limits.min)
        max_value = min(max_value, to_type_limits.max)
    value_range = abs(min_value) + max_value

    return min_value, max_value, value_range


@pytest.mark.parametrize(
    "to_type",
    [
        pytest.param(TensorProto.UINT8, id='to uint8'),
        pytest.param(TensorProto.UINT16, id='to uint16'),
        pytest.param(TensorProto.UINT32, id='to uint32'),

        pytest.param(TensorProto.INT8, id='to int8'),
        pytest.param(TensorProto.INT16, id='to int16'),
        pytest.param(TensorProto.INT32, id='to int32'),
        pytest.param(TensorProto.INT64, id='to int64'),

        pytest.param(TensorProto.FLOAT16, id='to float16'),
        pytest.param(TensorProto.FLOAT, id='to float32'),
        pytest.param(TensorProto.DOUBLE, id='to float64'),

        pytest.param(TensorProto.BOOL, id='to bool'),
    ])
def test_convert_cast_from_int8(to_type: TensorProto.DataType):
    x_shape = [256]

    x_data = np.arange(-128, 128).astype(np.int8)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cast', ['x'], ['y'], to=to_type)],
        'Cast test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.INT8, x_shape)],
        [onnx.helper.make_tensor_value_info('y', to_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "to_type",
    [
        pytest.param(TensorProto.UINT8, id='to uint8'),
        pytest.param(TensorProto.UINT16, id='to uint16'),
        pytest.param(TensorProto.UINT32, id='to uint32'),

        pytest.param(TensorProto.INT8, id='to int8'),
        pytest.param(TensorProto.INT16, id='to int16'),
        pytest.param(TensorProto.INT32, id='to int32'),
        pytest.param(TensorProto.INT64, id='to int64'),

        pytest.param(TensorProto.FLOAT16, id='to float16'),
        pytest.param(TensorProto.FLOAT, id='to float32'),
        pytest.param(TensorProto.DOUBLE, id='to float64'),

        pytest.param(TensorProto.BOOL, id='to bool'),
    ])
def test_convert_cast_from_int16(to_type: TensorProto.DataType):
    x_shape = [1024]

    x_data = np.arange(-2 ** 15, 2 ** 15, (2 ** 16) / 1024).astype(np.int16)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cast', ['x'], ['y'], to=to_type)],
        'Cast test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.INT16, x_shape)],
        [onnx.helper.make_tensor_value_info('y', to_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "to_type",
    [
        pytest.param(TensorProto.UINT8, id='to uint8'),
        pytest.param(TensorProto.UINT16, id='to uint16'),
        pytest.param(TensorProto.UINT32, id='to uint32'),

        pytest.param(TensorProto.INT8, id='to int8'),
        pytest.param(TensorProto.INT16, id='to int16'),
        pytest.param(TensorProto.INT32, id='to int32'),
        pytest.param(TensorProto.INT64, id='to int64'),

        pytest.param(TensorProto.FLOAT16, id='to float16'),
        pytest.param(TensorProto.FLOAT, id='to float32'),
        pytest.param(TensorProto.DOUBLE, id='to float64'),

        pytest.param(TensorProto.BOOL, id='to bool'),
    ])
def test_convert_cast_from_int32(to_type: TensorProto.DataType):
    x_shape = [2 ** 12]

    x_data = np.arange(-2 ** 31, 2 ** 31, 2 ** 20).astype(np.int32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cast', ['x'], ['y'], to=to_type)],
        'Cast test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.INT32, x_shape)],
        [onnx.helper.make_tensor_value_info('y', to_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "to_type",
    [
        pytest.param(TensorProto.UINT8, id='to uint8'),
        pytest.param(TensorProto.UINT16, id='to uint16'),
        pytest.param(TensorProto.UINT32, id='to uint32'),

        pytest.param(TensorProto.INT8, id='to int8'),
        pytest.param(TensorProto.INT16, id='to int16'),
        pytest.param(TensorProto.INT32, id='to int32'),
        pytest.param(TensorProto.INT64, id='to int64'),

        pytest.param(TensorProto.FLOAT16, id='to float16'),
        pytest.param(TensorProto.FLOAT, id='to float32'),
        pytest.param(TensorProto.DOUBLE, id='to float64'),

        pytest.param(TensorProto.BOOL, id='to bool'),
    ])
def test_convert_cast_from_int64(to_type: TensorProto.DataType):
    x_shape = [2 ** 12]

    x_data = np.arange(-2 ** 63, 2 ** 63, 2 ** 52).astype(np.int64)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cast', ['x'], ['y'], to=to_type)],
        'Cast test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.INT64, x_shape)],
        [onnx.helper.make_tensor_value_info('y', to_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "to_type",
    [
        pytest.param(TensorProto.UINT8, id='to uint8'),
        pytest.param(TensorProto.UINT16, id='to uint16'),
        pytest.param(TensorProto.UINT32, id='to uint32'),

        pytest.param(TensorProto.INT8, id='to int8'),
        pytest.param(TensorProto.INT16, id='to int16'),
        pytest.param(TensorProto.INT32, id='to int32'),
        pytest.param(TensorProto.INT64, id='to int64'),

        pytest.param(TensorProto.FLOAT16, id='to float16'),
        pytest.param(TensorProto.FLOAT, id='to float32'),
        pytest.param(TensorProto.DOUBLE, id='to float64'),

        pytest.param(TensorProto.BOOL, id='to bool'),
    ])
def test_convert_cast_from_uint8(to_type: TensorProto.DataType):
    x_shape = [256]

    x_data = np.arange(0, 2 ** 8).astype(np.uint8)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cast', ['x'], ['y'], to=to_type)],
        'Cast test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.UINT8, x_shape)],
        [onnx.helper.make_tensor_value_info('y', to_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "to_type",
    [
        pytest.param(TensorProto.UINT8, id='to uint8'),
        pytest.param(TensorProto.UINT16, id='to uint16'),
        pytest.param(TensorProto.UINT32, id='to uint32'),

        pytest.param(TensorProto.INT8, id='to int8'),
        pytest.param(TensorProto.INT16, id='to int16'),
        pytest.param(TensorProto.INT32, id='to int32'),
        pytest.param(TensorProto.INT64, id='to int64'),

        pytest.param(TensorProto.FLOAT16, id='to float16'),
        pytest.param(TensorProto.FLOAT, id='to float32'),
        pytest.param(TensorProto.DOUBLE, id='to float64'),

        pytest.param(TensorProto.BOOL, id='to bool'),
    ])
def test_convert_cast_from_uint32(to_type: TensorProto.DataType):
    x_shape = [2 ** 12]

    x_data = np.arange(0, 2 ** 32, 2 ** 20).astype(np.uint32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cast', ['x'], ['y'], to=to_type)],
        'Cast test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.UINT32, x_shape)],
        [onnx.helper.make_tensor_value_info('y', to_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "to_type",
    [
        pytest.param(TensorProto.UINT8, id='to uint8'),
        pytest.param(TensorProto.UINT16, id='to uint16'),
        pytest.param(TensorProto.UINT32, id='to uint32'),

        pytest.param(TensorProto.INT8, id='to int8'),
        pytest.param(TensorProto.INT16, id='to int16'),
        pytest.param(TensorProto.INT32, id='to int32'),
        pytest.param(TensorProto.INT64, id='to int64'),

        pytest.param(TensorProto.FLOAT16, id='to float16'),
        pytest.param(TensorProto.FLOAT, id='to float32'),
        pytest.param(TensorProto.DOUBLE, id='to float64'),

        pytest.param(TensorProto.BOOL, id='to bool'),
    ])
def test_convert_cast_from_float16(to_type: TensorProto.DataType):
    # Avoid undefined behavior testing [https://onnx.ai/onnx/operators/onnx__Cast.html]
    #   Casting from floating point to:
    #       fixed point: undefined if OOR.
    min_value, max_value, value_range = _get_tensor_limits(np.dtype(np.float16), to_numpy_type(to_type))
    shape = min(value_range, 2 ** 10)
    x_shape = [shape]

    x_data = np.arange(min_value, max_value, value_range / shape).astype(np.float16)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cast', ['x'], ['y'], to=to_type)],
        'Cast test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT16, x_shape)],
        [onnx.helper.make_tensor_value_info('y', to_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "to_type",
    [
        pytest.param(TensorProto.UINT8, id='to uint8'),
        pytest.param(TensorProto.UINT16, id='to uint16'),
        pytest.param(TensorProto.UINT32, id='to uint32'),

        pytest.param(TensorProto.INT8, id='to int8'),
        pytest.param(TensorProto.INT16, id='to int16'),
        pytest.param(TensorProto.INT32, id='to int32'),
        pytest.param(TensorProto.INT64, id='to int64'),

        pytest.param(TensorProto.FLOAT16, id='to float16'),
        pytest.param(TensorProto.FLOAT, id='to float32'),
        pytest.param(TensorProto.DOUBLE, id='to float64'),

        pytest.param(TensorProto.BOOL, id='to bool'),
    ])
def test_convert_cast_from_float32(to_type: TensorProto.DataType):
    # Avoid undefined behavior testing [https://onnx.ai/onnx/operators/onnx__Cast.html]
    #   Casting from floating point to:
    #       fixed point: undefined if OOR.
    min_value, max_value, value_range = _get_tensor_limits(np.dtype(np.float32), to_numpy_type(to_type))
    shape = min(value_range, 2 ** 10)
    x_shape = [shape]

    x_data = np.arange(min_value, max_value, value_range / shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cast', ['x'], ['y'], to=to_type)],
        'Cast test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', to_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "to_type",
    [
        pytest.param(TensorProto.UINT8, id='to uint8'),
        pytest.param(TensorProto.UINT16, id='to uint16'),
        pytest.param(TensorProto.UINT32, id='to uint32'),

        pytest.param(TensorProto.INT8, id='to int8'),
        pytest.param(TensorProto.INT16, id='to int16'),
        pytest.param(TensorProto.INT32, id='to int32'),
        pytest.param(TensorProto.INT64, id='to int64'),

        pytest.param(TensorProto.FLOAT16, id='to float16'),
        pytest.param(TensorProto.FLOAT, id='to float32'),
        pytest.param(TensorProto.DOUBLE, id='to float64'),

        pytest.param(TensorProto.BOOL, id='to bool'),
    ])
def test_convert_cast_from_float64(to_type: TensorProto.DataType):
    # Avoid undefined behavior testing [https://onnx.ai/onnx/operators/onnx__Cast.html]
    #   Casting from floating point to:
    #       fixed point: undefined if OOR.
    min_value, max_value, value_range = _get_tensor_limits(np.dtype(np.float64), to_numpy_type(to_type))
    shape = min(value_range, 2 ** 10)
    x_shape = [shape]

    x_data = np.arange(min_value, max_value, value_range / shape).astype(np.float64)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cast', ['x'], ['y'], to=to_type)],
        'Cast test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.DOUBLE, x_shape)],
        [onnx.helper.make_tensor_value_info('y', to_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "to_type",
    [
        pytest.param(TensorProto.UINT8, id='to uint8'),
        pytest.param(TensorProto.UINT16, id='to uint16'),
        pytest.param(TensorProto.UINT32, id='to uint32'),

        pytest.param(TensorProto.INT8, id='to int8'),
        pytest.param(TensorProto.INT16, id='to int16'),
        pytest.param(TensorProto.INT32, id='to int32'),
        pytest.param(TensorProto.INT64, id='to int64'),

        pytest.param(TensorProto.FLOAT16, id='to float16'),
        pytest.param(TensorProto.FLOAT, id='to float32'),
        pytest.param(TensorProto.DOUBLE, id='to float64'),

        pytest.param(TensorProto.BOOL, id='to bool'),
    ])
def test_convert_cast_from_bool(to_type: TensorProto.DataType):
    x_shape = [32]

    x_data = np.random.random(32) > 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cast', ['x'], ['y'], to=to_type)],
        'Cast test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.BOOL, x_shape)],
        [onnx.helper.make_tensor_value_info('y', to_type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "to_type",
    [
        pytest.param(TensorProto.UINT8, id='to uint8'),
        pytest.param(TensorProto.UINT16, id='to uint16'),
        pytest.param(TensorProto.UINT32, id='to uint32'),
        pytest.param(TensorProto.UINT32, id='to uint64'),

        pytest.param(TensorProto.INT8, id='to int8'),
        pytest.param(TensorProto.INT16, id='to int16'),
        pytest.param(TensorProto.INT32, id='to int32'),
        pytest.param(TensorProto.INT64, id='to int64'),

        pytest.param(TensorProto.FLOAT, id='to float32'),

        pytest.param(TensorProto.BOOL, id='to bool'),
    ])
def test_convert_cast__data_inference(to_type: TensorProto.DataType):
    shape = [13, 37]
    x_data = np.random.random(shape).astype(np.float32) * 42

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Cast', ['x'], ['y'], to=to_type)],
        'Cast test',
        [],
        [onnx.helper.make_tensor_value_info('y', to_type, ())],
        [onnx.helper.make_tensor('x', TensorProto.FLOAT, shape, x_data)]
    )
    onnx_model = onnx.helper.make_model(graph)

    inferred_tensor_data = dict()
    ModelShapeInference.infer_shapes(onnx_model, inferred_tensor_data=inferred_tensor_data)

    assert 'y' in inferred_tensor_data.keys()
    data = inferred_tensor_data['y']
    np_type = to_numpy_type(to_type)
    assert data.dtype == np_type
    assert np.allclose(data, x_data.astype(np_type))
