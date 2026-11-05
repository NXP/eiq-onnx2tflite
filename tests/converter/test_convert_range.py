#
# Copyright 2024,2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.converter import convert
from tests import executors
from onnx2tflite.src import errors
from onnx2tflite.src.errors import ConversionError


@pytest.mark.parametrize(
    'start, limit, delta',
    [
        pytest.param(0, 10, 1, id='0 to 10, delta = 1'),
        pytest.param(2, 20, 5, id='2 to 20, delta = 5'),
        pytest.param(-10, 20, 5, id='-10 to 20, delta = 5'),
        pytest.param(10, -20, -5, id='10 to -20, delta = -5'),
        pytest.param(0, 0, 1, id='0 to 0 (output is empty)',
                     marks=pytest.mark.xfail(reason="Failing since TF 2.13.0. Seem like a bug in TFLite, see "
                                                    "https://github.com/tensorflow/tensorflow/issues/57084 and "
                                                    "https://github.com/tensorflow/tensorflow/issues/61899")
                     ),
        pytest.param(0, 10, -1, id='0 to 10, delta = -1 (wrong direction)',
                     marks=pytest.mark.xfail(reason="Failing since TF 2.13.0. Seem like a bug in TFLite, see "
                                                    "https://github.com/tensorflow/tensorflow/issues/57084 and "
                                                    "https://github.com/tensorflow/tensorflow/issues/61899")
                     ),
        pytest.param(-1, -100, 15, id='-1 to -100, delta = 15 (wrong direction)',
                     marks=pytest.mark.xfail(reason="Failing since TF 2.13.0. Seem like a bug in TFLite, see "
                                                    "https://github.com/tensorflow/tensorflow/issues/57084 and "
                                                    "https://github.com/tensorflow/tensorflow/issues/61899")
                     ),
    ])
def test_convert_range(start: float, limit: float, delta: float):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Range', ['start', 'limit', 'delta'], ['o'])],
        'Range test',
        [],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('start', TensorProto.FLOAT, [], [start]),
            onnx.helper.make_tensor('limit', TensorProto.FLOAT, [], [limit]),
            onnx.helper.make_tensor('delta', TensorProto.FLOAT, [], [delta])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    executors.convert_run_compare(onnx_model, {})


@pytest.mark.parametrize(
    'start, limit, delta',
    [
        pytest.param(0.1, 12.3, 0.1, id='0.1 to 12.3, delta = 0.1 (float32 and float64 counts differ)'),
        pytest.param(0.1, 1.1, 0.7, id='0.1 to 1.1, delta = 0.7'),
        pytest.param(-3.14159, 2.71828, 0.42, id='-pi to e, delta = 0.42'),
        pytest.param(-3.14159, 2.71828, -0.42, id='-pi to e, delta = -0.42 (wrong direction)',
                     marks=pytest.mark.xfail(reason="Failing since TF 2.13.0. Seem like a bug in TFLite, see "
                                                    "https://github.com/tensorflow/tensorflow/issues/57084 and "
                                                    "https://github.com/tensorflow/tensorflow/issues/61899")
                     ),
        pytest.param(3.14159, -2.71828, -0.42, id='pi to -e, delta = -0.42'),
    ])
def test_convert_range__fraction_values(start: float, limit: float, delta: float):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Range', ['start', 'limit', 'delta'], ['o'])],
        'Range test',
        [],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('start', TensorProto.FLOAT, [], [start]),
            onnx.helper.make_tensor('limit', TensorProto.FLOAT, [], [limit]),
            onnx.helper.make_tensor('delta', TensorProto.FLOAT, [], [delta])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    executors.convert_run_compare(onnx_model, {})


def test_convert_range__count_mismatch__internal_tensor():
    # The float32 and float64 element counts differ (122 vs 123). The `Range` output is consumed by `Add` (broadcast).
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Range', ['start', 'limit', 'delta'], ['r']),
            onnx.helper.make_node('Add', ['r', 'x'], ['o'])
        ],
        'Range test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('start', TensorProto.FLOAT, [], [0.1]),
            onnx.helper.make_tensor('limit', TensorProto.FLOAT, [], [12.3]),
            onnx.helper.make_tensor('delta', TensorProto.FLOAT, [], [0.1])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    input_data = (np.random.random([1]) + 1.).astype(np.float32)
    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    'start, limit, delta, input_type',
    [
        pytest.param(0, 10, -1, TensorProto.INT32, id='int32, 0 to 10, delta = -1'),
        pytest.param(-3.14159, 2.71828, -0.42, TensorProto.FLOAT, id='float32, -pi to e, delta = -0.42'),
    ])
def test_convert_range__wrong_direction__keeps_range(start, limit, delta, input_type,
                                                     intermediate_tflite_model_provider):
    # Wrong-direction ranges are empty in ONNX. They must use the `limit = start` path, and not be replaced by static
    #  data. (Running the model is not possible due to a TFLite bug, see the xfail tests above.)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Range', ['start', 'limit', 'delta'], ['o'])],
        'Range test',
        [],
        [onnx.helper.make_tensor_value_info('o', input_type, ())],
        [
            onnx.helper.make_tensor('start', input_type, [], [start]),
            onnx.helper.make_tensor('limit', input_type, [], [limit]),
            onnx.helper.make_tensor('delta', input_type, [], [delta])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    convert.convert_model(onnx_model)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.RANGE])
    range_op = intermediate_tflite_model_provider.get_operators()[0]
    assert range_op.tmp_inputs[1] is range_op.tmp_inputs[0]


def test_convert_range__zero_delta():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Range', ['start', 'limit', 'delta'], ['o'])],
        'Range test',
        [],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('start', TensorProto.FLOAT, [], [0]),
            onnx.helper.make_tensor('limit', TensorProto.FLOAT, [], [1]),
            onnx.helper.make_tensor('delta', TensorProto.FLOAT, [], [0])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)

    assert exc_info.value.code == errors.Code.INVALID_ONNX_OPERATOR


@pytest.mark.parametrize(
    "input_type",
    [
        pytest.param(TensorProto.FLOAT, id="float32"),
        pytest.param(TensorProto.INT32, id="int32"),
        pytest.param(TensorProto.INT64, id="int64")
        # Other types are not supported.
    ])
def test_convert_range__types(input_type: TensorProto.DataType):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Range', ['start', 'limit', 'delta'], ['o'])],
        'Range test',
        [],
        [onnx.helper.make_tensor_value_info('o', input_type, ())],
        [
            onnx.helper.make_tensor('start', input_type, [], [1]),
            onnx.helper.make_tensor('limit', input_type, [], [10]),
            onnx.helper.make_tensor('delta', input_type, [], [2])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    executors.convert_run_compare(onnx_model, {})


def test_convert_range__unsupported_type():
    input_type = TensorProto.FLOAT16

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Range', ['start', 'limit', 'delta'], ['o'])],
        'Range test',
        [],
        [onnx.helper.make_tensor_value_info('o', input_type, ())],
        [
            onnx.helper.make_tensor('start', input_type, [], [1]),
            onnx.helper.make_tensor('limit', input_type, [], [10]),
            onnx.helper.make_tensor('delta', input_type, [], [2])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)

    assert exc_info.value.code == errors.Code.INVALID_ONNX_MODEL


def test_convert_range__different_input_types():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Range', ['start', 'limit', 'delta'], ['o'])],
        'Range test',
        [],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('start', TensorProto.FLOAT, [], [1]),
            onnx.helper.make_tensor('limit', TensorProto.FLOAT, [], [10]),
            onnx.helper.make_tensor('delta', TensorProto.DOUBLE, [], [2])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)

    assert exc_info.value.code == errors.Code.INVALID_ONNX_OPERATOR


def test_convert_range__dynamic_inputs():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Range', ['start', 'limit', 'delta'], ['o'])],
        'Range test',
        [
            onnx.helper.make_tensor_value_info('start', TensorProto.FLOAT, []),
            onnx.helper.make_tensor_value_info('limit', TensorProto.FLOAT, []),
            onnx.helper.make_tensor_value_info('delta', TensorProto.DOUBLE, [])
        ],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(errors.ConversionError) as exc_info:
        convert.convert_model(onnx_model)

    assert exc_info.value.code == errors.Code.SHAPE_INFERENCE_ERROR

# I couldn't think of an operator which outputs a scalar value, to test this.
# Everything that I tried outputs a tensor of shape [1], but ONNX (and TFLite) needs shape [] for scalars.
#
# def test_convert_range__inferred_dynamic_inputs():
#     graph = onnx.helper.make_graph(
#         [
#             onnx.helper.make_node('ConstantOfShape', ['shape'], ['start'],
#                                   value=onnx.helper.make_tensor('s', TensorProto.FLOAT, [], [-5])),
#             onnx.helper.make_node('ConstantOfShape', ['shape'], ['limit'],
#                                   value=onnx.helper.make_tensor('l', TensorProto.FLOAT, [], [5])),
#             onnx.helper.make_node('ConstantOfShape', ['shape'], ['delta'],
#                                   value=onnx.helper.make_tensor('d', TensorProto.FLOAT, [], [2])),
#             onnx.helper.make_node('Range', ['start', 'limit', 'delta'], ['o'])
#         ],
#         'Range test',
#         [],
#         [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
#         [onnx.helper.make_tensor('shape', TensorProto.INT64, [0], [])]
#     )
#
#     onnx_model = onnx.helper.make_model(graph)
#     executors.convert_run_compare(onnx_model, {})
