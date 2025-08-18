#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from tests import executors


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
        pytest.param(0.1, 12.3, 0.1, id='0.1 to 12.3, delta = 0.1'),
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
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_logs()['shape_inference'][0]['error_code'] == logger.Code.INVALID_ONNX_OPERATOR


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

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL


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

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR


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

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_logs()['shape_inference'][0]['error_code'] == logger.Code.SHAPE_INFERENCE_ERROR

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
