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

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


# noinspection SpellCheckingInspection
@pytest.mark.parametrize(
    "scales",
    [
        [1., 1., 2., 2.],
        [1., 1., 1., 2.25],
        [1., 1., 2.5, 1.75],
        [1., 1., 1.5, 2.5],
    ])
def test_convert_upsample__v7__nearest(scales: list[float]):
    shape = [2, 4, 6, 8]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Upsample', ['x'], ['o'], mode='nearest', scales=scales)],
        'Upsample test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 7)])

    np.random.seed(42)
    data = np.random.random(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, data)


# noinspection SpellCheckingInspection
@pytest.mark.parametrize(
    "scales",
    [
        [1., 1., 2., 2.],
        [1., 1., 1., 2.25],
        [1., 1., 2.5, 1.75],
        [1., 1., 1.5, 2.5],
    ])
def test_convert_upsample__v9__nearest(scales: list[float]):
    shape = [2, 4, 6, 8]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Upsample', ['x', 'scales'], ['o'], mode='nearest')],
        'Upsample test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('scales', TensorProto.FLOAT, [len(scales)], scales)]
    )

    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 9)])

    np.random.seed(42)
    data = np.random.random(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, data)


# noinspection SpellCheckingInspection
@pytest.mark.parametrize(
    "scales",
    [
        [1., 1., 2., 2.],
        [1., 1., 1., 2.25],
        [1., 1., 2.5, 1.75],
        [1., 1., 1.5, 2.5],
    ])
def test_convert_upsample__v7__linear(scales: list[float]):
    shape = [2, 4, 6, 8]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Upsample', ['x'], ['o'], mode='linear', scales=scales)],
        'Upsample test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 7)])

    np.random.seed(42)
    data = np.random.random(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, data)


# noinspection SpellCheckingInspection
@pytest.mark.parametrize(
    "scales",
    [
        [1., 1., 2., 2.],
        [1., 1., 1., 2.25],
        [1., 1., 2.5, 1.75],
        [1., 1., 1.5, 2.5],
    ])
def test_convert_upsample__v9__linear(scales: list[float]):
    shape = [2, 4, 6, 8]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Upsample', ['x', 'scales'], ['o'], mode='linear')],
        'Upsample test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('scales', TensorProto.FLOAT, [len(scales)], scales)]
    )

    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 9)])

    np.random.seed(42)
    data = np.random.random(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, data)


# noinspection SpellCheckingInspection
@pytest.mark.parametrize(
    "_type",
    [
        TensorProto.FLOAT, TensorProto.INT8, TensorProto.UINT8
    ], ids=name_for_onnx_type)
def test_convert_upsample__types(_type):
    shape = [2, 4, 6, 8]
    scales = [1., 1., 2., 2.]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Upsample', ['x'], ['o'], mode='linear', scales=scales)],
        'Upsample test',
        [onnx.helper.make_tensor_value_info('x', _type, shape)],
        [onnx.helper.make_tensor_value_info('o', _type, ())],
    )

    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 7)])

    np.random.seed(42)
    data = np.random.random(shape).astype(to_numpy_type(_type))

    executors.convert_run_compare(onnx_model, data)


# noinspection SpellCheckingInspection
def test_convert_upsample__unsupported_type():
    _type = TensorProto.DOUBLE

    shape = [2, 4, 6, 8]
    scales = [1., 1., 2., 2.]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Upsample', ['x'], ['o'], mode='linear', scales=scales)],
        'Upsample test',
        [onnx.helper.make_tensor_value_info('x', _type, shape)],
        [onnx.helper.make_tensor_value_info('o', _type, ())],
        []
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 7)])

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


# noinspection SpellCheckingInspection
def test_convert_upsample__unconvertible_scales():
    shape = [2, 4, 6, 8]
    scales = [2., 2., 2., 2.]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Upsample', ['x'], ['o'], mode='linear', scales=scales)],
        'Upsample test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('o', TensorProto.FLOAT, ())],
        []
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 7)])

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'scales' in logger.conversion_log.get_node_error_message(0)
