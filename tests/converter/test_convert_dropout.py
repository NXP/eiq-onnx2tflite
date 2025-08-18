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
from tests import executors


def test_convert_dropout__skipped(intermediate_tflite_model_provider):
    shape = [5, 10, 15]

    np.random.seed(42)

    x_data = np.random.rand(*shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Add', ['x', 'one'], ['y1']),
            onnx.helper.make_node('Dropout', ['y1'], ['y'])
        ],
        'Dropout test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('one', TensorProto.FLOAT, [1], [1.])]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.ADD])


def test_convert_dropout__not_skipped(intermediate_tflite_model_provider):
    shape = [5, 10, 15]

    np.random.seed(42)

    x_data = np.random.rand(*shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Dropout', ['x'], ['y'])],
        'Dropout test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.TRANSPOSE])


def test_convert_dropout__with_seed_attribute():
    shape = [5, 10, 15]

    np.random.seed(42)

    x_data = np.random.rand(*shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Dropout', ['x'], ['y'], seed=42)],
        'Dropout test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data)


def test_convert_dropout__with_ratio_attribute():
    shape = [5, 10, 15]

    np.random.seed(42)

    x_data = np.random.rand(*shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Dropout', ['x'], ['y'], ratio=0.42)],
        'Dropout test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )

    # Version 10 is the last one with `ratio` as attribute.
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 10)])

    executors.convert_run_compare(onnx_model, x_data)


def test_convert_dropout__training_mode():
    shape = [5, 10, 15]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Dropout', ['x', '', 'training_mode'], ['y'])],
        'Dropout test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('training_mode', TensorProto.BOOL, [], [1])]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_dropout__invalid_type():
    type_ = TensorProto.INT8

    shape = [5, 10, 15]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Dropout', ['x'], ['y'])
        ],
        'Dropout test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL
    assert 'INT8' in logger.conversion_log.get_node_error_message(0)
