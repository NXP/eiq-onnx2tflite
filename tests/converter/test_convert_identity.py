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
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "shape",
    [
        [256], [10, 20], [5, 10, 10], [2, 4, 6, 8], [2, 4, 6, 8, 10], [1, 2, 3, 4, 5, 6]
    ])
def test_convert_identity__not_skipped(shape: list[int], intermediate_tflite_model_provider):
    # The `Identity` will be turned to a `Transpose` operator.
    np.random.seed(42)
    data = np.random.rand(*shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Identity', ['x'], ['y'])],
        'Identity test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.TRANSPOSE])


def test_convert_identity__not_skipped__7d(intermediate_tflite_model_provider):
    # The `Identity` will be turned to a `FlexTranspose` operator.
    shape = [2, 1, 3, 1, 2, 4, 2]

    np.random.seed(42)
    data = np.random.rand(*shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Identity', ['x'], ['y'])],
        'Identity test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
    )

    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig({'allow_select_ops': True})
    executors.convert_run_compare(onnx_model, data, conversion_config=config)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.CUSTOM])


@pytest.mark.parametrize(
    "shape",
    [
        [256], [10, 20], [5, 10, 10], [2, 4, 6, 8], [2, 4, 6, 8, 10], [1, 2, 3, 4, 5, 6]
    ])
def test_convert_identity__skipped(shape: list[int], intermediate_tflite_model_provider):
    np.random.seed(42)
    data = np.random.rand(*shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Identity', ['x'], ['x1']),
            onnx.helper.make_node('Add', ['x1', 'one'], ['y'])
        ],
        'Identity test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
        [
            onnx.helper.make_tensor('x', TensorProto.FLOAT, shape, data),
            onnx.helper.make_tensor('one', TensorProto.FLOAT, [1], [1.0])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.ADD])


def test_convert_identity__inferred_data(intermediate_tflite_model_provider):
    shape = [42, 37]

    np.random.seed(42)
    data = np.random.rand(*shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Constant', [], ['x1'],
                                  value=onnx.helper.make_tensor('data', TensorProto.FLOAT, shape, data)),
            onnx.helper.make_node('Identity', ['x1'], ['x2']),
            onnx.helper.make_node('Add', ['x2', 'one'], ['y'])

        ],
        'Identity test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
        [
            onnx.helper.make_tensor('x', TensorProto.FLOAT, shape, data),
            onnx.helper.make_tensor('one', TensorProto.FLOAT, [1], [1.0])
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.ADD])


def test_convert_identity__chain(intermediate_tflite_model_provider):
    # The first 2 identities are skipped and the last one is turned to `Transpose`.
    shape = [42, 37]

    np.random.seed(42)
    data = np.random.rand(*shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Identity', ['x'], ['x1']),
            onnx.helper.make_node('Identity', ['x1'], ['x2']),
            onnx.helper.make_node('Identity', ['x2'], ['y'])
        ],
        'Identity test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor('x', TensorProto.FLOAT, shape, data)]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.TRANSPOSE])


@pytest.mark.parametrize(
    "type_",
    [
        TensorProto.FLOAT,
        TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
        TensorProto.UINT8, TensorProto.BOOL
    ],
    ids=name_for_onnx_type
)
def test_convert_identity__types(type_: TensorProto.DataType, intermediate_tflite_model_provider):
    shape = [42]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(to_numpy_type(type_))

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Identity', ['x'], ['y'])],
        'Identity test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.TRANSPOSE])


def test_convert_identity__invalid_type():
    type_ = TensorProto.DOUBLE

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Identity', ['x'], ['y'])],
        'Identity test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_identity__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Identity', ['x1'], ['x2']),
            onnx.helper.make_node('Reshape', ['x2', 'flat_shape'], ['x3']),
            onnx.helper.make_node('DequantizeLinear', ['x3', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [1], [np.prod(shape)])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
