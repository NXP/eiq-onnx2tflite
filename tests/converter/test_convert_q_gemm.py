#
# Copyright 2024-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type
from tests import executors


def _get_q_gemm_model(nodes: list[onnx.NodeProto], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp,
                      y_s, y_zp, c_data=None) -> onnx.ModelProto:
    if c_data is None:
        c_data = [0] * c_shape[0]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            'QGemm test',
            [
                onnx.helper.make_tensor_value_info('a', data_type, a_shape),
                onnx.helper.make_tensor_value_info('b', data_type, b_shape)
            ],
            [onnx.helper.make_tensor_value_info('y', data_type, ())],
            [
                onnx.helper.make_tensor('as', TensorProto.FLOAT, [], a_s),
                onnx.helper.make_tensor('azp', data_type, [], a_zp),
                onnx.helper.make_tensor('bs', TensorProto.FLOAT, [len(b_s)], b_s),
                onnx.helper.make_tensor('bzp', data_type, [len(b_zp)], b_zp),
                onnx.helper.make_tensor('c', TensorProto.INT32, c_shape, c_data),
                onnx.helper.make_tensor('ys', TensorProto.FLOAT, [], y_s),
                onnx.helper.make_tensor('yzp', data_type, [], y_zp),
            ]
        ),
    )
    onnx_model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))  # Add the opset with QGemm

    return onnx_model


def _get_input_data(data_type, a_shape, b_shape) -> dict[int, np.ndarray]:
    return {
        0: np.arange(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(data_type)),
        1: np.arange(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(data_type)),
    }


def _get_q_gemm_shapes(m, k, n, trans_a=False, trans_b=False):
    a_shape = [m, k] if not trans_a else [k, m]
    b_shape = [k, n] if not trans_b else [n, k]
    c_shape = [n]

    return a_shape, b_shape, c_shape


@pytest.mark.parametrize(
    'm, k, n, a_s, a_zp, b_s, b_zp, y_s, y_zp',  # K and N must be the same for this test
    [
        (5, 10, 10, [0.01], [10], [0.1], [0], [0.05], [-5]),
        (20, 30, 30, [1.23], [-100], [0.5], [1], [0.042], [-100])
    ])
def test_convert_int_q_gemm(m, k, n, a_s, a_zp, b_s, b_zp, y_s, y_zp, intermediate_tflite_model_provider):
    data_type = TensorProto.INT8
    a_shape, b_shape, c_shape = _get_q_gemm_shapes(m, k, n, trans_b=True)

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transB=True)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp)

    executors.convert_run_compare(onnx_model, _get_input_data(data_type, a_shape, b_shape))
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.FULLY_CONNECTED])


@pytest.mark.parametrize(
    'm, k, n, a_s, a_zp, b_s, b_zp, y_s, y_zp',  # K and N must be the same for this test
    [
        (5, 10, 10, [0.01], [128], [0.1], [130], [0.05], [120]),
        (20, 30, 30, [1.23], [100], [0.5], [180], [0.042], [10])
    ])
def test_convert_uint_q_gemm(m, k, n, a_s, a_zp, b_s, b_zp, y_s, y_zp, intermediate_tflite_model_provider):
    data_type = TensorProto.UINT8
    a_shape, b_shape, c_shape = _get_q_gemm_shapes(m, k, n, trans_b=True)

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transB=True)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp)

    executors.convert_run_compare(onnx_model, _get_input_data(data_type, a_shape, b_shape))
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.FULLY_CONNECTED])


def test_convert_q_gemm_default_attributes():
    data_type = TensorProto.INT8

    a_shape, b_shape, c_shape = _get_q_gemm_shapes(10, 20, 30)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.01], [10], [0.1], [0], [0.05], [-5]

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft')
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp)

    executors.convert_run_compare(onnx_model, _get_input_data(data_type, a_shape, b_shape))


@pytest.mark.parametrize(
    'c_shape',
    [
        ([4]),
        ([1, 4]),
    ])
def test_convert_int_8_q_gemm_with_fusible_c(c_shape, intermediate_tflite_model_provider):
    data_type = TensorProto.INT8

    a_shape, b_shape, _ = _get_q_gemm_shapes(10, 20, 4, trans_b=True)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.01], [10], [0.1], [0], [0.05], [-5]

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transB=True)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp, c_data=[1, 20, -50, 100])

    executors.convert_run_compare(onnx_model, _get_input_data(data_type, a_shape, b_shape))
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.FULLY_CONNECTED])


@pytest.mark.parametrize(
    'c_shape',
    [
        ([4]),
        ([1, 4]),
    ])
def test_convert_uint_8_q_gemm_with_fusible_c(c_shape, intermediate_tflite_model_provider):
    data_type = TensorProto.UINT8

    a_shape, b_shape, _ = _get_q_gemm_shapes(10, 20, 4, trans_b=True)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.01], [100], [0.1], [128], [0.05], [130]

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transB=True)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp, c_data=[1, 20, -50, 100])

    executors.convert_run_compare(onnx_model, _get_input_data(data_type, a_shape, b_shape))
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.FULLY_CONNECTED])


def test_convert_int_8_q_gemm_with_non_fusible_c():
    np.random.seed(2)
    c_shape = [10, 4]

    data_type = TensorProto.INT8
    c_data = (np.random.random(np.prod(c_shape)) * 3000.).astype(np.int32)

    a_shape, b_shape, _ = _get_q_gemm_shapes(10, 20, 4, trans_b=True)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.01], [10], [0.1], [0], [0.05], [-5]

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transB=True)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp, c_data)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_uint_8_q_gemm_with_non_fusible_c():
    np.random.seed(2)

    c_shape = [10, 4]
    data_type = TensorProto.UINT8
    c_data = (np.random.random(np.prod(c_shape)) * 3000.).astype(np.int32)

    a_shape, b_shape, _ = _get_q_gemm_shapes(10, 20, 4, trans_b=True)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.01], [140], [0.1], [128], [0.05], [100]

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transB=True)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp, c_data)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize(
    'trans_a, trans_b',
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ])
def test_convert_q_gemm_with_trans_x_combinations(trans_a: bool, trans_b: bool):
    data_type = TensorProto.INT8
    a_shape, b_shape, c_shape = _get_q_gemm_shapes(10, 20, 30, trans_a, trans_b)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.01], [10], [0.1], [0], [0.05], [-5]

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transA=trans_a, transB=trans_b)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp)

    executors.convert_run_compare(onnx_model, _get_input_data(data_type, a_shape, b_shape))


@pytest.mark.parametrize('trans_a', [True, False])
def test_convert_q_gemm__per_channel(trans_a: bool):
    data_type = TensorProto.INT8
    trans_b = True
    a_shape, b_shape, c_shape = _get_q_gemm_shapes(1, 2, 4, trans_a, trans_b)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.05], [-50], [0.05, 0.051, 0.052, 0.053], [0, 0, 0, 0], [0.044], [-51],

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transA=trans_a, transB=trans_b)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp)

    executors.convert_run_compare(onnx_model, _get_input_data(data_type, a_shape, b_shape))


@pytest.mark.parametrize('trans_a', [True, False])
def test_convert_q_gemm__per_channel__with_bias(trans_a: bool):
    data_type = TensorProto.INT8
    trans_b = True
    a_shape, b_shape, c_shape = _get_q_gemm_shapes(1, 2, 4, trans_a, trans_b)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.05], [-50], [0.05, 0.051, 0.052, 0.053], [0, 0, 0, 0], [0.044], [-51],
    c_data = [37] * c_shape[0]

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transA=trans_a, transB=trans_b)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp, c_data=c_data)

    executors.convert_run_compare(onnx_model, _get_input_data(data_type, a_shape, b_shape))


def test_convert_q_gemm__per_channel__unsupported_zero_points():
    data_type = TensorProto.INT8
    trans_a = True
    trans_b = True
    a_shape, b_shape, c_shape = _get_q_gemm_shapes(1, 2, 4, trans_a, trans_b)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.05], [-50], [0.05, 0.051, 0.052, 0.053], [0, 1, 0, 0], [0.044], [-51],

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transA=trans_a, transB=trans_b)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_q_gemm__per_channel__unsupported_trans_b():
    data_type = TensorProto.INT8
    trans_a = True
    trans_b = False
    a_shape, b_shape, c_shape = _get_q_gemm_shapes(1, 2, 4, trans_a, trans_b)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.05], [-50], [0.05, 0.051, 0.052, 0.053], [0, 0, 0, 0], [0.044], [-51],

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transA=trans_a, transB=trans_b)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_q_gemm__per_channel__unsupported_type():
    data_type = TensorProto.UINT8
    trans_a = True
    trans_b = True
    a_shape, b_shape, c_shape = _get_q_gemm_shapes(1, 2, 4, trans_a, trans_b)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.05], [50], [0.05, 0.051, 0.052, 0.053], [128, 128, 128, 128], [0.044], [51],

    onnx_model = _get_q_gemm_model([
        onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                              domain='com.microsoft', transA=trans_a, transB=trans_b)
    ], data_type, a_shape, b_shape, c_shape, a_s, a_zp, b_s, b_zp, y_s, y_zp)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize('alpha', [2.0, 20.0, 0.1])
def test_convert_q_gemm_with_alpha_and_dynamic_input_a(alpha: float, intermediate_tflite_model_provider):
    data_type = TensorProto.INT8
    a_shape, b_shape, c_shape = _get_q_gemm_shapes(10, 20, 30, trans_b=True)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.01], [10], [0.1], [0], [0.05], [-5]
    c_data = [37] * c_shape[0]

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('QuantizeLinear', ['a', 's', 'zp'], ['a1']),
                onnx.helper.make_node('QGemm', ['a1', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                                      domain='com.microsoft', alpha=alpha, transB=True)
            ],
            'QGemm test',
            [
                onnx.helper.make_tensor_value_info('a', TensorProto.FLOAT, a_shape),
                onnx.helper.make_tensor_value_info('b', data_type, b_shape)
            ],
            [onnx.helper.make_tensor_value_info('y', data_type, ())],
            [
                onnx.helper.make_tensor('as', TensorProto.FLOAT, [], a_s),
                onnx.helper.make_tensor('azp', data_type, [], a_zp),
                onnx.helper.make_tensor('bs', TensorProto.FLOAT, [], b_s),
                onnx.helper.make_tensor('bzp', data_type, [], b_zp),
                onnx.helper.make_tensor('c', TensorProto.INT32, c_shape, c_data),
                onnx.helper.make_tensor('ys', TensorProto.FLOAT, [], y_s),
                onnx.helper.make_tensor('yzp', data_type, [], y_zp),
                onnx.helper.make_tensor('s', TensorProto.FLOAT, [], a_s),
                onnx.helper.make_tensor('zp', data_type, [], a_zp),
            ]
        ),
    )
    onnx_model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))  # Add the opset with QGemm

    input_data = {
        0: np.arange(math.prod(a_shape)).reshape(a_shape).astype(np.float32),
        1: np.arange(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(data_type)),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)
    intermediate_tflite_model_provider.assert_converted_model_has_operators(
        [BuiltinOperator.QUANTIZE, BuiltinOperator.MUL, BuiltinOperator.FULLY_CONNECTED])


@pytest.mark.parametrize(
    'alpha',
    [
        2.0,
        20.0,
        0.1,
        0.0
    ])
def test_convert_q_gemm_with_alpha_and_static_input_a(alpha: float, intermediate_tflite_model_provider):
    data_type = TensorProto.INT8
    a_shape, b_shape, c_shape = _get_q_gemm_shapes(10, 20, 30, trans_b=True)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.01], [10], [0.1], [0], [0.05], [-5]

    a_data = np.arange(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(data_type))
    c_data = [0] * c_shape[0]

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                                      domain='com.microsoft', alpha=alpha, transB=True)
            ],
            'QGemm test',
            [
                onnx.helper.make_tensor_value_info('b', data_type, b_shape)
            ],
            [onnx.helper.make_tensor_value_info('y', data_type, ())],
            [
                onnx.helper.make_tensor('a', data_type, a_shape, a_data),
                onnx.helper.make_tensor('as', TensorProto.FLOAT, [], a_s),
                onnx.helper.make_tensor('azp', data_type, [], a_zp),
                onnx.helper.make_tensor('bs', TensorProto.FLOAT, [], b_s),
                onnx.helper.make_tensor('bzp', data_type, [], b_zp),
                onnx.helper.make_tensor('c', TensorProto.INT32, c_shape, c_data),
                onnx.helper.make_tensor('ys', TensorProto.FLOAT, [], y_s),
                onnx.helper.make_tensor('yzp', data_type, [], y_zp),
            ]
        ),
    )
    onnx_model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))  # Add the opset with QGemm

    executors.convert_run_compare(onnx_model,
                                  np.arange(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(data_type)),
                                  atol=1)
    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.FULLY_CONNECTED])


def test_convert_q_gemm_with_mismatched_types():
    a_type = TensorProto.INT8
    b_type = TensorProto.UINT8

    a_shape, b_shape, c_shape = _get_q_gemm_shapes(10, 20, 30)
    a_s, a_zp, b_s, b_zp, y_s, y_zp = [0.01], [10], [0.1], [0], [0.05], [-5]
    c_data = [0] * c_shape[0]

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c', 'ys', 'yzp'], ['y'],
                                      domain='com.microsoft')
            ],
            'QGemm test',
            [

                onnx.helper.make_tensor_value_info('a', a_type, a_shape),
                onnx.helper.make_tensor_value_info('b', b_type, b_shape)
            ],
            [onnx.helper.make_tensor_value_info('y', a_type, ())],
            [
                onnx.helper.make_tensor('as', TensorProto.FLOAT, [], a_s),
                onnx.helper.make_tensor('azp', a_type, [], a_zp),
                onnx.helper.make_tensor('bs', TensorProto.FLOAT, [], b_s),
                onnx.helper.make_tensor('bzp', b_type, [], b_zp),
                onnx.helper.make_tensor('c', TensorProto.INT32, c_shape, c_data),
                onnx.helper.make_tensor('ys', TensorProto.FLOAT, [], y_s),
                onnx.helper.make_tensor('yzp', a_type, [], y_zp),
            ]
        ),
    )
    onnx_model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))  # Add the opset with QGemm

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR


def test_convert_q_gemm_with_float_output():
    data_type = TensorProto.INT8

    a_shape, b_shape, c_shape = _get_q_gemm_shapes(10, 20, 30)
    a_s, a_zp, b_s, b_zp = [0.01], [10], [0.1], [0]
    c_data = [0] * c_shape[0]

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('QGemm', ['a', 'as', 'azp', 'b', 'bs', 'bzp', 'c'], ['y'],
                                      domain='com.microsoft')
            ],
            'QGemm test',
            [

                onnx.helper.make_tensor_value_info('a', data_type, a_shape),
                onnx.helper.make_tensor_value_info('b', data_type, b_shape)
            ],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
            [
                onnx.helper.make_tensor('as', TensorProto.FLOAT, [], a_s),
                onnx.helper.make_tensor('azp', data_type, [], a_zp),
                onnx.helper.make_tensor('bs', TensorProto.FLOAT, [], b_s),
                onnx.helper.make_tensor('bzp', data_type, [], b_zp),
                onnx.helper.make_tensor('c', TensorProto.INT32, c_shape, c_data),
            ]
        ),
    )
    onnx_model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))  # Add the opset with QGemm

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
