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

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


@pytest.mark.parametrize("_type", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_conv_transpose__quantized(_type: TensorProto.DataType):
    shape = [2, 4, 6, 8]
    w_shape = [4, 4, 4, 4]
    b_shape = [4]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('ConvTranspose', ['x1', 'w', 'b'], ['y'])
        ],
        'Quantized input test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('w', _type, w_shape),
            onnx.helper.make_tensor_value_info('b', _type, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', _type, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.INVALID_ONNX_MODEL
    assert 'ConvTranspose' in logger.conversion_log.get_node_error_message(1)


def test_convert_conv_transpose__unsupported_type():
    _type = TensorProto.DOUBLE

    shape = [2, 4, 6, 8]
    w_shape = [4, 4, 4, 4]
    b_shape = [4]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('ConvTranspose', ['x', 'w', 'b'], ['y'])
        ],
        'ConvTranspose test',
        [
            onnx.helper.make_tensor_value_info('x', _type, shape),
            onnx.helper.make_tensor_value_info('w', _type, w_shape),
            onnx.helper.make_tensor_value_info('b', _type, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', _type, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("input_shape", [[1, 2, 3, 3], [1, 3, 4, 3]], ids=lambda x: f"input_shape={x}")
@pytest.mark.parametrize("kernel_shape", [[3, 3], [4, 5]], ids=lambda x: f"kernel_shape={x}")
@pytest.mark.parametrize("auto_pad", ["VALID", "SAME_UPPER", "SAME_LOWER", None], ids=lambda x: f"auto_pad={x}")
@pytest.mark.parametrize("strides", [[1, 2], [3, 2], None], ids=lambda x: f"strides={x}")
@pytest.mark.parametrize("output_channels", [2, 3], ids=lambda x: f"output_channels={x}")
def test_convert_2d_conv_transpose__with_auto_pad(input_shape, kernel_shape, auto_pad, strides, output_channels):
    weight_shape = [input_shape[1], output_channels] + kernel_shape
    bias_shape = [weight_shape[1]]

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w", "bias"], ["output"],
                               auto_pad=auto_pad, strides=strides)],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("input_shape", [[1, 2, 3, 3], [1, 3, 4, 3]], ids=lambda x: f"input_shape={x}")
@pytest.mark.parametrize("kernel_shape", [[3, 3], [4, 5]], ids=lambda x: f"kernel_shape={x}")
@pytest.mark.parametrize("pads", [[0, 1, 0, 1], [1, 1, 1, 1], [2, 1, 0, 2]], ids=lambda x: f"pads={x}")
@pytest.mark.parametrize("strides", [[1, 2], [3, 2], None], ids=lambda x: f"strides={x}")
@pytest.mark.parametrize("output_channels", [2, 3], ids=lambda x: f"output_channels={x}")
def test_convert_2d_conv_transpose__with_pads(input_shape, kernel_shape, pads, strides, output_channels):
    weight_shape = [input_shape[1], output_channels] + kernel_shape
    bias_shape = [weight_shape[1]]

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w", "bias"], ["output"],
                               pads=pads, strides=strides)],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_2d_conv_transpose__without_bias():
    input_shape = [1, 4, 3, 3]
    weight_shape = [4, 2, 3, 3]

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w"], ["output"])],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_2d_conv_transpose__with_bias():
    input_shape = [1, 1, 2, 2]
    weight_shape = [1, 2, 2, 2]
    bias_shape = [weight_shape[1]]

    np.random.seed(42)
    weights = np.arange(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)
    bias = np.arange(bias_shape[0]).reshape(bias_shape).astype(np.float32) + 0.548787654709

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w", "bias"], ["output"])],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("kernel_shape", [[3, 3], [4, 5]], ids=lambda x: f"kernel_shape={x}")
def test_convert_2d_conv_transpose__various_kernel_shapes(kernel_shape):
    input_shape = [1, 4, 3, 3]
    weight_shape = [4, 2] + kernel_shape
    bias_shape = [weight_shape[1]]

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w", "bias"], ["output"])],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("auto_pad", ["VALID", "SAME_UPPER", "SAME_LOWER"], ids=lambda x: f"auto_pad={x}")
def test_convert_2d_conv_transpose__various_auto_pad(auto_pad):
    input_shape = [1, 1, 2, 2]
    weight_shape = [1, 1, 2, 2]

    np.random.seed(42)
    weights = np.arange(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w"], ["output"], auto_pad=auto_pad)],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_2d_conv_transpose__with_group():
    input_shape = [1, 4, 3, 3]
    weight_shape = [4, 2, 4, 5]

    weights = np.arange(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w"], ["output"], group=2)],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    log = logger.conversion_log
    assert log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert "ConvTranspose with 'group' other than '1' cannot be converted to TFLite." in log.get_node_error_message(0)


def test_convert_2d_conv_transpose__with_dilations():
    input_shape = [1, 4, 3, 3]
    weight_shape = [4, 2, 4, 5]

    weights = np.arange(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w"], ["output"], dilations=[1, 2])],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    log = logger.conversion_log
    assert log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert "ConvTranspose with 'dilations' other than '1' cannot be converted" in log.get_node_error_message(0)


def test_convert_1d_conv_transpose():
    input_shape = [1, 4, 3]
    weight_shape = [4, 2, 4]

    weights = np.arange(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ConvTranspose", ["x", "w"], ["output"], group=2)],
        'ConvTranspose test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    log = logger.conversion_log
    assert log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert "Conversion of ONNX ConvTranspose with a 1D kernel is not implemented." in log.get_node_error_message(0)
