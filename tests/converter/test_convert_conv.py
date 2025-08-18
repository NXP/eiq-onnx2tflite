#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx.helper
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import SkipShapeInferenceConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.tflite_generator.builtin_options import depthwise_conv_2d_options, conv_2d_options, conv_3d_options
from tests import executors


def test_convert_conv__unsupported_type():
    _type = TensorProto.DOUBLE

    shape = [2, 4, 6, 8]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Conv', ['x', 'w'], ['y'])
        ],
        'Conv test',
        [
            onnx.helper.make_tensor_value_info('x', _type, shape),
            onnx.helper.make_tensor_value_info('w', _type, shape),
        ],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("kernel_shape", [[3, 3], [4, 5]], ids=lambda x: f"kernel_shape = {x}")
@pytest.mark.parametrize("auto_pad", ["VALID", "SAME_UPPER", "SAME_LOWER", None], ids=lambda x: f"auto_pad = {x}")
@pytest.mark.parametrize("pads", [None, [0, 1, 0, 1], [1, 1, 1, 1], [2, 1, 0, 2]], ids=lambda x: f"pads = {x}")
@pytest.mark.parametrize("strides", [[1, 2], [3, 2], None], ids=lambda x: f"strides = {x}")
@pytest.mark.parametrize("dilations", [[1, 2], [3, 2], None], ids=lambda x: f"dilations = {x}")
@pytest.mark.parametrize("group", [1, 2, 4], ids=lambda x: f"group = {x}")
def test_convert_2d_conv(kernel_shape: list[int], auto_pad: str | None, pads: list[int] | None,
                         strides: list[int] | None, dilations: list[int] | None, group: int,
                         intermediate_tflite_model_provider):
    if auto_pad is not None and pads is not None:
        # Only 1 can be set at a time
        return

    if (auto_pad in {"SAME_UPPER", "SAME_LOWER"}) and dilations is not None:
        # ONNX Runtime doesn't support this combination
        return

    np.random.seed(42)

    weight_shape = [20, 10] + kernel_shape
    input_shape = [5, weight_shape[1] * group, 15, 20]
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    bias_shape = [weight_shape[0]]
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad,
                               pads=pads, strides=strides, dilations=dilations, group=group)],
        'Conv test',
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

    assert intermediate_tflite_model_provider.get_op_count(conv_2d_options.Conv2D) == group


@pytest.mark.parametrize("kernel_shape", [[3], [4]], ids=lambda x: f"kernel_shape = {x}")
@pytest.mark.parametrize("auto_pad", ["VALID", "SAME_UPPER", "SAME_LOWER", None], ids=lambda x: f"auto_pad = {x}")
@pytest.mark.parametrize("pads", [None, [0, 1], [1, 0], [1, 1], [0, 2]], ids=lambda x: f"pads = {x}")
@pytest.mark.parametrize("strides", [[2], None], ids=lambda x: f"strides = {x}")
@pytest.mark.parametrize("dilations", [[2], None], ids=lambda x: f"dilations = {x}")
def test_convert_1d_conv(kernel_shape: list[int], auto_pad: str | None, pads: list[int] | None,
                         strides: list[int] | None, dilations: list[int] | None):
    if auto_pad is not None and pads is not None:
        # Only 1 can be set at a time
        return

    if (auto_pad in {"SAME_UPPER", "SAME_LOWER"}) and dilations is not None:
        # ONNX Runtime doesn't support this combination
        return

    np.random.seed(23)

    input_shape = [5, 20, 30]
    weight_shape = [42, 20] + kernel_shape
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    bias_shape = [weight_shape[0]]
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad,
                               pads=pads, strides=strides, dilations=dilations)],
        'Conv test',
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


@pytest.mark.parametrize("kernel_shape", [[3, 3, 3], [4, 5, 6]], ids=lambda x: f"kernel_shape = {x}")
@pytest.mark.parametrize("auto_pad", ["VALID", "SAME_UPPER", "SAME_LOWER", None], ids=lambda x: f"auto_pad = {x}")
@pytest.mark.parametrize("pads", [None, [0, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1], [2, 1, 0, 2, 0, 1]],
                         ids=lambda x: f"pads = {x}")
@pytest.mark.parametrize("strides", [[1, 2, 3], [3, 2, 2], None], ids=lambda x: f"strides = {x}")
@pytest.mark.parametrize("dilations", [[1, 2, 1], [3, 2, 2], None], ids=lambda x: f"dilations = {x}")
def test_convert_3d_conv(kernel_shape: list[int], auto_pad: str | None, pads: list[int] | None,
                         strides: list[int] | None, dilations: list[int] | None):
    if auto_pad is not None and pads is not None:
        # Only 1 can be set at a time
        return

    if (auto_pad in {"SAME_UPPER", "SAME_LOWER"}) and dilations is not None:
        # ONNX Runtime doesn't support this combination
        return

    np.random.seed(42)

    input_shape = [5, 10, 15, 20, 12]
    weight_shape = [20, 10] + kernel_shape
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    bias_shape = [weight_shape[0]]
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "b"], ["o"], kernel_shape=kernel_shape, auto_pad=auto_pad,
                               pads=pads, strides=strides, dilations=dilations)],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("b", TensorProto.FLOAT, bias_shape, bias),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_1d_conv_with_dynamic_weights():
    kernel_shape = [3]
    np.random.seed(1337)

    input_shape = [5, 20, 30]
    weight_shape = [42, 20] + kernel_shape

    bias_shape = [weight_shape[0]]
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape,
                               pads=[4, 1], dilations=[2])],
        'Conv test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.arange(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_3d_conv_with_dynamic_weights():
    kernel_shape = [3, 4, 5]
    np.random.seed(1337)

    input_shape = [4, 8, 12, 16, 20]
    weight_shape = [42, 8] + kernel_shape

    bias_shape = [weight_shape[0]]
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape)],
        'Conv test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("bias", TensorProto.FLOAT, bias_shape, bias)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.arange(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32),
    }

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_2d_conv_without_bias():
    input_shape = [5, 10, 23, 42]
    kernel_shape = [5, 7]

    weight_shape = [20, 10] + kernel_shape

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w"], ["output"], kernel_shape=kernel_shape,
                               pads=[0, 1, 2, 3], strides=[2, 3])],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_2d_conv__omitted_bias():
    input_shape = [5, 10, 23, 42]
    kernel_shape = [5, 7]

    weight_shape = [20, 10] + kernel_shape

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", ""], ["output"], kernel_shape=kernel_shape,
                               pads=[0, 1, 2, 3], strides=[2, 3])],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_2d_conv__separated_and_omitted_bias():
    input_shape = [5, 30, 23, 5]
    kernel_shape = [2, 3]

    weight_shape = [21, 10] + kernel_shape

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", ""], ["output"], kernel_shape=kernel_shape,
                               pads=[0, 1, 2, 3], strides=[2, 3], group=3)],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_1d_conv_without_bias():
    input_shape = [5, 37, 15]
    kernel_shape = [3]

    weight_shape = [13, 37] + kernel_shape

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w"], ["output"], kernel_shape=kernel_shape,
                               pads=[1, 3], strides=[3])],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_1d_conv__omitted_bias():
    input_shape = [5, 37, 15]
    kernel_shape = [3]

    weight_shape = [13, 37] + kernel_shape

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", ""], ["output"], kernel_shape=kernel_shape,
                               pads=[1, 3], strides=[3])],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_1d_conv__separated_and_omitted_bias():
    input_shape = [5, 30, 15]
    kernel_shape = [3]

    weight_shape = [12, 10] + kernel_shape

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", ""], ["output"], kernel_shape=kernel_shape,
                               pads=[1, 3], strides=[3], group=3)],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_3d_conv_without_bias():
    input_shape = [4, 8, 12, 16, 20]
    kernel_shape = [2, 3, 4]

    weight_shape = [5, 8] + kernel_shape

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w"], ["output"], kernel_shape=kernel_shape,
                               pads=[1, 0, 1, 3, 2, 4], strides=[3, 2, 1], dilations=[2, 1, 4])],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_3d_conv__omitted_bias():
    input_shape = [4, 8, 12, 16, 20]
    kernel_shape = [2, 3, 4]

    weight_shape = [5, 8] + kernel_shape

    np.random.seed(42)
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", ""], ["output"], kernel_shape=kernel_shape,
                               pads=[1, 0, 1, 3, 2, 4], strides=[3, 2, 1], dilations=[2, 1, 4])],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


# TODO Not working for some reason. (probably unrelated to the omitted bias).
# def test_convert_3d_conv__separated_and_omitted_bias():
#     input_shape = [4, 27, 12, 16, 20]
#     kernel_shape = [2, 3, 4]
#
#     weight_shape = [9, 9] + kernel_shape
#
#     np.random.seed(42)
#     weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)
#
#     graph = onnx.helper.make_graph(
#         [onnx.helper.make_node("Conv", ["x", "w", ""], ["output"], kernel_shape=kernel_shape,
#                                pads=[1, 0, 1, 3, 2, 4], strides=[3, 2, 1], dilations=[2, 1, 4], group=3)],
#         'Conv test',
#         [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
#         [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
#         [onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights)]
#     )
#
#     onnx_model = onnx.helper.make_model(graph)
#
#     input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
#
#     executors.convert_run_compare(onnx_model, input_data)


def test_convert_conv_with_invalid_rank():
    # Try with 6D
    input_shape = [2, 4, 6, 8, 10, 12]
    kernel_shape = input_shape[2:]

    np.random.seed(42)

    weight_shape = [7, 4] + kernel_shape
    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    bias_shape = [weight_shape[0]]
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "b"], ["output"], kernel_shape=kernel_shape)],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("b", TensorProto.FLOAT, bias_shape, bias),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model)
    assert e.value.error_code == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_conv_with_not_enough_inputs():
    input_shape = [2, 4, 6, 8]
    kernel_shape = input_shape[2:]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x"], ["output"], kernel_shape=kernel_shape)],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model, conversion_config=SkipShapeInferenceConfig())
    assert e.value.error_code == logger.Code.SHAPE_INFERENCE_ERROR  # onnx.checker catches the error first


def test_convert_conv_with_strides_and_skewed_padding():
    # Convolution with strides=2 and padding only along one dimension (the H dimension in NxCxHxW tensor)
    # and only at begin of the axis - i.e. pad 2 rows at beginning
    node = onnx.helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[2, 1, 0, 1],
        strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
    )

    x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.],
                    [25., 26., 27., 28., 29.],
                    [30., 31., 32., 33., 34.]]]]).astype(np.float32)
    w = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(np.float32)
    graph = onnx.helper.make_graph(
        [node],
        "conv",
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 1, 7, 5)),
            onnx.helper.make_tensor_value_info("W", TensorProto.FLOAT, (1, 1, 3, 3))
        ],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ())]
    )

    onnx_model = onnx.helper.make_model(graph, producer_name="onnx-conv")

    input_data = {
        0: x,
        1: w,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize("kernel_shape", [[3, 3], [4, 5]], ids=lambda x: f"kernel_shape={x}")
@pytest.mark.parametrize("auto_pad", ["VALID", "SAME_UPPER", "SAME_LOWER", None], ids=lambda x: f"auto_pad={x}")
@pytest.mark.parametrize("pads", [None, [0, 1, 0, 1], [2, 1, 0, 2]], ids=lambda x: f"pads={x}")
@pytest.mark.parametrize("strides", [[1, 2], None], ids=lambda x: f"strides={x}")
@pytest.mark.parametrize("dilations", [[1, 2], [3, 2], None], ids=lambda x: f"dilations={x}")
def test_convert_conv_2D_into_depthwise_conv_2d(kernel_shape: list[int], auto_pad: str | None, pads: list[int] | None,
                                                strides: list[int] | None, dilations: list[int] | None,
                                                intermediate_tflite_model_provider):
    if auto_pad is not None and pads is not None:
        # Only 1 can be set at a time
        return

    if (auto_pad in {"SAME_UPPER", "SAME_LOWER"}) and dilations is not None:
        # ONNX Runtime doesn't support this combination
        return

    np.random.seed(42)

    input_shape = [1, 32, 15, 17]  # [batch, input_channels, height, width]
    weight_shape = [32, 1] + kernel_shape  # [output_channels, kernel_channels, kernel_height, kernel_width]
    group = 32

    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    bias_shape = [weight_shape[0]]
    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "bias"], ["output"], kernel_shape=kernel_shape, auto_pad=auto_pad,
                               pads=pads, strides=strides, dilations=dilations, group=group)],
        'Conv test',
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

    assert intermediate_tflite_model_provider.get_op_count(depthwise_conv_2d_options.DepthwiseConv2D) == 1


@pytest.mark.parametrize("group", [2, 4, 8], ids=lambda x: f"group={x}")
def test_convert_3d_conv_with_group__static_weights(group, intermediate_tflite_model_provider):
    kernel_shape = [2, 2, 2]
    dilations = [1, 1, 1]
    pads = [0, 0, 0, 0, 0, 0]
    strides = [1, 1, 1]

    weight_shape = [16, 16 // group] + kernel_shape
    input_shape = [1, weight_shape[1] * group, 2, 2, 2]
    bias_shape = [weight_shape[0]]

    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "b"], ["o"], kernel_shape=kernel_shape,
                               group=group, pads=pads, strides=strides, dilations=dilations)],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("b", TensorProto.FLOAT, bias_shape, bias),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)

    assert intermediate_tflite_model_provider.get_op_count(conv_3d_options.Conv3D) == group


@pytest.mark.parametrize("group", [2, 4, 8], ids=lambda x: f"group={x}")
def test_convert_3d_conv_with_group__dynamic_weights(group, intermediate_tflite_model_provider):
    kernel_shape = [2, 2, 2]
    dilations = [1, 1, 1]
    pads = [0, 0, 0, 0, 0, 0]
    strides = [1, 1, 1]

    weight_shape = [16, 16 // group] + kernel_shape
    input_shape = [1, weight_shape[1] * group, 2, 2, 2]
    bias_shape = [weight_shape[0]]

    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "b"], ["o"], kernel_shape=kernel_shape,
                               group=group, pads=pads, strides=strides, dilations=dilations)],
        'Conv test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("b", TensorProto.FLOAT, bias_shape, bias)],
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)
    }

    executors.convert_run_compare(onnx_model, input_data)

    assert intermediate_tflite_model_provider.get_op_count(conv_3d_options.Conv3D) == group


@pytest.mark.parametrize("group", [5, 12], ids=lambda x: f"group={x}")
def test_convert_3d_conv_with_unsupported_group(group):
    kernel_shape = [2, 2, 2]
    dilations = [1, 1, 1]
    pads = [0, 0, 0, 0, 0, 0]
    strides = [1, 1, 1]

    input_shape = [1, 4, 2, 2, 2]
    weight_shape = [24, 60 // group] + kernel_shape
    bias_shape = [weight_shape[0]]

    weights = np.random.random(np.prod(weight_shape)).reshape(weight_shape).astype(np.float32)

    bias = np.random.random(bias_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w", "b"], ["o"], kernel_shape=kernel_shape,
                               group=group, pads=pads, strides=strides, dilations=dilations)],
        'Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("w", TensorProto.FLOAT, weight_shape, weights),
            onnx.helper.make_tensor("b", TensorProto.FLOAT, bias_shape, bias),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
