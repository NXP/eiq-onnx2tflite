#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math
from typing import List

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.tflite_generator.builtin_options import concatenation_options, conv_2d_options, \
    depthwise_conv_2d_options, split_options
from tests import executors


def _create_q_linear_conv_model(input_shape: List[int], weights_shape: List[int],
                                dilations: List[int], strides: List[int], auto_pad: str, pads: List[int], group: int,
                                weights_data, bias_data, weights_scale, weights_zero_point,
                                input_type, output_type, weights_type) -> onnx.ModelProto:
    kernel_shape = weights_shape[2:]
    input_scale = [0.01865844801068306]
    input_zero_point = [114]
    output_scale = [0.12412]
    output_zero_point = [0]

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("QLinearConv",
                                      ["input", "input_scale", "input_zero_point", "weights", "weights_scale",
                                       "weights_zero_point", "output_scale", "output_zero_point", "bias"], ["output"],
                                      dilations=dilations, strides=strides, kernel_shape=kernel_shape, pads=pads,
                                      group=group, auto_pad=auto_pad)
            ],
            name="QLinearConv_test",
            inputs=[onnx.helper.make_tensor_value_info("input", input_type, input_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", output_type, ())],
            initializer=[
                onnx.helper.make_tensor("weights", weights_type, weights_shape, weights_data),
                onnx.helper.make_tensor("bias", onnx.TensorProto.INT32, [len(bias_data)], bias_data),
                onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [len(input_scale)], input_scale),
                onnx.helper.make_tensor("input_zero_point", input_type, [len(input_zero_point)],
                                        input_zero_point),
                onnx.helper.make_tensor("weights_scale", onnx.TensorProto.FLOAT, [len(weights_scale)], weights_scale),
                onnx.helper.make_tensor("weights_zero_point", weights_type, [len(weights_zero_point)],
                                        weights_zero_point),
                onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [len(output_scale)], output_scale),
                onnx.helper.make_tensor("output_zero_point", output_type, [len(output_zero_point)],
                                        output_zero_point),

            ]
        )
    )
    return onnx_model


@pytest.mark.parametrize(
    "input_shape,weights_shape,dilations,strides,auto_pad,pads,group,weights_scale,weights_zero_point", [
        pytest.param([1, 3, 224, 224], [96, 3, 11, 11], [1, 1], [4, 4], None, [0, 0, 0, 0], 1, [0.124],
                     [0], id="per tensor, explicit padding"),
        pytest.param([1, 384, 12, 12], [256, 192, 3, 3], [1, 1], [1, 1], "SAME_LOWER", None, 2,
                     [0.05], [0], id="per tensor"),
        pytest.param([1, 384, 12, 12], [256, 96, 3, 3], [1, 1], [1, 1], "SAME_LOWER", None, 4,
                     [0.05], [0], id="per tensor"),
        pytest.param([1, 256, 10, 10], [3, 256, 3, 3], [5, 5], [1, 1], None, [5, 5, 5, 5], 1,
                     [0.1, 0.2, 0.05], [0, 0, 0], id="per channel"),
    ])
def test_convert_2d_q_linear_conv_with_signed_types(input_shape: List[int], weights_shape: List[int],
                                                    dilations: List[int], strides: List[int],
                                                    auto_pad: str, pads: List[int], group: int,
                                                    weights_scale: List[float], weights_zero_point: List[int]):
    weights_data = np.arange(math.prod(weights_shape)).reshape(weights_shape).astype(np.int8)
    bias_data = np.arange(weights_shape[0]).astype(np.int32)

    onnx_model = _create_q_linear_conv_model(input_shape, weights_shape, dilations, strides, auto_pad,
                                             pads, group, weights_data, bias_data, weights_scale, weights_zero_point,
                                             onnx.TensorProto.INT8, onnx.TensorProto.INT8, onnx.TensorProto.INT8)

    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(np.int8)

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize("input_shape,weights_scale,group", [
    pytest.param([1, 12, 10, 10], [0.124], 1, id="per tensor,group=1"),  # single traditional convolution
    pytest.param([1, 12, 10, 10], [0.124], 3, id="per tensor,group=3"),  # separated convolutions
    pytest.param([1, 12, 10, 10], [0.1, 0.2, 0.05], 1, id="per channel,group=1"),  # single traditional convolution
    pytest.param([1, 12, 10, 10], [0.1, 0.2, 0.05], 3, id="per channel,group=3"),  # group convolution
    pytest.param([1, 3, 10, 10], [0.124], 3, id="per tensor,group=3"),  # depthwise convolution
    pytest.param([1, 3, 10, 10], [0.1, 0.2, 0.05], 3, id="per channel,depthwise"),  # depthwise convolution
])
@pytest.mark.parametrize("input_type,output_type,weight_type", [
    pytest.param(onnx.TensorProto.INT8, onnx.TensorProto.INT8, onnx.TensorProto.INT8, id="I:INT8,O:INT8,W:INT8"),
    pytest.param(onnx.TensorProto.UINT8, onnx.TensorProto.UINT8, onnx.TensorProto.INT8, id="I:UINT8,O:UINT8,W:INT8"),
    pytest.param(onnx.TensorProto.UINT8, onnx.TensorProto.UINT8, onnx.TensorProto.UINT8, id="I:UINT8,O:UINT8,W:UINT8")
])
def test_convert_2d_q_linear_conv__io_types_variants(
        input_shape: List[int], weights_scale: List[float], input_type, output_type, weight_type, group: int):
    weights_shape = [3, input_shape[1] // group, 3, 3]  # [out_channels, kernel_channels, kernel_height, kernel_width]

    bias_data = np.arange(weights_shape[0]).astype(np.int32)

    if weight_type == onnx.TensorProto.INT8:
        weights_zero_point = [0] * len(weights_scale)
        weights_data = np.random.randint(-128, 127, weights_shape).astype(np.int8)
    else:
        weights_zero_point = [128] * len(weights_scale)
        weights_data = np.random.randint(0, 255, weights_shape).astype(np.uint8)

    # noinspection PyTypeChecker
    onnx_model = _create_q_linear_conv_model(input_shape, weights_shape, None, None, "SAME_LOWER",
                                             None, group, weights_data, bias_data, weights_scale, weights_zero_point,
                                             input_type, output_type, weight_type)

    if input_type == onnx.TensorProto.INT8:
        input_data = np.random.randint(-20, 20, input_shape).astype(np.int8)
    else:
        input_data = np.random.randint(0, 20, input_shape).astype(np.uint8)

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize(
    "input_shape, weights_shape, dilations, strides, auto_pad, pads, group, weights_scale, "
    "weights_zero_point",
    [
        pytest.param([1, 32, 224, 224], [3, 32, 3, 3], [1, 1], [2, 2], None, [1, 1, 1, 1], 1,
                     [0.1, 0.2, 0.3], [128, 128, 128], id="per tensor"),
        pytest.param([1, 384, 12, 12], [4, 192, 3, 3], [1, 1], [1, 1], "SAME_LOWER", None, 2,
                     [0.001, 0.1, 0.5, 0.2], [128] * 4, id="per channel"),
    ])
def test_convert_2d_q_linear_conv_with_unsigned_types(input_shape: List[int], weights_shape: List[int],
                                                      dilations: List[int], strides: List[int], auto_pad: str,
                                                      pads: List[int], group: int,
                                                      weights_scale: List[float], weights_zero_point: List[int]):
    weights_data = np.arange(math.prod(weights_shape)).reshape(weights_shape).astype(np.uint8)
    bias_data = np.arange(weights_shape[0]).astype(np.int32)

    onnx_model = _create_q_linear_conv_model(input_shape, weights_shape, dilations, strides, auto_pad,
                                             pads, group, weights_data, bias_data, weights_scale, weights_zero_point,
                                             onnx.TensorProto.UINT8, onnx.TensorProto.UINT8, onnx.TensorProto.UINT8)
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(np.uint8)

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize(
    "input_shape, weights_shape, dilations, strides, auto_pad, pads, group, weights_scale, "
    "weights_zero_point",
    [
        pytest.param([1, 32, 224, 224], [3, 32, 3, 3], [1, 1], [2, 2], None, [1, 1, 1, 1], 1,
                     [0.1, 0.2, 0.3], [0, 0, 0], id="per tensor"),
        pytest.param([1, 384, 12, 12], [4, 192, 3, 3], [1, 1], [1, 1], "SAME_LOWER", None, 2,
                     [0.001, 0.1, 0.5, 0.2], [0] * 4, id="per channel"),
    ])
def test_convert_2d_q_linear_conv_with_unsigned_input_and_signed_weights(input_shape: List[int],
                                                                         weights_shape: List[int],
                                                                         dilations: List[int],
                                                                         strides: List[int], auto_pad: str,
                                                                         pads: List[int], group: int,
                                                                         weights_scale: List[float],
                                                                         weights_zero_point: List[int]):
    weights_data = np.arange(math.prod(weights_shape)).reshape(weights_shape).astype(np.int8)
    bias_data = np.arange(weights_shape[0]).astype(np.int32)

    onnx_model = _create_q_linear_conv_model(input_shape, weights_shape, dilations, strides, auto_pad,
                                             pads, group, weights_data, bias_data, weights_scale, weights_zero_point,
                                             onnx.TensorProto.UINT8, onnx.TensorProto.UINT8, onnx.TensorProto.INT8)
    input_data = np.random.randint(0, 100, math.prod(input_shape)).reshape(input_shape).astype(np.uint8)

    executors.convert_run_compare(onnx_model, input_data, atol=1)


def test_convert_2d_q_linear_conv_without_optional_attributes():
    # Test corresponds to thirdparty/onnx/onnx/backend/test/case/node/qlinearconv.py, but with static quantization
    x = np.array(
        [
            [255, 174, 162, 25, 203, 168, 58],
            [15, 59, 237, 95, 129, 0, 64],
            [56, 242, 153, 221, 168, 12, 166],
            [232, 178, 186, 195, 237, 162, 237],
            [188, 39, 124, 77, 80, 102, 43],
            [127, 230, 21, 83, 41, 40, 134],
            [255, 154, 92, 141, 42, 148, 247],
        ],
        dtype=np.uint8,
    ).reshape((1, 1, 7, 7))

    x_scale = np.float32(0.00369204697)
    x_zero_point = np.uint8(132)

    w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))

    w_scale = np.array([0.00172794575], dtype=np.float32)
    w_zero_point = np.array([128], dtype=np.uint8)

    y_scale = np.float32(0.00162681262)
    y_zero_point = np.uint8(123)

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[onnx.helper.make_node(
                "QLinearConv",
                inputs=[
                    "input",
                    "x_scale",
                    "x_zero_point",
                    "w",
                    "w_scale",
                    "w_zero_point",
                    "y_scale",
                    "y_zero_point",
                ],
                outputs=["output"])],
            name="QLinearConv_test",
            inputs=[onnx.helper.make_tensor_value_info("input", onnx.TensorProto.UINT8, x.shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.UINT8, ())],
            initializer=[
                onnx.helper.make_tensor("w", onnx.TensorProto.UINT8, w.shape, w),
                onnx.helper.make_tensor("x_scale", onnx.TensorProto.FLOAT, [1], [x_scale]),
                onnx.helper.make_tensor("x_zero_point", onnx.TensorProto.UINT8, [1],
                                        [x_zero_point]),
                onnx.helper.make_tensor("w_scale", onnx.TensorProto.FLOAT, w_scale.shape, w_scale),
                onnx.helper.make_tensor("w_zero_point", onnx.TensorProto.UINT8, w_zero_point.shape,
                                        w_zero_point),
                onnx.helper.make_tensor("y_scale", onnx.TensorProto.FLOAT, [1], [y_scale]),
                onnx.helper.make_tensor("y_zero_point", onnx.TensorProto.UINT8, [1],
                                        [y_zero_point]),
            ]
        )
    )

    executors.convert_run_compare(onnx_model, x)


@pytest.mark.parametrize("dilations,strides,auto_pad,pads,group,weights_scale,weights_zero_point", [
    pytest.param([1, 1], [4, 4], None, [0, 0, 0, 0], 1, [0.124], [0], id="per tensor, explicit padding"),
    pytest.param([1, 1], [1, 1], "SAME_LOWER", None, 2, [0.05], [0], id="per tensor"),
    pytest.param([5, 5], [1, 1], None, [5, 5, 5, 5], 1, [0.1, 0.2, 0.05, 0.4, 0.5], [0, 0, 0, 0, 0], id="per channel"),
])
def test_convert_2d_q_linear_conv_into_depthwise_conv_2d_static_weights(
        dilations: List[int], strides: List[int], auto_pad: str, pads: List[int], group: int,
        weights_scale: List[float], weights_zero_point: List[int], intermediate_tflite_model_provider):
    kernel_shape = [3, 3]
    input_shape = [1, 5, 15, 17]  # [batch, input_channels, height, width]
    weights_shape = [5, 1] + kernel_shape  # [output_channels, kernel_channels, kernel_height, kernel_width]
    group = 5

    weights_data = np.arange(math.prod(weights_shape)).reshape(weights_shape).astype(np.int8)
    bias_data = np.arange(weights_shape[0]).astype(np.int32)

    # noinspection PyTypeChecker
    onnx_model = _create_q_linear_conv_model(input_shape, weights_shape, dilations, strides, auto_pad,
                                             pads, group, weights_data, bias_data, weights_scale, weights_zero_point,
                                             onnx.TensorProto.INT8, onnx.TensorProto.INT8, onnx.TensorProto.INT8)

    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(np.int8)

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    assert intermediate_tflite_model_provider.get_op_count(depthwise_conv_2d_options.DepthwiseConv2D) == 1


@pytest.mark.parametrize("dilations,strides,auto_pad,pads,weights_scale,weights_zero_point", [
    pytest.param([1, 1], [4, 4], None, [0, 0, 0, 0], [0.124], [0], id="per tensor, explicit padding"),
    pytest.param([1, 1], [1, 1], "SAME_LOWER", None, [0.05], [0], id="per tensor"),
    pytest.param([5, 5], [1, 1], None, [5, 5, 5, 5], [0.1, 0.2, 0.05, 0.4, 0.5], [0, 0, 0, 0, 0], id="per channel"),
])
def test_convert_2d_q_linear_conv_into_depthwise_conv_2d_dynamic_int8_weights(
        dilations: List[int], strides: List[int], auto_pad: str, pads: List[int],
        weights_scale: List[float], weights_zero_point: List[int], intermediate_tflite_model_provider):
    kernel_shape = [3, 3]
    input_shape = [1, 5, 15, 17]  # [batch, input_channels, height, width]
    weights_shape = [5, 1] + kernel_shape  # [output_channels, kernel_channels, kernel_height, kernel_width]
    group = 5

    weights_data = np.arange(math.prod(weights_shape)).reshape(weights_shape).astype(np.int8)
    bias_data = np.arange(weights_shape[0]).astype(np.int32)

    kernel_shape = weights_shape[2:]
    input_scale = [0.01865844801068306]
    input_zero_point = [114]
    output_scale = [0.12412]
    output_zero_point = [0]

    q_linear_conv_node = onnx.helper.make_node(
        "QLinearConv",
        ["input", "input_scale", "input_zero_point", "weights", "weights_scale", "weights_zero_point", "output_scale",
         "output_zero_point", "bias"],
        ["output"],
        dilations=dilations, strides=strides, kernel_shape=kernel_shape, pads=pads, group=group, auto_pad=auto_pad)

    graph = onnx.helper.make_graph(
        nodes=[q_linear_conv_node],
        name="QLinearConv_test",
        inputs=[
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.INT8, input_shape),
            onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.INT8, weights_shape),
        ],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT8, ())],
        initializer=[
            onnx.helper.make_tensor("bias", onnx.TensorProto.INT32, [len(bias_data)], bias_data),
            onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [len(input_scale)], input_scale),
            onnx.helper.make_tensor("input_zero_point", onnx.TensorProto.INT8, [len(input_zero_point)],
                                    input_zero_point),
            onnx.helper.make_tensor("weights_scale", onnx.TensorProto.FLOAT, [len(weights_scale)], weights_scale),
            onnx.helper.make_tensor("weights_zero_point", onnx.TensorProto.INT8, [len(weights_zero_point)],
                                    weights_zero_point),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [len(output_scale)], output_scale),
            onnx.helper.make_tensor("output_zero_point", onnx.TensorProto.INT8, [len(output_zero_point)],
                                    output_zero_point),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(math.prod(input_shape)).reshape(input_shape).astype(np.int8),
        1: weights_data
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    assert intermediate_tflite_model_provider.get_op_count(depthwise_conv_2d_options.DepthwiseConv2D) == 1


@pytest.mark.parametrize("dilations,strides,auto_pad,pads,weights_scale,weights_zero_point", [
    pytest.param([1, 1], [4, 4], None, [0, 0, 0, 0], [0.124], [128], id="per tensor, explicit padding"),
    pytest.param([1, 1], [1, 1], "SAME_LOWER", None, [0.05], [128], id="per tensor"),
])
def test_convert_2d_q_linear_conv_into_depthwise_conv_2d_dynamic_uint8_weights(
        dilations: List[int], strides: List[int], auto_pad: str, pads: List[int],
        weights_scale: List[float], weights_zero_point: List[int], intermediate_tflite_model_provider):
    kernel_shape = [3, 3]
    input_shape = [1, 5, 3, 3]  # [batch, input_channels, height, width]
    weights_shape = [5, 1] + kernel_shape  # [output_channels, kernel_channels, kernel_height, kernel_width]
    group = 5

    bias_data = np.arange(weights_shape[0]).astype(np.int32)

    kernel_shape = weights_shape[2:]
    input_scale = [0.01865844801068306]
    input_zero_point = [114]
    output_scale = [0.12412]
    output_zero_point = [0]

    q_linear_conv_node = onnx.helper.make_node(
        "QLinearConv",
        ["input", "input_scale", "input_zero_point", "weights", "weights_scale", "weights_zero_point", "output_scale",
         "output_zero_point", "bias"],
        ["output"],
        dilations=dilations, strides=strides, kernel_shape=kernel_shape, pads=pads, group=group, auto_pad=auto_pad)

    graph = onnx.helper.make_graph(
        nodes=[q_linear_conv_node],
        name="QLinearConv_test",
        inputs=[
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.UINT8, input_shape),
            onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.UINT8, weights_shape),
        ],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.UINT8, ())],
        initializer=[
            onnx.helper.make_tensor("bias", onnx.TensorProto.INT32, [len(bias_data)], bias_data),
            # onnx.helper.make_tensor("weights", onnx.TensorProto.UINT8, weights_shape, weights_data),
            onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [len(input_scale)], input_scale),
            onnx.helper.make_tensor("input_zero_point", onnx.TensorProto.UINT8, [len(input_zero_point)],
                                    input_zero_point),
            onnx.helper.make_tensor("weights_scale", onnx.TensorProto.FLOAT, [len(weights_scale)], weights_scale),
            onnx.helper.make_tensor("weights_zero_point", onnx.TensorProto.UINT8, [len(weights_zero_point)],
                                    weights_zero_point),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [len(output_scale)], output_scale),
            onnx.helper.make_tensor("output_zero_point", onnx.TensorProto.UINT8, [len(output_zero_point)],
                                    output_zero_point),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.random.randint(0, 20, input_shape).reshape(input_shape).astype(np.uint8),
        1: np.random.randint(0, 20, weights_shape).reshape(weights_shape).astype(np.uint8)
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    assert intermediate_tflite_model_provider.get_op_count(depthwise_conv_2d_options.DepthwiseConv2D) == 1


def test_convert_2d_q_linear_conv_unsupported_nonzero_zero_point():
    weights_scale = [0.1, 0.2, 0.05, 0.4, 0.5]
    weights_zero_point = [1, 1, 1, 1, 1]
    kernel_shape = [3, 3]
    input_shape = [1, 5, 15, 17]  # [batch, input_channels, height, width]
    weights_shape = [5, 1] + kernel_shape  # [output_channels, kernel_channels, kernel_height, kernel_width]

    bias_data = np.arange(weights_shape[0]).astype(np.int32)

    kernel_shape = weights_shape[2:]
    input_scale = [0.01865844801068306]
    input_zero_point = [114]
    output_scale = [0.12412]
    output_zero_point = [0]

    q_linear_conv_node = onnx.helper.make_node(
        "QLinearConv",
        ["input", "input_scale", "input_zero_point", "weights", "weights_scale", "weights_zero_point", "output_scale",
         "output_zero_point", "bias"],
        ["output"], kernel_shape=kernel_shape, )

    graph = onnx.helper.make_graph(
        nodes=[q_linear_conv_node],
        name="QLinearConv_test",
        inputs=[
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.INT8, input_shape),
            onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.INT8, weights_shape),
        ],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT8, ())],
        initializer=[
            onnx.helper.make_tensor("bias", onnx.TensorProto.INT32, [len(bias_data)], bias_data),
            onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [len(input_scale)], input_scale),
            onnx.helper.make_tensor("input_zero_point", onnx.TensorProto.INT8, [len(input_zero_point)],
                                    input_zero_point),
            onnx.helper.make_tensor("weights_scale", onnx.TensorProto.FLOAT, [len(weights_scale)], weights_scale),
            onnx.helper.make_tensor("weights_zero_point", onnx.TensorProto.INT8, [len(weights_zero_point)],
                                    weights_zero_point),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [len(output_scale)], output_scale),
            onnx.helper.make_tensor("output_zero_point", onnx.TensorProto.INT8, [len(output_zero_point)],
                                    output_zero_point),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("strides,auto_pad,pads,weights_scale,weights_zero_point", [
    pytest.param([4, 4], None, [0, 0, 0, 0], [0.124], [0], id="per tensor-explicit zero padding"),
    pytest.param([4, 4], None, [0, 1, 1, 0], [0.124], [0], id="per tensor-explicit non-zero padding"),
    pytest.param([1, 1], "SAME_LOWER", None, [0.05], [0], id="per tensor-SAME_LOWER padding"),
])
@pytest.mark.parametrize("group", [2, 3, 6], ids=lambda x: f"group={x}")
def test_convert_2d_q_linear_conv_group__into_multiple_convolutions__with_static_weights(
        strides: List[int], auto_pad: str, pads: List[int], weights_scale: List[float], weights_zero_point: List[int],
        group: int, intermediate_tflite_model_provider):
    kernel_shape = [3, 3]
    input_shape = [1, 24, 15, 17]  # [batch, input_channels, height, width]
    weights_shape = [6, 24 // group] + kernel_shape  # [output_channels, kernel_channels, kernel_height, kernel_width]
    bias_shape = [weights_shape[0]]

    weights_data = np.arange(math.prod(weights_shape)).reshape(weights_shape).astype(np.int8)
    bias_data = np.arange(bias_shape[0]).astype(np.int32)

    kernel_shape = weights_shape[2:]
    input_scale = [0.01865844801068306]
    input_zero_point = [114]
    output_scale = [0.12412]
    output_zero_point = [0]

    q_linear_conv_node = onnx.helper.make_node(
        "QLinearConv",
        ["input", "input_scale", "input_zero_point", "weights", "weights_scale", "weights_zero_point", "output_scale",
         "output_zero_point", "bias"],
        ["output"],
        strides=strides, kernel_shape=kernel_shape, pads=pads, group=group, auto_pad=auto_pad)

    graph = onnx.helper.make_graph(
        nodes=[q_linear_conv_node],
        name="QLinearConv_test",
        inputs=[
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.INT8, input_shape),
            onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.INT8, weights_shape),
            onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.INT32, bias_shape),
        ],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT8, ())],
        initializer=[
            onnx.helper.make_tensor("weights", onnx.TensorProto.INT8, weights_shape, weights_data),
            onnx.helper.make_tensor("bias", onnx.TensorProto.INT32, [len(bias_data)], bias_data),
            onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [len(input_scale)], input_scale),
            onnx.helper.make_tensor("input_zero_point", onnx.TensorProto.INT8, [len(input_zero_point)],
                                    input_zero_point),
            onnx.helper.make_tensor("weights_scale", onnx.TensorProto.FLOAT, [len(weights_scale)], weights_scale),
            onnx.helper.make_tensor("weights_zero_point", onnx.TensorProto.INT8, [len(weights_zero_point)],
                                    weights_zero_point),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [len(output_scale)], output_scale),
            onnx.helper.make_tensor("output_zero_point", onnx.TensorProto.INT8, [len(output_zero_point)],
                                    output_zero_point),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(np.int8)

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    assert intermediate_tflite_model_provider.get_op_count(conv_2d_options.Conv2D) == group
    assert intermediate_tflite_model_provider.get_op_count(split_options.Split) == 1  # input
    assert intermediate_tflite_model_provider.get_op_count(concatenation_options.Concatenation) == 1


@pytest.mark.parametrize("strides,auto_pad,pads,weights_scale,weights_zero_point", [
    pytest.param([4, 4], None, [0, 0, 0, 0], [0.124], [0], id="per tensor-explicit zero padding"),
    pytest.param([4, 4], None, [0, 1, 1, 0], [0.124], [0], id="per tensor-explicit non-zero padding"),
    pytest.param([1, 1], "SAME_LOWER", None, [0.05], [0], id="per tensor-SAME_LOWER padding"),
])
@pytest.mark.parametrize("group", [2, 3, 6], ids=lambda x: f"group={x}")
def test_convert_2d_q_linear_conv_group__into_multiple_convolutions__with_dynamic_weights(
        strides: List[int], auto_pad: str, pads: List[int], weights_scale: List[float], weights_zero_point: List[int],
        group: int, intermediate_tflite_model_provider):
    kernel_shape = [3, 3]
    input_shape = [1, 24, 15, 17]  # [batch, input_channels, height, width]
    weights_shape = [6, 24 // group] + kernel_shape  # [output_channels, kernel_channels, kernel_height, kernel_width]
    bias_shape = [weights_shape[0]]

    kernel_shape = weights_shape[2:]
    input_scale = [0.01865844801068306]
    input_zero_point = [114]
    output_scale = [0.12412]
    output_zero_point = [0]

    q_linear_conv_node = onnx.helper.make_node(
        "QLinearConv",
        ["input", "input_scale", "input_zero_point", "weights", "weights_scale", "weights_zero_point", "output_scale",
         "output_zero_point", "bias"],
        ["output"],
        strides=strides, kernel_shape=kernel_shape, pads=pads, group=group, auto_pad=auto_pad)

    graph = onnx.helper.make_graph(
        nodes=[q_linear_conv_node],
        name="QLinearConv_test",
        inputs=[
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.INT8, input_shape),
            onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.INT8, weights_shape),
            onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.INT32, bias_shape),
        ],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT8, ())],
        initializer=[
            onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [len(input_scale)], input_scale),
            onnx.helper.make_tensor("input_zero_point", onnx.TensorProto.INT8, [len(input_zero_point)],
                                    input_zero_point),
            onnx.helper.make_tensor("weights_scale", onnx.TensorProto.FLOAT, [len(weights_scale)], weights_scale),
            onnx.helper.make_tensor("weights_zero_point", onnx.TensorProto.INT8, [len(weights_zero_point)],
                                    weights_zero_point),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [len(output_scale)], output_scale),
            onnx.helper.make_tensor("output_zero_point", onnx.TensorProto.INT8, [len(output_zero_point)],
                                    output_zero_point),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(math.prod(input_shape)).reshape(input_shape).astype(np.int8),
        1: np.arange(math.prod(weights_shape)).reshape(weights_shape).astype(np.int8),
        2: np.arange(bias_shape[0]).astype(np.int32),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    assert intermediate_tflite_model_provider.get_op_count(conv_2d_options.Conv2D) == group
    assert intermediate_tflite_model_provider.get_op_count(split_options.Split) == 3  # input, weight, bias
    assert intermediate_tflite_model_provider.get_op_count(concatenation_options.Concatenation) == 1


@pytest.mark.parametrize("strides,auto_pad,pads,group,weights_scale,weights_zero_point", [
    pytest.param([4, 4], None, [0, 0, 0, 0], 12, [0.124], [0], id="group=12")
])
def test_convert_2d_q_linear_conv_group__nonconvertible_into_multiple_convolutions(
        strides: List[int], auto_pad: str, pads: List[int], group: int, weights_scale: List[float],
        weights_zero_point: List[int], intermediate_tflite_model_provider):
    kernel_shape = [3, 3]
    input_shape = [1, 24, 15, 17]  # [batch, input_channels, height, width]
    weights_shape = [36, 24 // group] + kernel_shape  # [output_channels, kernel_channels, kernel_height, kernel_width]
    bias_shape = [weights_shape[0]]

    kernel_shape = weights_shape[2:]
    input_scale = [0.01865844801068306]
    input_zero_point = [114]
    output_scale = [0.12412]
    output_zero_point = [0]

    q_linear_conv_node = onnx.helper.make_node(
        "QLinearConv",
        ["input", "input_scale", "input_zero_point", "weights", "weights_scale", "weights_zero_point", "output_scale",
         "output_zero_point", "bias"],
        ["output"],
        strides=strides, kernel_shape=kernel_shape, pads=pads, group=group, auto_pad=auto_pad)

    graph = onnx.helper.make_graph(
        nodes=[q_linear_conv_node],
        name="QLinearConv_test",
        inputs=[
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.INT8, input_shape),
            onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.INT8, weights_shape),
            onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.INT32, bias_shape),
        ],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT8, ())],
        initializer=[
            onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [len(input_scale)], input_scale),
            onnx.helper.make_tensor("input_zero_point", onnx.TensorProto.INT8, [len(input_zero_point)],
                                    input_zero_point),
            onnx.helper.make_tensor("weights_scale", onnx.TensorProto.FLOAT, [len(weights_scale)], weights_scale),
            onnx.helper.make_tensor("weights_zero_point", onnx.TensorProto.INT8, [len(weights_zero_point)],
                                    weights_zero_point),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [len(output_scale)], output_scale),
            onnx.helper.make_tensor("output_zero_point", onnx.TensorProto.INT8, [len(output_zero_point)],
                                    output_zero_point),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(math.prod(input_shape)).reshape(input_shape).astype(np.int8),
        1: np.arange(math.prod(weights_shape)).reshape(weights_shape).astype(np.int8),
        2: np.arange(bias_shape[0]).astype(np.int32),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    assert intermediate_tflite_model_provider.get_op_count(conv_2d_options.Conv2D) == 1
    assert intermediate_tflite_model_provider.get_op_count(split_options.Split) == 0
    assert intermediate_tflite_model_provider.get_op_count(concatenation_options.Concatenation) == 0


@pytest.mark.parametrize("strides,auto_pad,pads,group,weights_scale,weights_zero_point", [
    pytest.param([4, 4], None, [0, 1, 1, 0], 2, [0.124] * 36, [0] * 36, id="per channel weight quantization")
])
def test_convert_2d_q_linear_conv_group__convertible_into_multiple_convolutions(
        strides: List[int], auto_pad: str, pads: List[int], group: int, weights_scale: List[float],
        weights_zero_point: List[int], intermediate_tflite_model_provider):
    kernel_shape = [3, 3]
    input_shape = [1, 24, 15, 17]  # [batch, input_channels, height, width]
    weights_shape = [36, 24 // group] + kernel_shape  # [output_channels, kernel_channels, kernel_height, kernel_width]
    bias_shape = [weights_shape[0]]

    kernel_shape = weights_shape[2:]
    input_scale = [0.01865844801068306]
    input_zero_point = [114]
    output_scale = [0.12412]
    output_zero_point = [0]

    q_linear_conv_node = onnx.helper.make_node(
        "QLinearConv",
        ["input", "input_scale", "input_zero_point", "weights", "weights_scale", "weights_zero_point", "output_scale",
         "output_zero_point", "bias"],
        ["output"],
        strides=strides, kernel_shape=kernel_shape, pads=pads, group=group, auto_pad=auto_pad)

    graph = onnx.helper.make_graph(
        nodes=[q_linear_conv_node],
        name="QLinearConv_test",
        inputs=[
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.INT8, input_shape),
            onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.INT8, weights_shape),
            onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.INT32, bias_shape),
        ],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT8, ())],
        initializer=[
            onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [len(input_scale)], input_scale),
            onnx.helper.make_tensor("input_zero_point", onnx.TensorProto.INT8, [len(input_zero_point)],
                                    input_zero_point),
            onnx.helper.make_tensor("weights_scale", onnx.TensorProto.FLOAT, [len(weights_scale)], weights_scale),
            onnx.helper.make_tensor("weights_zero_point", onnx.TensorProto.INT8, [len(weights_zero_point)],
                                    weights_zero_point),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [len(output_scale)], output_scale),
            onnx.helper.make_tensor("output_zero_point", onnx.TensorProto.INT8, [len(output_zero_point)],
                                    output_zero_point),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(math.prod(input_shape)).reshape(input_shape).astype(np.int8),
        1: np.arange(math.prod(weights_shape)).reshape(weights_shape).astype(np.int8),
        2: np.arange(bias_shape[0]).astype(np.int32),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)

    assert intermediate_tflite_model_provider.get_op_count(conv_2d_options.Conv2D) == 2
    assert intermediate_tflite_model_provider.get_op_count(split_options.Split) == 3
    assert intermediate_tflite_model_provider.get_op_count(concatenation_options.Concatenation) == 1


@pytest.mark.parametrize(
    "input_shape,weights_shape,dilations,strides,auto_pad,pads,group,weights_scale,weights_zero_point", [
        pytest.param([1, 3, 224, 224], [96, 3, 11, 11], [1, 1], [4, 4], None, [0, 0, 0, 0], 1, [0.124],
                     [0], id="per tensor, explicit padding"),
        pytest.param([1, 384, 12, 12], [256, 192, 3, 3], [1, 1], [1, 1], "SAME_LOWER", None, 2,
                     [0.05], [0], id="per tensor"),
        pytest.param([1, 384, 12, 12], [256, 96, 3, 3], [1, 1], [1, 1], "SAME_LOWER", None, 4,
                     [0.05], [0], id="per tensor"),
        pytest.param([1, 256, 10, 10], [3, 256, 3, 3], [5, 5], [1, 1], None, [5, 5, 5, 5], 1,
                     [0.1, 0.2, 0.05], [0, 0, 0], id="per channel"),
    ])
def test_convert_2d_q_linear_conv__default_bias(input_shape: List[int], weights_shape: List[int],
                                                dilations: List[int], strides: List[int],
                                                auto_pad: str, pads: List[int], group: int,
                                                weights_scale: List[float], weights_zero_point: List[int]):
    # Bias has name ''.

    weights_data = np.arange(math.prod(weights_shape)).reshape(weights_shape).astype(np.int8)
    bias_data = np.arange(weights_shape[0]).astype(np.int32)

    kernel_shape = weights_shape[2:]
    input_scale = [0.01865844801068306]
    input_zero_point = [114]
    output_scale = [0.12412]
    output_zero_point = [0]

    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("QLinearConv",
                                      ["input", "input_scale", "input_zero_point", "weights", "weights_scale",
                                       "weights_zero_point", "output_scale", "output_zero_point", ''], ["output"],
                                      dilations=dilations, strides=strides, kernel_shape=kernel_shape, pads=pads,
                                      group=group, auto_pad=auto_pad)
            ],
            name="QLinearConv_test",
            inputs=[onnx.helper.make_tensor_value_info("input", TensorProto.INT8, input_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
            initializer=[
                onnx.helper.make_tensor("weights", TensorProto.INT8, weights_shape, weights_data),
                onnx.helper.make_tensor("bias", onnx.TensorProto.INT32, [len(bias_data)], bias_data),
                onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [len(input_scale)], input_scale),
                onnx.helper.make_tensor("input_zero_point", TensorProto.INT8, [len(input_zero_point)],
                                        input_zero_point),
                onnx.helper.make_tensor("weights_scale", onnx.TensorProto.FLOAT, [len(weights_scale)], weights_scale),
                onnx.helper.make_tensor("weights_zero_point", TensorProto.INT8, [len(weights_zero_point)],
                                        weights_zero_point),
                onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [len(output_scale)], output_scale),
                onnx.helper.make_tensor("output_zero_point", TensorProto.INT8, [len(output_zero_point)],
                                        output_zero_point),

            ]
        )
    )
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(np.int8)

    executors.convert_run_compare(onnx_model, input_data, atol=1)
