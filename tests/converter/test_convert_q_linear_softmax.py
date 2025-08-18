#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto

import onnx2tflite.src.logger as logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type
from tests import executors


@pytest.mark.parametrize("input_shape,axis", [
    pytest.param((10,), 0, id="1D,axis=0"),
    pytest.param((10,), -1, id="1D,axis=-1"),
    pytest.param((2, 3), -1, id="2D,axis=-1"),
    pytest.param((2, 3), 0, id="2D,axis=0"),
    pytest.param((2, 3), 1, id="2D,axis=1"),
    pytest.param((2, 3, 4), 0, id="3D,axis=0"),
    pytest.param((2, 3, 4), 1, id="3D,axis=1"),
    pytest.param((2, 3, 4), 2, id="3D,axis=2"),
    pytest.param((2, 3, 4, 5), -4, id="4D,axis=-4"),
    pytest.param((2, 3, 4, 5), -1, id="4D,axis=-1"),
    pytest.param((2, 3, 4, 5), 0, id="4D,axis=0"),
    pytest.param((2, 3, 4, 5), 1, id="4D,axis=1"),
    pytest.param((2, 3, 4, 5), 2, id="4D,axis=2"),
    pytest.param((2, 3, 4, 5), 3, id="4D,axis=3"),
    pytest.param((2, 3, 4, 5, 6), -2, id="5D,axis=-2"),
    pytest.param((2, 3, 4, 5, 6), 0, id="5D,axis=0"),
    pytest.param((2, 3, 4, 5, 6), 3, id="5D,axis=3"),
])
@pytest.mark.parametrize("opset", [11, 13], ids=(lambda x: f"opset={x}"))
@pytest.mark.parametrize("io_type", [TensorProto.INT8, TensorProto.UINT8], ids=(lambda x: f"types={x}"))
def test_convert_q_linear_softmax(input_shape, axis, opset, io_type):
    node = onnx.helper.make_node("QLinearSoftmax",
                                 ["x", "x_scale", "x_FS_ZP", "y_scale", "y_FS_ZP"],
                                 ["y"],
                                 axis=axis, opset=opset, domain="com.microsoft")

    # Limitations for i8 types 
    #   scale, zero_point: 1.0/256.0, -128
    # Limitations for u8
    #   scale, zero_point: 1.0/256, 0
    FS_ZP_on_type = -128 if (io_type == TensorProto.INT8) else 0
    graph = onnx.helper.make_graph(
        [node],
        "graph-qlinear-softmax",
        inputs=[onnx.helper.make_tensor_value_info("x", io_type, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info("y", io_type, ())],
        initializer=[
            onnx.helper.make_tensor("x_scale", onnx.TensorProto.FLOAT, [], [0.003906]),
            onnx.helper.make_tensor("x_FS_ZP", io_type, [], [FS_ZP_on_type]),
            onnx.helper.make_tensor("y_scale", onnx.TensorProto.FLOAT, [], [0.003906]),
            onnx.helper.make_tensor("y_FS_ZP", io_type, [], [FS_ZP_on_type]),
        ]
    )

    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-qlinear-softmax",
        opset_imports=[onnx.helper.make_opsetid("", opset)])

    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add the opset with QLinearSoftmax
    onnx.checker.check_model(model)
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(io_type))
    executors.convert_run_compare(model, input_data, atol=1)


@pytest.mark.parametrize("axis", list(range(-4, 4)), ids=(lambda x: f"axis={x}"))
@pytest.mark.parametrize("io_type", [TensorProto.INT8, TensorProto.UINT8], ids=(lambda x: f"types={x}"))
@pytest.mark.parametrize("opset", [11, 13], ids=(lambda x: f"opset={x}"))
def test_convert_q_linear_softmax_after_conv(axis: int, io_type, opset: int):
    FS_SCALE = 1.0 / 256.0
    FS_ZP = -128 if (io_type == TensorProto.INT8) else 0

    q_shape = [2, 3, 92, 92]
    w_shape = [8, 3, 11, 11]
    x_shape = [2, 8, 54, 54]
    dilations = [1, 1]
    strides = [4, 4]
    auto_pad = None
    pads = [0, 0, 0, 0]
    group = 1
    w_scale = [FS_SCALE]
    w_FS_ZP = [0 if (io_type == TensorProto.INT8) else 128]
    kernel_shape = w_shape[2:]
    q_scale = [FS_SCALE]
    q_FS_ZP = [0]
    x_scale = [FS_SCALE]
    x_FS_ZP = [FS_ZP]
    w_data = np.arange(math.prod(w_shape)).reshape(w_shape).astype(to_numpy_type(io_type))
    b_data = np.arange(w_shape[0]).astype(np.int32)
    x_sm_scale = [FS_SCALE]
    x_sm_FS_ZP = [FS_ZP]
    y_scale = [FS_SCALE]
    y_FS_ZP = [FS_ZP]

    node_ql_conv = onnx.helper.make_node(
        "QLinearConv",
        ["q", "q_scale", "q_FS_ZP", "W", "w_scale", "w_FS_ZP", "x_scale", "x_FS_ZP", "bias"],
        ["x"],
        dilations=dilations, strides=strides, kernel_shape=kernel_shape,
        pads=pads, group=group, auto_pad=auto_pad
    )

    node_ql_softmax = onnx.helper.make_node(
        "QLinearSoftmax",
        ["x", "x_sm_scale", "x_sm_FS_ZP", "y_scale", "y_FS_ZP"],
        ["y"],
        axis=axis, opset=opset, domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node_ql_conv, node_ql_softmax],
        'ql_conv+ql_softmax',
        [onnx.helper.make_tensor_value_info("q", io_type, q_shape)],
        [onnx.helper.make_tensor_value_info("y", io_type, ())],
        initializer=[
            onnx.helper.make_tensor("W", io_type, w_shape, w_data),
            onnx.helper.make_tensor("bias", TensorProto.INT32, [len(b_data)], b_data),
            onnx.helper.make_tensor("q_scale", TensorProto.FLOAT, [len(q_scale)], q_scale),
            onnx.helper.make_tensor("q_FS_ZP", io_type, [len(q_FS_ZP)], q_FS_ZP),
            onnx.helper.make_tensor("w_scale", TensorProto.FLOAT, [len(w_scale)], w_scale),
            onnx.helper.make_tensor("w_FS_ZP", io_type, [len(w_FS_ZP)], w_FS_ZP),
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [len(x_scale)], x_scale),
            onnx.helper.make_tensor("x_FS_ZP", io_type, [len(x_FS_ZP)], x_FS_ZP),
            onnx.helper.make_tensor("x_sm_scale", TensorProto.FLOAT, [len(x_sm_scale)], x_sm_scale),
            onnx.helper.make_tensor("x_sm_FS_ZP", io_type, [len(x_sm_FS_ZP)], x_sm_FS_ZP),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [len(y_scale)], y_scale),
            onnx.helper.make_tensor("y_FS_ZP", io_type, [len(y_FS_ZP)], y_FS_ZP),
        ]
    )

    original_model = onnx.helper.make_model(
        graph,
        producer_name="onnx-ql_conv-ql_softmax",
        opset_imports=[onnx.helper.make_opsetid("", opset)]
    )

    original_model.opset_import.append(
        onnx.helper.make_opsetid("com.microsoft", 1))  # Add the opset with QLinearSoftmax
    onnx.checker.check_model(original_model)
    input_data = np.arange(math.prod(q_shape)).reshape(q_shape).astype(to_numpy_type(io_type))
    executors.convert_run_compare(original_model, input_data, atol=1)


@pytest.mark.parametrize("opset", [11, 13], ids=(lambda x: f"opset={x}"))
def test_convert_q_linear_softmax_over_dim_val_1(opset: int):
    FS_SCALE = 1.0 / 256.0
    FS_ZP = -128
    input_shape = (1, 3, 4, 1)
    axis = 0

    node = onnx.helper.make_node(
        "QLinearSoftmax",
        ["x", "x_scale", "x_zero_point", "y_scale", "y_zero_point"],
        ["y"],
        axis=axis, opset=opset, domain="com.microsoft")

    graph = onnx.helper.make_graph(
        [node],
        "graph-qlinear-softmax",
        inputs=[onnx.helper.make_tensor_value_info("x", TensorProto.INT8, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info("y", TensorProto.INT8, ())],
        initializer=[
            onnx.helper.make_tensor("x_scale", onnx.TensorProto.FLOAT, [], [FS_SCALE]),
            onnx.helper.make_tensor("x_zero_point", TensorProto.INT8, [], [FS_ZP]),
            onnx.helper.make_tensor("y_scale", onnx.TensorProto.FLOAT, [], [FS_SCALE]),
            onnx.helper.make_tensor("y_zero_point", TensorProto.INT8, [], [FS_ZP]),
        ]
    )

    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-qlinear-softmax",
        opset_imports=[onnx.helper.make_opsetid("", opset)])

    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add the opset with QLinearSoftmax
    onnx.checker.check_model(model)

    with pytest.raises(logger.Error):
        convert.convert_model(model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE


@pytest.mark.parametrize("opset", [11, 13], ids=(lambda x: f"opset={x}"))
@pytest.mark.parametrize("scale", [1.2 / 256.0, 0.9 / 256.0], ids=(lambda x: f"scale={x}"))
@pytest.mark.parametrize("zero_point", [64, 5], ids=(lambda x: f"zero_point={x}"))
@pytest.mark.parametrize("io_type", [TensorProto.INT8, TensorProto.UINT8], ids=(lambda x: f"types={x}"))
def test_convert_q_linear_softmax__invalid_output_q_params(opset: int, scale: float, zero_point: int, io_type):
    input_shape = (2, 3, 3, 16)

    node = onnx.helper.make_node(
        "QLinearSoftmax",
        ["x", "x_scale", "x_zero_point", "y_scale", "y_zero_point"],
        ["y"],
        axis=-1, opset=opset, domain="com.microsoft")

    graph = onnx.helper.make_graph(
        [node],
        "graph-qlinear-softmax",
        inputs=[onnx.helper.make_tensor_value_info("x", io_type, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info("y", io_type, ())],
        initializer=[
            onnx.helper.make_tensor("x_scale", onnx.TensorProto.FLOAT, [], [scale]),
            onnx.helper.make_tensor("x_zero_point", io_type, [], [zero_point]),
            onnx.helper.make_tensor("y_scale", onnx.TensorProto.FLOAT, [], [scale]),
            onnx.helper.make_tensor("y_zero_point", io_type, [], [zero_point]),
        ]
    )

    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-qlinear-softmax",
        opset_imports=[onnx.helper.make_opsetid("", opset)])

    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))  # Add the opset with QLinearSoftmax
    onnx.checker.check_model(model)

    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(io_type))
    executors.convert_run_compare(model, input_data, atol=1)


@pytest.mark.parametrize('io_type', [TensorProto.INT8, TensorProto.UINT8], ids=(lambda x: f'types={x}'))
def test_convert_q_linear_softmax__default_zero_point(io_type):
    # Represent the optional zero point using name ''.

    input_shape = [10, 15]
    opset = 13

    node = onnx.helper.make_node('QLinearSoftmax',
                                 ['x', 'x_scale', '', 'y_scale', 'y_FS_ZP'],
                                 ['y'], opset=opset, domain='com.microsoft')

    # Limitations for i8 types 
    #   scale, zero_point: 1.0/256.0, -128
    # Limitations for u8
    #   scale, zero_point: 1.0/256, 0
    fs_zp_on_type = -128 if (io_type == TensorProto.INT8) else 0
    graph = onnx.helper.make_graph(
        [node],
        'graph-qlinear-softmax',
        inputs=[onnx.helper.make_tensor_value_info('x', io_type, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info('y', io_type, ())],
        initializer=[
            onnx.helper.make_tensor('x_scale', onnx.TensorProto.FLOAT, [], [0.003906]),
            onnx.helper.make_tensor('y_scale', onnx.TensorProto.FLOAT, [], [0.003906]),
            onnx.helper.make_tensor('y_FS_ZP', io_type, [], [fs_zp_on_type]),
        ]
    )

    model = onnx.helper.make_model(
        graph,
        producer_name='onnx-qlinear-softmax',
        opset_imports=[onnx.helper.make_opsetid('', opset)])

    model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))  # Add the opset with QLinearSoftmax
    onnx.checker.check_model(model)
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(io_type))
    executors.convert_run_compare(model, input_data, atol=1)
