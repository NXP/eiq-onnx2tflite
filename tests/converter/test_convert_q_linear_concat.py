#
# Copyright 2023-2024 NXP
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

from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape, output_scale, output_zero_point, input_1_scale,"
    "input_1_zero_point, input_2_scale, input_2_zero_point, axis, data_type",
    [
        pytest.param([1, 24, 56, 5], [1, 24, 56, 5], [0.01], [1], [0.01], [10], [0.05], [10], -1, TensorProto.UINT8,
                     id="4D,UINT8,axis=-1"),
        pytest.param([1, 24, 56, 5], [1, 24, 56, 5], [0.01], [10], [0.01], [10], [0.01], [10], 0, TensorProto.INT8,
                     id="4D,INT8,same quantization params,axis=0"),
        pytest.param([1, 24, 56, 5], [1, 24, 56, 5], [0.01], [1], [0.01], [1], [0.01], [10], 1, TensorProto.INT8,
                     id="4D,INT8,different quantization params,axis=1"),
        pytest.param([1, 24, 56, 2], [1, 24, 56, 1], [0.01], [1], [0.01], [1], [0.01], [10], -1, TensorProto.INT8,
                     id="4D,INT8,different quantization params,axis=-1"),
        pytest.param([1, 24, 2], [1, 24, 1], [0.01], [1], [0.01], [1], [0.01], [10], 2, TensorProto.INT8,
                     id="3D,INT8,different quantization params,axis=-1"),
    ])
def test_convert_q_linear_concat(
        input_1_shape: List[int], input_2_shape: List[int],
        output_scale, output_zero_point,
        input_1_scale, input_1_zero_point,
        input_2_scale, input_2_zero_point,
        axis,
        data_type: TensorProto.DataType,
):
    q_linear_concat_node = onnx.helper.make_node(
        op_type="QLinearConcat",
        inputs=["output_scale", "output_zero_point",
                "input_1", "input_1_scale", "input_1_zero_point",
                "input_2", "input_2_scale", "input_2_zero_point"],
        outputs=["output"],
        domain="com.microsoft",
        axis=axis)

    graph = onnx.helper.make_graph(
        nodes=[q_linear_concat_node],
        name="QLinearConcat model",
        inputs=[onnx.helper.make_tensor_value_info("input_1", data_type, input_1_shape),
                onnx.helper.make_tensor_value_info("input_2", data_type, input_2_shape)],
        outputs=[onnx.helper.make_tensor_value_info("output", data_type, ())],
        initializer=[
            onnx.helper.make_tensor("input_1_scale", onnx.TensorProto.FLOAT, [], input_1_scale),
            onnx.helper.make_tensor("input_1_zero_point", data_type, [], input_1_zero_point),
            onnx.helper.make_tensor("input_2_scale", onnx.TensorProto.FLOAT, [], input_2_scale),
            onnx.helper.make_tensor("input_2_zero_point", data_type, [], input_2_zero_point),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [], output_scale),
            onnx.helper.make_tensor("output_zero_point", data_type, [], output_zero_point), ])

    onnx_model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid(domain="com.microsoft", version=1),
            onnx.helper.make_opsetid(domain="", version=8)
        ])

    input_data = {
        0: np.arange(math.prod(input_1_shape)).reshape(input_1_shape).astype(to_numpy_type(data_type)),
        1: np.arange(math.prod(input_2_shape)).reshape(input_2_shape).astype(to_numpy_type(data_type)),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape, output_scale, output_zero_point, input_1_scale,"
    "input_1_zero_point, input_2_scale, input_2_zero_point, data_type",
    [
        pytest.param([1, 24, 56, 2], [1, 24, 56, 2], [0.01], [1], [0.01], [10], [0.01], [10],
                     TensorProto.INT8),
    ])
def test_convert_q_linear_concat_static_input(
        input_1_shape: List[int], input_2_shape: List[int],
        output_scale, output_zero_point,
        input_1_scale, input_1_zero_point,
        input_2_scale, input_2_zero_point,
        data_type: TensorProto.DataType
):
    input_1 = np.arange(math.prod(input_1_shape)).reshape(input_1_shape).astype(to_numpy_type(data_type))

    q_linear_concat_node = onnx.helper.make_node(
        op_type="QLinearConcat",
        inputs=["output_scale", "output_zero_point",
                "input_1", "input_1_scale", "input_1_zero_point",
                "input_2", "input_2_scale", "input_2_zero_point"],
        outputs=["output"],
        domain="com.microsoft",
        axis=-1)

    graph = onnx.helper.make_graph(
        nodes=[q_linear_concat_node],
        name="QLinearConcat model",
        inputs=[onnx.helper.make_tensor_value_info("input_2", data_type, input_2_shape)],
        outputs=[onnx.helper.make_tensor_value_info("output", data_type, ())],
        initializer=[
            onnx.helper.make_tensor("input_1", data_type, input_1_shape, input_1),
            onnx.helper.make_tensor("input_1_scale", onnx.TensorProto.FLOAT, [], input_1_scale),
            onnx.helper.make_tensor("input_1_zero_point", data_type, [], input_1_zero_point),
            onnx.helper.make_tensor("input_2_scale", onnx.TensorProto.FLOAT, [], input_2_scale),
            onnx.helper.make_tensor("input_2_zero_point", data_type, [], input_2_zero_point),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [], output_scale),
            onnx.helper.make_tensor("output_zero_point", data_type, [], output_zero_point),
        ])
    onnx_model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid(domain="com.microsoft", version=1),
            onnx.helper.make_opsetid(domain="", version=8)
        ])

    input_data = np.arange(math.prod(input_2_shape)).reshape(input_2_shape).astype(to_numpy_type(data_type))

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize(
    "input_1_shape, input_2_shape, output_scale, output_zero_point, input_1_scale,"
    "input_1_zero_point, input_2_scale, input_2_zero_point, axis, data_type",
    [
        pytest.param([1, 24, 56, 5], [1, 24, 56, 5], [0.01], [1], [0.01], [10], [0.05], [10], -1, TensorProto.UINT8,
                     id="4D,UINT8,axis=-1"),
        pytest.param([1, 24, 4, 3], [1, 24, 2, 3], [0.01], [1], [0.01], [10], [0.05], [10], 2, TensorProto.INT8,
                     id="4D,INT8,axis=2"),
    ])
def test_convert_q_linear_concat_channel_first(
        input_1_shape: List[int], input_2_shape: List[int],
        output_scale, output_zero_point,
        input_1_scale, input_1_zero_point,
        input_2_scale, input_2_zero_point,
        axis,
        data_type: TensorProto.DataType,
):
    q_linear_concat_node = onnx.helper.make_node(
        op_type="QLinearConcat",
        inputs=["output_scale", "output_zero_point",
                "input_1", "input_1_scale", "input_1_zero_point",
                "input_2", "input_2_scale", "input_2_zero_point"],
        outputs=["concat_output"],
        domain="com.microsoft",
        axis=axis)

    q_linear_global_average_pool_node = onnx.helper.make_node(
        "QLinearGlobalAveragePool",
        ["concat_output", "output_scale", "output_zero_point", "output_scale", "output_zero_point"],
        ["avg_pool_output"],
        domain="com.microsoft")

    graph = onnx.helper.make_graph(
        nodes=[q_linear_concat_node, q_linear_global_average_pool_node],
        name="QLinearConcat model",
        inputs=[onnx.helper.make_tensor_value_info("input_1", data_type, input_1_shape),
                onnx.helper.make_tensor_value_info("input_2", data_type, input_2_shape)],
        outputs=[onnx.helper.make_tensor_value_info("avg_pool_output", data_type, ())],
        initializer=[
            onnx.helper.make_tensor("input_1_scale", onnx.TensorProto.FLOAT, [], input_1_scale),
            onnx.helper.make_tensor("input_1_zero_point", data_type, [], input_1_zero_point),
            onnx.helper.make_tensor("input_2_scale", onnx.TensorProto.FLOAT, [], input_2_scale),
            onnx.helper.make_tensor("input_2_zero_point", data_type, [], input_2_zero_point),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [], output_scale),
            onnx.helper.make_tensor("output_zero_point", data_type, [], output_zero_point),
        ])

    onnx_model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid(domain="com.microsoft", version=1),
            onnx.helper.make_opsetid(domain="", version=8)
        ])

    input_data = {
        0: np.arange(math.prod(input_1_shape)).reshape(input_1_shape).astype(to_numpy_type(data_type)),
        1: np.arange(math.prod(input_2_shape)).reshape(input_2_shape).astype(to_numpy_type(data_type)),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)


@pytest.mark.parametrize("data_type", [TensorProto.INT8, TensorProto.UINT8])
def test_convert_q_linear_concat_3_inputs(data_type):
    input_1_shape = [1, 3, 56, 5]
    input_2_shape = [1, 2, 56, 5]
    input_3_shape = [1, 1, 56, 5]

    input_1_scale, input_1_zero_point = [0.01], [1]
    input_2_scale, input_2_zero_point = [0.01], [10]
    input_3_scale, input_3_zero_point = [0.05], [10]
    output_scale, output_zero_point = [0.05], [10]

    axis = 1

    q_linear_concat_node = onnx.helper.make_node(
        op_type="QLinearConcat",
        inputs=["output_scale", "output_zero_point",
                "input_1", "input_1_scale", "input_1_zero_point",
                "input_2", "input_2_scale", "input_2_zero_point",
                "input_3", "input_3_scale", "input_3_zero_point"],
        outputs=["output"],
        domain="com.microsoft",
        axis=axis)

    graph = onnx.helper.make_graph(
        nodes=[q_linear_concat_node],
        name="QLinearConcat model",
        inputs=[onnx.helper.make_tensor_value_info("input_1", data_type, input_1_shape),
                onnx.helper.make_tensor_value_info("input_2", data_type, input_2_shape),
                onnx.helper.make_tensor_value_info("input_3", data_type, input_3_shape)],
        outputs=[onnx.helper.make_tensor_value_info("output", data_type, ())],
        initializer=[
            onnx.helper.make_tensor("input_1_scale", onnx.TensorProto.FLOAT, [], input_1_scale),
            onnx.helper.make_tensor("input_1_zero_point", data_type, [], input_1_zero_point),
            onnx.helper.make_tensor("input_2_scale", onnx.TensorProto.FLOAT, [], input_2_scale),
            onnx.helper.make_tensor("input_2_zero_point", data_type, [], input_2_zero_point),
            onnx.helper.make_tensor("input_3_scale", onnx.TensorProto.FLOAT, [], input_3_scale),
            onnx.helper.make_tensor("input_3_zero_point", data_type, [], input_3_zero_point),
            onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [], output_scale),
            onnx.helper.make_tensor("output_zero_point", data_type, [], output_zero_point), ])

    onnx_model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid(domain="com.microsoft", version=1),
            onnx.helper.make_opsetid(domain="", version=8)
        ])

    input_data = {
        0: np.arange(math.prod(input_1_shape)).reshape(input_1_shape).astype(to_numpy_type(data_type)),
        1: np.arange(math.prod(input_2_shape)).reshape(input_2_shape).astype(to_numpy_type(data_type)),
        2: np.arange(math.prod(input_3_shape)).reshape(input_3_shape).astype(to_numpy_type(data_type)),
    }

    executors.convert_run_compare(onnx_model, input_data, atol=1)
