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


def _create_onnx_model(input_shape: List[int], input_scale, input_zero_point, output_scale,
                       output_zero_point, input_type: TensorProto.DataType,
                       output_type: TensorProto.DataType) -> onnx.ModelProto:
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("QLinearGlobalAveragePool",
                                      ["input", "input_scale", "input_zero_point", "output_scale", "output_zero_point"],
                                      ["output"],
                                      domain="com.microsoft")  # Specify which opset QLinearGlobalAveragePool is from
            ],
            name="QLinearGlobalAveragePool_test",
            inputs=[onnx.helper.make_tensor_value_info("input", input_type, input_shape)],
            outputs=[onnx.helper.make_tensor_value_info("output", output_type, ())],
            initializer=[
                onnx.helper.make_tensor("input_scale", onnx.TensorProto.FLOAT, [], input_scale),
                onnx.helper.make_tensor("input_zero_point", input_type, [], input_zero_point),
                onnx.helper.make_tensor("output_scale", onnx.TensorProto.FLOAT, [], output_scale),
                onnx.helper.make_tensor("output_zero_point", output_type, [], output_zero_point),
            ]
        ),
    )

    # Add the opset with QLinearGlobalAveragePool
    onnx_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    return onnx_model


@pytest.mark.parametrize(
    "input_shape , input_scale, input_zero_point, output_scale, output_zero_point, "
    "input_type, output_type",
    [
        pytest.param([1, 12, 128, 256], [0.1], [0], [0.1], [0],
                     TensorProto.INT8, TensorProto.INT8, id="signed types"),
        pytest.param([1, 12, 128, 256], [0.1], [128], [0.1], [128],
                     TensorProto.UINT8, TensorProto.UINT8, id="unsigned types"),
        pytest.param([1, 12, 128, 256], [0.5], [-20], [0.05], [20],
                     TensorProto.INT8, TensorProto.INT8, id="signed + different quantization parameters"),
        pytest.param([1, 12, 128, 256], [0.05], [100], [0.5], [156],
                     TensorProto.UINT8, TensorProto.UINT8, id="unsigned + different quantization parameters"),

        pytest.param([1, 12, 12, 16, 18], [0.1], [0], [0.1], [0],
                     TensorProto.INT8, TensorProto.INT8, id="5D signed"),
        pytest.param([1, 3, 64, 6, 4], [0.1], [128], [0.1], [128],
                     TensorProto.UINT8, TensorProto.UINT8, id="5D unsigned"),
        pytest.param([1, 15, 10, 15, 17], [0.5], [-20], [0.05], [20],
                     TensorProto.INT8, TensorProto.INT8, id="5D signed + different quantization parameters"),
        pytest.param([1, 5, 3, 15, 12], [0.05], [100], [0.5], [156],
                     TensorProto.UINT8, TensorProto.UINT8, id="5D unsigned + different quantization parameters"),

        pytest.param([1, 12, 12], [0.04], [0], [0.04], [0],
                     TensorProto.INT8, TensorProto.INT8, id="3D signed"),
        pytest.param([1, 3, 64], [0.09], [128], [0.09], [128],
                     TensorProto.UINT8, TensorProto.UINT8, id="3D unsigned"),
        pytest.param([1, 15, 10], [0.123], [-30], [0.0321], [35],
                     TensorProto.INT8, TensorProto.INT8, id="3D signed + different quantization parameters"),
        pytest.param([1, 5, 3], [0.0421], [90], [0.124], [170],
                     TensorProto.UINT8, TensorProto.UINT8, id="3D unsigned + different quantization parameters"),
    ])
def test_convert_q_linear_global_average_pool(input_shape: List[int], input_scale, input_zero_point, output_scale,
                                              output_zero_point, input_type: TensorProto.DataType,
                                              output_type: TensorProto.DataType):
    onnx_model = _create_onnx_model(input_shape, input_scale, input_zero_point, output_scale,
                                    output_zero_point, input_type, output_type)

    input_data = np.arange(math.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(input_type))

    executors.convert_run_compare(onnx_model, input_data, atol=1)
