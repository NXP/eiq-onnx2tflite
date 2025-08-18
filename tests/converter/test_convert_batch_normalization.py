#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import List

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import SkipShapeInferenceConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from tests import executors


def _make_operands(input_shape):
    np.random.seed(input_shape[-1])

    if len(input_shape) > 1:
        channels = [input_shape[1]]
    else:
        channels = [1]

    scale = np.random.rand(*channels)
    bias = np.random.rand(*channels)
    mean = np.random.rand(*channels)
    var = np.random.rand(*channels)

    return [
        onnx.helper.make_tensor("scale", TensorProto.FLOAT, channels, scale),
        onnx.helper.make_tensor("bias", TensorProto.FLOAT, channels, bias),
        onnx.helper.make_tensor("mean", TensorProto.FLOAT, channels, mean),
        onnx.helper.make_tensor("var", TensorProto.FLOAT, channels, var),
    ]


@pytest.mark.parametrize("_type", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_batch_normalization__quantized(_type: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('BatchNormalization', ['x1', "scale", "bias", "mean", "var"], ['x2']),
            onnx.helper.make_node('DequantizeLinear', ['x2', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', _type, [1], [12]),
            onnx.helper.make_tensor('scale', _type, [1], [10]),
            onnx.helper.make_tensor('bias', _type, [1], [20]),
            onnx.helper.make_tensor('mean', _type, [1], [30]),
            onnx.helper.make_tensor('var', _type, [1], [40]),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.INVALID_ONNX_MODEL


@pytest.mark.parametrize(
    "input_shape",
    [
        # [42], # ONNX Runtime crashes with a 1D input despite the fact that the documentation supports it.
        [13, 37],
        [3, 14, 159],
        [10, 42, 23, 17],
        [2, 71, 8, 28, 18],
    ], ids=lambda x: f"{len(x)}D")
def test_convert_batch_normalization(input_shape: List[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("BatchNormalization", ["input", "scale", "bias", "mean", "var"], ["output"])],
        'BatchNormalization test',
        [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        _make_operands(input_shape)
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_batch_normalization_with_invalid_inputs():
    input_shape = [5, 10, 15, 20]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("BatchNormalization", ["input", "scale", "bias", "mean"], ["output"])],
        'BatchNormalization test',
        [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        _make_operands(input_shape)[:-1]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model, conversion_config=SkipShapeInferenceConfig())
    assert e.value.error_code == logger.Code.SHAPE_INFERENCE_ERROR


def test_convert_batch_normalization_with_invalid_outputs():
    input_shape = [5, 10, 15, 20]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("BatchNormalization", ["input", "scale", "bias", "mean", "var"],
                               ["output", "running_mean", "running_var"])],
        'BatchNormalization test',
        [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        _make_operands(input_shape)
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model, conversion_config=SkipShapeInferenceConfig())
    assert e.value.error_code == logger.Code.SHAPE_INFERENCE_ERROR


@pytest.mark.parametrize(
    "input_shape",
    [
        [13, 37],
        [3, 14, 159],
        [10, 42, 23, 17],
        [2, 71, 8, 28, 18],
    ], ids=lambda x: f"{len(x)}D")
def test_convert_batch_normalization_with_dynamic_operands(input_shape: List[int]):
    np.random.seed(input_shape[-1])

    channels = [input_shape[1]]

    scale = np.random.rand(*channels).astype(np.float32)
    bias = np.random.rand(*channels).astype(np.float32)
    mean = np.random.rand(*channels).astype(np.float32)
    var = np.random.rand(*channels).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("BatchNormalization", ["input", "scale", "bias", "mean", "var"], ["output"])],
        'BatchNormalization test',
        [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("scale", TensorProto.FLOAT, channels),
            onnx.helper.make_tensor_value_info("bias", TensorProto.FLOAT, channels),
            onnx.helper.make_tensor_value_info("mean", TensorProto.FLOAT, channels),
            onnx.helper.make_tensor_value_info("var", TensorProto.FLOAT, channels),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: scale,
        2: bias,
        3: mean,
        4: var,
    }

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape",
    [
        [13, 37],
        [3, 14, 159],
        [10, 42, 23, 17],
        [2, 71, 8, 28, 18],
    ], ids=lambda x: f"{len(x)}, spatial=0")
def test_convert_batch_normalization_with_zero_spatial_argument(input_shape: List[int]):
    operand_shape = input_shape[1:]

    np.random.seed(input_shape[-1])
    scale = np.random.rand(*operand_shape)
    bias = np.random.rand(*operand_shape)
    mean = np.random.rand(*operand_shape)
    var = np.random.rand(*operand_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("BatchNormalization", ["input", "scale", "bias", "mean", "var"], ["output"], spatial=0)],
        'BatchNormalization test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("scale", TensorProto.FLOAT, operand_shape, scale),
            onnx.helper.make_tensor("bias", TensorProto.FLOAT, operand_shape, bias),
            onnx.helper.make_tensor("mean", TensorProto.FLOAT, operand_shape, mean),
            onnx.helper.make_tensor("var", TensorProto.FLOAT, operand_shape, var),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import[0].version = 7  # Only v7 has the `spatial` attribute

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape",
    [
        [13, 37],
        [3, 14, 159],
        [10, 42, 23, 17],
        [2, 71, 8, 28, 18],
    ], ids=lambda x: f"{len(x)}D, spatial=0, dynamic operands")
def test_convert_batch_normalization_with_dynamic_operands_and_zero_spatial_argument(input_shape: List[int]):
    np.random.seed(input_shape[-1])

    channels = input_shape[1:]

    scale = np.random.rand(*channels).astype(np.float32)
    bias = np.random.rand(*channels).astype(np.float32)
    mean = np.random.rand(*channels).astype(np.float32)
    var = np.random.rand(*channels).astype(np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("BatchNormalization", ["input", "scale", "bias", "mean", "var"], ["output"], spatial=0)],
        'BatchNormalization test',
        [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("scale", TensorProto.FLOAT, channels),
            onnx.helper.make_tensor_value_info("bias", TensorProto.FLOAT, channels),
            onnx.helper.make_tensor_value_info("mean", TensorProto.FLOAT, channels),
            onnx.helper.make_tensor_value_info("var", TensorProto.FLOAT, channels),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx_model.opset_import[0].version = 7  # Only v7 has the `spatial` attribute

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: scale,
        2: bias,
        3: mean,
        4: var,
    }

    executors.convert_run_compare(onnx_model, input_data)
