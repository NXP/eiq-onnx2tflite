#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

import numpy as np
import onnx.helper
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import SkipShapeInferenceConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from onnx2tflite.src.tflite_generator.builtin_options import transpose_options
from tests import executors


@pytest.mark.parametrize(
    "output_shape",
    [
        ([5, 768]),
        ([5, 3, 256]),
        ([5, 48, 16]),
        ([5, 12, 4, 16]),
        ([10, 6, 2, 8, 4]),
        ([15, 256]),
        ([1, 3840]),
    ])
def test_convert_reshape(output_shape: List[int]):
    input_shape = (5, 3, 16, 16)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Reshape', ['input', 'new_shape'], ['output'])],
        'reshape test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("new_shape", TensorProto.INT64, [len(output_shape)], output_shape)]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_reshape_quantized():
    input_shape = (5, 3, 16, 16)
    output_shape = ([5, 768])
    scale = [1.0]
    zero_point = [0]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["y"], axis=1),
            onnx.helper.make_node('Reshape', ['y', 'new_shape'], ['output'])],
        'reshape test quantized',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT8, ())],
        initializer=[
            onnx.helper.make_tensor("new_shape", TensorProto.INT64, [len(output_shape)], output_shape),
            onnx.helper.make_tensor("scale", TensorProto.FLOAT, [len(scale)], scale),
            onnx.helper.make_tensor("zero_point", TensorProto.INT8, [len(zero_point)], zero_point)
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    tflite_executor, _ = executors.convert_run_compare(onnx_model, input_data)
    output_quant_params = tflite_executor.get_output_details(0)['quantization_parameters']
    assert output_quant_params['scales'] == [1.0]
    assert output_quant_params['zero_points'] == [0]


@pytest.mark.parametrize(
    "new_shape, output_shape",
    [
        # Input shape is always (5, 3, 16, 16)
        pytest.param([0, 3, 16, 16], [5, 3, 16, 16], id="'0' on index 0"),
        pytest.param([5, 0, 16, 16], [5, 3, 16, 16], id="'0' on index 1"),
        pytest.param([5, 3, 16, 0], [5, 3, 16, 16], id="'0' on index 3"),
        pytest.param([10, 0, 128], [10, 3, 128], id="'0' with reduced rank"),

        pytest.param([-1, 3, 16, 16], [5, 3, 16, 16], id="'-1' on index 0"),
        pytest.param([5, 3, 16, -1], [5, 3, 16, 16], id="'-1' on index 3"),
        pytest.param([10, -1, 12], [10, 32, 12], id="'-1' with reduced rank"),
        pytest.param([2, 3, -1, 4, 2], [2, 3, 80, 4, 2], id="'-1' with increased rank"),

        pytest.param([-1, 0, 16, 16], [5, 3, 16, 16], id="'-1' and '0' combined"),
        pytest.param([10, -1, 0], [10, 24, 16], id="'-1' and '0' with reduced rank"),
        pytest.param([2, 0, 4, -1, 2], [2, 3, 4, 80, 2], id="'-1' and '0' with increased rank"),

    ])
def test_convert_reshape_with_special_values(new_shape: List[int], output_shape: List[int]):
    input_shape = (5, 3, 16, 16)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Reshape', ['input', 'new_shape'], ['output'])],
        'reshape test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("new_shape", TensorProto.INT64, [len(new_shape)], new_shape)]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "new_shape, output_shape",
    [
        # Input shape is always (5, 3, 14, 16)
        pytest.param([0, 3, 14, 16], [5, 3, 14, 16], id="to channels first: '0' on index 0"),
        pytest.param([5, 0, 14, 16], [5, 3, 14, 16], id="to channels first: '0' on index 1"),
        pytest.param([5, 3, 14, 0], [5, 3, 14, 16], id="to channels first: '0' on index 3"),
        pytest.param([10, 0, 112], [10, 3, 112], id="to channels first: '0' with reduced rank"),
        pytest.param([3, 2, 5, 0, 7], [3, 2, 5, 16, 7], id="to channels first: '0' with increased rank"),
        pytest.param([0, 672], [5, 672], id="to formatless: '0'"),

        pytest.param([-1, 3, 14, 16], [5, 3, 14, 16], id="to channels first: '-1' on index 0"),
        pytest.param([5, 3, 16, -1], [5, 3, 16, 14], id="to channels first: '-1' on index 3"),
        pytest.param([10, -1, 12], [10, 28, 12], id="to channels first: '-1' with reduced rank"),
        pytest.param([2, 3, -1, 4, 2], [2, 3, 70, 4, 2], id="to channels first: '-1' with increased rank"),
        pytest.param([20, -1], [20, 168], id="to formatless: '-1'"),

        pytest.param([-1, 0, 14, 16], [5, 3, 14, 16], id="to channels first: '-1' and '0' combined"),
        pytest.param([10, -1, 0], [10, 24, 14], id="to channels first: '-1' and '0' with reduced rank"),
        pytest.param([2, 0, 4, -1, 2], [2, 3, 4, 70, 2], id="to channels first: '-1' and '0' with increased rank"),
        pytest.param([-1, 0], [1120, 3], id="to formatless: '-1' and '0'"),

    ])
def test_convert_reshape_from_channels_first(new_shape: List[int], output_shape: List[int]):
    input_shape = [5, 3, 14, 16]
    kernel_shape = [1] * (len(input_shape) - 2)
    if new_shape is None:
        new_shape = output_shape

    onnx_model = onnx.helper.make_model(
        graph=onnx.helper.make_graph(
            [
                onnx.helper.make_node('MaxPool', ['input'], ['max_pool_out'], kernel_shape=kernel_shape),
                onnx.helper.make_node('Reshape', ['max_pool_out', 'new_shape'], ['output'])
            ],
            'conv + reshape',
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[onnx.helper.make_tensor("new_shape", TensorProto.INT64, [len(new_shape)], new_shape)]
        )
    )

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, new_shape, output_shape",
    [
        pytest.param([2, 3, 14, 16], [2, 3, 13, 15], [2, 3, 11, 13], id="4D->4D,BC=preserved"),
        pytest.param([5, 3, 14, 16], [0, 3, 13, 15], [5, 3, 12, 14], id="4D->4D,BC=preserved,batch=0"),
        pytest.param([2, 3, 14, 16], [3, 1, 26, 15], [3, 1, 24, 13], id="4D->4D,BC=mixed"),
        pytest.param([5, 4, 14, 16], [-1, 13, 15], [20, 11, 14], id="4D->3D,BC=mixed,batch=-1"),
        pytest.param([5, 4, 14, 16], [20, 13, 15], [20, 11, 14], id="4D->3D,BC=mixed-0"),
        pytest.param([2, 3, 14, 16], [3, 26, 15], [20, 11, 14], id="4D->3D,BC=mixed-1"),
    ])
def test_convert_reshape_both_io_channels_first(input_shape: List[int], new_shape: List[int], output_shape: List[int]):
    kernel_shape_pre = [2] * (len(input_shape) - 2)
    kernel_shape_post = [2] * (len(output_shape) - 2)

    onnx_model = onnx.helper.make_model(
        graph=onnx.helper.make_graph(
            [
                onnx.helper.make_node('MaxPool', ['input'], ['max_pre_pool_out'], kernel_shape=kernel_shape_pre),
                onnx.helper.make_node('Reshape', ['max_pre_pool_out', 'new_shape'], ['reshape_output']),
                onnx.helper.make_node('MaxPool', ['reshape_output'], ['max_pool_post_out'],
                                      kernel_shape=kernel_shape_post),
            ],
            'MaxPool+Reshape+MaxPool',
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
            [onnx.helper.make_tensor_value_info("max_pool_post_out", TensorProto.FLOAT, ())],
            initializer=[onnx.helper.make_tensor("new_shape", TensorProto.INT64, [len(new_shape)], new_shape)]
        )
    )

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "input_shape, new_shape, transpose_ops_count",
    [
        # 3D Unsqueeze
        pytest.param([2, 3, 4], [1, 2, 3, 4], 3, id="3D-Unsqueeze-index=0"),
        pytest.param([2, 3, 4], [2, 1, 3, 4], 3, id="3D-Unsqueeze-index=1"),
        pytest.param([2, 3, 4], [2, 3, 1, 4], 2, id="3D-Unsqueeze-index=2"),
        pytest.param([2, 3, 4], [2, 3, 4, 1], 2, id="3D-Unsqueeze-index=3"),

        # 4D Squeeze
        pytest.param([1, 2, 3, 4], [2, 3, 4], 3, id="4D-Squeeze-index=0"),
        pytest.param([2, 1, 3, 4], [2, 3, 4], 3, id="4D-Squeeze-index=1"),
        pytest.param([2, 3, 1, 4], [2, 3, 4], 2, id="4D-Squeeze-index=2"),
        pytest.param([2, 3, 4, 1], [2, 3, 4], 2, id="4D-Squeeze-index=3"),

        # 5D Squeeze
        pytest.param([1, 2, 3, 4, 5], [2, 3, 4, 5], 3, id="5D-Squeeze-index=0"),
        pytest.param([2, 1, 3, 4, 5], [2, 3, 4, 5], 3, id="5D-Squeeze-index=1"),
        pytest.param([2, 3, 1, 4, 5], [2, 3, 4, 5], 2, id="5D-Squeeze-index=2"),
        pytest.param([2, 3, 4, 1, 5], [2, 3, 4, 5], 2, id="5D-Squeeze-index=3"),
        pytest.param([2, 3, 4, 5, 1], [2, 3, 4, 5], 2, id="5D-Squeeze-index=4"),

        # 4D Unsqueeze
        pytest.param([2, 3, 4, 5], [1, 2, 3, 4, 5], 3, id="4D-Unsqueeze-index=0"),
        pytest.param([2, 3, 4, 5], [2, 1, 3, 4, 5], 3, id="4D-Unsqueeze-index=1"),
        pytest.param([2, 3, 4, 5], [2, 3, 1, 4, 5], 2, id="4D-Unsqueeze-index=2"),
        pytest.param([2, 3, 4, 5], [2, 3, 4, 1, 5], 2, id="4D-Unsqueeze-index=3"),
        pytest.param([2, 3, 4, 5], [2, 3, 4, 5, 1], 2, id="4D-Unsqueeze-index=4"),

        # Mixed dimensions
        pytest.param([2, 3, 2, 10], [2, 3, 4, 5, 1], 2, id="4D-Unsqueeze-index=4-mixed"),
        pytest.param([2, 3, 15], [2, 3, 1, 3, 5], 2, id="3D-Unsqueeze-index=2-mixed"),
        pytest.param([2, 3, 10, 1, 4], [2, 3, 2, 20], 2, id="5D-Unsqueeze-index=3-mixed"),
    ])
def test_convert_reshape_both_io_channels_first__single_unitary_change(
        input_shape: List[int], new_shape: List[int], transpose_ops_count, intermediate_tflite_model_provider
):
    np.random.seed(input_shape[-1])

    channels_pre = [input_shape[1]]
    channels_post = [new_shape[1]]

    scale_pre = np.random.rand(*channels_pre).astype(np.float32)
    bias_pre = np.random.rand(*channels_pre).astype(np.float32)
    mean_pre = np.random.rand(*channels_pre).astype(np.float32)
    var_pre = np.random.rand(*channels_pre).astype(np.float32)

    scale_post = np.random.rand(*channels_post).astype(np.float32)
    bias_post = np.random.rand(*channels_post).astype(np.float32)
    mean_post = np.random.rand(*channels_post).astype(np.float32)
    var_post = np.random.rand(*channels_post).astype(np.float32)

    # Using BatchNorm because it supports up to 5D as an input
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("BatchNormalization",
                                  ["input", "scale_pre", "bias_pre", "mean_pre", "var_pre"],
                                  ["y"]),
            onnx.helper.make_node('Reshape', ['y', 'new_shape'], ['reshape_output']),
            onnx.helper.make_node("BatchNormalization",
                                  ["reshape_output", "scale_post", "bias_post", "mean_post", "var_post"],
                                  ["z"]),
        ],
        'BatchNormalization+Reshape+BatchNormalization',
        inputs=[
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("scale_pre", TensorProto.FLOAT, channels_pre),
            onnx.helper.make_tensor_value_info("bias_pre", TensorProto.FLOAT, channels_pre),
            onnx.helper.make_tensor_value_info("mean_pre", TensorProto.FLOAT, channels_pre),
            onnx.helper.make_tensor_value_info("var_pre", TensorProto.FLOAT, channels_pre),
            onnx.helper.make_tensor_value_info("scale_post", TensorProto.FLOAT, channels_post),
            onnx.helper.make_tensor_value_info("bias_post", TensorProto.FLOAT, channels_post),
            onnx.helper.make_tensor_value_info("mean_post", TensorProto.FLOAT, channels_post),
            onnx.helper.make_tensor_value_info("var_post", TensorProto.FLOAT, channels_post),
        ],
        outputs=[onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("new_shape", TensorProto.INT64, [len(new_shape)], new_shape)])

    onnx_model = onnx.helper.make_model(graph=graph)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: scale_pre,
        2: bias_pre,
        3: mean_pre,
        4: var_pre,
        5: scale_post,
        6: bias_post,
        7: mean_post,
        8: var_post,
    }

    executors.convert_run_compare(onnx_model, input_data, atol=4e-6)

    assert intermediate_tflite_model_provider.get_op_count(transpose_options.Transpose) == transpose_ops_count


@pytest.mark.parametrize(
    "new_shape, output_shape",
    [
        # Input shape is always (48, 60)  3*16 * 3*4*5
        pytest.param([0, 3, 4, 5], [48, 3, 4, 5], id="to channels first: '0' on index 0"),
        pytest.param([2, 0, 3, 8], [2, 60, 3, 8], id="to channels first: '0' on index 1"),

        pytest.param([-1, 3, 12, 5], [16, 3, 12, 5], id="to channels first: '-1' on index 0"),
        pytest.param([2, 32, -1, 15], [2, 32, 3, 15], id="to channels first: '-1' on index 2"),
        pytest.param([20, 12, 12, -1], [20, 12, 12, 1], id="to channels first: '-1' on index 3"),

        pytest.param([-1, 0, 2, 8], [3, 60, 2, 8], id="to channels first: '0' and '-1'"),
    ])
def test_convert_reshape_from_formatless(new_shape: List[int], output_shape: List[int]):
    input_shape = [48, 60]
    kernel_shape = [1] * (len(new_shape) - 2)
    if new_shape is None:
        new_shape = output_shape

    onnx_model = onnx.helper.make_model(
        graph=onnx.helper.make_graph(
            [
                onnx.helper.make_node('Reshape', ['input', 'new_shape'], ['reshape_output']),
                onnx.helper.make_node('MaxPool', ['reshape_output'], ['output'], kernel_shape=kernel_shape),
            ],
            'reshape + max_pool',
            [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
            [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
            initializer=[onnx.helper.make_tensor("new_shape", TensorProto.INT64, [len(new_shape)], new_shape)]
        )
    )

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_reshape_dynamic():
    input_shape = (1, 768, 14, 14)
    new_shape_shape = (3,)
    output_shape = (1, 768, 196)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Reshape', ['input', 'new_shape'], ['output'])],
        'reshape test',
        [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("new_shape", TensorProto.INT64, new_shape_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = {
        0: np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.array([1, 768, -1], dtype=np.int64).reshape(new_shape_shape)
    }

    executors.convert_run_compare(onnx_model, input_data, conversion_config=SkipShapeInferenceConfig())


@pytest.mark.parametrize("type_", [
    TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT8, TensorProto.INT16, TensorProto.INT32,
    TensorProto.INT64, TensorProto.UINT8, TensorProto.UINT32, TensorProto.UINT64, TensorProto.STRING, TensorProto.BOOL
], ids=name_for_onnx_type)
def test_convert_reshape__types(type_: TensorProto.DataType):
    input_shape = [2, 4, 6]
    output_shape = [8, 6]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Reshape', ['x', 'new_shape'], ['y'])],
        'reshape test',
        [onnx.helper.make_tensor_value_info("x", type_, input_shape)],
        [onnx.helper.make_tensor_value_info("y", type_, ())],
        [onnx.helper.make_tensor('new_shape', TensorProto.INT64, [len(output_shape)], output_shape)]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(to_numpy_type(type_))

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_reshape__invalid_type():
    type_ = TensorProto.UINT16

    input_shape = [2, 4, 6]
    output_shape = [8, 6]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Reshape', ['x', 'new_shape'], ['y'])],
        'Invalid type test test',
        [onnx.helper.make_tensor_value_info('x', type_, input_shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('new_shape', TensorProto.INT64, [len(output_shape)], output_shape)]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'UINT16' in logger.conversion_log.get_node_error_message(0)
