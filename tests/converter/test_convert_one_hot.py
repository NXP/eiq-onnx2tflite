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

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig, SkipShapeInferenceConfig
from onnx2tflite.src.converter import convert
from tests import executors


@pytest.mark.parametrize("in_type, out_type, depth_type", [
    (TensorProto.INT64, TensorProto.INT64, TensorProto.INT64),
    (TensorProto.INT64, TensorProto.FLOAT, TensorProto.INT64),
    (TensorProto.INT32, TensorProto.FLOAT, TensorProto.INT32),

    (TensorProto.INT64, TensorProto.FLOAT, TensorProto.INT32)  # From a real model.
])
def test_convert_one_hot__static_inputs__types(in_type, out_type, depth_type):
    input_shape = [4]
    input_data = [3, 1, 0, 2]
    depth = 2
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('OneHot', ['indices', 'depth', 'values'], ['y'])],
        'OneHot test',
        [],
        [onnx.helper.make_tensor_value_info('y', out_type, ())],
        [
            onnx.helper.make_tensor('indices', in_type, input_shape, input_data),
            onnx.helper.make_tensor('depth', depth_type, [1], [depth]),
            onnx.helper.make_tensor('values', out_type, [2], [13, 37]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})


def test_convert_one_hot__invalid_types():
    in_type, out_type, depth_type = TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('OneHot', ['indices', 'depth', 'values'], ['y'])],
        'OneHot test',
        [],
        [onnx.helper.make_tensor_value_info('y', out_type, [4, 2])],
        [
            onnx.helper.make_tensor('indices', in_type, [4], [3, 1, 0, 2]),
            onnx.helper.make_tensor('depth', depth_type, [1], [2]),
            onnx.helper.make_tensor('values', out_type, [2], [13, 37]),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=SkipShapeInferenceConfig())
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_convert_one_hot__axis(axis):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('OneHot', ['indices', 'depth', 'values'], ['y'], axis=axis)],
        'OneHot test',
        [],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, [4, 8], [1, 2, 3, 4] * 8),
            onnx.helper.make_tensor('depth', TensorProto.INT64, [1], [4]),
            onnx.helper.make_tensor('values', TensorProto.INT64, [2], [13, 37]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})


@pytest.mark.parametrize("depth", [1, 2, 3, 10])
@pytest.mark.parametrize("depth_shape", [[], [1]])  # Can be a scalar or 1-element tensor.
def test_convert_one_hot__depth(depth, depth_shape):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('OneHot', ['indices', 'depth', 'values'], ['y'])],
        'OneHot test',
        [],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, [4, 8], [1, 2, 3, 4] * 8),
            onnx.helper.make_tensor('depth', TensorProto.INT64, depth_shape, [depth]),
            onnx.helper.make_tensor('values', TensorProto.INT64, [2], [13, 37]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})


def test_convert_one_hot__static_negative_indices():
    input_shape = [2, 4, 6, 8]
    np.random.seed(42)
    input_data = (np.random.random(input_shape) * 10 - 5).astype(np.int64)
    depth = 3

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('OneHot', ['indices', 'depth', 'values'], ['y'])],
        'OneHot test',
        [],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, input_shape, input_data),
            onnx.helper.make_tensor('depth', TensorProto.INT64, [1], [depth]),
            onnx.helper.make_tensor('values', TensorProto.INT64, [2], [13, 37]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})


def test_convert_one_hot__inferred_negative_indices(intermediate_tflite_model_provider):
    input_shape = [2, 4, 6, 8]
    np.random.seed(42)
    input_data = (np.random.random(input_shape) * 10 - 5).astype(np.int64)
    depth = 3

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Reshape', ['indices', 'shape'], ['dynamic_indices']),
            onnx.helper.make_node('OneHot', ['dynamic_indices', 'depth', 'values'], ['y']),
        ],
        'OneHot test',
        [],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, input_shape, input_data),
            onnx.helper.make_tensor('depth', TensorProto.INT64, [1], [depth]),
            onnx.helper.make_tensor('values', TensorProto.INT64, [2], [13, 37]),
            onnx.helper.make_tensor('shape', TensorProto.INT64, [len(input_shape)], input_shape),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig()
    config.dont_skip_nodes_with_known_outputs = True  # Make sure the `Reshape` doesn't just get skipped.
    executors.convert_run_compare(onnx_model, {}, conversion_config=config)

    message = logger.conversion_log.get_logs()['node_1'][0]['message']
    assert message == 'Using inferred static data for `OneHot` input tensor `indices` named `dynamic_indices`.'


def test_convert_one_hot__dynamic_indices(intermediate_tflite_model_provider):
    input_shape = [2, 4, 6, 8]
    depth = 3

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('OneHot', ['indices', 'depth', 'values'], ['y'])],
        'OneHot test',
        [onnx.helper.make_tensor_value_info('indices', TensorProto.INT64, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
        [
            onnx.helper.make_tensor('depth', TensorProto.INT64, [1], [depth]),
            onnx.helper.make_tensor('values', TensorProto.INT64, [2], [13, 37]),
            onnx.helper.make_tensor('shape', TensorProto.INT64, [len(input_shape)], input_shape),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    message = logger.conversion_log.get_node_error_message(0)
    assert 'Conversion of ONNX `OneHot` with dynamic `indices` input is not possible' in message
    assert '--guarantee-non-negative-indices' in message


def test_convert_one_hot__static_negative_indices__dynamic_depth():
    input_shape = [2, 4, 6, 8]
    np.random.seed(42)
    input_data = (np.random.random(input_shape) * 10 - 5).astype(np.int64)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('OneHot', ['indices', 'depth', 'values'], ['y'])],
        'OneHot test',
        [onnx.helper.make_tensor_value_info('depth', TensorProto.INT64, ())],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, input_shape, input_data),
            onnx.helper.make_tensor('values', TensorProto.INT64, [2], [13, 37]),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_logs()['shape_inference'][0]['error_code'] == logger.Code.SHAPE_INFERENCE_ERROR


def test_convert_one_hot__dynamic_values(intermediate_tflite_model_provider):
    input_shape = [2, 4, 6, 8]
    np.random.seed(42)
    input_data = (np.random.random(input_shape) * 10 - 5).astype(np.int64)
    depth = 3

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('OneHot', ['indices', 'depth', 'values'], ['y'])],
        'OneHot test',
        [onnx.helper.make_tensor_value_info('values', TensorProto.INT64, [2])],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
        [
            onnx.helper.make_tensor('depth', TensorProto.INT64, [1], [depth]),
            onnx.helper.make_tensor('indices', TensorProto.INT64, input_shape, input_data),
            onnx.helper.make_tensor('shape', TensorProto.INT64, [len(input_shape)], input_shape),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'Conversion of ONNX `OneHot` with a dynamic `values` input is not yet supported.' in \
           logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize("axis", [0, 1, 2, 3, 4])
def test_convert_one_hot__channels_first_input(axis):
    input_shape = [2, 4, 6, 8]

    np.random.seed(42)
    input_data = (np.random.random(input_shape) * 5).astype(np.int64)
    depth = 3

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['indices'], ['dynamic_indices'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Cast', ['dynamic_indices'], ['dynamic_indices_2'], to=TensorProto.INT64),
            onnx.helper.make_node('OneHot', ['dynamic_indices_2', 'depth', 'values'], ['y'], axis=axis)
        ],
        'OneHot test',
        [],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.FLOAT, input_shape, input_data),
            onnx.helper.make_tensor('depth', TensorProto.INT64, [1], [depth]),
            onnx.helper.make_tensor('values', TensorProto.FLOAT, [2], [13, 37]),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig()
    config.guarantee_non_negative_indices = True
    executors.convert_run_compare(onnx_model, {}, conversion_config=config)


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_convert_one_hot__channels_first_output(axis):
    input_shape = [2, 4, 6]

    np.random.seed(42)
    input_data = (np.random.random(input_shape) * 5).astype(np.int64)
    depth = 3

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('OneHot', ['indices', 'depth', 'values'], ['y1'], axis=axis),
            onnx.helper.make_node('MaxPool', ['y1'], ['y'], kernel_shape=[1, 1])
        ],
        'OneHot test',
        [],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, input_shape, input_data),
            onnx.helper.make_tensor('depth', TensorProto.INT64, [1], [depth]),
            onnx.helper.make_tensor('values', TensorProto.FLOAT, [2], [13, 37]),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {})


@pytest.mark.parametrize("axis", [0, 1, 2, 3, 4])
def test_convert_one_hot__channels_first_inputs_and_outputs(axis):
    input_shape = [2, 4, 6, 8]
    depth = 3

    np.random.seed(42)
    input_data = (np.random.random(input_shape) * 5).astype(np.int64)

    w1_shape = [4, 4, 1, 1]
    w1_data = np.random.random(w1_shape).astype(np.float32)

    w2_shape = [4, 4, 1, 1, 1]
    if axis == 0:  # A new batch size is created by the OneHot which pushes the original batch 2 to become the channels.
        w2_shape[1] = 2
    if axis == 1:  # A new channels dimension of size 3 gets created by the OneHot.
        w2_shape[1] = 3
    w2_data = np.random.random(w2_shape).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Conv', ['indices', 'w1'], ['y1'], kernel_shape=[1, 1], auto_pad='SAME_UPPER'),
            onnx.helper.make_node('Cast', ['y1'], ['y2'], to=TensorProto.INT64),
            onnx.helper.make_node('OneHot', ['y2', 'depth', 'values'], ['y3'], axis=axis),
            onnx.helper.make_node('Conv', ['y3', 'w2'], ['y'], kernel_shape=[1, 1, 1], auto_pad='SAME_UPPER')
        ],
        'OneHot test',
        [],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.FLOAT, input_shape, input_data),
            onnx.helper.make_tensor('depth', TensorProto.INT64, [1], [depth]),
            onnx.helper.make_tensor('values', TensorProto.FLOAT, [2], [13, 37]),
            onnx.helper.make_tensor('w1', TensorProto.FLOAT, w1_shape, w1_data),
            onnx.helper.make_tensor('w2', TensorProto.FLOAT, w2_shape, w2_data),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig()
    config.guarantee_non_negative_indices = True
    executors.convert_run_compare(onnx_model, {}, conversion_config=config)
