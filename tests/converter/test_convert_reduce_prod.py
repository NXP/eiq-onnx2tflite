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
from onnx2tflite.src.conversion_config import SkipShapeInferenceConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    "axes",
    [
        [0], [1], [2], [-1], [-2], [-3],
        [-1, 1],
        [0, 1, 2], [-2, 0, -1],
    ], ids=lambda x: f'axes={x}')
def test_convert_reduce_prod__v18(axes: list[int]):
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReduceProd', ['x', 'axes'], ['y'])],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('axes', TensorProto.INT64, [len(axes)], axes)]
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "axes",
    [
        [0], [1], [2], [-1], [-2], [-3],
        [-1, 1],
        [0, 1, 2], [-2, 0, -1],
    ], ids=lambda x: f'axes={x}')
def test_convert_reduce_prod__v18_channels_first_static_axes(axes: list[int]):
    x_shape = [5, 10, 15, 20]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('ReduceProd', ['y1', 'axes'], ['y'])
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('axes', TensorProto.INT64, [len(axes)], axes)]
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize("axes", [[0], [1], [2], [3]], ids=lambda x: f'axes={x}')
def test_convert_reduce_prod__channels_first_input_and_output(axes: list[int]):
    x_shape = [2, 3, 4, 5]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('ReduceProd', ['y1', 'axes'], ['y2'], keepdims=0),
            onnx.helper.make_node('MaxPool', ['y2'], ['y'], kernel_shape=[1]),
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('axes', TensorProto.INT64, [len(axes)], axes)]
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize("axes", [[0], [1], [2], [3]], ids=lambda x: f'axes={x}')
def test_convert_reduce_prod__channels_first_output(axes: list[int]):
    x_shape = [2, 3, 4, 5]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('ReduceProd', ['x', 'axes'], ['y2'], keepdims=0),
            onnx.helper.make_node('MaxPool', ['y2'], ['y'], kernel_shape=[1]),
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('axes', TensorProto.INT64, [len(axes)], axes)]
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "axes",
    [
        [0], [1], [2], [-1], [-2], [-3],
        [-1, 1],
        [0, 1, 2], [-2, 0, -1],
    ], ids=lambda x: f'axes={x}')
def test_convert_reduce_prod__v18_channels_first__keepdims_false__different_axes(axes: list[int]):
    x_shape = [2, 3, 4, 5]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('ReduceProd', ['y1', 'axes'], ['y'], keepdims=0)
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('axes', TensorProto.INT64, [len(axes)], axes)]
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    executors.convert_run_compare(onnx_model, x_data)


def test_convert_reduce_prod__v18_dynamic_axes():
    x_shape = [5, 10, 15, 20]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('ReduceProd', ['x', 'axes'], ['y'])
        ],
        'ReduceProd test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('axes', TensorProto.INT64, ())
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_logs()['shape_inference'][0]['error_code'] == logger.Code.SHAPE_INFERENCE_ERROR


@pytest.mark.parametrize(
    'indices',
    [
        pytest.param([0], id='axes = [1]'),
        pytest.param([1], id='axes = [2]'),
        pytest.param([2], id='axes = [3]'),
        pytest.param([0, 2], id='axes = [1, 3]'),
        pytest.param([2, 0, 1], id='axes = [3, 1, 2]'),
    ])
def test_convert_reduce_prod__v18_inferred_dynamic_axes(indices: list[int]):
    x_shape = [1, 2, 3, 20]

    np.random.seed(42)
    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Shape', ['x'], ['shape']),
            onnx.helper.make_node('Gather', ['shape', 'indices'], ['axes']),
            onnx.helper.make_node('ReduceProd', ['x', 'axes'], ['y'])
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('indices', TensorProto.INT64, [len(indices)], indices)]
    )

    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    'indices',
    [
        pytest.param([0], id='axes = [1]'),
        pytest.param([1], id='axes = [2]'),
        pytest.param([2], id='axes = [3]'),
        pytest.param([0, 2], id='axes = [1, 3]'),
        pytest.param([2, 0, 1], id='axes = [3, 1, 2]'),
    ])
def test_convert_reduce_prod__v18_channels_first_inferred_dynamic_axes(indices: list[int]):
    x_shape = [1, 2, 3, 20]

    np.random.seed(42)
    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('Shape', ['x'], ['shape']),
            onnx.helper.make_node('Gather', ['shape', 'indices'], ['axes']),
            onnx.helper.make_node('ReduceProd', ['x1', 'axes'], ['y'])
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('indices', TensorProto.INT64, [len(indices)], indices)]
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    executors.convert_run_compare(onnx_model, x_data)


def test_convert_reduce_prod__v18_dynamic_axes__cast_operator():
    # If the shape inference is skipped, there is no inferred data for the axis. Conversion is supported by adding a
    #  `cast` operator.

    shape = [1, 2, 3, 20]
    axes = [-2, 0]
    output_shape = [1, 2, 1, 20]

    np.random.seed(42)

    input_data = {
        0: np.random.rand(*shape).astype(np.float32) - 0.5,
        1: np.asarray(axes, np.int64)
    }

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('ReduceProd', ['x', 'axes'], ['y'])
        ],
        'ReduceProd test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('axes', TensorProto.INT64, [len(axes)])
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, output_shape)]
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])

    executors.convert_run_compare(onnx_model, input_data, conversion_config=SkipShapeInferenceConfig())


def test_convert_reduce_prod__v18_default_axes():
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReduceProd', ['x', ''], ['y'])],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "axes",
    [
        [0], [1], [2], [-1], [-2], [-3],
        [-1, 1],
        [0, 1, 2], [-2, 0, -1],
    ], ids=lambda x: f'axes={x}')
def test_convert_reduce_prod__v13(axes: list[int]):
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReduceProd', ['x'], ['y'], axes=axes)],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "axes",
    [
        [0], [1], [2], [-1], [-2], [-3],
        [-1, 1],
        [0, 1, 2], [-2, 0, -1],
    ], ids=lambda x: f'axes={x}')
def test_convert_reduce_prod__v13_channels_first_input(axes: list[int]):
    x_shape = [5, 10, 15, 20]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('ReduceProd', ['x1'], ['y'], axes=axes)
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize("axes", [[0], [1], [2], [-1], [-2]], ids=lambda x: f'axes={x}')
def test_convert_reduce_prod__v13_channels_first_output(axes: list[int]):
    x_shape = [2, 3, 4, 5]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('ReduceProd', ['x'], ['y1'], keepdims=0, axes=axes),
            onnx.helper.make_node('MaxPool', ['y1'], ['y'], kernel_shape=[1]),
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize("axes", [[0], [1], [2], [-1], [-2]], ids=lambda x: f'axes={x}')
def test_convert_reduce_prod__v13_channels_first_input_and_output(axes: list[int]):
    x_shape = [2, 3, 4, 5]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('ReduceProd', ['y1'], ['y2'], keepdims=0, axes=axes),
            onnx.helper.make_node('MaxPool', ['y2'], ['y'], kernel_shape=[1]),
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize("keepdims", [0, 1], ids=lambda x: f"keepdims={x}")
def test_convert_reduce_prod__v13_channels_first_different_keepdims(keepdims):
    x_shape = [1, 1, 2, 3]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('ReduceProd', ['x1'], ['y'], axes=[1], keepdims=keepdims)
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    executors.convert_run_compare(onnx_model, x_data)


@pytest.mark.parametrize(
    "axes",
    [
        [0], [1], [2], [-1], [-2], [-3],
        [-1, 1],
        [0, 1, 2], [-2, 0, -1],
    ], ids=lambda x: f'axes={x}')
def test_convert_reduce_prod__v13_channels_first__keepdims_false__different_axes(axes):
    x_shape = [2, 3, 4, 5]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['x1'], kernel_shape=[1, 1]),
            onnx.helper.make_node('ReduceProd', ['x1'], ['y'], axes=axes, keepdims=0)
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    executors.convert_run_compare(onnx_model, x_data)


def test_convert_reduce_prod__v13_default_axes():
    x_shape = [5, 10, 15]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReduceProd', ['x'], ['y'])],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    executors.convert_run_compare(onnx_model, x_data)


def test_convert_reduce_prod__noop_with_empty_axes(intermediate_tflite_model_provider):
    x_shape = [5, 10, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('ReduceProd', ['x'], ['y'], noop_with_empty_axes=1),
        ],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR


def test_convert_reduce_prod__keepdims_true():
    x_shape = [5, 10, 15]

    np.random.seed(42)
    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReduceProd', ['x', 'axes'], ['y'], keepdims=True)],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('axes', TensorProto.INT64, [2], [0, 2])]
    )
    onnx_model = onnx.helper.make_model(graph)
    executors.convert_run_compare(onnx_model, x_data)


def test_convert_reduce_prod__keepdims_false():
    x_shape = [5, 10, 15]

    np.random.seed(42)
    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReduceProd', ['x', 'axes'], ['y'], keepdims=False)],
        'ReduceProd test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('axes', TensorProto.INT64, [2], [0, 2])]
    )
    onnx_model = onnx.helper.make_model(graph)
    executors.convert_run_compare(onnx_model, x_data)


def test_convert_reduce_sum__invalid_type():
    type_ = TensorProto.DOUBLE

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReduceSum', ['x'], ['y'], axes=[0])],
        'Invalid type test test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)])

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize(
    "type_",
    [
        TensorProto.INT32, TensorProto.INT64, TensorProto.FLOAT,
    ], ids=name_for_onnx_type)
def test_convert_reduce_sum__types(type_: TensorProto.DataType):
    shape = [42]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ReduceSum', ['x'], ['y'], axes=[0])],
        'Type test test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)])

    np.random.seed(42)
    data = (np.random.random(shape) * 20).astype(to_numpy_type(type_))
    executors.convert_run_compare(onnx_model, data)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_reduce_sum__quantized(type_: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('ReduceSum', ['x1', 'axes'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('axes', TensorProto.INT64, [1], [0])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.INVALID_ONNX_MODEL
