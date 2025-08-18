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
from onnx2tflite.src.converter import convert
from tests import executors


def noise(shape: list[int]) -> np.ndarray:
    """ Return random float32 noise in the range of -0.05 to 0.05 """
    return (np.random.rand(*shape) - .5).astype(np.float32) / 10.


def _get_rnn_shapes(seq_length=100, batch_size=2, input_size=100, hidden_size=16, num_directions=1):
    x_shape = [seq_length, batch_size, input_size]
    w_shape = [num_directions, hidden_size, input_size]
    r_shape = [num_directions, hidden_size, hidden_size]
    b_shape = [num_directions, 2 * hidden_size]

    return x_shape, w_shape, r_shape, b_shape


def _get_rnn_model(nodes: list[onnx.NodeProto], x_shape, w_shape, r_shape, b_shape, seq_length=1,
                   batch_size=1):
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5
    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    graph = onnx.helper.make_graph(
        nodes,
        'RNN test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, [batch_size], sl_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    return onnx_model


@pytest.mark.parametrize(
    "seq_length, batch_size, input_size, hidden_size",
    [
        (3, 2, 50, 32),
        (12, 3, 4, 25),
        (40, 5, 200, 40),
    ])
def test_convert_simple_forward_rnn(seq_length: int, batch_size: int, input_size: int, hidden_size: int):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes(seq_length, batch_size, input_size, hidden_size)

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], direction='forward', hidden_size=hidden_size)
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=6.e-6)


@pytest.mark.parametrize(
    "seq_length, batch_size, input_size, hidden_size",
    [
        (15, 23, 42, 12),
        (2, 100, 100, 1),
        (2, 2, 2, 2),
    ])
def test_convert_full_forward_rnn(seq_length: int, batch_size: int, input_size: int, hidden_size: int):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes(seq_length, batch_size, input_size, hidden_size)

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model(
        [onnx.helper.make_node('RNN', ['x', 'w', 'r', 'b', 'sl'], ['y'], direction='forward', hidden_size=hidden_size)],
        x_shape, w_shape, r_shape, b_shape, seq_length, batch_size
    )
    executors.convert_run_compare(onnx_model, x_data, atol=5.e-6)


@pytest.mark.parametrize(
    "activations",
    [
        pytest.param(['Tanh'], id="Tanh"),
        pytest.param(['Relu'], id="Relu"),
    ])
def test_convert_forward_rnn_with_activations(activations: list[str]):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], direction='forward', hidden_size=16,
                              activations=activations)
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=5.e-6)


@pytest.mark.parametrize(
    "activations",
    [
        pytest.param(['Sigmoid'], id='Sigmoid'),
        pytest.param(['Invalid'], id='Invalid activation function'),
    ])
def test_convert_forward_rnn_with_unsupported_activations(activations: list[str]):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    np.random.seed(42)

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], direction='forward', hidden_size=16,
                              activations=activations)
    ], x_shape, w_shape, r_shape, b_shape)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_forward_rnn_with_channels_first_input():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1]),
        onnx.helper.make_node('RNN', ['y1', 'w', 'r'], ['y'], hidden_size=16, direction='forward')
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-6)


def test_convert_forward_rnn_with_channels_first_output():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y1'], hidden_size=16, direction='forward'),
        onnx.helper.make_node('MaxPool', ['y1'], ['y'], kernel_shape=[1, 1])
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-6)


def test_convert_forward_rnn_with_channels_first_input_and_output():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1]),
        onnx.helper.make_node('RNN', ['y1', 'w', 'r'], ['y2'], hidden_size=16, direction='forward'),
        onnx.helper.make_node('MaxPool', ['y2'], ['y'], kernel_shape=[1, 1])
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-6)


def test_convert_rnn_with_layout():
    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], hidden_size=16, layout=1)
    ], *_get_rnn_shapes())

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR


def test_convert_rnn_with_clip():
    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], hidden_size=16, clip=1.)
    ], *_get_rnn_shapes())

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_rnn_with_initial_h():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5
    sl_data = np.asarray([100] * 2, np.int32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('RNN', ['x', 'w', 'r', 'b', 'sl', 'ih'], ['y'], hidden_size=16)],
        'RNN test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, [2], sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, [1, 2, 16],
                                    np.broadcast_to(1., [1, 2, 16]).astype(np.float32)),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_rnn_with_2_outputs():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y1', 'y2'], hidden_size=16)],
        'RNN test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [
            onnx.helper.make_tensor_value_info('y1', TensorProto.FLOAT, ()),
            onnx.helper.make_tensor_value_info('y2', TensorProto.FLOAT, ()),
        ],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


# ------------------------------------------ REVERSE ------------------------------------------

@pytest.mark.parametrize(
    "seq_length, batch_size, input_size, hidden_size",
    [
        (3, 2, 50, 32),
        (12, 3, 4, 25),
        (40, 5, 200, 40),
    ])
def test_convert_simple_reverse_rnn(seq_length: int, batch_size: int, input_size: int, hidden_size: int):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes(seq_length, batch_size, input_size,
                                                         hidden_size)

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], direction='reverse', hidden_size=hidden_size)
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=5.e-6)


@pytest.mark.parametrize(
    "seq_length, batch_size, input_size, hidden_size",
    [
        (15, 23, 42, 12),
        (2, 100, 100, 1),
        (2, 2, 2, 2),
    ])
def test_convert_full_reverse_rnn(seq_length: int, batch_size: int, input_size: int, hidden_size: int):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes(seq_length, batch_size, input_size,
                                                         hidden_size)

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model(
        [onnx.helper.make_node('RNN', ['x', 'w', 'r', 'b', 'sl'], ['y'], direction='reverse', hidden_size=hidden_size)],
        x_shape, w_shape, r_shape, b_shape, seq_length, batch_size
    )
    executors.convert_run_compare(onnx_model, x_data, atol=5.e-6)


@pytest.mark.parametrize(
    "activations",
    [
        pytest.param(['Tanh'], id="Tanh"),
        pytest.param(['Relu'], id="Relu"),
    ])
def test_convert_reverse_rnn_with_activations(activations: list[str]):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], direction='reverse', hidden_size=16,
                              activations=activations)
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=5.e-6)


@pytest.mark.parametrize(
    "activations",
    [
        pytest.param(['Sigmoid'], id='Sigmoid'),
        pytest.param(['Invalid'], id='Invalid activation function'),
    ])
def test_convert_reverse_rnn_with_unsupported_activations(activations: list[str]):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    np.random.seed(42)

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], direction='reverse', hidden_size=16,
                              activations=activations)
    ], x_shape, w_shape, r_shape, b_shape)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_reverse_rnn_with_channels_first_input():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1]),
        onnx.helper.make_node('RNN', ['y1', 'w', 'r'], ['y'], hidden_size=16, direction='reverse')
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-6)


def test_convert_reverse_rnn_with_channels_first_output():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y1'], hidden_size=16, direction='reverse'),
        onnx.helper.make_node('MaxPool', ['y1'], ['y'], kernel_shape=[1, 1])
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-6)


def test_convert_reverse_rnn_with_channels_first_input_and_output():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1]),
        onnx.helper.make_node('RNN', ['y1', 'w', 'r'], ['y2'], hidden_size=16, direction='reverse'),
        onnx.helper.make_node('MaxPool', ['y2'], ['y'], kernel_shape=[1, 1])
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-6)


# ------------------------------------------ BIDIRECTIONAL ------------------------------------------

@pytest.mark.parametrize(
    "seq_length, batch_size, input_size, hidden_size",
    [
        (31, 41, 5, 9),
        (4, 2, 4, 2),
        (12, 34, 5, 6),
    ])
def test_convert_simple_bidirectional_rnn(seq_length: int, batch_size: int, input_size: int, hidden_size: int):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes(seq_length, batch_size, input_size,
                                                         hidden_size, 2)

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], direction='bidirectional', hidden_size=hidden_size)
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=5.e-6)


@pytest.mark.parametrize(
    "seq_length, batch_size, input_size, hidden_size",
    [
        (15, 215, 21, 2),
        (7, 8, 9, 10),
        (1, 1, 1, 1),
    ])
def test_convert_full_bidirectional_rnn(seq_length: int, batch_size: int, input_size: int, hidden_size: int):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes(seq_length, batch_size, input_size,
                                                         hidden_size, 2)

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model(
        [onnx.helper.make_node('RNN', ['x', 'w', 'r', 'b', 'sl'], ['y'], direction='bidirectional',
                               hidden_size=hidden_size)],
        x_shape, w_shape, r_shape, b_shape, seq_length, batch_size
    )
    executors.convert_run_compare(onnx_model, x_data, atol=5.e-6)


@pytest.mark.parametrize(
    "activations",
    [
        pytest.param(['Tanh', 'Tanh'], id="Tanh"),
        pytest.param(['Relu', 'Relu'], id="Relu"),
    ])
def test_convert_bidirectional_rnn_with_activations(activations: list[str]):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes(num_directions=2)

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], direction='bidirectional', hidden_size=16,
                              activations=activations)
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=5.e-6)


@pytest.mark.parametrize(
    "activations",
    [
        pytest.param(['Sigmoid', 'Sigmoid'], id='Sigmoid'),
        pytest.param(['Tanh', 'Relu'], id='Different fw and bw activations'),
    ])
def test_convert_bidirectional_rnn_with_unsupported_activations(activations: list[str]):
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes(num_directions=2)

    np.random.seed(42)

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], direction='bidirectional', hidden_size=16,
                              activations=activations)
    ], x_shape, w_shape, r_shape, b_shape)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_bidirectional_rnn_with_channels_first_input():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes(num_directions=2)

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1]),
        onnx.helper.make_node('RNN', ['y1', 'w', 'r'], ['y'], hidden_size=16, direction='bidirectional')
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-6)


def test_convert_bidirectional_rnn_with_channels_first_output():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes(num_directions=2)

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y1'], hidden_size=16, direction='bidirectional'),
        onnx.helper.make_node('MaxPool', ['y1'], ['y'], kernel_shape=[1, 1])
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-6)


def test_convert_bidirectional_rnn_with_channels_first_input_and_output():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes(num_directions=2)

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1]),
        onnx.helper.make_node('RNN', ['y1', 'w', 'r'], ['y2'], hidden_size=16, direction='bidirectional'),
        onnx.helper.make_node('MaxPool', ['y2'], ['y'], kernel_shape=[1, 1])
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-6)


def test_convert_simple_rnn_with_unused_extra_outputs():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    np.random.seed(42)
    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    onnx_model = _get_rnn_model([
        onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y', 'y_h'], hidden_size=16),
    ], x_shape, w_shape, r_shape, b_shape)

    executors.convert_run_compare(onnx_model, x_data, atol=9.e-7)


def test_convert_rnn_with_dynamic_but_convertible_initial_h():
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()
    initial_h_shape = [1, 2, 16]

    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5
    sl_data = np.asarray([100] * 2, np.int32)

    np.random.seed(42)
    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('ConstantOfShape', ['initial_h_shape'], ['initial_h'],
                                  value=onnx.helper.make_tensor('initial_h', TensorProto.FLOAT, [1], [0.])),
            onnx.helper.make_node('RNN', ['x', 'w', 'r', 'b', 'sl', 'initial_h'], ['y'], hidden_size=16)
        ],
        'RNN test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, [2], sl_data),
            onnx.helper.make_tensor('initial_h_shape', TensorProto.INT64, [3], initial_h_shape),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    executors.convert_run_compare(onnx_model, x_data, atol=9.e-7)


def test_convert_rnn__invalid_type():
    type_ = TensorProto.DOUBLE
    x_shape, w_shape, r_shape, b_shape = _get_rnn_shapes()

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('RNN', ['x', 'w', 'r'], ['y'], hidden_size=16, clip=1.)],
        'RNN test',
        [onnx.helper.make_tensor_value_info('x', type_, x_shape),
         onnx.helper.make_tensor_value_info('w', type_, w_shape),
         onnx.helper.make_tensor_value_info('r', type_, r_shape),
         onnx.helper.make_tensor_value_info('b', type_, b_shape),
         onnx.helper.make_tensor_value_info('sl', TensorProto.INT32, [2])],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)
