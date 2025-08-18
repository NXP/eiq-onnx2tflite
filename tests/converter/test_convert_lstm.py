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


@pytest.mark.parametrize(
    "seq_length, batch_size, input_size, hidden_size",
    [
        (3, 2, 50, 32),
        (12, 3, 4, 25),
        (40, 5, 200, 40),
    ])
def test_convert_simple_forward_lstm(seq_length: int, batch_size: int, input_size: int, hidden_size: int):
    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]

    np.random.seed(42)

    lin_space = np.linspace(0, 6.28, np.prod(x_shape).item())
    x_data = (np.sin(lin_space).reshape(x_shape).astype(np.float32) + noise(x_shape)) * 5.

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y'], hidden_size=hidden_size, direction='forward')],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-7)


@pytest.mark.parametrize(
    "seq_length, batch_size, input_size, hidden_size",
    [
        (3, 2, 50, 32),
        (12, 3, 4, 25),
        (40, 5, 200, 40),
    ])
def test_convert_simple_forward_lstm_with_random_data(seq_length: int, batch_size: int, input_size: int,
                                                      hidden_size: int):
    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5

    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5

    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y'], hidden_size=hidden_size, direction='forward')],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=4e-7)


@pytest.mark.parametrize(
    "activations",
    [
        pytest.param(['Sigmoid', 'Tanh', 'Tanh'], id="Default activation functions."),
        pytest.param(['Sigmoid', 'Relu', 'Relu'], id="Sigmoid + Relu."),
    ])
def test_convert_forward_lstm_with_activations(activations: list[str]):
    seq_length = 100
    batch_size = 2
    input_size = 100
    hidden_size = 16

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]

    np.random.seed(42)

    lin_space = np.linspace(0, 6.28, np.prod(x_shape).item())
    x_data = (np.sin(lin_space).reshape(x_shape).astype(np.float32) * 5. + noise(x_shape)) * 5.

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y'], hidden_size=hidden_size, direction='forward',
                               activations=activations)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-6)


@pytest.mark.parametrize(
    "activations",
    [
        pytest.param(['Sigmoid', 'Tanh', 'Relu'], id="Different 'g' and 'h'."),
        pytest.param(['Tanh', 'Tanh', 'Tanh'], id="Unsupported 'f'."),
        pytest.param(['Sigmoid', 'Sigmoid', 'Sigmoid'], id="Unsupported 'g'/'h'."),
    ])
def test_convert_forward_lstm_with_unsupported_activations(activations: list[str]):
    seq_length = 100
    batch_size = 2
    input_size = 100
    hidden_size = 16

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]

    np.random.seed(42)

    lin_space = np.linspace(0, 6.28, np.prod(x_shape).item())
    x_data = (np.sin(lin_space).reshape(x_shape).astype(np.float32) * 5. + noise(x_shape)) * 5.

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y'], hidden_size=hidden_size, direction='forward',
                               activations=activations)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_forward_lstm_with_clipping():
    seq_length = 3
    batch_size = 2
    input_size = 50
    hidden_size = 32

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]

    np.random.seed(42)

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y'], hidden_size=hidden_size, clip=1.0)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_forward_lstm_with_input_forget():
    seq_length = 100
    batch_size = 2
    input_size = 100
    hidden_size = 16

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]

    np.random.seed(42)

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y'], hidden_size=hidden_size, input_forget=1)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_lstm_with_layout():
    seq_length = 16
    batch_size = 2
    input_size = 100
    hidden_size = 100

    x_shape = [batch_size, seq_length, input_size]
    w_shape = [2, 4 * hidden_size, input_size]
    r_shape = [2, 4 * hidden_size, hidden_size]
    b_shape = [2, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [2, batch_size, hidden_size]
    p_shape = [2, 3 * hidden_size]

    np.random.seed(42)

    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = np.zeros(initial_c_shape, np.float32)

    p_data = noise(p_shape) * 10.

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'ih', 'ic', 'p'], ['y'],
                               hidden_size=hidden_size, direction='bidirectional', layout=1)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
            onnx.helper.make_tensor('p', TensorProto.FLOAT, p_shape, p_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_OPERATOR


def test_convert_forward_lstm_with_bias():
    seq_length = 16
    batch_size = 2
    input_size = 100
    hidden_size = 100

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]

    np.random.seed(42)

    lin_space = np.linspace(0, 6.28, np.prod(x_shape).item())
    x_data = (np.sin(lin_space).reshape(x_shape).astype(np.float32) + noise(x_shape)) * 5.

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    b_data = np.asarray([[i + 1] * hidden_size for i in range(8)]).flatten() / 8. - 0.5 + noise([8 * hidden_size])

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b'], ['y'], hidden_size=hidden_size)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=3.e-7)


def test_convert_forward_lstm_with_uniform_sequence_lens():
    seq_length = 16
    batch_size = 2
    input_size = 100
    hidden_size = 100

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]

    np.random.seed(42)

    lin_space = np.linspace(0, 6.28, np.prod(x_shape).item())
    x_data = (np.sin(lin_space).reshape(x_shape).astype(np.float32) + noise(x_shape)) * 5.

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    b_data = np.asarray([[i + 1] * hidden_size for i in range(8)]).flatten() / 8. - 0.5 + noise([8 * hidden_size])

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl'], ['y'], hidden_size=hidden_size)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=3.e-7)


def test_convert_forward_lstm_with_inconvertible_sequence_lens():
    seq_length = 16
    batch_size = 2
    input_size = 100
    hidden_size = 100

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]

    np.random.seed(42)

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    b_data = np.asarray([[i + 1] * hidden_size for i in range(8)]).flatten() / 8. - 0.5 + noise([8 * hidden_size])

    sl_data = np.asarray([seq_length] * batch_size, np.int32)
    sl_data[-2] = 42  # Change the length of 1 sequence

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl'], ['y'], hidden_size=hidden_size)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_lstm_with_inconvertible_initial_h():
    seq_length = 16
    batch_size = 2
    input_size = 100
    hidden_size = 100

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = [1, batch_size, hidden_size]

    np.random.seed(42)

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    b_data = np.asarray([[i + 1] * hidden_size for i in range(8)]).flatten() / 8. - 0.5 + noise([8 * hidden_size])

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = noise(initial_h_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'ih'], ['y'], hidden_size=hidden_size)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_lstm_with_inconvertible_initial_c():
    seq_length = 16
    batch_size = 2
    input_size = 100
    hidden_size = 100

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [1, batch_size, hidden_size]

    np.random.seed(42)

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    b_data = np.asarray([[i + 1] * hidden_size for i in range(8)]).flatten() / 8. - 0.5 + noise([8 * hidden_size])

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = noise(initial_c_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'ih', 'ic'], ['y'], hidden_size=hidden_size)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_forward_lstm_with_zero_initial_h_and_c():
    seq_length = 16
    batch_size = 2
    input_size = 100
    hidden_size = 100

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [1, batch_size, hidden_size]

    np.random.seed(42)

    lin_space = np.linspace(0, 6.28, np.prod(x_shape).item())
    x_data = (np.sin(lin_space).reshape(x_shape).astype(np.float32) + noise(x_shape)) * 5.

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    b_data = np.asarray([[i + 1] * hidden_size for i in range(8)]).flatten() / 8. - 0.5 + noise([8 * hidden_size])

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = np.zeros(initial_c_shape, np.float32)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'ih', 'ic'], ['y'], hidden_size=hidden_size)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=3.e-7)


def test_convert_forward_lstm_with_peephole_weights():
    seq_length = 16
    batch_size = 2
    input_size = 100
    hidden_size = 100

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [1, batch_size, hidden_size]
    p_shape = [1, 3 * hidden_size]

    np.random.seed(42)

    lin_space = np.linspace(0, 6.28, np.prod(x_shape).item())
    x_data = (np.sin(lin_space).reshape(x_shape).astype(np.float32) + noise(x_shape)) * 5.

    # Use different range of weights for each group of connections. (input to input, input to output...)
    w_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 4. - 0.5
    w_data = np.broadcast_to(w_data, w_shape) + noise(w_shape)

    r_data = np.asarray([[i + 1] * hidden_size for i in range(4)]).reshape([1, 4 * hidden_size, 1]) / 8. - 0.25
    r_data = np.broadcast_to(r_data, r_shape) + noise(r_shape)

    b_data = np.asarray([[i + 1] * hidden_size for i in range(8)]).flatten() / 8. - 0.5 + noise([8 * hidden_size])

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = np.zeros(initial_c_shape, np.float32)

    p_data = noise(p_shape) * 10.

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'ih', 'ic', 'p'], ['y'], hidden_size=hidden_size)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
            onnx.helper.make_tensor('p', TensorProto.FLOAT, p_shape, p_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=1.2e-7)


@pytest.mark.parametrize(
    "seq_length, batch_size, input_size, hidden_size",
    [
        (3, 4, 50, 32),
        (12, 3, 4, 25),
        (40, 5, 200, 40),
    ])
def test_convert_simple_bidirectional_lstm(seq_length: int, batch_size: int, input_size: int, hidden_size: int):
    x_shape = [seq_length, batch_size, input_size]
    w_shape = [2, 4 * hidden_size, input_size]
    r_shape = [2, 4 * hidden_size, hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y'], hidden_size=hidden_size, direction='bidirectional')],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=4e-7)


@pytest.mark.parametrize(
    "activations",
    [
        pytest.param(['Sigmoid', 'Tanh', 'Tanh'] * 2, id="Default activation functions."),
        pytest.param(['Sigmoid', 'Relu', 'Relu'] * 2, id="Sigmoid + Relu."),
    ])
def test_convert_bidirectional_lstm_with_activations(activations: list[str]):
    seq_length = 100
    batch_size = 2
    input_size = 100
    hidden_size = 16

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [2, 4 * hidden_size, input_size]
    r_shape = [2, 4 * hidden_size, hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y'], hidden_size=hidden_size, direction='bidirectional',
                               activations=activations)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-6)


@pytest.mark.parametrize(
    "activations",
    [
        pytest.param(['Sigmoid', 'Tanh', 'Relu'] * 2, id="Different 'g' and 'h'."),
        pytest.param(['Tanh', 'Tanh', 'Tanh'] * 2, id="Unsupported 'f'."),
        pytest.param(['Sigmoid', 'Sigmoid', 'Sigmoid'] * 2, id="Unsupported 'g'/'h'."),
        pytest.param(['Sigmoid', 'Tanh', 'Tanh', 'Tanh', 'Tanh', 'Tanh'], id="Different 'f' in forward and backward."),
        pytest.param(['Sigmoid', 'Tanh', 'Tanh', 'Sigmoid', 'Relu', 'Tanh'],
                     id="Different 'g' in forward and backward."),
        pytest.param(['Sigmoid', 'Tanh', 'Tanh', 'Sigmoid', 'Tanh', 'Relu'],
                     id="Different 'h' in forward and backward."),
    ])
def test_convert_bidirectional_lstm_with_unsupported_activations(activations: list[str]):
    seq_length = 100
    batch_size = 2
    input_size = 100
    hidden_size = 16

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [2, 4 * hidden_size, input_size]
    r_shape = [2, 4 * hidden_size, hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y'], hidden_size=hidden_size, direction='bidirectional',
                               activations=activations)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_bidirectional_lstm_with_bias():
    seq_length = 16
    batch_size = 2
    input_size = 100
    hidden_size = 100

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [2, 4 * hidden_size, input_size]
    r_shape = [2, 4 * hidden_size, hidden_size]
    b_shape = [2, 8 * hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b'], ['y'], hidden_size=hidden_size,
                               direction='bidirectional')],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=6.e-7)


def test_convert_bidirectional_lstm_with_peephole_weights():
    seq_length = 16
    batch_size = 2
    input_size = 100
    hidden_size = 100

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [2, 4 * hidden_size, input_size]
    r_shape = [2, 4 * hidden_size, hidden_size]
    b_shape = [2, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [2, batch_size, hidden_size]
    p_shape = [2, 3 * hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = np.zeros(initial_c_shape, np.float32)

    p_data = noise(p_shape) * 10.

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'ih', 'ic', 'p'], ['y'],
                               hidden_size=hidden_size, direction='bidirectional')],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
            onnx.helper.make_tensor('p', TensorProto.FLOAT, p_shape, p_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=7e-7)


@pytest.mark.parametrize(
    "seq_length, batch_size, input_size, hidden_size",
    [
        (3, 4, 50, 32),
        (12, 3, 4, 25),
        (40, 5, 200, 40),
    ])
def test_convert_simple_reverse_lstm(seq_length: int, batch_size: int, input_size: int, hidden_size: int):
    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y'], hidden_size=hidden_size, direction='reverse')],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=4e-7)


@pytest.mark.parametrize(
    "seq_length, batch_size, input_size, hidden_size",
    [
        (2, 48, 1, 17),
        (50, 2, 99, 1),
        (1, 1, 1, 1),
        (42, 42, 42, 42),
    ])
def test_convert_full_reverse_lstm(seq_length: int, batch_size: int, input_size: int, hidden_size: int):
    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [1, batch_size, hidden_size]
    p_shape = [1, 3 * hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = np.zeros(initial_c_shape, np.float32)

    p_data = noise(p_shape) * 10.

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'ih', 'ic', 'p'], ['y'],
                               hidden_size=hidden_size, direction='reverse')],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
            onnx.helper.make_tensor('p', TensorProto.FLOAT, p_shape, p_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=6e-7)


def test_convert_forward_lstm_with_channels_first_input():
    seq_length = 2
    batch_size = 16
    input_size = 32
    hidden_size = 50

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [1, batch_size, hidden_size]
    p_shape = [1, 3 * hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = np.zeros(initial_c_shape, np.float32)

    p_data = noise(p_shape) * 10.
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1]),
            onnx.helper.make_node('LSTM', ['y1', 'w', 'r', 'b', 'sl', 'ih', 'ic', 'p'], ['y'], hidden_size=hidden_size)
        ],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
            onnx.helper.make_tensor('p', TensorProto.FLOAT, p_shape, p_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-7)


def test_convert_forward_lstm_with_channels_first_output():
    seq_length = 2
    batch_size = 16
    input_size = 32
    hidden_size = 50

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [1, batch_size, hidden_size]
    p_shape = [1, 3 * hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = np.zeros(initial_c_shape, np.float32)

    p_data = noise(p_shape) * 10.
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'ih', 'ic', 'p'], ['y1'], hidden_size=hidden_size),
            onnx.helper.make_node('MaxPool', ['y1'], ['y'], kernel_shape=[1, 1])
        ],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
            onnx.helper.make_tensor('p', TensorProto.FLOAT, p_shape, p_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-7)


def test_convert_bidirectional_lstm_with_channels_first_input():
    seq_length = 3
    batch_size = 16
    input_size = 32
    hidden_size = 50

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [2, 4 * hidden_size, input_size]
    r_shape = [2, 4 * hidden_size, hidden_size]
    b_shape = [2, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [2, batch_size, hidden_size]
    p_shape = [2, 3 * hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = np.zeros(initial_c_shape, np.float32)

    p_data = noise(p_shape) * 10.
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1]),
            onnx.helper.make_node('LSTM', ['y1', 'w', 'r', 'b', 'sl', 'ih', 'ic', 'p'], ['y'], hidden_size=hidden_size,
                                  direction='bidirectional')
        ],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
            onnx.helper.make_tensor('p', TensorProto.FLOAT, p_shape, p_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-7)


def test_convert_bidirectional_lstm_with_channels_first_output():
    seq_length = 2
    batch_size = 16
    input_size = 32
    hidden_size = 50

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [2, 4 * hidden_size, input_size]
    r_shape = [2, 4 * hidden_size, hidden_size]
    b_shape = [2, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [2, batch_size, hidden_size]
    p_shape = [2, 3 * hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = np.zeros(initial_c_shape, np.float32)

    p_data = noise(p_shape) * 10.
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'ih', 'ic', 'p'], ['y1'], hidden_size=hidden_size,
                                  direction='bidirectional'),
            onnx.helper.make_node('MaxPool', ['y1'], ['y'], kernel_shape=[1, 1])
        ],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
            onnx.helper.make_tensor('p', TensorProto.FLOAT, p_shape, p_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-7)


def test_convert_reverse_lstm_with_channels_first_input():
    seq_length = 2
    batch_size = 16
    input_size = 32
    hidden_size = 50

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [1, batch_size, hidden_size]
    p_shape = [1, 3 * hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = np.zeros(initial_c_shape, np.float32)

    p_data = noise(p_shape) * 10.
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['x'], ['y1'], kernel_shape=[1]),
            onnx.helper.make_node('LSTM', ['y1', 'w', 'r', 'b', 'sl', 'ih', 'ic', 'p'], ['y'], hidden_size=hidden_size,
                                  direction='reverse')
        ],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
            onnx.helper.make_tensor('p', TensorProto.FLOAT, p_shape, p_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-7)


def test_convert_reverse_lstm_with_channels_first_output():
    seq_length = 2
    batch_size = 16
    input_size = 32
    hidden_size = 50

    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [1, batch_size, hidden_size]
    p_shape = [1, 3 * hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5

    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    initial_h_data = np.zeros(initial_h_shape, np.float32)
    initial_c_data = np.zeros(initial_c_shape, np.float32)

    p_data = noise(p_shape) * 10.
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'ih', 'ic', 'p'], ['y1'], hidden_size=hidden_size,
                                  direction='reverse'),
            onnx.helper.make_node('MaxPool', ['y1'], ['y'], kernel_shape=[1, 1])
        ],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data),
            onnx.helper.make_tensor('ih', TensorProto.FLOAT, initial_h_shape, initial_h_data),
            onnx.helper.make_tensor('ic', TensorProto.FLOAT, initial_c_shape, initial_c_data),
            onnx.helper.make_tensor('p', TensorProto.FLOAT, p_shape, p_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=2e-7)


def test_convert_simple_lstm_with_unused_extra_outputs():
    seq_length, batch_size, input_size, hidden_size = 5, 10, 15, 20
    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y', 'y_h', 'y_c'], hidden_size=hidden_size)],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=8.e-8)


def test_convert_lstm_with_dynamic_but_convertible_initial_h():
    seq_length, batch_size, input_size, hidden_size = 5, 10, 15, 20
    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = [1, batch_size, hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5
    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('ConstantOfShape', ['shape'], ['initial_h'],
                                  value=onnx.helper.make_tensor('zero', TensorProto.FLOAT, [1], [0.])),
            onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'initial_h'], ['y'], hidden_size=hidden_size)
        ],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('shape', TensorProto.INT64, [3], initial_h_shape),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data)
        ],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=9.e-8)


def test_convert_lstm_with_dynamic_but_convertible_initial_c():
    seq_length, batch_size, input_size, hidden_size = 5, 10, 15, 20
    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]
    b_shape = [1, 8 * hidden_size]
    sl_shape = [batch_size]
    initial_h_shape = initial_c_shape = [1, batch_size, hidden_size]

    np.random.seed(42)

    x_data = np.random.rand(*x_shape).astype(np.float32) - 0.5
    w_data = np.random.rand(*w_shape).astype(np.float32) - 0.5
    r_data = np.random.rand(*r_shape).astype(np.float32) - 0.5
    b_data = np.random.rand(*b_shape).astype(np.float32) - 0.5
    sl_data = np.asarray([seq_length] * batch_size, np.int32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('ConstantOfShape', ['initial_h_shape'], ['initial_h'],
                                  value=onnx.helper.make_tensor('zero', TensorProto.FLOAT, [1], [0.])),
            onnx.helper.make_node('ConstantOfShape', ['initial_c_shape'], ['initial_c'],
                                  value=onnx.helper.make_tensor('zero', TensorProto.FLOAT, [1], [0.])),
            onnx.helper.make_node('LSTM', ['x', 'w', 'r', 'b', 'sl', 'initial_h', 'initial_c'], ['y'],
                                  hidden_size=hidden_size)
        ],
        'LSTM test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('w', TensorProto.FLOAT, w_shape, w_data),
            onnx.helper.make_tensor('r', TensorProto.FLOAT, r_shape, r_data),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, b_data),
            onnx.helper.make_tensor('initial_h_shape', TensorProto.INT64, [3], initial_h_shape),
            onnx.helper.make_tensor('initial_c_shape', TensorProto.INT64, [3], initial_c_shape),
            onnx.helper.make_tensor('sl', TensorProto.INT32, sl_shape, sl_data)
        ],
    )

    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, x_data, atol=9.e-8)


def test_convert_lstm__invalid_type():
    type_ = TensorProto.INT8

    seq_length, batch_size, input_size, hidden_size = 2, 3, 4, 5
    x_shape = [seq_length, batch_size, input_size]
    w_shape = [1, 4 * hidden_size, input_size]
    r_shape = [1, 4 * hidden_size, hidden_size]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('LSTM', ['x', 'w', 'r'], ['y'], hidden_size=hidden_size, direction='forward')],
        'LSTM test',
        [
            onnx.helper.make_tensor_value_info('x', type_, x_shape),
            onnx.helper.make_tensor_value_info('w', type_, w_shape),
            onnx.helper.make_tensor_value_info('r', type_, r_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', type_, ())]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.INVALID_ONNX_MODEL
    assert 'INT8' in logger.conversion_log.get_node_error_message(0)
