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
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type, name_for_onnx_type
from tests import executors


@pytest.mark.parametrize(
    "equation",
    [
        # 1 input
        'ij->ij',
        'ijk->j',

        # 2 2D inputs, 2D output
        'ij,jk->ik',
        'ij,jk->kj',
        'ik,kj->',

        # 2 2D inputs, 3D output
        'ij,jk->jik',
        'ij,jk->kji',
        'ij,jk->ikj',

        # 2 3D inputs, 4D output
        'ijk,jkl->ijkl',
        'ijk,lki->klij',

        # Mixed ranks
        'ij,jkl->ilk',
        'ij,jkl->lk',
        'lijk,jil->ki',
        'ij,kl->klij',
        'jilk,l->iljk',
        'jilk,klij->k',

        # Recurring dimensions
        'ii,i->i',
        'ikki,kikj->ji',
        'ikki,kikj->',
        'jjj,kk->k'
    ])
def test_convert_einsum__equations(equation: str):
    inputs = equation.split('->')[0].split(',')
    input_names = [f'input_{i}' for i in range(len(inputs))]

    shape_values = {
        'i': 4,
        'j': 6,
        'k': 8,
        'l': 10
    }
    shapes: list[tuple[str, list[int]]] = [
        (name, [shape_values[dim] for dim in inpt]) for name, inpt in zip(input_names, inputs)
    ]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Einsum', input_names, ['y'], equation=equation)],
        'Einsum test',
        [
            onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in shapes
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = {
        i: (np.random.random(shape).astype(np.float32) - 0.5) * 10 for i, (_, shape) in enumerate(shapes)
    }
    executors.convert_run_compare(onnx_model, input_data,
                                  conversion_config=ConversionConfig({'allow_select_ops': True}))


@pytest.mark.parametrize(
    "equation",
    [
        'ij,jk',
        'kilk,ijlk',
        'j,kjl',
        'lijk,ill',
        'lij,ljk'
    ])
def test_convert_einsum__no_output_in_equation(equation: str):
    inputs = equation.split('->')[0].split(',')
    input_names = [f'input_{i}' for i in range(len(inputs))]

    shape_values = {
        'i': 4,
        'j': 6,
        'k': 8,
        'l': 10
    }
    shapes: list[tuple[str, list[int]]] = [
        (name, [shape_values[dim] for dim in inpt]) for name, inpt in zip(input_names, inputs)
    ]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Einsum', input_names, ['y'], equation=equation)],
        'Einsum test',
        [
            onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in shapes
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = {
        i: (np.random.random(shape).astype(np.float32) - 0.5) * 10 for i, (_, shape) in enumerate(shapes)
    }
    executors.convert_run_compare(onnx_model, input_data,
                                  conversion_config=ConversionConfig({'allow_select_ops': True}))


def test_convert_einsum__spaces_in_equation():
    i, j, k = 2, 4, 6
    shapes = [[i, j], [j, k]]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Einsum', ['a', 'b'], ['y'], equation='   i  j ,   j k -   >  i   k ')],
        'Einsum test',
        [
            onnx.helper.make_tensor_value_info('a', TensorProto.FLOAT, shapes[0]),
            onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT, shapes[1])
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = {
        i: (np.random.random(shape).astype(np.float32) - 0.5) * 10 for i, shape in enumerate(shapes)
    }
    executors.convert_run_compare(onnx_model, input_data,
                                  conversion_config=ConversionConfig({'allow_select_ops': True}))


def test_convert_einsum__no_select_ops():
    i, j, k = 2, 4, 6
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Einsum', ['a', 'b'], ['y'], equation='ij,jk->ik')],
        'Einsum test',
        [
            onnx.helper.make_tensor_value_info('a', TensorProto.FLOAT, [i, j]),
            onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT, [j, k])
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig()
    config.allow_select_ops = False

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=config)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert '--allow-select-ops' in logger.conversion_log.get_node_error_message(0)


def test_convert_einsum__3_inputs():
    i, j, k = 2, 4, 6
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Einsum', ['a', 'b', 'c'], ['y'], equation='ij,jk,ki->ijk')],
        'Einsum test',
        [
            onnx.helper.make_tensor_value_info('a', TensorProto.FLOAT, [i, j]),
            onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT, [j, k]),
            onnx.helper.make_tensor_value_info('c', TensorProto.FLOAT, [k, i])
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=ConversionConfig({'allow_select_ops': True}))
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'Conversion of ONNX `Einsum` with 3 inputs is not possible.' in logger.conversion_log.get_node_error_message(
        0)


def test_convert_einsum__ellipsis():
    i, j, k = 2, 4, 6
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Einsum', ['a', 'b'], ['y'], equation='...ij,...jk->...ik')],
        'Einsum test',
        [
            onnx.helper.make_tensor_value_info('a', TensorProto.FLOAT, [i, j]),
            onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT, [j, k])
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=ConversionConfig({'allow_select_ops': True}))

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED
    assert 'ellipsis' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize(
    "_type",
    [
        TensorProto.FLOAT, TensorProto.DOUBLE,
        TensorProto.INT32, TensorProto.INT64,
    ])
def test_convert_einsum__types(_type: TensorProto.DataType):
    i, j, k = 2, 4, 6
    shapes = [[i, j], [j, k]]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Einsum', ['a', 'b'], ['y'], equation='ij,jk->ik')],
        'Einsum test',
        [
            onnx.helper.make_tensor_value_info('a', _type, shapes[0]),
            onnx.helper.make_tensor_value_info('b', _type, shapes[1])
        ],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np_type = to_numpy_type(_type)
    np.random.seed(42)
    input_data = {
        i: ((np.random.random(shape) - 0.5) * 10).astype(np_type) for i, shape in enumerate(shapes)
    }
    executors.convert_run_compare(onnx_model, input_data,
                                  conversion_config=ConversionConfig({'allow_select_ops': True}))


def test_convert_einsum__unsupported_type():
    i, j, k = 2, 4, 6
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Einsum', ['a', 'b'], ['y'], equation='ij,jk->ik')],
        'Einsum test',
        [
            onnx.helper.make_tensor_value_info('a', TensorProto.FLOAT16, [i, j]),
            onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT16, [j, k])
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT16, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=ConversionConfig({'allow_select_ops': True}))

    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT16' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize(
    "a_shape, b_shape, equation",
    [
        pytest.param([2, 4, 6, 8], [8, 6], 'ijkl,lk->jik', id='second input 2D'),
        pytest.param([2, 4, 6, 8], [8, 4, 2], 'ijkl,lji->k', id='second input 3D'),
        pytest.param([2, 4, 6, 8], [4, 8, 2, 6], 'ijkl,jlik->jk', id='second input 4D'),
    ])
def test_convert_einsum__channels_first_inputs__dynamic(a_shape: list[int], b_shape: list[int], equation: str):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['a'], ['a1'], kernel_shape=[1] * (len(a_shape) - 2)),
            onnx.helper.make_node('Einsum', ['a1', 'b'], ['y'], equation=equation)
        ],
        'Einsum test',
        [
            onnx.helper.make_tensor_value_info('a', TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    shapes = [a_shape, b_shape]
    input_data = {
        i: ((np.random.random(shape) - 0.5) * 10).astype(np.float32) for i, shape in enumerate(shapes)
    }
    executors.convert_run_compare(onnx_model, input_data,
                                  conversion_config=ConversionConfig({'allow_select_ops': True}))


def test_convert_einsum__channels_first_inputs__static():
    i, j, k, l = 2, 4, 6, 8
    a_shape = [i, j, k, l]
    b_shape = [j, k, k]
    equation = 'ijkl,jkk->li'

    np.random.seed(42)
    shapes = [a_shape, b_shape]
    input_data = {
        i: ((np.random.random(shape) - 0.5) * 10).astype(np.float32) for i, shape in enumerate(shapes)
    }

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('MaxPool', ['a'], ['a1'], kernel_shape=[1] * (len(a_shape) - 2)),
            onnx.helper.make_node('Einsum', ['a1', 'b'], ['y'], equation=equation)
        ],
        'Einsum test',
        [],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('a', TensorProto.FLOAT, a_shape, input_data[0]),
            onnx.helper.make_tensor('b', TensorProto.FLOAT, b_shape, input_data[1])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, {}, conversion_config=ConversionConfig({'allow_select_ops': True}))


def test_convert_einsum__channels_first_output():
    i, j, k, l = 2, 4, 6, 8
    a_shape = [i, j, k, l]
    b_shape = [j, k, k]
    equation = 'ijkl,jkk->lijk'

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Einsum', ['a', 'b'], ['y1'], equation=equation),
            onnx.helper.make_node('MaxPool', ['y1'], ['y'], kernel_shape=[1, 1]),
        ],
        'Einsum test',
        [
            onnx.helper.make_tensor_value_info('a', TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info('b', TensorProto.FLOAT, b_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    shapes = [a_shape, b_shape]
    input_data = {
        i: ((np.random.random(shape) - 0.5) * 10).astype(np.float32) for i, shape in enumerate(shapes)
    }
    executors.convert_run_compare(onnx_model, input_data,
                                  conversion_config=ConversionConfig({'allow_select_ops': True}))


@pytest.mark.parametrize("_type", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_einsum__quantized(_type: TensorProto.DataType):
    shape = [3, 14, 15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Einsum', ['x1', 'x1'], ['y'], equation='ijk,ijk->i')
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', _type, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [1.42]),
            onnx.helper.make_tensor('zp', _type, [1], [12])
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(1) == logger.Code.NOT_IMPLEMENTED
