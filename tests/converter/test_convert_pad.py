#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig, SkipShapeInferenceConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize("type_", [
    TensorProto.FLOAT, TensorProto.UINT8, TensorProto.INT8, TensorProto.INT32, TensorProto.INT64
], ids=name_for_onnx_type)
def test_convert_pad__types(type_: TensorProto.DataType):
    shape = [2, 3, 4, 5]
    pads = [1] * 8
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads', ''], ['o'], mode='constant')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", type_, shape)],
        [onnx.helper.make_tensor_value_info("o", type_, ())],
        [onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads)]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = (np.random.random(shape) * 1).astype(to_numpy_type(type_))

    config = ConversionConfig({"ignore_opset_version": True})
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


def test_convert_pad__unsupported_type():
    type_ = TensorProto.DOUBLE

    shape = [2, 3, 4, 5]
    pads = [1] * 8
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads', ''], ['o'], mode='constant')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", type_, shape)],
        [onnx.helper.make_tensor_value_info("o", type_, ())],
        [onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads)]
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig({"ignore_opset_version": True})
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=config)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize(
    "shape, pads",
    [
        pytest.param([42], [1, 17], id="1D"),
        pytest.param([10, 20], [0, 2, 4, 1], id="2D"),
        pytest.param([10, 20, 4], [0, 2, 4, 1, 0, 2], id="3D"),
        pytest.param([10, 5, 3, 4], [0, 2, 0, 0, 4, 1, 0, 2], id="4D"),
        pytest.param([7, 4, 5, 3, 4], [0, 2, 0, 1, 19, 0, 4, 1, 0, 2], id="5D"),
    ])
def test_convert_pad_v2_with_constant_zero(shape: list[int], pads: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x'], ['o'], mode='constant', pads=pads, value=0.0)],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 2)])

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    config = ConversionConfig({"ignore_opset_version": True})
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


def test_convert_pad_v2_with_negative_pads():
    shape = [13, 37]
    pads = [0, 0, 0, -1]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x'], ['o'], mode='constant', pads=pads, value=0.0)],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 2)])

    config = ConversionConfig({"ignore_opset_version": True})
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=config)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize(
    "shape, pads, value",
    [
        pytest.param([42], [1, 17], 1.2, id="1D"),
        pytest.param([10, 20], [0, 2, 4, 1], 2.3, id="2D"),
        pytest.param([10, 20, 4], [0, 2, 4, 1, 0, 2], 3.4, id="3D"),
        pytest.param([10, 5, 3, 4], [0, 2, 0, 0, 4, 1, 0, 2], 4.5, id="4D"),
        pytest.param([7, 4, 5, 3, 4], [0, 2, 0, 1, 19, 0, 4, 1, 0, 2], 5.6, id="5D"),
    ])
def test_convert_pad_v2_with_constant(shape: list[int], pads: list[int], value: float):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x'], ['o'], mode='constant', pads=pads, value=value)],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 2)])

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    config = ConversionConfig({"ignore_opset_version": True})
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


@pytest.mark.parametrize(
    "shape, pads",
    [
        pytest.param([10, 20, 4], [0, 2, 4, 1, 0, 2], id="3D"),
        pytest.param([10, 5, 3, 4], [0, 2, 0, 0, 4, 1, 0, 2], id="4D"),
    ])
def test_convert_pad_v2_with_channels_first_format(shape: list[int], pads: list[int]):
    kernel_shape = [1] * (len(shape) - 2)
    print(kernel_shape)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MaxPool", ['x'], ['o1'], kernel_shape=kernel_shape),
            onnx.helper.make_node("Pad", ['o1'], ['o'], mode='constant', pads=pads, value=0.0)
        ],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 2)])

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    config = ConversionConfig({"ignore_opset_version": True})
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


def test_convert_pad_v2_skipping(intermediate_tflite_model_provider):
    shape = [4, 10]
    pads = [0, 0, 0, 0]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Pad", ['x'], ['o1'], pads=pads),
            onnx.helper.make_node("Pad", ['o1'], ['o'], pads=pads),
        ],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 2)])

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    config = ConversionConfig({"ignore_opset_version": True})
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 1
    assert ops[0].builtin_options.operator_type == BuiltinOperator.PAD


def test_convert_pad_v2_bugged_reflect():
    shape = [2, 2]
    pads = [2, 2, 2, 2]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x'], ['o'], mode='reflect', pads=pads, value=1.0)],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 2)])

    config = ConversionConfig({"ignore_opset_version": True})
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=config)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize(
    "shape, pads",
    [
        pytest.param([42], [41, 41], id="1D"),
        pytest.param([3, 3], [2, 2, 2, 2], id="2D"),
        pytest.param([10, 20, 4], [0, 2, 3, 1, 0, 2], id="3D"),
        pytest.param([10, 5, 3, 4], [0, 2, 0, 0, 4, 1, 0, 2], id="4D"),
    ])
def test_convert_pad_v2_reflect(shape, pads):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x'], ['o'], mode='reflect', pads=pads, value=1.0)],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 2)])

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    config = ConversionConfig({"ignore_opset_version": True})
    executors.convert_run_compare(onnx_model, input_data, conversion_config=config)


def test_convert_pad_v2_edge():
    shape = [2, 2]
    pads = [2, 2, 2, 2]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x'], ['o'], mode='edge', pads=pads, value=1.0)],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 2)])

    config = ConversionConfig({"ignore_opset_version": True})
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=config)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize(
    "shape, pads",
    [
        pytest.param([42], [1, 17], id="1D"),
        pytest.param([10, 20], [0, 2, 4, 1], id="2D"),
        pytest.param([10, 20, 4], [0, 2, 4, 1, 0, 2], id="3D"),
        pytest.param([10, 5, 3, 4], [0, 2, 0, 0, 4, 1, 0, 2], id="4D"),
        pytest.param([7, 4, 5, 3, 4], [0, 2, 0, 1, 19, 0, 4, 1, 0, 2], id="5D"),
    ])
def test_convert_pad_v19_constant_mode(shape: list[int], pads: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads', 'constant'], ['o'], mode='constant')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
            onnx.helper.make_tensor('constant', TensorProto.FLOAT, [1], [3.14159]),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_pad_v19_dynamic_pads():
    shape = [2, 2]
    pads = [2, 2, 2, 2]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads'], ['o'], mode='reflect')],
        'Pad test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("pads", TensorProto.INT64, [len(pads)]),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, shape)],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model, conversion_config=SkipShapeInferenceConfig())
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


def test_convert_pad_v19_dynamic_axes():
    shape = [2, 2]
    axes = [0, 2]
    pads = [1, 1, 1, 1]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads', 'constant', 'axes'], ['o'], mode='reflect')],
        'Pad test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("axes", TensorProto.INT64, [len(axes)]),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
            onnx.helper.make_tensor('constant', TensorProto.FLOAT, [1], [1.1]),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


def test_convert_pad_v19__default_axes():
    shape = [5, 10, 15, 20]
    pads = list(range(len(shape) * 2))  # Different padding everywhere.

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads', 'constant', ''], ['o'], mode='reflect')],
        'Pad test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
            onnx.helper.make_tensor('constant', TensorProto.FLOAT, [1], [1.1]),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "shape, pads",
    [
        pytest.param([42], [1, 17], id="1D"),
        pytest.param([10, 20], [0, 2, 4, 1], id="2D"),
        pytest.param([10, 20, 4], [0, 2, 4, 1, 0, 2], id="3D"),
        pytest.param([10, 5, 3, 4], [0, 2, 0, 0, 4, 1, 0, 2], id="4D"),
        pytest.param([7, 4, 5, 3, 4], [0, 2, 0, 1, 19, 0, 4, 1, 0, 2], id="5D"),
    ])
def test_convert_pad_v19_implicit_operands(shape: list[int], pads: list[int]):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads'], ['o'], mode='constant')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads)]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    "data_type, code",
    [
        pytest.param(TensorProto.FLOAT16, logger.Code.CONVERSION_IMPOSSIBLE, id="FLOAT16"),
        pytest.param(TensorProto.FLOAT, None, id="FLOAT"),
        pytest.param(TensorProto.DOUBLE, logger.Code.CONVERSION_IMPOSSIBLE, id="DOUBLE - unsupported"),

        pytest.param(TensorProto.INT8, None, id="INT8"),
        pytest.param(TensorProto.INT16, logger.Code.NOT_IMPLEMENTED, id="INT16 - unsupported"),
        pytest.param(TensorProto.INT32, None, id="INT32"),
        pytest.param(TensorProto.INT64, None, id="INT64"),

        pytest.param(TensorProto.UINT8, None, id="UINT8"),
        pytest.param(TensorProto.UINT16, logger.Code.CONVERSION_IMPOSSIBLE, id="UINT16"),
        pytest.param(TensorProto.UINT32, logger.Code.CONVERSION_IMPOSSIBLE, id="UINT32"),
        pytest.param(TensorProto.UINT64, logger.Code.CONVERSION_IMPOSSIBLE, id="UINT64"),

        pytest.param(TensorProto.BOOL, logger.Code.CONVERSION_IMPOSSIBLE, id="BOOL"),
        pytest.param(TensorProto.STRING, logger.Code.CONVERSION_IMPOSSIBLE, id="STRING"),
    ])
def test_convert_pad_v19_implicit_constant_different_types(data_type: TensorProto.DataType, code: logger.Code | None):
    shape = [2, 3, 4, 5]
    pads = [1] * 8
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads', ''], ['o'], mode='constant')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", data_type, shape)],
        [onnx.helper.make_tensor_value_info("o", data_type, ())],
        [onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads)]
    )
    onnx_model = onnx.helper.make_model(graph)

    if code is None:
        input_data = np.arange(np.prod(shape)).reshape(shape).astype(to_numpy_type(data_type))
        executors.convert_run_compare(onnx_model, input_data)

    else:
        with pytest.raises(logger.Error):
            convert.convert_model(onnx_model)
        assert logger.conversion_log.get_node_error_code(0) == code


def test_convert_pad_with_quantized_input():
    shape = [3, 3]
    pads = [2] * (2 * len(shape))
    padded_shape = [el + 4 for el in shape]
    new_shape = padded_shape[::-1]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 'scale', 'zp'], ['o1']),
            onnx.helper.make_node('Pad', ['o1', 'pads'], ['o2'], mode='constant'),
            onnx.helper.make_node('Reshape', ['o2', 'new_shape'], ['o3']),
            onnx.helper.make_node('DequantizeLinear', ['o3', 'scale', 'zp'], ['o']),
        ],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
            onnx.helper.make_tensor('new_shape', TensorProto.INT64, [len(new_shape)], new_shape),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [1], [17]),
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [1], [0.52]),
        ],

    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_pad_with_quantized_input__non_zero_constant():
    shape = [3, 3]
    pads = [2] * (2 * len(shape))
    padded_shape = [el + 4 for el in shape]
    new_shape = padded_shape[::-1]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 'scale', 'zp'], ['o1']),
            onnx.helper.make_node('Pad', ['o1', 'pads', 'constant'], ['o2'], mode='constant'),
            onnx.helper.make_node('Reshape', ['o2', 'new_shape'], ['o3']),
            onnx.helper.make_node('DequantizeLinear', ['o3', 'scale', 'zp'], ['o']),
        ],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
            onnx.helper.make_tensor('new_shape', TensorProto.INT64, [len(new_shape)], new_shape),
            onnx.helper.make_tensor('zp', TensorProto.INT8, [1], [17]),
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [1], [0.52]),
            onnx.helper.make_tensor('constant', TensorProto.INT8, [1], [20])
        ],

    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_pad_v19_bugged_reflect_mode():
    shape = [2, 2]
    pads = [2] * 4
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads'], ['o'], mode='reflect')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize(
    "shape, pads",
    [
        pytest.param([42], [1, 17], id="1D"),
        pytest.param([10, 20], [0, 2, 4, 1], id="2D"),
        pytest.param([10, 20, 4], [0, 2, 3, 1, 0, 2], id="3D"),
        pytest.param([10, 5, 3, 4], [0, 2, 0, 0, 4, 1, 0, 2], id="4D"),
        pytest.param([7, 4, 5, 3, 20], [0, 2, 0, 1, 19, 0, 3, 1, 0, 2], id="5D"),
    ])
def test_convert_pad_v19_reflect_mode(shape, pads):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads'], ['o'], mode='reflect')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    input_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_pad_v19_edge_mode():
    shape = [2, 2]
    pads = [2] * 4
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads'], ['o'], mode='edge')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)
    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


def test_convert_pad_v19_wrap_mode():
    shape = [2, 2]
    pads = [2] * 4
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Pad", ['x', 'pads'], ['o'], mode='wrap')],
        'Pad test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_pad__quantized(type_: TensorProto.DataType):
    shape = [2, 4, 6, 8]

    flat_size = np.prod([s + 2 for s in shape])

    np.random.seed(42)
    data = (np.random.random(shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Pad', ['x1', 'pads'], ['x2']),
            onnx.helper.make_node('Reshape', ['x2', 'flat_shape'], ['x3']),
            onnx.helper.make_node('DequantizeLinear', ['x3', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [1], [flat_size]),
            onnx.helper.make_tensor('pads', TensorProto.INT64, [8], [1] * 8)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
