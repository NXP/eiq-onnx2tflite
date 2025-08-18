#
# Copyright 2024-2025 NXP
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
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests.executors import OnnxExecutor, TFLiteExecutor, _assert_array_shape_equals_output_shape


# Since the `Multinomial` operator is RNG based, we cannot compare the TFLite and ONNX outputs directly. The tests below
#  only statistically verify the expected behavior.

def _convert_run_and_compare_distributions(onnx_model: onnx.ModelProto, data: np.ndarray, shape: list[int],
                                           sample_size: int, atol: float) -> (np.ndarray, np.ndarray):
    onnx_model = ModelShapeInference.infer_shapes(onnx_model)

    tfl_model = convert.convert_model(onnx_model)

    onnx_executor = OnnxExecutor(onnx_model.SerializeToString())
    onnx_output = onnx_executor.inference(data)

    tflite_executor = TFLiteExecutor(model_content=bytes(tfl_model))
    tflite_output = tflite_executor.inference(data)

    assert tflite_output.dtype == onnx_output.dtype
    assert tflite_output.shape == onnx_output.shape
    _assert_array_shape_equals_output_shape(onnx_model.graph.output[0], onnx_output, True)

    tflite_distribution = []
    onnx_distribution = []
    for batch in range(shape[0]):
        tflite_distribution.append([])
        onnx_distribution.append([])
        for class_ in range(shape[1]):
            tflite_distribution[batch].append(
                (tflite_output[batch] == class_).sum() / sample_size
            )
            onnx_distribution[batch].append(
                (onnx_output[batch] == class_).sum() / sample_size
            )

    tflite_distribution = np.asarray(tflite_distribution, 'float32')
    onnx_distribution = np.asarray(onnx_distribution, 'float32')

    logger.d(f'Maximum distribution difference = `{np.abs(tflite_distribution - onnx_distribution).max()}`.')
    assert np.allclose(tflite_distribution, onnx_distribution, atol=atol)

    return tflite_output, onnx_output


def test_convert_multinomial_v19():
    sample_size = 50000  # Take many samples to increase statistical significance.

    shape = [5, 3]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Multinomial', ['x'], ['y'], sample_size=sample_size, seed=42.)],
        'Multinomial test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT32, ())],
    )
    onnx_model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_operatorsetid("", 19)]
    )
    onnx.checker.check_model(onnx_model)

    data = np.array([
        [1.2, 1.9, 8.1],
        [18., 2.2, 7.7],
        [15.1, 100.5, 42.42],
        [1., 2., 3.],
        [0.2, 0.5, 0.3]
    ], 'float32')

    _convert_run_and_compare_distributions(onnx_model, data, shape, sample_size, atol=0.0046)


@pytest.mark.xfail(reason="Multinomial v22 not yet implemented in ORT")
def test_convert_multinomial_v22():
    sample_size = 50000  # Take many samples to increase statistical significance.

    shape = [5, 3]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Multinomial', ['x'], ['y'], sample_size=sample_size, seed=42.)],
        'Multinomial test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT32, ())],
    )
    onnx_model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_operatorsetid("", 22)]
    )
    onnx.checker.check_model(onnx_model)

    data = np.array([
        [1.2, 1.9, 8.1],
        [18., 2.2, 7.7],
        [15.1, 100.5, 42.42],
        [1., 2., 3.],
        [0.2, 0.5, 0.3]
    ], 'float32')

    _convert_run_and_compare_distributions(onnx_model, data, shape, sample_size, atol=0.0046)


@pytest.mark.parametrize("dtype", [TensorProto.INT32, TensorProto.INT64], ids=name_for_onnx_type)
def test_convert_multinomial__dtype(dtype: TensorProto.DataType):
    sample_size = 50000  # Take many samples to increase statistical significance.

    shape = [1, 3]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Multinomial', ['x'], ['y'], sample_size=sample_size, dtype=dtype)],
        'Multinomial test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', dtype, ())],
    )
    onnx_model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_operatorsetid("", 19)]
    )
    onnx.checker.check_model(onnx_model)

    data = np.array([
        [-1., 2.3, 8.6],
    ], 'float32')

    _convert_run_and_compare_distributions(onnx_model, data, shape, sample_size, atol=0.0029)


def test_convert_multinomial__invalid_dtype():
    dtype = TensorProto.INT16

    shape = [3, 3]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Multinomial', ['x'], ['y'], dtype=dtype)],
        'Multinomial test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', dtype, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    # onnx.check_model() catches it before our converter can.
    with pytest.raises(logger.Error) as e:
        convert.convert_model(onnx_model, conversion_config=SkipShapeInferenceConfig())
    assert e.value.error_code == logger.Code.SHAPE_INFERENCE_ERROR
    assert 'int32 or int64' in e.value.msg


@pytest.mark.parametrize("type_", [TensorProto.FLOAT], ids=name_for_onnx_type)
def test_convert_multinomial__types(type_: TensorProto.DataType):
    sample_size = 50000  # Take many samples to increase statistical significance.

    shape = [1, 3]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Multinomial', ['x'], ['y'], sample_size=sample_size)],
        'Multinomial test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT32, ())],
    )
    onnx_model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_operatorsetid("", 19)]
    )
    onnx.checker.check_model(onnx_model)

    data = np.array([
        [-1., 2.3, 8.6],
    ], to_numpy_type(type_))

    _convert_run_and_compare_distributions(onnx_model, data, shape, sample_size, atol=0.0029)


def test_convert_multinomial__invalid_type():
    type_ = TensorProto.DOUBLE

    shape = [3, 3]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Multinomial', ['x'], ['y'])],
        'Multinomial test',
        [onnx.helper.make_tensor_value_info('x', type_, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT32, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


def test_convert_multinomial__seed_mapping():
    # Model contains 2 parallel `Multinomial` nodes with the same input and seed, so they should output the same data.

    sample_size = 1000

    shape = [1, 3]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Multinomial', ['x'], ['y1'], sample_size=sample_size, seed=42.),
            onnx.helper.make_node('Multinomial', ['x'], ['y2'], sample_size=sample_size, seed=42.)
        ],
        'Multinomial test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [
            onnx.helper.make_tensor_value_info('y1', TensorProto.INT32, ()),
            onnx.helper.make_tensor_value_info('y2', TensorProto.INT32, ())
        ],
    )
    onnx_model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_operatorsetid("", 19)]
    )
    onnx.checker.check_model(onnx_model)

    data = np.array([
        [1.2, -1.9, 8.1]

    ], 'float32')

    onnx_model = ModelShapeInference.infer_shapes(onnx_model)

    tfl_model = convert.convert_model(onnx_model)

    onnx_executor = OnnxExecutor(onnx_model.SerializeToString())
    onnx_output = onnx_executor.inference(data)

    tflite_executor = TFLiteExecutor(model_content=bytes(tfl_model))
    tflite_output = tflite_executor.inference(data)

    onnx_y1 = onnx_output['y1']
    onnx_y2 = onnx_output['y2']
    assert np.allclose(onnx_y1, onnx_y2)

    tflite_y1 = tflite_output['y1']
    tflite_y2 = tflite_output['y2']
    assert np.allclose(tflite_y1, tflite_y2)
