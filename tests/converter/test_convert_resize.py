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
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type, to_numpy_type
from tests import executors


@pytest.mark.parametrize(
    'output_shape',
    [
        pytest.param([1, 3, 42, 42], id='Down-sampling to 42x42.'),
        pytest.param([1, 3, 10, 11], id='Down-sampling to 10x11.'),
        pytest.param([1, 3, 23, 81], id='Down-sampling to 23x81.'),

        pytest.param([1, 3, 120, 120], id='Up-sampling to 120x120.'),
        pytest.param([1, 3, 137, 156], id='Up-sampling to 137x156.'),
        pytest.param([1, 3, 217, 200], id='Up-sampling to 217x200.'),

        pytest.param([1, 3, 42, 142], id='Re-sizing to 42x142.'),
        pytest.param([1, 3, 237, 23], id='Re-sizing to 237x23.'),
    ])
def test_convert_resize__linear__sizes(output_shape: list[int]):
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], mode='linear')],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], output_shape)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, atol=1.3e-5)


@pytest.mark.parametrize(
    'type_',
    [TensorProto.FLOAT, TensorProto.UINT8, TensorProto.INT8],
    ids=name_for_onnx_type
)
def test_convert_resize__linear__types(type_: TensorProto.DataType):
    input_shape = [1, 3, 100, 100]
    output_shape = [1, 3, 120, 120]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], mode='linear')],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', type_, input_shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], output_shape)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(to_numpy_type(type_))

    executors.convert_run_compare(onnx_model, input_data, atol=8.8e-6)


def test_convert_resize__invalid_type():
    type_ = TensorProto.DOUBLE

    input_shape = [1, 3, 100, 100]
    output_shape = [1, 3, 120, 120]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], mode='linear')],
        'Invalid type test test',
        [onnx.helper.make_tensor_value_info('x', type_, input_shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], output_shape)]
    )

    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert 'FLOAT64' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize(
    'scales',
    [
        pytest.param([1., 1., 0.5, 0.5], id='0.5 x 0.5'),
        pytest.param([1., 1., .37, .42], id='0.37 x 0.42'),
        pytest.param([1., 1., 2., 2.], id='2.0 x 2.0'),
        pytest.param([1., 1., 1.42, 1.37], id='1.42 x 1.37'),
        pytest.param([1., 1., 1.65, 0.12], id='1.65 x 0.12'),
        pytest.param([1., 1., 0.78, 2.49], id='0.78 x 2.49'),
    ])
def test_convert_resize__linear__scales(scales: list[int]):
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', 'scales', ''], ['y'], mode='linear')],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('scales', TensorProto.FLOAT, [4], scales)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, atol=1.3e-5)


@pytest.mark.parametrize(
    'sizes, axes',
    [
        # Testing only positive axes, because ONNX Runtime crashes with negative axes.
        # (Windows fatal exception: code 0xc0000374)

        pytest.param([42, 37], [2, 3]),
        pytest.param([120], [3]),
        pytest.param([250, 10], [3, 2]),
        pytest.param([142, 1, 10], [3, 0, 2]),
    ])
def test_convert_resize__linear__sizes_and_axes(sizes: list[int], axes: list[int]):
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], mode='linear', axes=axes)],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [len(sizes)], sizes)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, atol=1.3e-5)


@pytest.mark.parametrize(
    'scales, axes',
    [
        # Testing only positive axes, because ONNX Runtime crashes with negative axes.
        # (Windows fatal exception: code 0xc0000374)

        pytest.param([0.42, 0.42], [2, 3]),
        pytest.param([1.2], [3]),
        pytest.param([2.5, 0.1], [3, 2]),
        pytest.param([1.42, 1.0, 0.1], [3, 0, 2]),
    ])
def test_convert_resize__linear__scales_and_axes(scales: list[int], axes: list[int]):
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', 'scales', ''], ['y'], mode='linear', axes=axes)],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('scales', TensorProto.FLOAT, [len(scales)], scales)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, atol=1.3e-5)


@pytest.mark.parametrize(
    'coordinate_transformation_mode',
    [
        'align_corners',
        'half_pixel',
        'pytorch_half_pixel',
        'asymmetric',
    ])
def test_convert_resize__linear__coordinate_transformation_mode(coordinate_transformation_mode: str):
    sizes = [1, 3, 42, 42]
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], mode='linear',
                               coordinate_transformation_mode=coordinate_transformation_mode)],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], sizes)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, atol=1.3e-5)


@pytest.mark.parametrize(
    'sizes',
    [
        pytest.param([1, 3, 42, 42], id='Down-sampling to 42x42.'),
        pytest.param([1, 3, 10, 11], id='Down-sampling to 10x11.'),
        pytest.param([1, 3, 23, 81], id='Down-sampling to 23x81.'),

        pytest.param([1, 3, 120, 120], id='Up-sampling to 120x120.'),
        pytest.param([1, 3, 137, 156], id='Up-sampling to 137x156.'),
        pytest.param([1, 3, 217, 200], id='Up-sampling to 217x200.'),

        pytest.param([1, 3, 42, 142], id='Re-sizing to 42x142.'),
        pytest.param([1, 3, 237, 23], id='Re-sizing to 237x23.'),
    ])
def test_convert_resize__nearest__sizes(sizes: list[int]):
    input_shape = [1, 3, 100, 100]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], mode='nearest',
                               nearest_mode='round_prefer_ceil')],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], sizes)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    'type_',
    [TensorProto.FLOAT, TensorProto.UINT8, TensorProto.INT8],
    ids=name_for_onnx_type
)
def test_convert_resize__nearest__types(type_: TensorProto.DataType):
    input_shape = [1, 3, 100, 100]
    output_shape = [1, 3, 120, 120]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], mode='nearest',
                               nearest_mode='round_prefer_ceil')],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', type_, input_shape)],
        [onnx.helper.make_tensor_value_info('y', type_, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], output_shape)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(to_numpy_type(type_))

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    'scales',
    [
        pytest.param([1., 1., 0.5, 0.5], id='0.5 x 0.5'),
        pytest.param([1., 1., .37, .42], id='0.37 x 0.42'),
        pytest.param([1., 1., 2., 2.], id='2.0 x 2.0'),
        pytest.param([1., 1., 1.42, 1.37], id='1.42 x 1.37'),
        pytest.param([1., 1., 1.65, 0.12], id='1.65 x 0.12'),
        pytest.param([1., 1., 0.78, 2.49], id='0.78 x 2.49'),
    ])
def test_convert_resize__nearest__scales(scales: list[int]):
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', 'scales', ''], ['y'], mode='nearest',
                               nearest_mode='round_prefer_ceil')],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('scales', TensorProto.FLOAT, [4], scales)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    'sizes, axes',
    [
        # Testing only positive axes, because ONNX Runtime crashes with negative axes.
        # (Windows fatal exception: code 0xc0000374)

        pytest.param([42, 37], [2, 3]),
        pytest.param([120], [3]),
        pytest.param([250, 10], [3, 2]),
        pytest.param([142, 1, 10], [3, 0, 2]),
    ])
def test_convert_resize__nearest__sizes_and_axes(sizes: list[int], axes: list[int]):
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], mode='nearest', axes=axes,
                               nearest_mode='round_prefer_ceil')],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [len(sizes)], sizes)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    'scales, axes',
    [
        # Testing only positive axes, because ONNX Runtime crashes with negative axes.
        # (Windows fatal exception: code 0xc0000374)

        pytest.param([0.42, 0.42], [2, 3]),
        pytest.param([1.2], [3]),
        pytest.param([2.5, 0.1], [3, 2]),
        pytest.param([1.42, 1.0, 0.1], [3, 0, 2]),
    ])
def test_convert_resize__nearest__scales_and_axes(scales: list[int], axes: list[int]):
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', 'scales', ''], ['y'], mode='nearest', axes=axes,
                               nearest_mode='round_prefer_ceil')],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('scales', TensorProto.FLOAT, [len(scales)], scales)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


@pytest.mark.parametrize(
    'coordinate_transformation_mode, nearest_mode',
    [
        ('align_corners', 'round_prefer_ceil'),
        ('half_pixel', 'round_prefer_ceil'),
        ('pytorch_half_pixel', 'round_prefer_ceil'),

        # For some reason `coordinate_transformation_mode=asymmetric` is the only time TFLite uses the
        #  floor rounding approach.
        ('asymmetric', 'floor'),
    ])
def test_convert_resize__nearest__coordinate_transformation_mode(coordinate_transformation_mode: str,
                                                                 nearest_mode: str):
    sizes = [1, 3, 142, 137]
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], mode='nearest',
                               nearest_mode=nearest_mode,
                               coordinate_transformation_mode=coordinate_transformation_mode)],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], sizes)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data)


def test_convert_resize__nearest__coordinate_transformation_mode__accept_error():
    # These attributes are used in a customer model.
    coordinate_transformation_mode = 'half_pixel'
    nearest_mode = 'round_prefer_floor'

    sizes = [1, 3, 142, 137]
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], mode='nearest',
                               nearest_mode=nearest_mode,
                               coordinate_transformation_mode=coordinate_transformation_mode)],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], sizes)]
    )
    onnx_model = onnx.helper.make_model(graph)

    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    config = ConversionConfig()
    config.accept_resize_rounding_error = True

    # The maximum error is about 0.966.
    # About 97.87% of output values are exactly correct.
    executors.convert_run_compare(onnx_model, input_data, atol=.97, conversion_config=config)


def test_convert_resize__nearest__coordinate_transformation_mode__print_error():
    # These attributes are used in a customer model.
    coordinate_transformation_mode = 'half_pixel'
    nearest_mode = 'round_prefer_floor'

    sizes = [1, 3, 142, 137]
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], mode='nearest',
                               nearest_mode=nearest_mode,
                               coordinate_transformation_mode=coordinate_transformation_mode)],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], sizes)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE
    assert '--accept-resize-rounding-error' in logger.conversion_log.get_node_error_message(0)


@pytest.mark.parametrize(
    'scales',
    [
        pytest.param([1., 1., 0.515, 0.7], id='0.515 x 0.7'),
        pytest.param([1., 1., 0.511, 0.7], id='0.511 x 0.7'),
        pytest.param([1., 1., 0.5, 1.095], id='0.5 x 1.095'),
    ])
def test_convert_resize__invalid_scales(scales: list[int]):
    # The input_shape * scales will not result in a whole number.

    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', 'scales', ''], ['y'])],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('scales', TensorProto.FLOAT, [4], scales)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_resize__invalid_channels_scale():
    scales = [1., 2., 1., 1.]  # The C is not 1.

    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', 'scales', ''], ['y'])],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('scales', TensorProto.FLOAT, [4], scales)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_convert_resize__unsupported_coordinate_transformation_mode():
    coordinate_transformation_mode = 'half_pixel_symmetric'

    sizes = [1, 3, 50, 50]
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'],
                               coordinate_transformation_mode=coordinate_transformation_mode)],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], sizes)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


# noinspection SpellCheckingInspection
def test_convert_resize__antialias_down_sampling():
    sizes = [1, 3, 42, 37]
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'], antialias=1)],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], sizes)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


@pytest.mark.parametrize(
    'keep_aspect_ratio_policy',
    ['not_larger', 'not_smaller']
)
def test_convert_resize__keep_aspect_ratio_policy(keep_aspect_ratio_policy: str):
    input_shape = [1, 3, 100, 100]
    sizes = [1, 3, 42, 137]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', '', 'sizes'], ['y'],
                               keep_aspect_ratio_policy=keep_aspect_ratio_policy)],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], sizes)]
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert.convert_model(onnx_model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.NOT_IMPLEMENTED


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_resize__linear__quantized(type_: TensorProto.DataType):
    input_shape = [1, 3, 100, 100]
    output_shape = [1, 3, 120, 120]

    np.random.seed(42)
    data = (np.random.random(input_shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Resize', ['x1', '', '', 'sizes'], ['x2'], mode='linear'),
            onnx.helper.make_node('Reshape', ['x2', 'flat_shape'], ['x3']),
            onnx.helper.make_node('DequantizeLinear', ['x3', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [1], [np.prod(output_shape)]),
            onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], output_shape)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data, atol=0.042)


@pytest.mark.parametrize("type_", [TensorProto.INT8, TensorProto.UINT8], ids=name_for_onnx_type)
def test_convert_resize__nearest__quantized(type_: TensorProto.DataType):
    input_shape = [1, 3, 100, 100]
    output_shape = [1, 3, 120, 120]

    np.random.seed(42)
    data = (np.random.random(input_shape) * 100).astype(np.float32)  # [0, 100)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('QuantizeLinear', ['x', 's', 'zp'], ['x1']),
            onnx.helper.make_node('Resize', ['x1', '', '', 'sizes'], ['x2'], mode='nearest',
                                  nearest_mode='round_prefer_ceil'),
            onnx.helper.make_node('Reshape', ['x2', 'flat_shape'], ['x3']),
            onnx.helper.make_node('DequantizeLinear', ['x3', 's', 'zp'], ['y'])
        ],
        'Quantized input test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('s', TensorProto.FLOAT, [1], [.0042]),
            onnx.helper.make_tensor('zp', type_, [1], [12]),
            onnx.helper.make_tensor('flat_shape', TensorProto.INT64, [1], [np.prod(output_shape)]),
            onnx.helper.make_tensor('sizes', TensorProto.INT64, [4], output_shape)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    executors.convert_run_compare(onnx_model, data)
