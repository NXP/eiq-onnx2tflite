#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
from onnx import TensorProto

from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from tests import executors


def test_convert_resize_qdq__nearest_neighbor(intermediate_tflite_model_provider):
    scales, axes = [2.5, 0.1], [3, 2]
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

    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE, BuiltinOperator.RESIZE_NEAREST_NEIGHBOR,
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])


def test_convert_resize_qdq__linear(intermediate_tflite_model_provider):
    scales, axes = [2.5, 0.1], [3, 2]
    input_shape = [1, 3, 100, 100]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Resize', ['x', '', 'scales', ''], ['y'], mode='linear', axes=axes)],
        'Resize test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('scales', TensorProto.FLOAT, [len(scales)], scales)]
    )
    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data, atol=4e-3)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.TRANSPOSE, BuiltinOperator.QUANTIZE, BuiltinOperator.RESIZE_BILINEAR,
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.TRANSPOSE
    ])
