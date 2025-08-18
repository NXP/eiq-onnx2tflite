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


def test_convert_arg_min_qdq(intermediate_tflite_model_provider):
    axis, keepdims = 3, 0
    shape = [4, 5, 6, 7]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ArgMin', ['x'], ['y'], axis=axis, keepdims=keepdims)],
        'ArgMin test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.INT64, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    # Quantize ONNX model
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.ARG_MIN
    ])
