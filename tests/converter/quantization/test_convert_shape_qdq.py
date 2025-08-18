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


def test_convert_shape_qdq__node_removed(intermediate_tflite_model_provider):
    start = 1
    end = 4
    input_shape = [11, 3, 5]
    shape_output_shape = [2]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Shape', ['input'], ['y'], start=start, end=end),
            onnx.helper.make_node('Add', ['y', 'x'], ['output']),
        ],
        'Shape test',
        [
            onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
            onnx.helper.make_tensor_value_info("x", TensorProto.INT64, shape_output_shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.INT64, shape_output_shape)],
        value_info=[onnx.helper.make_tensor_value_info("y", TensorProto.INT64, shape_output_shape)]
    )

    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({
        "input": InputSpec(input_shape, np.float32),
        "x": InputSpec(shape_output_shape, np.int64),
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = {
        0: np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32),
        1: np.random.random(np.prod(shape_output_shape)).reshape(shape_output_shape).astype(np.int64),
    }

    executors.convert_run_compare(quantized_model, input_data, input_data_tflite=input_data[1])

    intermediate_tflite_model_provider.assert_converted_model_has_operators([BuiltinOperator.ADD])
