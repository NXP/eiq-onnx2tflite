#
# Copyright 2026 NXP
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


def test_convert_topk_qdq(intermediate_tflite_model_provider):
    input_shape = [2, 4, 4, 6]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("TopK", ["x", "k"], ["y", "z"])],
        "TopK test",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape), ],
        [
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, ()),
            onnx.helper.make_tensor_value_info("z", TensorProto.INT64, ()),
        ],
        [onnx.helper.make_tensor("k", TensorProto.INT64, [1], [3])],
    )
    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})

    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE,
        BuiltinOperator.TOPK_V2,
        BuiltinOperator.CAST,
        BuiltinOperator.DEQUANTIZE,
    ])