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


def test_convert_tile_qdq(intermediate_tflite_model_provider):
    x_shape = [2, 3, 4]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Tile', ['x', 'repeats'], ['y'])],
        'Tile test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('repeats', TensorProto.INT64, [3], [1, 2, 3])]
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(x_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    input_data = (np.random.random(np.prod(x_shape))).reshape(x_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.TILE, BuiltinOperator.DEQUANTIZE
    ])


def test_convert_tile__different_q_params():
    quant_type = TensorProto.INT8
    shape = [5, 4, 1]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"]),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["x_dequant"]),
            onnx.helper.make_node("Tile", ["x_dequant", "repeats"], ["y"]),
            onnx.helper.make_node("QuantizeLinear", ["y", "y_scale", "y_zero_point"], ["y_output"]),
            onnx.helper.make_node("DequantizeLinear", ["y_output", "y_scale", "y_zero_point"], ["output"]),
        ],
        'Tile test quantized',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.003825969761237502]),
            onnx.helper.make_tensor("x_zero_point", quant_type, [], [-128]),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.004825969761237502]),
            onnx.helper.make_tensor("y_zero_point", quant_type, [], [-128]),
            onnx.helper.make_tensor('repeats', TensorProto.INT64, [3], [1, 2, 3]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    input_data = np.random.random(np.prod(shape)).reshape(shape).astype(np.float32)

    executors.convert_run_compare(onnx_model, input_data, atol=0.0049)
