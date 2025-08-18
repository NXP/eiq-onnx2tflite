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


def test_convert_scatter_nd__dynamic_updates(intermediate_tflite_model_provider):
    x_shape = [8]
    indices_shape = [4, 1]
    updates_shape = [4]

    indices = [4, 3, 1, 7]
    updates = [9., 10., 11., 12.]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'])],
        'ScatterND test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info('updates', TensorProto.FLOAT, updates_shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices_shape, indices)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    # Quantize ONNX model
    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec(x_shape, np.float32),
        "updates": InputSpec(updates_shape, np.float32)
    })
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    np.random.seed(42)
    input_data = {
        0: (np.random.random(x_shape).astype(np.float32) - 0.5) * 20,  # <-10, 10)
        1: np.array(updates, np.float32)
    }

    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE, BuiltinOperator.SCATTER_ND,
        BuiltinOperator.SCATTER_ND, BuiltinOperator.SELECT_V2, BuiltinOperator.DEQUANTIZE
    ])


def test_convert_scatter_nd__static_updates(intermediate_tflite_model_provider):
    x_shape, indices_shape, updates_shape = [10, 20], [5, 2], [5]

    indices = np.random.choice(np.arange(min(x_shape)), indices_shape, replace=False)
    updates = np.random.random(updates_shape)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'])],
        'ScatterND test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices_shape, indices),
            onnx.helper.make_tensor('updates', TensorProto.FLOAT, updates_shape, updates)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    # Quantize ONNX model
    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(x_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    np.random.seed(42)
    input_data = (np.random.random(x_shape).astype(np.float32) - 0.5) * 20  # <-10, 10)

    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.SCATTER_ND, BuiltinOperator.SCATTER_ND,
        BuiltinOperator.SELECT_V2, BuiltinOperator.DEQUANTIZE
    ])


def test_convert_scatter_nd__static_x(intermediate_tflite_model_provider):
    x_shape, indices_shape, updates_shape = [10, 20], [5, 2], [5]

    indices = np.random.choice(np.arange(min(x_shape)), indices_shape, replace=False)
    x_data = (np.random.random(x_shape).astype(np.float32) - 0.5) * 20

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('ScatterND', ['x', 'indices', 'updates'], ['y'])],
        'ScatterND test',
        [onnx.helper.make_tensor_value_info('updates', TensorProto.FLOAT, updates_shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor('indices', TensorProto.INT64, indices_shape, indices),
            onnx.helper.make_tensor('x', TensorProto.FLOAT, x_shape, x_data)
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    # Quantize ONNX model
    calibration_data_reader = RandomDataCalibrationDataReader({"updates": InputSpec(updates_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    quantized_model = ModelShapeInference.infer_shapes(quantized_model)

    np.random.seed(42)
    input_data = np.random.random(updates_shape).astype(np.float32)

    executors.convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.SCATTER_ND, BuiltinOperator.SCATTER_ND,
        BuiltinOperator.SELECT_V2, BuiltinOperator.DEQUANTIZE
    ])
