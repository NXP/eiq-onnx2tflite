#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
import pytest
import tensorflow.lite as tflite
from onnx import TensorProto

from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2tflite.lib.tflite import (
    BuiltinOperator as tflBuiltinOperator
)
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.tflite_generator.builtin_options import broadcast_to_options, dequantize_options, quantize_options
from tests import executors


@pytest.fixture(scope="module")
def qdq_model_with_expand():
    input_shape = [2, 2, 10, 1]
    new_shape = [3]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Expand', ['input', 'new_shape'], ['output'])],
        'expand test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor("new_shape", TensorProto.INT64, [len(new_shape)], new_shape)]
    )

    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({"input": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


def test_convert_expand_qdq(qdq_model_with_expand, intermediate_tflite_model_provider):
    input_shape = [2, 2, 10, 1]

    input_data = np.random.random(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

    executors.convert_run_compare(qdq_model_with_expand, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 4
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert (intermediate_tflite_model_provider.get_operator_code_at_index(1) ==
            tflBuiltinOperator.BuiltinOperator.BROADCAST_ARGS)
    assert isinstance(ops[2].builtin_options, broadcast_to_options.BroadcastTo)
    assert isinstance(ops[3].builtin_options, dequantize_options.Dequantize)


def test_convert_expand_different_q_params():
    quant_type = TensorProto.INT8
    shape = [5, 4, 1]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_quant"]),
            onnx.helper.make_node("DequantizeLinear", ["x_quant", "x_scale", "x_zero_point"], ["x_dequant"]),
            onnx.helper.make_node("Expand", ["x_dequant", "new_shape"], ["y"]),
            onnx.helper.make_node("QuantizeLinear", ["y", "y_scale", "y_zero_point"], ["output"])
        ],
        'Expand test quantized',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("output", quant_type, ())],
        [
            onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.003825969761237502]),
            onnx.helper.make_tensor("x_zero_point", quant_type, [], [-128]),
            onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.004825969761237502]),
            onnx.helper.make_tensor("y_zero_point", quant_type, [], [-128]),
            onnx.helper.make_tensor("new_shape", TensorProto.INT64, [1], [3]),
        ]
    )

    onnx_model = onnx.helper.make_model(graph)
    input_data = np.random.random(np.prod(shape)).reshape(shape).astype(np.float32)

    # To make the test deterministic and avoid rounding differences in XNNPACK vs ONNX Runtime,
    # use TensorFlow Lite Reference kernels
    executors.convert_run_compare(onnx_model, input_data,
                                  tflite_op_resolver_type=tflite.experimental.OpResolverType.BUILTIN_REF)
