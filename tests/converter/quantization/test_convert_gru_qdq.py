#
# Copyright 2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
from onnx import TensorProto

from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2tflite.src.converter import convert
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options, quantize_options
from onnx2tflite.src.tflite_generator.custom_options import onnx_gru_options


# noinspection PyUnresolvedReferences
def test_convert_gru_qdq(intermediate_tflite_model_provider):
    input_size = 5
    hidden_size = 8
    number_of_gates = 3

    input_data = np.random.randn(1, number_of_gates, input_size).astype(np.float32)

    node = onnx.helper.make_node(
        "GRU",
        inputs=["X", "W", "R", "B", "", "initial_h"],
        outputs=["Y", "Y_h"],
        hidden_size=hidden_size,
        direction="forward",
    )

    W = np.random.randn(1, number_of_gates * hidden_size, input_size).astype(np.float32)
    R = np.random.randn(1, number_of_gates * hidden_size, hidden_size).astype(np.float32)

    W_B = np.random.randn(1, number_of_gates * hidden_size).astype(np.float32)
    R_B = np.random.randn(1, number_of_gates * hidden_size).astype(np.float32)
    B = np.concatenate((W_B, R_B), axis=1) * 0.05

    initial_h = np.random.randn(1, number_of_gates, hidden_size).astype(np.float32) * 0.005

    graph = onnx.helper.make_graph(
        [node],
        'Gru test',
        [onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, input_data.shape)],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, ())],
        initializer=[
            onnx.helper.make_tensor("W", TensorProto.FLOAT, W.shape, W),
            onnx.helper.make_tensor("R", TensorProto.FLOAT, R.shape, R),
            onnx.helper.make_tensor("B", TensorProto.FLOAT, B.shape, B),
            onnx.helper.make_tensor("initial_h", TensorProto.FLOAT, initial_h.shape, initial_h),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    config = RandomDataCalibrationDataReader({"X": InputSpec(input_data.shape, np.float32)}).to_config()
    quantized_model = QDQQuantizer().quantize_model(onnx_model, config)

    # Just verify model was converted. OnnxGRU is custom TFLite op, and it requires custom TFLite delegate.
    tflite_model = convert.convert_model(quantized_model)

    assert len(tflite_model) > 0

    ops = intermediate_tflite_model_provider.get_operators()
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[1].custom_options, onnx_gru_options.OnnxGRU)
    assert isinstance(ops[2].builtin_options, dequantize_options.Dequantize)

    gru_op = ops[1]
    assert len(gru_op.tmp_inputs) == 6
    assert gru_op.tmp_inputs[0].shape.vector == [1, 3, 5]  # x
    assert gru_op.tmp_inputs[0].quantization is not None
    assert gru_op.tmp_inputs[0].quantization.is_per_tensor() is not None

    assert gru_op.tmp_inputs[1].shape.vector == [1, 24, 5]  # w
    assert gru_op.tmp_inputs[1].quantization is not None
    assert gru_op.tmp_inputs[1].quantization.is_per_channel() is not None

    assert gru_op.tmp_inputs[2].shape.vector == [1, 24, 8]  # r
    assert gru_op.tmp_inputs[2].quantization is not None
    assert gru_op.tmp_inputs[2].quantization.is_per_channel() is not None

    assert gru_op.tmp_inputs[3].shape.vector == [1, 48]  # b
    assert gru_op.tmp_inputs[3].quantization is not None
    assert gru_op.tmp_inputs[3].quantization.quantized_dimension == 1
    assert gru_op.tmp_inputs[3].quantization.is_per_channel() is not None

    assert gru_op.tmp_inputs[5].shape.vector == [1, 3, 8]  # initial_h
    assert gru_op.tmp_inputs[5].quantization is not None
    assert gru_op.tmp_inputs[5].quantization.is_per_channel() is not None

    assert len(gru_op.tmp_outputs) == 2
    assert gru_op.tmp_outputs[0].shape.vector == [1, 1, 3, 8]  # y
    assert gru_op.tmp_outputs[0].quantization is not None
    assert gru_op.tmp_outputs[0].quantization.is_per_tensor() is not None

    assert gru_op.tmp_outputs[1].shape.vector == [1, 3, 8]  # y_h
    assert gru_op.tmp_outputs[1].quantization is not None
    assert gru_op.tmp_outputs[1].quantization.is_per_channel() is not None
