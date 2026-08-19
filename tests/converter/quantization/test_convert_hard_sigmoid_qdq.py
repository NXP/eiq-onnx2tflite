#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options, quantize_options
from tests import executors


@pytest.mark.parametrize(
    "attributes, expected_ops",
    [
        pytest.param(
            {},
            [BuiltinOperator.MUL, BuiltinOperator.ADD, BuiltinOperator.MINIMUM, BuiltinOperator.RELU],
            id="default_attributes",
        ),
        pytest.param(
            {"alpha": 0.25, "beta": 0.6},
            [BuiltinOperator.MUL, BuiltinOperator.ADD, BuiltinOperator.MINIMUM, BuiltinOperator.RELU],
            id="alpha_and_beta",
        ),
        pytest.param(
            {"alpha": 1.0, "beta": 0.5},
            [BuiltinOperator.ADD, BuiltinOperator.MINIMUM, BuiltinOperator.RELU],
            id="neutral_alpha",
        ),
        pytest.param(
            {"alpha": 0.2, "beta": 0.0},
            [BuiltinOperator.MUL, BuiltinOperator.MINIMUM, BuiltinOperator.RELU],
            id="neutral_beta",
        ),
        pytest.param(
            {"alpha": 1.0, "beta": 0.0},
            [BuiltinOperator.MINIMUM, BuiltinOperator.RELU],
            id="neutral_alpha_and_beta",
        ),
    ])
def test_convert_hard_sigmoid_qdq(attributes, expected_ops, intermediate_tflite_model_provider):
    input_shape = [4, 8, 16]

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("HardSigmoid", ["x"], ["output"], **attributes)],
        'HardSigmoid test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    calibration_data_reader = RandomDataCalibrationDataReader({"x": InputSpec(input_shape, np.float32)})
    quantized_model = QDQQuantizer().quantize_model(onnx_model, calibration_data_reader.to_config())

    input_data = (np.random.random(input_shape).astype(np.float32) - 0.5) * 20  # Values from -10 to 10

    executors.convert_run_compare(quantized_model, input_data, atol=0.004)

    ops = intermediate_tflite_model_provider.get_operators()
    # A leading `Quantize` and a trailing `Dequantize` wrap the quantized chain.
    assert isinstance(ops[0].builtin_options, quantize_options.Quantize)
    assert isinstance(ops[-1].builtin_options, dequantize_options.Dequantize)

    op_codes = [
        intermediate_tflite_model_provider.get_operator_code_at_index(op.opcode_index) for op in ops[1:-1]
    ]
    assert op_codes == expected_ops

    # All tensors of the intermediate operators (between the leading `Quantize` and trailing
    # `Dequantize`) must be quantized, i.e. carry quantization parameters and use an integer type.
    for op in ops[1:-1]:
        for tensor in list(op.tmp_inputs) + list(op.tmp_outputs):
            assert tensor.quantization is not None, (
                f"Intermediate tensor '{tensor.name}' is not quantized (missing quantization parameters)."
            )
            assert tensor.type in {TensorType.INT8, TensorType.UINT8}, (
                f"Intermediate tensor '{tensor.name}' has non-quantized type {tensor.type}."
            )


def _make_qdq_tensor(name: str, scale: float, zero_point: int, zp_type: int) -> list:
    """Create the initializers describing a QDQ quantization for a tensor named 'name'."""
    return [
        onnx.helper.make_tensor(f"{name}_scale", TensorProto.FLOAT, [], [scale]),
        onnx.helper.make_tensor(f"{name}_zero_point", zp_type, [], [zero_point]),
    ]


def test_convert_hard_sigmoid_qdq__mismatched_input_output_quantization(intermediate_tflite_model_provider):
    """Verify a QDQ `HardSigmoid` whose input and output use different quantization parameters converts correctly.

    The QDQ cluster is constructed manually so the input `DequantizeLinear` and the output `QuantizeLinear` use
     deliberately different scale/zero-point values, which is not something the automatic calibration would
     necessarily produce.
    """
    input_shape = [4, 8, 16]

    # Deliberately different quantization parameters for the input and the output tensors.
    x_scale, x_zp = 0.08, -5
    y_scale, y_zp = 0.004, -110

    nodes = [
        onnx.helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_q"]),
        onnx.helper.make_node("DequantizeLinear", ["x_q", "x_scale", "x_zero_point"], ["x_dq"]),
        onnx.helper.make_node("HardSigmoid", ["x_dq"], ["hs"], alpha=0.25, beta=0.6),
        onnx.helper.make_node("QuantizeLinear", ["hs", "y_scale", "y_zero_point"], ["hs_q"]),
        onnx.helper.make_node("DequantizeLinear", ["hs_q", "y_scale", "y_zero_point"], ["output"]),
    ]

    initializers = []
    initializers.extend(_make_qdq_tensor("x", x_scale, x_zp, TensorProto.INT8))
    initializers.extend(_make_qdq_tensor("y", y_scale, y_zp, TensorProto.INT8))

    graph = onnx.helper.make_graph(
        nodes,
        "HardSigmoid mismatched q-params test",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        initializer=initializers,
    )
    quantized_model = onnx.helper.make_model(graph)

    input_data = (np.random.random(input_shape).astype(np.float32) - 0.5) * 20  # Values from -10 to 10

    executors.convert_run_compare(quantized_model, input_data)

    ops = intermediate_tflite_model_provider.get_operators()
    op_codes = [intermediate_tflite_model_provider.get_operator_code_at_index(op.opcode_index) for op in ops]
    assert op_codes == [BuiltinOperator.QUANTIZE, BuiltinOperator.MUL, BuiltinOperator.ADD, BuiltinOperator.MINIMUM, 
                        BuiltinOperator.RELU, BuiltinOperator.DEQUANTIZE]

    for op in ops[1:-1]:
        for tensor in list(op.tmp_inputs) + list(op.tmp_outputs):
            assert tensor.quantization is not None, (
                f"Intermediate tensor '{tensor.name}' is not quantized (missing quantization parameters)."
            )
            assert tensor.type in {TensorType.INT8, TensorType.UINT8}, (
                f"Intermediate tensor '{tensor.name}' has non-quantized type {tensor.type}."
            )
