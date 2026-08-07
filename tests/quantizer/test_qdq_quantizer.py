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
from onnx2quant.quantization_config import QuantizationConfig


def test_quantizer__conv_dynamic_weight__weight_zero_point_is_zero():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", inputs=["x", "w"], outputs=["y"], pads=[1, 1, 1, 1])],
        'Dynamic weight conv test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8]),
            onnx.helper.make_tensor_value_info("w", TensorProto.FLOAT, [2, 4, 3, 3]),
        ],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 8, 8])],
    )
    onnx_model = onnx.helper.make_model(graph)

    q_config = QuantizationConfig(RandomDataCalibrationDataReader({
        'x': InputSpec([1, 4, 8, 8], np.float32),
        'w': InputSpec([2, 4, 3, 3], np.float32),
    }))
    model = QDQQuantizer().quantize_model(onnx_model, q_config)
    weight_dq = model.graph.node[2]

    zp_init = next((t for t in model.graph.initializer if t.name == weight_dq.input[2]), None)
    assert zp_init is not None, f"Zero point initializer '{weight_dq.input[2]}' not found."
    assert np.all(np.array(onnx.numpy_helper.to_array(zp_init)).flatten() == 0), (
        "Conv dynamic weight zero_point must be 0 (INT8). "
        "TFLite Conv2D/DepthwiseConv2D require symmetric weight quantization."
    )


def test_quantizer__conv_static_weight__weight_zero_point_is_zero():
    weight_data = np.random.randn(2, 4, 3, 3).astype(np.float32)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", inputs=["x", "w"], outputs=["y"], pads=[1, 1, 1, 1])],
        'Static weight conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 8, 8])],
        initializer=[onnx.numpy_helper.from_array(weight_data, name="w")],
    )
    onnx_model = onnx.helper.make_model(graph)

    q_config = QuantizationConfig(RandomDataCalibrationDataReader({
        'x': InputSpec([1, 4, 8, 8], np.float32),
    }))
    model = QDQQuantizer().quantize_model(onnx_model, q_config)
    weight_dq = model.graph.node[0]

    zp_init = next((t for t in model.graph.initializer if t.name == weight_dq.input[2]), None)
    assert zp_init is not None, f"Zero point initializer '{weight_dq.input[2]}' not found."
    assert np.all(np.array(onnx.numpy_helper.to_array(zp_init)).flatten() == 0), (
        "Static Conv weight zero_point must be 0 (INT8 symmetric)."
    )


def _build_conv_relu_conv_model() -> onnx.ModelProto:
    """Conv('conv0') -> Relu('relu0') -> Conv('conv1')."""
    w0 = np.random.randn(2, 4, 3, 3).astype(np.float32)
    w1 = np.random.randn(2, 2, 3, 3).astype(np.float32)

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Conv", inputs=["x", "w0"], outputs=["c0"], name="conv0"),
            onnx.helper.make_node("Relu", inputs=["c0"], outputs=["r0"], name="relu0"),
            onnx.helper.make_node("Conv", inputs=["r0", "w1"], outputs=["y"], name="conv1"),
        ],
        'Conv Relu Conv test',
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 8, 8])],
        initializer=[
            onnx.numpy_helper.from_array(w0, name="w0"),
            onnx.numpy_helper.from_array(w1, name="w1"),
        ],
    )
    return onnx.helper.make_model(graph)


def _input_quantized(model: onnx.ModelProto, node_name: str, input_idx: int) -> bool:
    dq_outputs = {node.output[0] for node in model.graph.node if node.op_type == "DequantizeLinear"}
    op = next(node for node in model.graph.node if node.name == node_name)
    return op.input[input_idx] in dq_outputs


def test_quantizer__nodes_to_exclude__excluded_conv_not_quantized():
    onnx_model = _build_conv_relu_conv_model()

    qc = QuantizationConfig(
        RandomDataCalibrationDataReader({'x': InputSpec([1, 4, 8, 8], np.float32)}),
        {"nodes_to_exclude": ["conv1"]},
    )
    model = QDQQuantizer().quantize_model(onnx_model, qc)

    assert _input_quantized(model, "conv0", 1), "Non-excluded 'conv0' should be quantized."
    assert not _input_quantized(model, "conv1", 1), (
        "Excluded node 'conv1' must not have a quantized weight."
    )


def test_quantizer__op_types_to_quantize__only_listed_op_type_quantized():
    onnx_model = _build_conv_relu_conv_model()

    qc = QuantizationConfig(
        RandomDataCalibrationDataReader({'x': InputSpec([1, 4, 8, 8], np.float32)}),
        {"op_types_to_quantize": ["Relu"]},
    )
    model = QDQQuantizer().quantize_model(onnx_model, qc)
    onnx.save_model(model, "quantized.onnx")

    assert not _input_quantized(model, "conv0", 1), (
        "Only 'Relu' was requested; 'Conv' nodes must stay unquantized."
    )
    assert not _input_quantized(model, "conv1", 1), (
        "Only 'Relu' was requested; 'Conv' nodes must stay unquantized."
    )
    assert _input_quantized(model, "relu0", 0), (
        "Requested 'relu0' should be quantized."
    )


def test_quantizer__nodes_to_quantize__only_listed_node_quantized():
    onnx_model = _build_conv_relu_conv_model()

    qc = QuantizationConfig(
        RandomDataCalibrationDataReader({'x': InputSpec([1, 4, 8, 8], np.float32)}),
        {"nodes_to_quantize": ["conv0"]},
    )
    model = QDQQuantizer().quantize_model(onnx_model, qc)

    assert _input_quantized(model, "conv0", 1), "Requested 'conv0' should be quantized."
    assert not _input_quantized(model, "conv1", 1), (
        "Only 'conv0' was requested for quantization; 'conv1' must stay unquantized."
    )
