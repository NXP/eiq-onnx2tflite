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
