#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import os
import pathlib

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from onnx2quant.qdq_quantization import InputSpec, QDQClustersRecognizer, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2tflite.src import model_inspector
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.onnx_parser import onnx_model

_ARTIFACTS_DIR = pathlib.Path(__file__).parent.parent.parent.joinpath("artifacts")


@pytest.fixture(scope="session")
def single_add_node_qdq():
    quantizer = QDQQuantizer()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "y"], ["output"])],
        'Add test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5]),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 5]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)
    calibration_data_reader = RandomDataCalibrationDataReader({
        "x": InputSpec([2, 5], np.float32),
        "y": InputSpec([2, 5], np.float32),
    })
    quantized_model = quantizer.quantize_model(onnx_model, calibration_data_reader.to_config())
    return ModelShapeInference.infer_shapes(quantized_model)


@pytest.fixture(scope="session")
def alexnet_qdq_conv_only():
    model = onnx.load(os.path.join(_ARTIFACTS_DIR, "downloaded", "bvlcalexnet-12", "model.onnx"))

    quantizer = QDQQuantizer(op_types_to_quantize=["Conv"])
    calibration_data_reader = RandomDataCalibrationDataReader({"data_0": InputSpec([1, 3, 224, 224], np.float32)})
    quantized_model = quantizer.quantize_model(model, calibration_data_reader.to_config())

    return ModelShapeInference.infer_shapes(quantized_model)


def test_quantize_add_model(single_add_node_qdq):
    ops = [node.op_type for node in single_add_node_qdq.graph.node]

    assert ops.count("Add") == 1
    assert ops.count("QuantizeLinear") == 3
    assert ops.count("DequantizeLinear") == 3


def test_recognize_qdq_ops_in_add_model(single_add_node_qdq):
    model = onnx_model.ModelProto(single_add_node_qdq)
    onnx_inspector = model_inspector.ONNXModelInspector(model)
    recognizer = QDQClustersRecognizer(onnx_inspector)
    recognized_ops = recognizer.recognize_ops()

    assert (recognized_ops.standalone_quantization_ops ==
            ["x_QuantizeLinear_0", "y_QuantizeLinear_1", "output_DequantizeLinear_6"])
    assert (recognized_ops.qdq_cluster_quantization_ops ==
            ["x_DequantizeLinear_2", "y_DequantizeLinear_3", "output_QuantizeLinear_5"])
    assert recognized_ops.quantized_float_ops == ["Add_4"]


def test_recognize_qdq_conv_ops_in_alexnet_model(alexnet_qdq_conv_only):
    quantized_model = onnx_model.ModelProto(alexnet_qdq_conv_only)
    onnx_inspector = model_inspector.ONNXModelInspector(quantized_model)
    recognizer = QDQClustersRecognizer(onnx_inspector, supported_qdq_ops=["Conv"])
    recognized_ops = recognizer.recognize_ops()

    assert len(recognized_ops.quantized_float_ops) == 5  # Model has 5 Conv nodes
    assert len(recognized_ops.qdq_cluster_quantization_ops) == 20  # Every Conv node surrounded by 4 q-ops
    assert len(recognized_ops.standalone_quantization_ops) == 10  # Two for each Conv node
    assert len(recognized_ops.non_quantized_float_ops) == 17  # Non-Conv (non-quantized) float nodes


def test_not_recognize_unsupported_qdq_ops_in_alexnet_model(alexnet_qdq_conv_only):
    quantized_model = onnx_model.ModelProto(alexnet_qdq_conv_only)
    onnx_inspector = model_inspector.ONNXModelInspector(quantized_model)
    recognizer = QDQClustersRecognizer(onnx_inspector, supported_qdq_ops=["NonExistent"])
    recognized_ops = recognizer.recognize_ops()

    assert len(recognized_ops.qdq_cluster_quantization_ops) == 0  # No q-ops recognized as part of a cluster
