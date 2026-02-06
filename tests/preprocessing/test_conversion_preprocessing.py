#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import numpy as np
import onnx
from onnx import TensorProto

from onnx2quant.qdq_quantization import QDQQuantizer, RandomDataCalibrationDataReader, InputSpec
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.converter.preprocessing_steps.duplicate_dequantize_linear_for_each_consumer import (
    DuplicateDequantizeLinearForEachConsumer,
)
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from tests.executors import OnnxExecutor, convert_run_compare


def test_duplicate_dequantize_linear___duplicate_for_each_consumer():
    shape = [3, 16, 16]

    graph = onnx.helper.make_graph(
        [
            # Tensor name 'static_1_duplicated_1' is used intentionally to test potential tensor name conflict
            onnx.helper.make_node('Add', ['static_1_duplicated_1', 'x'], ['a']),
            onnx.helper.make_node(
                'DequantizeLinear',
                ['static_1', 'scale', 'zero_point'], ['dequant_out'],
                name="original_dequant"
            ),
            onnx.helper.make_node('Add', ['dequant_out', 'a'], ['y1']),
            onnx.helper.make_node('Add', ['dequant_out', 'static_2'], ['y2']),
            onnx.helper.make_node('Add', ['dequant_out', 'static_3'], ['y3']),
        ],
        name='Duplicate DequantizeLinear test',
        inputs=[
            onnx.helper.make_tensor_value_info('static_1_duplicated_1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info('y1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('y2', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('y3', TensorProto.FLOAT, shape),
        ],
        initializer=[
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [1], [0.0042]),
            onnx.helper.make_tensor('zero_point', TensorProto.INT8, [1], [12]),
            onnx.helper.make_tensor('static_1', TensorProto.INT8, [], [10]),
            onnx.helper.make_tensor('static_2', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
            onnx.helper.make_tensor('static_3', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
        ]
    )
    # noinspection DuplicatedCode
    onnx_model = onnx.helper.make_model(graph)
    onnx_model = ModelShapeInference.infer_shapes(onnx_model)

    onnx.checker.check_model(onnx_model)

    preprocessing_step = DuplicateDequantizeLinearForEachConsumer(onnx_model)
    preprocessing_step.run()

    # Check that the original DequantizeLinear was removed
    dequant_nodes = [node for node in onnx_model.graph.node if node.name == "original_dequant"]
    assert len(dequant_nodes) == 0, "Original DequantizeLinear node should be removed"

    # Check that new DequantizeLinear nodes were created for each consumer
    new_dequant_nodes = [node for node in onnx_model.graph.node if node.op_type == "DequantizeLinear"]
    assert len(new_dequant_nodes) == 3, f"Expected 3 new DequantizeLinear nodes, got {len(new_dequant_nodes)}"

    # Check that each new node has unique tensor names
    expected_outputs = ["dequant_out_duplicated_0", "dequant_out_duplicated_1", "dequant_out_duplicated_2"]
    actual_outputs = sorted([node.output[0] for node in new_dequant_nodes])
    assert actual_outputs == expected_outputs, f"Expected outputs {expected_outputs}, got {actual_outputs}"

    # Check that the value_info contains the new output tensors
    output_names = [vi.name for vi in onnx_model.graph.value_info]
    for expected_output in expected_outputs:
        assert expected_output in output_names, f"Expected output {expected_output} not found in value_info"

    # Check that the consumers use the new DequantizeLinear outputs
    add_nodes = [node for node in onnx_model.graph.node if node.op_type == "Add"]
    assert len(add_nodes) == 4

    consumer_inputs = [node.input[0] for node in add_nodes]
    assert "dequant_out_duplicated_0" in consumer_inputs, "One Add node should use dequant_out_duplicated_0"
    assert "dequant_out_duplicated_1" in consumer_inputs, "One Add node should use dequant_out_duplicated_1"
    assert "dequant_out_duplicated_2" in consumer_inputs, "One Add node should use dequant_out_duplicated_2"

    # Dry run model
    executor = OnnxExecutor(onnx_model.SerializeToString())
    executor.inference({
        0: np.random.random(shape).astype('float32'),
        1: np.random.random(shape).astype('float32'),
    })


def test_duplicate_dequantize_linear___duplicate_for_each_consumer__e2e(intermediate_tflite_model_provider):
    shape = [2, 3]

    onnx_graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Add", ["x1", "static_1"], ["y1"]),
            onnx.helper.make_node("Add", ["x2", "static_1"], ["y2"]),
            onnx.helper.make_node("Add", ["x3", "static_1"], ["y3"])
        ],
        name="Duplicate Dequantize e2e",
        inputs=[
            onnx.helper.make_tensor_value_info("x1", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("x2", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("x3", TensorProto.FLOAT, shape),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("y1", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("y2", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("y3", TensorProto.FLOAT, shape),
        ],
        initializer=[
            onnx.helper.make_tensor("static_1", TensorProto.FLOAT, [], [1.0]),
        ],
    )

    onnx_model = onnx.helper.make_model(onnx_graph)
    onnx.checker.check_model(onnx_model)

    config = RandomDataCalibrationDataReader({
        "x1": InputSpec(shape, np.float32),
        "x2": InputSpec(shape, np.float32),
        "x3": InputSpec(shape, np.float32)
    }).to_config()
    quantized_model = QDQQuantizer().quantize_model(onnx_model, config)

    input_data = {
        0: np.random.random(shape).astype(np.float32),
        1: np.random.random(shape).astype(np.float32),
        2: np.random.random(shape).astype(np.float32),
    }

    convert_run_compare(quantized_model, input_data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE, BuiltinOperator.QUANTIZE,
        BuiltinOperator.ADD, BuiltinOperator.ADD, BuiltinOperator.ADD,
        BuiltinOperator.DEQUANTIZE, BuiltinOperator.DEQUANTIZE, BuiltinOperator.DEQUANTIZE
    ])


def test_duplicate_dequantize_linear__no_duplicate__single_consumer():
    shape = [3, 16, 16]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                'DequantizeLinear',
                ['static_1', 'scale', 'zero_point'], ['dequant_out'],
                name="original_dequant"
            ),
            # DequantizeLinear has only 1 consumer
            onnx.helper.make_node('Add', ['dequant_out', 'static_2'], ['y']),
        ],
        name='No duplicate test',
        inputs=[onnx.helper.make_tensor_value_info('x1', TensorProto.INT8, shape)],
        outputs=[onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, shape)],
        initializer=[
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [1], [0.0042]),
            onnx.helper.make_tensor('zero_point', TensorProto.INT8, [1], [12]),
            onnx.helper.make_tensor('static_1', TensorProto.INT8, [1], [12]),
            onnx.helper.make_tensor('static_2', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
        ]
    )
    # noinspection DuplicatedCode
    onnx_model = onnx.helper.make_model(graph)
    onnx_model = ModelShapeInference.infer_shapes(onnx_model)

    onnx.checker.check_model(onnx_model)

    preprocessing_step = DuplicateDequantizeLinearForEachConsumer(onnx_model)
    preprocessing_step.run()

    # Check that the original DequantizeLinear was preserved
    dequant_nodes = [node for node in onnx_model.graph.node if node.name == "original_dequant"]
    assert len(dequant_nodes) == 1, "Only original DequantizeLinear node should be preserved"


def test_duplicate_dequantize_linear__no_duplicate__quantized_per_channel():
    shape = [2]

    # noinspection DuplicatedCode
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Add', ['x1', 'x2'], ['x3']),
            # Dequantize linear quantized per-channel
            onnx.helper.make_node(
                'DequantizeLinear',
                ['x', 'scale', 'zero_point'], ['dequant_out'],
                name="original_dequant"
            ),
            onnx.helper.make_node('Add', ['dequant_out', 'static_2'], ['y1']),
            onnx.helper.make_node('Add', ['dequant_out', 'static_3'], ['y2']),
            onnx.helper.make_node('Add', ['dequant_out', 'static_4'], ['y3']),
        ],
        name='Duplicate DequantizeLinear test',
        inputs=[
            onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info('y1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('y2', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('y3', TensorProto.FLOAT, shape),
        ],
        initializer=[
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [2], [0.0042, 0.0042]),
            onnx.helper.make_tensor('zero_point', TensorProto.INT8, [2], [10, 23]),
            onnx.helper.make_tensor('x', TensorProto.INT8, [2], [10, 20]),
            onnx.helper.make_tensor('static_2', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
            onnx.helper.make_tensor('static_3', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
            onnx.helper.make_tensor('static_4', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
        ]
    )
    # noinspection DuplicatedCode
    onnx_model = onnx.helper.make_model(graph)
    onnx_model = ModelShapeInference.infer_shapes(onnx_model)

    onnx.checker.check_model(onnx_model)

    preprocessing_step = DuplicateDequantizeLinearForEachConsumer(onnx_model)
    preprocessing_step.run()

    # Check that the original DequantizeLinear was preserved
    dequant_nodes = [node for node in onnx_model.graph.node if node.name == "original_dequant"]
    assert len(dequant_nodes) == 1, "Only original DequantizeLinear node should be preserved"


def test_duplicate_dequantize_linear__no_duplicate__input_not_initializer():
    shape = [3, 16, 16]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Add', ['x1', 'x2'], ['x3']),
            onnx.helper.make_node('DequantizeLinear', ['x', 'scale', 'zero_point'], ['dequant_out'],
                                  name="original_dequant"),
            onnx.helper.make_node('Add', ['dequant_out', 'static_1'], ['y1']),
            onnx.helper.make_node('Add', ['dequant_out', 'static_2'], ['y2']),
            onnx.helper.make_node('Add', ['dequant_out', 'static_3'], ['y3']),
        ],
        name='Duplicate DequantizeLinear test',
        inputs=[
            onnx.helper.make_tensor_value_info('x', TensorProto.INT8, shape),
            onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info('y1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('y2', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('y3', TensorProto.FLOAT, shape),
        ],
        initializer=[
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [1], [0.0042]),
            onnx.helper.make_tensor('zero_point', TensorProto.INT8, [1], [10]),
            onnx.helper.make_tensor('static_1', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
            onnx.helper.make_tensor('static_2', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
            onnx.helper.make_tensor('static_3', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
        ]
    )
    # noinspection DuplicatedCode
    onnx_model = onnx.helper.make_model(graph)
    onnx_model = ModelShapeInference.infer_shapes(onnx_model)

    onnx.checker.check_model(onnx_model)

    preprocessing_step = DuplicateDequantizeLinearForEachConsumer(onnx_model)
    preprocessing_step.run()

    # Check that the original DequantizeLinear was preserved
    dequant_nodes = [node for node in onnx_model.graph.node if node.name == "original_dequant"]
    assert len(dequant_nodes) == 1, "Only original DequantizeLinear node should be preserved"


def test_duplicate_dequantize_linear__no_duplicate__unknown_output_shape():
    shape = [3, 16, 16]

    # noinspection DuplicatedCode
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Add', ['x1', 'x2'], ['x3']),
            onnx.helper.make_node('DequantizeLinear', ['x', 'scale', 'zero_point'], ['dequant_out'],
                                  name="original_dequant"),
            onnx.helper.make_node('Add', ['dequant_out', 'static_2'], ['y1']),
            onnx.helper.make_node('Add', ['dequant_out', 'static_3'], ['y2']),
            onnx.helper.make_node('Add', ['dequant_out', 'static_4'], ['y3']),
        ],
        name='Duplicate DequantizeLinear test',
        inputs=[
            onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info('y1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('y2', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('y3', TensorProto.FLOAT, shape),
        ],
        initializer=[
            onnx.helper.make_tensor('scale', TensorProto.FLOAT, [1], [0.0042]),
            onnx.helper.make_tensor('zero_point', TensorProto.INT8, [1], [10]),
            onnx.helper.make_tensor('x', TensorProto.INT8, [], [10]),
            onnx.helper.make_tensor('static_2', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
            onnx.helper.make_tensor('static_3', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
            onnx.helper.make_tensor('static_4', TensorProto.FLOAT, shape, np.random.random(shape).astype('float32')),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    # No model shape inference

    onnx.checker.check_model(onnx_model)

    preprocessing_step = DuplicateDequantizeLinearForEachConsumer(onnx_model)
    preprocessing_step.run()

    # Check that the original DequantizeLinear was preserved
    dequant_nodes = [node for node in onnx_model.graph.node if node.name == "original_dequant"]
    assert len(dequant_nodes) == 1, "Only original DequantizeLinear node should be preserved"
