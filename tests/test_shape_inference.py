#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import os.path
import pathlib

import onnx
import onnx.shape_inference
import pytest
from onnx import TensorProto
from onnxruntime.tools.symbolic_shape_infer import get_shape_from_type_proto

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter.convert import convert_model
from onnx2tflite.src.logger import conversion_log
from onnx2tflite.src.model_shape_inference import ModelShapeInference


@pytest.mark.parametrize("input_shape,output_shape", [
    pytest.param(("batch", 1280, 7, 7), [1, 1280, 1, 1], id="symbolic input - (batch, 1280, 7, 7)"),
    pytest.param((1, 1280, 7, 7), [1, 1280, 1, 1], id="static input - (1, 1280, 7, 7)"),
])
def test_shape_inference_qlinear_global_average_pool(input_shape, output_shape):
    node = onnx.helper.make_node(
        op_type="QLinearGlobalAveragePool",
        inputs=["X_quantized", "X_scale", "X_zero_point", "Z_scale", "Z_zero_point"],
        outputs=["Z_quantized"],
        name="GlobalAveragePool_quant",
        domain="com.microsoft",
        channels_last=0)

    graph = onnx.helper.make_graph(
        nodes=[node],
        name="model",
        inputs=[onnx.helper.make_tensor_value_info("X_quantized", TensorProto.UINT8, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info("Z_quantized", TensorProto.UINT8, None)],
        initializer=[
            onnx.helper.make_tensor("X_scale", onnx.TensorProto.FLOAT, (), [0.0235294122248888]),
            onnx.helper.make_tensor("X_zero_point", onnx.TensorProto.UINT8, (), [0]),
            onnx.helper.make_tensor("Z_scale", onnx.TensorProto.FLOAT, (), [0.019110053777694702]),
            onnx.helper.make_tensor("Z_zero_point", onnx.TensorProto.UINT8, (), [0])
        ]
    )

    onnx_model = onnx.helper.make_model(
        graph,
        producer_name="onnx-QLinearGlobalAveragePool",
        opset_imports=[onnx.helper.make_opsetid(domain="com.microsoft", version=1),
                       onnx.helper.make_opsetid(domain="", version=8)])

    inferred_model = ModelShapeInference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model, full_check=True)

    inferred_output_shape = get_shape_from_type_proto(inferred_model.graph.output[0].type)
    assert inferred_output_shape == output_shape


def test_shape_inference_qlinear_global_average_pool_channel_last():
    input_shape = ("batch_size", 7, 7, 1280)
    output_shape = [1, 1, 1, 1280]

    node = onnx.helper.make_node(
        op_type="QLinearGlobalAveragePool",
        inputs=["X_quantized", "X_scale", "X_zero_point", "Z_scale", "Z_zero_point"],
        outputs=["Z_quantized"],
        name="GlobalAveragePool_quant",
        domain="com.microsoft",
        channels_last=1)

    graph = onnx.helper.make_graph(
        nodes=[node],
        name="model",
        inputs=[onnx.helper.make_tensor_value_info("X_quantized", TensorProto.UINT8, input_shape)],
        outputs=[onnx.helper.make_tensor_value_info("Z_quantized", TensorProto.UINT8, None)],
        initializer=[
            onnx.helper.make_tensor("X_scale", onnx.TensorProto.FLOAT, (), [0.0235294122248888]),
            onnx.helper.make_tensor("X_zero_point", onnx.TensorProto.UINT8, (), [0]),
            onnx.helper.make_tensor("Z_scale", onnx.TensorProto.FLOAT, (), [0.019110053777694702]),
            onnx.helper.make_tensor("Z_zero_point", onnx.TensorProto.UINT8, (), [0])
        ]
    )

    onnx_model = onnx.helper.make_model(
        graph,
        producer_name="onnx-QLinearGlobalAveragePool",
        opset_imports=[onnx.helper.make_opsetid(domain="com.microsoft", version=1),
                       onnx.helper.make_opsetid(domain="", version=8)])

    inferred_model = ModelShapeInference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model, full_check=True)

    inferred_output_shape = get_shape_from_type_proto(inferred_model.graph.output[0].type)
    assert inferred_output_shape == output_shape


def test_shape_inference__multiple_inputs_symbolic():
    input_shape = ("batch", 24, 56, 56)

    node = onnx.helper.make_node(
        op_type="QLinearAdd",
        inputs=["X_quantized", "X_scale", "X_zero_point",
                "Y_quantized", "Y_scale", "Y_zero_point",
                "Z_scale", "Z_zero_point"],
        outputs=["Z_quantized"],
        name="Add_15_quant",
        domain="com.microsoft")

    graph = onnx.helper.make_graph(
        nodes=[node],
        name="model",
        inputs=[
            onnx.helper.make_tensor_value_info("X_quantized", TensorProto.UINT8, input_shape),
            onnx.helper.make_tensor_value_info("Y_quantized", TensorProto.UINT8, input_shape),
        ],
        outputs=[onnx.helper.make_tensor_value_info("Z_quantized", TensorProto.UINT8, None)],
        initializer=[
            onnx.helper.make_tensor("X_scale", onnx.TensorProto.FLOAT, (), [0.039710190147161484]),
            onnx.helper.make_tensor("X_zero_point", onnx.TensorProto.UINT8, (), [107]),
            onnx.helper.make_tensor("Y_scale", onnx.TensorProto.FLOAT, (), [0.04609474167227745]),
            onnx.helper.make_tensor("Y_zero_point", onnx.TensorProto.UINT8, (), [127]),
            onnx.helper.make_tensor("Z_scale", onnx.TensorProto.FLOAT, (), [0.05797068774700165]),
            onnx.helper.make_tensor("Z_zero_point", onnx.TensorProto.UINT8, (), [125]),
        ]
    )

    onnx_model = onnx.helper.make_model(
        graph,
        producer_name="onnx-QLinearAdd",
        opset_imports=[onnx.helper.make_opsetid(domain="com.microsoft", version=1),
                       onnx.helper.make_opsetid(domain="", version=8)])

    with pytest.raises(logger.Error) as e:
        ModelShapeInference.infer_shapes(onnx_model)

    assert "Model inputs contain following symbolic dimensions: 'batch'." in e.value.msg


@pytest.mark.parametrize("input_shape,axis", [
    pytest.param(("batch_size", 3), 0, id="2D,axis=0"),
    pytest.param(("batch_size", 3, 10), 1, id="3D,axis=1"),
])
def test_symbolic_batch_dim_making_static(input_shape, axis):
    node = onnx.helper.make_node("Softmax", ["x"], ["y"], axis=axis)

    graph = onnx.helper.make_graph(
        [node],
        "graph-softmax",
        [onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, input_shape)],
    )

    model = onnx.helper.make_model(graph, producer_name="onnx-softmax")
    onnx.checker.check_model(model)

    inferred_model = ModelShapeInference.infer_shapes(model)

    assert inferred_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1


def test_ignoring_automatic_symbolic_batch_dim_for_model_with_2_inputs():
    shape = ["N", 2, 6]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["x", "y"], ["o"])],
        'Mul test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("o", TensorProto.FLOAT, ())],
    )
    model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error) as e:
        ModelShapeInference.infer_shapes(model)

    assert "Model inputs contain following symbolic dimensions: 'N'." in e.value.msg


def test_undefined_symbolic_dimensions_logged():
    x_shape = ["batch", "w", 15]
    y_shape = [15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Add", ["x", "y"], ["z"]),
            onnx.helper.make_node('Softmax', ['z'], ['output']),
        ],
        'Add test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape)
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    with pytest.raises(logger.Error):
        convert_model(onnx_model)

    error_log = logger.conversion_log.get_logs()["shape_inference"][0]["message"]
    assert "[Code.INVALID_INPUT] - Model inputs contain following symbolic dimensions:" in error_log
    assert 'batch' in error_log
    assert 'w' in error_log


def test_symbolic_dimensions_definition():
    x_shape = ["batch", "w", 15]
    y_shape = [15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Add", ["x", "y"], ["z"]),
            onnx.helper.make_node('Softmax', ['z'], ['output']),
        ],
        'Add test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape)
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    cc = ConversionConfig()
    cc.symbolic_dimensions_mapping = {
        "batch": 1,
        "w": 10
    }
    _ = convert_model(onnx_model, conversion_config=cc)

    assert len(conversion_log.get_logs()) == 0


def test_partial_symbolic_dimensions_definition():
    x_shape = ["batch", "w", 15]
    y_shape = [15]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Add", ["x", "y"], ["z"]),
            onnx.helper.make_node('Softmax', ['z'], ['output']),
        ],
        'Add test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape)
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())]
    )

    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    cc = ConversionConfig()
    cc.symbolic_dimensions_mapping = {"batch": 1}

    with pytest.raises(logger.Error):
        convert_model(onnx_model, conversion_config=cc)

    error_log = logger.conversion_log.get_logs()["shape_inference"][0]["message"]
    assert "[Code.INVALID_INPUT] - Model inputs contain following symbolic dimensions: 'w'." in error_log


def test_slice___negative_int_max_ends():
    shape = [4, 2]
    int_max = (2 ** 31 - 1)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Slice", ["data", "starts", "ends", "axes", "steps"], ["output"])],
        'Slice test',
        [
            onnx.helper.make_tensor_value_info("data", TensorProto.FLOAT, shape),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
        [
            onnx.helper.make_tensor("starts", TensorProto.INT64, [1], [-1]),
            onnx.helper.make_tensor("ends", TensorProto.INT64, [1], [-int_max]),
            onnx.helper.make_tensor("axes", TensorProto.INT64, [1], [0]),
            onnx.helper.make_tensor("steps", TensorProto.INT64, [1], [-1]),
        ]
    )
    model = onnx.helper.make_model(graph)

    inferred_model = ModelShapeInference.infer_shapes(model)

    assert inferred_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 4
    assert inferred_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 2


def test_incomplete_shape_inference__best_effort_model__generated():
    model_path = pathlib.Path(os.getcwd()).joinpath("sym_shape_infer_temp.onnx")
    if os.path.exists(model_path):  # Remove the file if it already exists.
        os.remove(model_path)

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Reshape', ['x', 'new_shape'], ['y'])],
        'Test shape inference',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('new_shape', TensorProto.INT64, [2]),
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert_model(onnx_model)
    error = logger.conversion_log.get_logs()['shape_inference'][0]

    assert error['error_code'] == logger.Code.SHAPE_INFERENCE_ERROR
    assert 'sym_shape_infer_temp.onnx' in error['message']
    assert os.path.isfile(model_path)

    os.remove(model_path)  # Remove the generated file.


def test_incomplete_shape_inference__best_effort_model__not_generated():
    model_path = pathlib.Path(os.getcwd()).joinpath("sym_shape_infer_temp.onnx")
    if os.path.exists(model_path):  # Remove the file if it exists.
        os.remove(model_path)

    shape = [42]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Reshape', ['x', 'new_shape'], ['y'])],
        'Test shape inference',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('new_shape', TensorProto.INT64, [2]),
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    config = ConversionConfig()
    config.generate_artifacts_after_failed_shape_inference = False  # Prohibit generation of the model.

    with pytest.raises(logger.Error):
        convert_model(onnx_model, conversion_config=config)
    error = logger.conversion_log.get_logs()['shape_inference'][0]

    assert error['error_code'] == logger.Code.SHAPE_INFERENCE_ERROR
    assert 'sym_shape_infer_temp.onnx' not in error['message']
    assert not os.path.isfile(model_path)  # Make sure the file wasn't generated.


def test_shape_inference__operator_with_impossible_shape_inference():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Unique', ['x'], ['y'])],
        'Test shape inference',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, [42])],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error):
        convert_model(onnx_model)
    error = logger.conversion_log.get_logs()['shape_inference'][0]

    assert error['error_code'] == logger.Code.SHAPE_INFERENCE_ERROR
    assert 'Unique' in error['message']


def test_shape_inference__unexpected_error():
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Add', ['x', 'x'], ['x1']),
            onnx.helper.make_node('Mul', [], ['x2']),  # No inputs! Shape inference will crash.
            onnx.helper.make_node('Div', ['x1', 'x1'], ['y'])
        ],
        'Test shape inference',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, [42])],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with pytest.raises(logger.Error) as e:
        ModelShapeInference.infer_shapes(onnx_model)

    assert e.value.error_code == logger.Code.SHAPE_INFERENCE_ERROR
    assert 'cause might be the `Mul` operator' in e.value.msg
    assert "output tensors ['x2']" in e.value.msg
