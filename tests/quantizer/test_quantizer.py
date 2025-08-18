#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto

# noinspection PyProtectedMember
from onnx2quant.__main__ import _quantize_model
from onnx2quant.qdq_quantization import InputSpec, QDQQuantizer, RandomDataCalibrationDataReader
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.src import logger
from onnx2tflite.src.tflite_generator import tflite_model
from tests import executors

_ROOT_DIR = pathlib.Path(__file__).parent.parent.parent


class randomCalibrationDataset:

    def __init__(self, shape, path="tmp_calibration_dataset", np_type=np.float32, items_count=5):
        self.path = os.path.join(_ROOT_DIR, path)
        self.shape = shape
        self.np_type = np_type
        self.items_count = items_count

    def __enter__(self):

        if os.path.exists(self.path):
            raise Exception(f"Directory with name '{self.path}' already exists!")
        os.mkdir(self.path)

        for x in range(self.items_count):
            input_vector = np.random.random(self.shape).reshape(self.shape).astype(self.np_type)
            np.save(os.path.join(self.path, f"{x}.npy"), input_vector)

        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path)


def assert_successful_quantization(result, model_dir=_ROOT_DIR, model_name="model_quant.onnx"):
    model_path = os.path.join(model_dir, model_name)
    assert result.returncode == 0, "Quantizer returned error!"
    assert "error" not in result.stderr.lower(), "Quantizer produced error message!"
    assert os.path.exists(model_path), "Quantizer hasn't produced quantized model file!"
    os.remove(model_path)


def exec_quantization(args: list[str]):
    return subprocess.run(['python', '-m', "onnx2quant"] + args, capture_output=True, text=True, cwd=_ROOT_DIR)


def test_cmd_print_help():
    result = exec_quantization(["-h"])

    assert result.returncode == 0
    assert len(result.stderr) == 0
    assert result.stdout.startswith("usage: onnx2quant")


def test_cmd_simple_quantization():
    model_path = os.path.join(_ROOT_DIR, "tests", "artifacts", "downloaded", "mnist-12", "model.onnx")

    with randomCalibrationDataset((1, 1, 28, 28), items_count=10):
        result = exec_quantization([model_path, "-c", f"Input3;tmp_calibration_dataset"])

    assert_successful_quantization(result)


def test_cmd_per_channel_quantization():
    model_dir = os.path.join(_ROOT_DIR, "tests", "artifacts", "downloaded", "googlenet-12")
    model_path = os.path.join(model_dir, "model.onnx")

    with randomCalibrationDataset((1, 3, 224, 224), items_count=10):
        result = exec_quantization(
            [model_path, "-c", f"data_0;tmp_calibration_dataset", "--per-channel"])

    quantized_model_path = os.path.join(_ROOT_DIR, "model_quant.onnx")
    model = onnx.load(quantized_model_path)

    assert_successful_quantization(result)

    conv_w_dequantize_name = 'conv1/7x7_s2_w_0_DequantizeLinear'
    conv_w_dequantize = [n for n in model.graph.node if n.name == conv_w_dequantize_name][0]

    w = [t for t in model.graph.initializer if t.name == conv_w_dequantize.input[0]][0]
    scale = [t for t in model.graph.initializer if t.name == conv_w_dequantize.input[1]][0]
    zp = [t for t in model.graph.initializer if t.name == conv_w_dequantize.input[2]][0]

    assert scale.dims[0] == zp.dims[0] == w.dims[0]
    assert scale.dims[0] != 1


def test_cmd_multi_input_model():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "y"], ["output"])],
        'Add test',
        [
            onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5]),
            onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 5]),
        ],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 5])],
    )
    onnx_model = onnx.helper.make_model(graph)

    onnx_model_name = "model_add.onnx"
    onnx.save(onnx_model, os.path.join(_ROOT_DIR, onnx_model_name))

    with randomCalibrationDataset((2, 5), path="calib_a", items_count=5):
        with randomCalibrationDataset((2, 5), path="calib_y", items_count=5):
            result = exec_quantization([onnx_model_name, "-c", f"x;calib_a", "-c", "y;calib_y"])

    assert_successful_quantization(result, model_name="model_add_quant.onnx")
    os.remove(os.path.join(_ROOT_DIR, onnx_model_name))


def test_cmd_dataset_size_not_correspond():
    model_path = os.path.join(_ROOT_DIR, "tests", "artifacts", "downloaded", "mnist-12", "model.onnx")

    with randomCalibrationDataset((1, 3, 224, 224), path="first_dataset", items_count=5):
        with randomCalibrationDataset((1, 3, 224, 224), path="another_dataset", items_count=4):
            result = exec_quantization([model_path, "-c", "Input3;first_dataset", "-c", "Inp;another_dataset"])

    assert result.returncode != 0
    assert "[ERROR] [Code.INVALID_INPUT] - Input dataset dirs don't contain same number of *.npy files." in result.stderr


def test_quantizer_via_code():
    model_path = os.path.join(_ROOT_DIR, "tests", "artifacts", "downloaded", "mnist-8", "model.onnx")
    model = onnx.load_model(model_path)
    with randomCalibrationDataset((1, 1, 28, 28), items_count=10) as dataset_dir:
        args = {
            "calibration_dataset_mapping": {"Input3": dataset_dir},
            "allow_opset_10_and_lower": True
        }

        _quantize_model(model, "model_quant.onnx", args)

    assert os.path.exists("model_quant.onnx")
    os.remove("model_quant.onnx")


def test_quantizer__set_input_shape__prog():
    input_shape = (1, None, None)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Softmax', ['input'], ['output'])],
        'Exp test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, "tmp_dataset_dir")
        output_model_path = os.path.join(temp_dir, "model_add_quant.onnx")

        with randomCalibrationDataset((1, 28, 28), path=dataset_path, items_count=10) as dataset_dir:
            args = {
                "calibration_dataset_mapping": {"input": dataset_dir},
                "allow_opset_10_and_lower": True,
                "input_shapes_mapping": {"input": (1, 28, 28)},
            }

            _quantize_model(onnx_model, output_model_path, args)

        assert os.path.exists(output_model_path)
        os.remove(output_model_path)


def test_quantizer__set_input_shape__cli():
    input_shape = (1, None, None)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Softmax', ['input'], ['output'])],
        'Exp test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, "tmp_dataset_dir")
        input_model_path = os.path.join(temp_dir, "model_softmax.onnx")
        output_model_path = os.path.join(temp_dir, "model_softmax_quant.onnx")

        onnx.save(onnx_model, input_model_path)

        with randomCalibrationDataset((1, 28, 28), path=dataset_path, items_count=10):
            result = exec_quantization([input_model_path,
                                        "-o", output_model_path,
                                        "-c", "input;" + dataset_path,
                                        "-m", "input:(1,28,28)"])
        assert_successful_quantization(result, pathlib.Path(temp_dir), "model_softmax_quant.onnx")


def test_quantizer__set_symbolic_dimension__cli():
    input_shape = ("N", 28, 28)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Softmax', ['input'], ['output'])],
        'Exp test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, "tmp_dataset_dir")
        input_model_path = os.path.join(temp_dir, "model_softmax.onnx")
        output_model_path = os.path.join(temp_dir, "model_softmax_quant.onnx")

        onnx.save(onnx_model, input_model_path)

        with randomCalibrationDataset((2, 28, 28), path=dataset_path, items_count=10):
            result = exec_quantization([input_model_path,
                                        "-o", output_model_path,
                                        "-c", "input;" + dataset_path,
                                        "-s", "N:2"])
        assert_successful_quantization(result, pathlib.Path(temp_dir), "model_softmax_quant.onnx")


def test_quantizer__incomplete_shape_inference__best_effort_model__generated():
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

    with pytest.raises(logger.Error) as e:
        # Quantize the model.
        q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
        QDQQuantizer().quantize_model(onnx_model, q_config)

    assert e.value.error_code == logger.Code.SHAPE_INFERENCE_ERROR
    assert 'sym_shape_infer_temp.onnx' in e.value.msg
    assert os.path.isfile(model_path)

    os.remove(model_path)  # Remove the generated file.


def test_quantizer__incomplete_shape_inference__best_effort_model__not_generated():
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

    with pytest.raises(logger.Error) as e:
        # Quantize the model.
        q_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
        q_config.generate_artifacts_after_failed_shape_inference = False  # Prohibit generation of the model.
        QDQQuantizer().quantize_model(onnx_model, q_config)

    assert e.value.error_code == logger.Code.SHAPE_INFERENCE_ERROR
    assert 'sym_shape_infer_temp.onnx' not in e.value.msg
    assert not os.path.isfile(model_path)  # Make sure the file wasn't generated.


def test_quantizer__random_data_calibration_data_reader__specific_ranges(intermediate_tflite_model_provider):
    shape = [13, 37]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Add', ['x1', 'x2'], ['y1']),
            onnx.helper.make_node('Cast', ['x3'], ['y2'], to=TensorProto.FLOAT),
            onnx.helper.make_node('Mul', ['y1', 'y2'], ['y'])
        ],
        'Quantizer test',
        [
            onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x3', TensorProto.INT64, shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    q_config = QuantizationConfig(RandomDataCalibrationDataReader({
        'x1': InputSpec(shape, np.float32, -4.2, 13.37),
        'x2': InputSpec(shape, np.float32, -42),
        'x3': InputSpec(shape, np.int64, max=420)  # INT64
    }))
    onnx_model = QDQQuantizer().quantize_model(onnx_model, q_config)

    np.random.seed(42)
    data = {
        0: np.random.uniform(-4.2, 13.37, shape).astype('float32'),
        1: np.random.uniform(-42, 1., shape).astype('float32'),
        2: np.random.uniform(0., 420, shape).astype('int64')
    }

    executors.convert_run_compare(onnx_model, data)

    tensors = intermediate_tflite_model_provider.get_tensors()
    x1 = [t for t in tensors if t.name == 'x1_DequantizeLinear_Output']
    x2 = [t for t in tensors if t.name == 'x2_DequantizeLinear_Output']
    x3 = [t for t in tensors if t.name == 'y2_DequantizeLinear_Output']
    assert len(x1) == len(x2) == len(x3) == 1

    def _compute_range(quantization: tflite_model.Quantization) -> (float, float):
        # Assuming int8 quantization.
        scale = quantization.scale.vector[0]
        zp = quantization.zero_point.vector[0]
        return (
            scale * (-128 - zp),
            scale * (127 - zp)
        )

    # Check that the quantization parameters of the inputs just about cover the specified ranges.
    assert np.allclose(_compute_range(x1[0].quantization), [-4.2, 13.37], atol=0.2)
    assert np.allclose(_compute_range(x2[0].quantization), [-42., 1.], atol=0.2)
    assert np.allclose(_compute_range(x3[0].quantization), [0., 420.], atol=1.1)
