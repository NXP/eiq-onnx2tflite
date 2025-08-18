#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import os
import pathlib
import subprocess
import tempfile

from onnx import TensorProto

from thirdparty.onnx import onnx

_CURRENT_DIR = pathlib.Path(__file__).parent
_DEFAULT_MODEL_PATH = pathlib.Path(os.path.join(_CURRENT_DIR.parent, "model.tflite"))

def assert_successful_conversion(result, model_path=_DEFAULT_MODEL_PATH):
    assert result.returncode == 0
    assert "Successfully converted" in result.stderr
    assert os.path.exists(model_path)
    os.remove(model_path)


def exec_conversion(args: list[str]):
    return subprocess.run(['python', '-m', "onnx2tflite"] + args, capture_output=True, text=True,
                          cwd=_CURRENT_DIR.parent)


def test_cmd_print_help():
    result = exec_conversion(["-h"])

    assert result.returncode == 0
    assert len(result.stderr) == 0
    assert result.stdout.startswith("usage: onnx2tflite")


def test_cmd_simple_conversion():
    _DEFAULT_MODEL_PATH.unlink(missing_ok=True)
    model_path = os.path.join(_CURRENT_DIR, "artifacts", "downloaded", "mnist-8", "model.onnx")

    result = exec_conversion([model_path, "--verbose"])

    assert_successful_conversion(result)


def test_cmd_symbolic_dimensions_conversion():
    _DEFAULT_MODEL_PATH.unlink(missing_ok=True)
    model_path = os.path.join(_CURRENT_DIR, "artifacts", "downloaded", "mnist-8", "model.onnx")

    result = exec_conversion([model_path, "-s", "batch:1", "--verbose"])

    assert_successful_conversion(result)


def test_cmd_invalid_symbolic_dimensions_conversion():
    model_path = os.path.join(_CURRENT_DIR, "artifacts", "downloaded", "mnist-8", "model.onnx")

    result = exec_conversion([model_path, "-s", "batch:", "--verbose"])

    assert result.returncode != 0
    assert "Symbolic dimension mapping 'batch:' in invalid format" in result.stderr


def test_cmd_qdq_aware_conversion():
    _DEFAULT_MODEL_PATH.unlink(missing_ok=True)
    model_path = os.path.join(_CURRENT_DIR, "artifacts", "downloaded", "squeezenet1.0-12-int8", "model.onnx")

    result = exec_conversion([model_path, "--qdq-aware-conversion", "--verbose"])

    assert_successful_conversion(result)


def test_cmd__dynamic_input_shape_error():
    input_shape = (1, None, None)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Exp', ['input'], ['output'])],
        'Exp test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.onnx")
        onnx.save_model(onnx_model, model_path)

        result = exec_conversion([model_path])

        assert result.returncode != 0
        assert "Model has dynamically defined inputs: 'input'." in result.stderr


def test_cmd__dynamic_and_symbolic_input():
    input_shape = (1, None, "x")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Exp', ['input'], ['output'])],
        'Exp test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.onnx")
        onnx.save_model(onnx_model, model_path)

        result = exec_conversion([model_path])

        assert result.returncode != 0
        assert "Model inputs are not statically defined." in result.stderr
        assert "They contain following symbolic dimensions: 'x'" in result.stderr
        assert "following input tensors are dynamic: 'input'." in result.stderr


def test_cmd__set_input_shape():
    input_shape = (1, None, None)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Exp', ['input'], ['output'])],
        'Exp test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.onnx")
        onnx.save_model(onnx_model, model_path)

        result = exec_conversion([model_path, "-m", "input:(1,5,4)", "--verbose"])

        assert_successful_conversion(result)


def test_cmd__set_input_shape__invalid_format():
    input_shape = (1, None, None)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Exp', ['input'], ['output'])],
        'Exp test',
        [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.onnx")
        onnx.save_model(onnx_model, model_path)

        result = exec_conversion([model_path, "-m", "input:(1a,5,4)"])

        assert result.returncode != 0
        assert "[Code.INVALID_INPUT] - Input shape definition 'input:(1a,5,4)' in invalid format." in result.stderr
