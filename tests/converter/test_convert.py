#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import os
import pathlib
from typing import Dict, Tuple

import onnx
import pytest
from onnx import numpy_helper

import tests.converter.enabled_onnx_tests as enabled_onnx_tests
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from tests.converter.full_model_testing_utils import load_test_artifacts, full_model_test, full_quantized_model_test, \
    load_full_model_specs
from tests.executors import OnnxExecutor, TFLiteExecutor, compare_output_arrays

_ARTIFACTS_DIR = pathlib.Path(__file__).parent.parent.joinpath("artifacts")
_ONNX_TESTS_DIR = os.path.join(pathlib.Path(__file__).parent.parent.parent,
                               "thirdparty", "onnx", "onnx", "backend", "test", "data")

ONNX_TEST_ARTIFACTS = load_test_artifacts(_ONNX_TESTS_DIR, "node", enabled_onnx_tests.NODE)
ONNX_TEST_ARTIFACTS += load_test_artifacts(_ONNX_TESTS_DIR, "pytorch-converted", enabled_onnx_tests.PYTORCH_CONVERTED)
ONNX_TEST_ARTIFACTS += load_test_artifacts(_ONNX_TESTS_DIR, "pytorch-operator", enabled_onnx_tests.PYTORCH_OPERATOR)
ONNX_TEST_ARTIFACTS += load_test_artifacts(_ONNX_TESTS_DIR, "simple", enabled_onnx_tests.SIMPLE)


def _load_input_tensors(test_data_dir) -> Tuple[Dict, Dict]:
    input_data_onnx = {}
    input_data_tflite = {}

    for input_file in os.listdir(test_data_dir):
        if input_file.startswith("input_"):
            index = int(input_file[6:-3])

            tensor = onnx.load_tensor(os.path.join(test_data_dir, input_file))
            input_data_onnx[index] = numpy_helper.to_array(tensor)
            input_data_tflite[index] = input_data_onnx[index]

    return input_data_onnx, input_data_tflite


@pytest.mark.parametrize("onnx_artifact_dir,test_args", ONNX_TEST_ARTIFACTS)
def test_output_of_test_onnx_model_corresponds_to_generated_tflite(onnx_artifact_dir: str, test_args: dict):
    model_path = os.path.join(onnx_artifact_dir, "model.onnx")
    conversion_args = test_args.get(enabled_onnx_tests.CONVERSION_ARGS, dict())
    comparison_args = test_args.get(enabled_onnx_tests.COMPARISON_ARGS, dict())
    conversion_error = test_args.get(enabled_onnx_tests.CONVERSION_ERROR, None)

    if conversion_error is not None:
        with pytest.raises(logger.Error):
            bytes(convert.convert_model(model_path, ConversionConfig(conversion_args)))
        assert (logger.conversion_log.get_node_error_code(conversion_error[enabled_onnx_tests.CONVERSION_ERROR_NODE]) ==
                conversion_error[enabled_onnx_tests.CONVERSION_ERROR_ERROR])
        assert (logger.conversion_log.get_node_error_message(
            conversion_error[enabled_onnx_tests.CONVERSION_ERROR_NODE]) ==
                conversion_error[enabled_onnx_tests.CONVERSION_ERROR_MSG])
        return
    else:
        converted_tflite = bytes(convert.convert_model(model_path, ConversionConfig(conversion_args)))

    input_data_onnx, input_data_tflite = _load_input_tensors(os.path.join(onnx_artifact_dir, "test_data_set_0"))

    onnx_executor = OnnxExecutor(model_path)
    onnx_output = onnx_executor.inference(input_data_onnx)

    tflite_executor = TFLiteExecutor(model_content=converted_tflite)
    tflite_output = tflite_executor.inference(input_data_tflite)

    if isinstance(tflite_output, dict):
        for output_name, tflite_out in tflite_output.items():
            compare_output_arrays(tflite_out, onnx_output[output_name], output_name, **comparison_args)
    else:
        compare_output_arrays(tflite_output, onnx_output, 'main output', **comparison_args)


ONNX_FULL_MODEL_SPECS = load_full_model_specs(_ARTIFACTS_DIR, "downloaded", enabled_onnx_tests.ONNX_ZOO_MODELS)


@pytest.mark.parametrize("onnx_artifact_dir, atol, conversion_args", ONNX_FULL_MODEL_SPECS)
@full_model_test
def test_output__full_onnx_models(onnx_artifact_dir: str, atol: float, conversion_args: dict):
    model_path = os.path.join(onnx_artifact_dir, "model.onnx")
    input_data_onnx, input_data_tflite = _load_input_tensors(os.path.join(onnx_artifact_dir, "test_data_set_0"))

    return model_path, input_data_onnx, input_data_tflite


ONNX_MODELS_QUANTIZABLE = load_full_model_specs(_ARTIFACTS_DIR, "downloaded",
                                                enabled_onnx_tests.ONNX_ZOO_MODELS_QUANTIZABLE)


@pytest.mark.parametrize("onnx_artifact_dir, atol, conversion_args", ONNX_MODELS_QUANTIZABLE)
@full_quantized_model_test
def test_output_of_onnx_models_quantized(onnx_artifact_dir: str, atol: float, conversion_args: dict):
    model_path = os.path.join(onnx_artifact_dir, "model.onnx")
    input_data_onnx, input_data_tflite = _load_input_tensors(os.path.join(onnx_artifact_dir, "test_data_set_0"))

    return model_path, input_data_onnx, input_data_tflite
