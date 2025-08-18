#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import functools
import os
import warnings
from typing import List, Optional, Dict

import cpuinfo
import numpy as np
import onnx
import pytest

import tests.converter.enabled_onnx_tests as enabled_onnx_tests
from onnx2quant.qdq_quantization import RandomDataCalibrationDataReader, QDQQuantizer
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from tests.executors import OnnxExecutor, TFLiteExecutor


def load_test_artifacts(root_dir, subdir, enabled_tests: Optional[Dict]) -> List[pytest.param]:
    """
    Load test artifacts stored in ONNX test format and return list of test specifications as
    pytest params. ONNX test package includes model named "model.onnx" and subdirectory
    "/test_data_set_0" with file "input_0.pb". Directories has to be organized in
    hierarchy <root_dir>/<subdir>/<package_dir>.


    :param root_dir: Parent directory, where ONNX test packages are stored.
    :param subdir: <root_dir> subdirectory where ONNX packages are stored.
    :param enabled_tests: List of test name used to filter tests available in <subdir>.
            If None, all tests in the directory are returned.
    :return: List of pytest.param description of found tests.
    """
    tests_dir = os.path.join(root_dir, subdir)

    artifact_dirs = []
    for test_path in [f.path for f in os.scandir(tests_dir) if f.is_dir()]:
        test_name = os.path.basename(test_path)
        if enabled_tests and test_name not in enabled_tests:
            continue

        test_args: dict = enabled_tests[test_name]

        param = pytest.param(os.path.join(tests_dir, test_path), test_args, id=str(f"{subdir}:{test_name}"))
        artifact_dirs.append(param)

    return artifact_dirs


def load_full_model_specs(root_dir, subdir, test_artifact_specs: Dict) -> List[pytest.param]:
    tests_dir = os.path.join(root_dir, subdir)

    pytest_spec = []
    for test_path in [f.path for f in os.scandir(tests_dir) if f.is_dir()]:
        test_name = os.path.basename(test_path)
        if test_name not in test_artifact_specs:
            continue

        atol = _load_atol_from_metadata(test_artifact_specs[test_name])
        marks = test_artifact_specs[test_name]["marks"] if "marks" in test_artifact_specs[test_name] else []
        conversion_args = test_artifact_specs[test_name].get(enabled_onnx_tests.CONVERSION_ARGS, dict())

        _id = str(f"{subdir}:{test_name}")
        param = pytest.param(os.path.join(tests_dir, test_path), atol, conversion_args, marks=marks,
                             id=_id)
        pytest_spec.append(param)

    all_spec_tests = set([str(f"{subdir}:{test_name}") for test_name in test_artifact_specs])
    all_available_tests = set([i.id for i in pytest_spec])
    diff = all_spec_tests.difference(all_available_tests)
    if diff:
        # TODO: python<3.12: Backslashes may not appear inside the expression portions of f-strings, so you cannot use
        #                    them, for example, to escape quotes inside f-strings:
        #                    https://peps.python.org/pep-0498/#escape-sequences
        #       python 3.12+: https://docs.python.org/3.12/whatsnew/3.12.html#pep-701-syntactic-formalization-of-f-strings
        nl = '\n'
        warnings.warn(f"Resources for following enabled tests are not available (outdated download_models.py?):\n"
                      f"{nl.join(map(str, diff))}")

    return pytest_spec


# Call to cpuinfo is expensive. Cache the result for speedup.
_cpu_supports_avx2 = "avx2" in cpuinfo.get_cpu_info()["flags"]


def _load_atol_from_metadata(test_spec_metadata: Dict) -> float:
    """
    Load atol (absolute tolerance) from test's metadata. If CPU
    has AVX2 extension, atol value with key "atol_avx2" is preferred
    over value under key "atol".

    :param test_spec_metadata: Dictionary with test's metadata.
    :return: Atol value. Returns 1.e-8 if no atol value is present in the metadata.
    """

    if "atol_avx2" in test_spec_metadata and _cpu_supports_avx2:
        return test_spec_metadata["atol_avx2"]
    elif "atol" in test_spec_metadata:
        return test_spec_metadata["atol"]
    else:
        return 1.e-8


def load_models_from_dir(rootdir, subdir, test_artifact_specs: Dict) -> List[pytest.param]:
    pytest_spec = []
    for onnx_file_path in test_artifact_specs:
        test_name = onnx_file_path
        atol = _load_atol_from_metadata(test_artifact_specs[test_name])
        marks = test_artifact_specs[test_name]["marks"] if "marks" in test_artifact_specs[test_name] else []
        conversion_args = test_artifact_specs[test_name].get(enabled_onnx_tests.CONVERSION_ARGS, dict())

        _id = str(f"{subdir}:{test_name}")
        param = pytest.param(os.path.join(rootdir, subdir, onnx_file_path), atol, conversion_args, marks=marks, id=_id)
        pytest_spec.append(param)

    return pytest_spec


def full_quantized_model_test(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        assert (kwargs['onnx_artifact_dir'])
        assert (kwargs['atol'])

        test_name = os.path.basename(kwargs['onnx_artifact_dir'])
        model_path, input_data_onnx, input_data_tflite = func(*args, **kwargs)
        onnx_model = onnx.load_model(model_path)
        onnx_model = ModelShapeInference.infer_shapes(onnx_model)

        qc = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model), {
            "allow_opset_10_and_lower": True
        })
        quantized_onnx_model = QDQQuantizer().quantize_model(onnx_model, qc)
        quantized_onnx_model = ModelShapeInference.infer_shapes(quantized_onnx_model)

        tfl_model = convert.convert_model(quantized_onnx_model)

        onnx_executor = OnnxExecutor(quantized_onnx_model.SerializeToString())
        output_onnx = onnx_executor.inference(input_data_onnx)

        tflite_executor = TFLiteExecutor(model_content=bytes(tfl_model))
        tflite_output = tflite_executor.inference(input_data_tflite)

        def _compare_output_tensors(onnx_tensor: np.ndarray, tflite_tensor: np.ndarray, tensor_name=None,
                                    atol=kwargs['atol']):
            print(
                f"[{test_name}][tensor: {tensor_name}] Maximum output difference: {np.max(np.abs(tflite_tensor - onnx_tensor))} (atol = {atol})")
            assert np.allclose(onnx_tensor, tflite_tensor, atol=atol)

        # If model has multiple outputs, they are stored as dictionary
        if isinstance(output_onnx, dict):
            for key, index in zip(output_onnx.keys(), range(len(output_onnx.keys()))):
                # TODO (robert): Fix output tensor naming in the converter. The converter shall preserve the output tensor names
                # as the system using the generated tflite model might rely on the naming.
                # test case: mobilenet_ssd_v1_onnx_ptq_uint8_converted.onnx
                # Once done remove the tfl_key search logic from here.
                tfl_key = [tk for tk in tflite_output.keys() if key in tk]
                assert len(tfl_key) == 1
                tfl_key = tfl_key[0]

                atol = kwargs['atol'][index] if isinstance(kwargs['atol'], list) else kwargs['atol']

                _compare_output_tensors(output_onnx[key], tflite_output[tfl_key], tensor_name=key, atol=atol)
        else:
            _compare_output_tensors(output_onnx, tflite_output)

    return inner


def full_model_test(func):
    """
    Decorator implementing the logic of full model testing with pytest.mark.parametrize. Decorator expects at least following
    kwargs are defined by pytest.mark.parametrize: 'onnx_artifact_dir', 'atol'.

    :param func: Decorated function, expected to return the: path to the ONNX model, onnx model input data and
    tflite model input data
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        assert (kwargs['onnx_artifact_dir'])
        assert (kwargs['atol'])

        conversion_args = kwargs.get(enabled_onnx_tests.CONVERSION_ARGS, dict())

        test_name = os.path.basename(kwargs['onnx_artifact_dir'])
        model_path, input_data_onnx, input_data_tflite = func(*args, **kwargs)
        converted_tflite = bytes(convert.convert_model(model_path, ConversionConfig(conversion_args)))

        onnx_executor = OnnxExecutor(model_path)
        output_onnx = onnx_executor.inference(input_data_onnx)

        tflite_executor = TFLiteExecutor(model_content=converted_tflite)
        tflite_output = tflite_executor.inference(input_data_tflite)

        def _compare_output_tensors(onnx_tensor: np.ndarray, tflite_tensor: np.ndarray, tensor_name=None,
                                    atol=kwargs['atol']):
            print(
                f"[{test_name}][tensor: {tensor_name}] Maximum output difference: {np.max(np.abs(tflite_tensor - onnx_tensor))} (atol = {atol})")
            assert np.allclose(onnx_tensor, tflite_tensor, atol=atol)

        # If model has multiple outputs, they are stored as dictionary
        if isinstance(output_onnx, dict):
            for key, index in zip(output_onnx.keys(), range(len(output_onnx.keys()))):
                # TODO (robert): Fix output tensor naming in the converter. The converter shall preserve the output tensor names
                # as the system using the generated tflite model might rely on the naming.
                # test case: mobilenet_ssd_v1_onnx_ptq_uint8_converted.onnx
                # Once done remove the tfl_key search logic from here.
                tfl_key = [tk for tk in tflite_output.keys() if key in tk]
                assert len(tfl_key) == 1
                tfl_key = tfl_key[0]

                atol = kwargs['atol'][index] if isinstance(kwargs['atol'], list) else kwargs['atol']

                _compare_output_tensors(output_onnx[key], tflite_output[tfl_key], tensor_name=key,
                                        atol=atol)
        else:
            _compare_output_tensors(output_onnx, tflite_output)

    return inner
