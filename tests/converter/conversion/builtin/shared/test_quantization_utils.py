#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import List

import flatbuffers
import numpy as np
import pytest

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
# noinspection PyProtectedMember
from onnx2tflite.src.converter.quantization_utils import _re_quantize_uint8_to_int8
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import quantize_options
from tests.executors import TFLiteExecutor
import tensorflow.lite as tflite

@pytest.mark.parametrize("scale,zero_point", [
    (0.1, 100),
    (12, 128),
    (-0.5, 200)  # Negative scale raise Segmenation Fault on XNNPACK since TF ~2.15.0. Default kernels and builtin
                 # kernels without default delegates works fine.
])
def test_static_uint8_to_int8_per_tensor_re_quantization(scale: float, zero_point: int):
    io_shape = [2, 3, 4, 5]
    output_zero_point = zero_point - 128
    input_data = np.arange(0, np.prod(io_shape)).reshape(io_shape).astype(np.uint8)

    model = tflite_model.Model(
        version=3,
        operator_codes=tflite_model.OperatorCodes([tflite_model.OperatorCode(BuiltinOperator.QUANTIZE)]),
        sub_graphs=tflite_model.SubGraphs([
            tflite_model.SubGraph(
                inputs=tflite_model.SubGraphInputs([1]), outputs=tflite_model.SubGraphOutputs([0]),
                tensors=tflite_model.Tensors([
                    tflite_model.Tensor(tflite_model.Shape(io_shape), "output", 0, TensorType.INT8,
                                        tflite_model.Quantization(scale=tflite_model.Scale([scale]),
                                                                  zero_point=tflite_model.ZeroPoint(
                                                                      [output_zero_point]))),
                    tflite_model.Tensor(tflite_model.Shape(io_shape), "input", 0, TensorType.UINT8,
                                        tflite_model.Quantization(scale=tflite_model.Scale([scale]),
                                                                  zero_point=tflite_model.ZeroPoint([zero_point])))
                ]),
                operators=tflite_model.Operators([
                    tflite_model.Operator(tflite_model.OperatorInputs([1]), tflite_model.OperatorOutputs([0]),
                                          quantize_options.Quantize())
                ])
            )
        ]),
        buffers=tflite_model.Buffers([tflite_model.Buffer()])
    )

    builder = flatbuffers.Builder()
    model.gen_tflite(builder)
    model = bytes(builder.Output())

    # Use reference kernels due to negative scale issue with XNNPACK
    tflite_executor = TFLiteExecutor(model_content=model, op_resolver_type=tflite.experimental.OpResolverType.BUILTIN_REF)
    tflite_output = tflite_executor.inference(input_data).astype(np.int8)

    translator_output = _re_quantize_uint8_to_int8(input_data)

    assert np.allclose(tflite_output, translator_output)


@pytest.mark.parametrize("scale, zero_point, axis", [
    ([0.1, 0.2], [120, 128], 0),
    ([0.1, 0.2, 0.3], [1, 2, 3], -2),
    ([0.1, 0.2, 0.3, -12], [-12, 0, - 140, 200], 2),
])
def test_static_uint8_to_int8_per_channel_re_quantization(scale: List[float], zero_point: List[int], axis: int):
    # TFLite Quantize operator doesn't seem to support per-channel re-quantization, so tests use a prepared
    # expected output.
    io_shape = [2, 3, 4]
    input_data = np.arange(118, np.prod(io_shape) + 118).reshape(io_shape).astype(np.uint8)

    # The output should only depend on the inputs and difference of zero points (128 in this case), so it is constant
    # for these tests.
    expected_output = [[[-10, -9, -8, -7], [-6, -5, -4, -3], [-2, -1, 0, 1]],
                       [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]]

    translator_output = _re_quantize_uint8_to_int8(input_data)

    # noinspection PyTypeChecker
    assert np.allclose(translator_output, expected_output)
