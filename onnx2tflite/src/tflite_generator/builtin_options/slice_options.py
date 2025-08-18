#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""
    slice_options

Representation of the TFLite operator 'Slice'.
"""

import flatbuffers as fb

import onnx2tflite.src.tflite_generator.meta.meta as meta
from onnx2tflite.lib.tflite import SliceOptions as libSliceOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions


class Slice(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(BuiltinOptions.SliceOptions, BuiltinOperator.SLICE)

    def gen_tflite(self, builder: fb.Builder):
        libSliceOptions.Start(builder)
        return libSliceOptions.End(builder)
