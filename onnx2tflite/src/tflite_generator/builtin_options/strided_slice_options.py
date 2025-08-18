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
from onnx2tflite.lib.tflite import StridedSliceOptions as libStridedSliceOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions


class StridedSlice(meta.BuiltinOptions):
    offset: bool

    def __init__(self, offset: bool = False) -> None:
        super().__init__(BuiltinOptions.StridedSliceOptions, BuiltinOperator.STRIDED_SLICE)
        self.offset = offset

    def gen_tflite(self, builder: fb.Builder):
        libStridedSliceOptions.Start(builder)
        libStridedSliceOptions.AddOffset(builder, self.offset)
        return libStridedSliceOptions.End(builder)
