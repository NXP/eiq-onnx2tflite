#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""quantize_options

Representation of a TFLite operator 'Quantize'.
"""

import flatbuffers as fb

import onnx2tflite.lib.tflite.BuiltinOperator as libBuiltinOperator
import onnx2tflite.lib.tflite.BuiltinOptions as libBuiltinOptions
import onnx2tflite.lib.tflite.QuantizeOptions as libQuantizeOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Quantize(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.QuantizeOptions,
                         libBuiltinOperator.BuiltinOperator.QUANTIZE)

    def gen_tflite(self, builder: fb.Builder):
        libQuantizeOptions.Start(builder)

        return libQuantizeOptions.End(builder)
