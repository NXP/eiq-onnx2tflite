#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers

from onnx2tflite.lib.tflite import BuiltinOperator, BuiltinOptions, PadOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Pad(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.BuiltinOptions.PadOptions, BuiltinOperator.BuiltinOperator.PAD)

    def gen_tflite(self, builder: flatbuffers.Builder) -> int:
        PadOptions.Start(builder)

        return PadOptions.End(builder)
