#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers

from onnx2tflite.lib.tflite import BuiltinOptions, BuiltinOperator, PadV2Options
from onnx2tflite.src.tflite_generator.meta import meta


class PadV2(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.BuiltinOptions.PadV2Options, BuiltinOperator.BuiltinOperator.PADV2)

    def gen_tflite(self, builder: flatbuffers.Builder):
        PadV2Options.Start(builder)

        return PadV2Options.End(builder)
