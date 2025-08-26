#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


import flatbuffers as fb

import onnx2tflite.lib.tflite.BuiltinOperator as libBuiltinOperator
import onnx2tflite.lib.tflite.BuiltinOptions as libBuiltinOptions
import onnx2tflite.lib.tflite.SelectV2Options as libSelectV2Options
from onnx2tflite.src.tflite_generator.meta import meta


class SelectV2(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.SelectV2Options,
                         libBuiltinOperator.BuiltinOperator.SELECT_V2)

    def gen_tflite(self, builder: fb.Builder) -> int:
        libSelectV2Options.Start(builder)
        return libSelectV2Options.End(builder)
