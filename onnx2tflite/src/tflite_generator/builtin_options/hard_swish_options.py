#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

from onnx2tflite.lib.tflite import HardSwishOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class HardSwish(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.HardSwishOptions, BuiltinOperator.HARD_SWISH)

    def gen_tflite(self, builder: fb.Builder) -> int:
        HardSwishOptions.Start(builder)

        return HardSwishOptions.End(builder)
