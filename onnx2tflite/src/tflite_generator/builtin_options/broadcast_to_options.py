#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

import onnx2tflite.lib.tflite.BroadcastToOptions as libBroadcastToOptions
import onnx2tflite.src.tflite_generator.meta.meta as meta
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions


class BroadcastTo(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.BroadcastToOptions,
                         BuiltinOperator.BROADCAST_TO)

    def gen_tflite(self, builder: fb.Builder):
        libBroadcastToOptions.Start(builder)

        return libBroadcastToOptions.End(builder)
