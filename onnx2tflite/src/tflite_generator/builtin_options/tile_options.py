#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

from onnx2tflite.lib.tflite import TileOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Tile(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.TileOptions, BuiltinOperator.TILE)

    def gen_tflite(self, builder: fb.Builder):
        TileOptions.Start(builder)

        return TileOptions.End(builder)
