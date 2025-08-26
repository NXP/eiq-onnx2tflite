#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

from onnx2tflite.lib.tflite import SpaceToDepthOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class SpaceToDepth(meta.BuiltinOptions):
    block_size: int

    def __init__(self, block_size: int) -> None:
        super().__init__(BuiltinOptions.SpaceToDepthOptions, BuiltinOperator.SPACE_TO_DEPTH)
        self.block_size = block_size

    def gen_tflite(self, builder: fb.Builder) -> int:
        SpaceToDepthOptions.Start(builder)

        SpaceToDepthOptions.SpaceToDepthOptionsAddBlockSize(builder, self.block_size)

        return SpaceToDepthOptions.End(builder)
