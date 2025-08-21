#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

from onnx2tflite.lib.tflite import DepthToSpaceOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class DepthToSpace(meta.BuiltinOptions):
    block_size: int

    def __init__(self, block_size: int) -> None:
        super().__init__(BuiltinOptions.DepthToSpaceOptions, BuiltinOperator.DEPTH_TO_SPACE)
        self.block_size = block_size

    def gen_tflite(self, builder: fb.Builder):
        DepthToSpaceOptions.Start(builder)

        DepthToSpaceOptions.DepthToSpaceOptionsAddBlockSize(builder, self.block_size)

        return DepthToSpaceOptions.End(builder)
