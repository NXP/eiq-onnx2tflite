#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers

from onnx2tflite.lib.tflite import BuiltinOptions, BuiltinOperator, MirrorPadOptions
from onnx2tflite.lib.tflite.MirrorPadMode import MirrorPadMode
from onnx2tflite.src.tflite_generator.meta import meta


# SYMMETRIC pads = [2, 2, 2, 2]
# input:
# [[0. 1.]
#  [2. 3.]]
#
# output:
# [[3. 2. 2. 3. 3. 2.]
#  [1. 0. 0. 1. 1. 0.]
#  [1. 0. 0. 1. 1. 0.]
#  [3. 2. 2. 3. 3. 2.]
#  [3. 2. 2. 3. 3. 2.]
#  [1. 0. 0. 1. 1. 0.]]

# REFLECT pads = [2, 2, 2, 2]
# input:
# [[0. 1.]
#  [2. 3.]]
#
# output: (Doesn't make sense to me, why the first row is all 0. Also last row is weird.)
#         (Also the element [0][0] is sometimes 9.462417e-28, so the computations seems non-deterministic!)
# [[0. 0. 0. 0. 0. 0.]
#  [0. 3. 2. 3. 2. 2.]
#  [2. 1. 0. 1. 0. 0.]
#  [0. 3. 2. 3. 2. 2.]
#  [2. 1. 0. 1. 0. 0.]
#  [2. 1. 0. 1. 0. 0.]]

class MirrorPad(meta.BuiltinOptions):
    mode: MirrorPadMode

    def __init__(self, mode: MirrorPadMode = MirrorPadMode.REFLECT) -> None:
        super().__init__(BuiltinOptions.BuiltinOptions.MirrorPadOptions, BuiltinOperator.BuiltinOperator.MIRROR_PAD)
        self.mode = mode

    def gen_tflite(self, builder: flatbuffers.Builder):
        MirrorPadOptions.Start(builder)

        MirrorPadOptions.AddMode(builder, self.mode)

        return MirrorPadOptions.End(builder)
