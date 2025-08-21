#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

from onnx2tflite.lib.tflite import RandomOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Multinomial(meta.BuiltinOptions):
    seed: int
    seed2: int

    def __init__(self, seed: int, seed2: int) -> None:
        super().__init__(BuiltinOptions.RandomOptions, BuiltinOperator.MULTINOMIAL)
        self.seed = seed
        self.seed2 = seed2

    def gen_tflite(self, builder: fb.Builder):
        RandomOptions.Start(builder)

        RandomOptions.AddSeed(builder, self.seed)
        RandomOptions.AddSeed2(builder, self.seed2)

        return RandomOptions.End(builder)
