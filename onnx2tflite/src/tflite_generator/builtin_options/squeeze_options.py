#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import flatbuffers as fb

import onnx2tflite.lib.tflite.SqueezeOptions as libSqueezeOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class SqueezeDims(meta.IntVector):
    def __init__(self, new_shape: list[int]) -> None:
        super().__init__(new_shape, libSqueezeOptions.StartSqueezeDimsVector)


class Squeeze(meta.BuiltinOptions):
    squeeze_dims: SqueezeDims | None

    def __init__(self, squeeze_dims: list[int] | None) -> None:
        super().__init__(BuiltinOptions.SqueezeOptions, BuiltinOperator.SQUEEZE)

        if squeeze_dims is not None:
            self.squeeze_dims = SqueezeDims(squeeze_dims)
        else:
            self.squeeze_dims = None

    def gen_tflite(self, builder: fb.Builder):
        if self.squeeze_dims is not None:
            tfl_squeeze_dims = self.squeeze_dims.gen_tflite(builder)
        else:
            tfl_squeeze_dims = None

        libSqueezeOptions.Start(builder)

        if tfl_squeeze_dims is not None:
            libSqueezeOptions.AddSqueezeDims(builder, tfl_squeeze_dims)

        return libSqueezeOptions.End(builder)
