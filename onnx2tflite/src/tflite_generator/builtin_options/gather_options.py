#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

import onnx2tflite.lib.tflite.GatherOptions as libGatherOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class Gather(meta.BuiltinOptions):
    axis: int
    batch_dims: int

    def __init__(self, axis: int, batch_dims: int = 0) -> None:
        super().__init__(BuiltinOptions.GatherOptions, BuiltinOperator.GATHER)
        self.axis = axis
        self.batch_dims = batch_dims

    def gen_tflite(self, builder: fb.Builder) -> int:
        libGatherOptions.Start(builder)

        libGatherOptions.AddAxis(builder, self.axis)
        libGatherOptions.AddBatchDims(builder, self.batch_dims)

        return libGatherOptions.End(builder)
