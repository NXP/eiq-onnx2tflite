#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


import flatbuffers as fb

from onnx2tflite.lib.tflite import CastOptions as libCastOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.tflite_generator.meta import meta


class Cast(meta.BuiltinOptions):
    in_data_type: TensorType
    out_data_type: TensorType

    def __init__(self, in_data_type: TensorType, out_data_type: TensorType) -> None:
        super().__init__(BuiltinOptions.CastOptions, BuiltinOperator.CAST)
        self.in_data_type = in_data_type
        self.out_data_type = out_data_type

    def gen_tflite(self, builder: fb.Builder):
        libCastOptions.Start(builder)

        libCastOptions.AddInDataType(builder, self.in_data_type)
        libCastOptions.AddOutDataType(builder, self.out_data_type)

        return libCastOptions.End(builder)
