#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""shape_options

Representation of a TFLite operator 'Shape'.
"""

import flatbuffers as fb

import onnx2tflite.lib.tflite.ShapeOptions as libShapeOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.tflite_generator.meta import meta


class Shape(meta.BuiltinOptions):
    out_type: TensorType

    def __init__(self, out_type: TensorType) -> None:
        super().__init__(BuiltinOptions.ShapeOptions, BuiltinOperator.SHAPE)
        self.out_type = out_type

    def gen_tflite(self, builder: fb.Builder):
        libShapeOptions.Start(builder)

        libShapeOptions.AddOutType(builder, self.out_type)

        return libShapeOptions.End(builder)
