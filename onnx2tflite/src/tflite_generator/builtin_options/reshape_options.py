#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""Reshape

Representation of the TFLite operator 'Reshape'.
"""


import flatbuffers as fb

import onnx2tflite.lib.tflite.BuiltinOperator as libBuiltinOperator
import onnx2tflite.lib.tflite.BuiltinOptions as libBuiltinOptions
import onnx2tflite.lib.tflite.ReshapeOptions as libReshapeOptions
from onnx2tflite.src.tflite_generator.meta import meta


class NewShape(meta.IntVector):
    def __init__(self, new_shape: list[int]) -> None:
        super().__init__(new_shape, libReshapeOptions.StartNewShapeVector)


class Reshape(meta.BuiltinOptions):
    new_shape: NewShape | None

    def __init__(self, new_shape: list[int] | None) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.ReshapeOptions,
                         libBuiltinOperator.BuiltinOperator.RESHAPE)
        if new_shape is not None:
            self.new_shape = NewShape(new_shape)
        else:
            self.new_shape = None

    def gen_tflite(self, builder: fb.Builder):
        if self.new_shape is not None:
            tfl_new_shape = self.new_shape.gen_tflite(builder)
        else:
            tfl_new_shape = None

        libReshapeOptions.Start(builder)

        if tfl_new_shape is not None:
            libReshapeOptions.AddNewShape(builder, tfl_new_shape)

        return libReshapeOptions.End(builder)
