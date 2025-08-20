#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

from onnx2tflite.lib.tflite import ResizeBilinearOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


# noinspection SpellCheckingInspection
class ResizeBilinear(meta.BuiltinOptions):
    align_corners: bool
    half_pixel_centers: bool

    def __init__(self, align_corners: bool, half_pixel_centers: bool) -> None:
        super().__init__(BuiltinOptions.ResizeBilinearOptions, BuiltinOperator.RESIZE_BILINEAR)
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers

    def gen_tflite(self, builder: fb.Builder) -> int:
        ResizeBilinearOptions.Start(builder)

        ResizeBilinearOptions.AddAlignCorners(builder, self.align_corners)
        ResizeBilinearOptions.AddHalfPixelCenters(builder, self.half_pixel_centers)

        return ResizeBilinearOptions.End(builder)
