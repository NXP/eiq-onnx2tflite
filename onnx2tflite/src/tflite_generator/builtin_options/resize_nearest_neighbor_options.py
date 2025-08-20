#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import flatbuffers as fb

from onnx2tflite.lib.tflite import ResizeNearestNeighborOptions
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


# noinspection SpellCheckingInspection
class ResizeNearestNeighbor(meta.BuiltinOptions):
    align_corners: bool
    half_pixel_centers: bool

    def __init__(self, align_corners: bool, half_pixel_centers: bool) -> None:
        super().__init__(BuiltinOptions.ResizeNearestNeighborOptions, BuiltinOperator.RESIZE_NEAREST_NEIGHBOR)
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers

    def gen_tflite(self, builder: fb.Builder) -> int:
        ResizeNearestNeighborOptions.Start(builder)

        ResizeNearestNeighborOptions.AddAlignCorners(builder, self.align_corners)
        ResizeNearestNeighborOptions.AddHalfPixelCenters(builder, self.half_pixel_centers)

        return ResizeNearestNeighborOptions.End(builder)
