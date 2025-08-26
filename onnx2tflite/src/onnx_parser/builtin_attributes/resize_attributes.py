#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta
from onnx2tflite.src.onnx_parser.meta.meta import ONNXIntListAttribute


# noinspection SpellCheckingInspection
class Resize(meta.ONNXOperatorAttributes):
    # V10+
    mode: str

    # V11+
    coordinate_transformation_mode: str
    cubic_coeff_a: float
    exclude_outside: int
    extrapolation_value: float
    nearest_mode: str

    # V18+
    antialias: int
    axes: ONNXIntListAttribute | None
    keep_aspect_ratio_policy: str

    def _default_values(self) -> None:
        self.mode = "nearest"

        self.coordinate_transformation_mode = "half_pixel"
        self.cubic_coeff_a = -0.75
        self.exclude_outside = 0
        self.extrapolation_value = 0.0
        self.nearest_mode = "round_prefer_floor"

        self.antialias = 0
        self.axes = None
        self.keep_aspect_ratio_policy = "stretch"

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "mode":
                self.mode = attr.s.decode("utf-8")

            elif attr.name == "coordinate_transformation_mode":
                self.coordinate_transformation_mode = attr.s.decode("utf-8")
            elif attr.name == "cubic_coeff_a":
                self.cubic_coeff_a = attr.f
            elif attr.name == "exclude_outside":
                self.exclude_outside = attr.i
            elif attr.name == "extrapolation_value":
                self.extrapolation_value = attr.f
            elif attr.name == "nearest_mode":
                self.nearest_mode = attr.s.decode("utf-8")

            elif attr.name == "antialias":
                self.antialias = attr.i
            elif attr.name == "axes":
                self.axes = ONNXIntListAttribute(attr)
            elif attr.name == "keep_aspect_ratio_policy":
                self.keep_aspect_ratio_policy = attr.s.decode("utf-8")

            else:
                logger.w(f"ONNX `Resize` attribute `{attr.name}` is not supported!")
