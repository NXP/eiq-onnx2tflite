#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class ConvTranspose(meta.ONNXOperatorAttributes):
    auto_pad: str
    dilations: meta.ONNXIntListAttribute | None
    group: int
    kernel_shape: meta.ONNXIntListAttribute | None
    output_padding: meta.ONNXIntListAttribute | None
    output_shape: meta.ONNXIntListAttribute | None
    pads: meta.ONNXIntListAttribute | None
    strides: meta.ONNXIntListAttribute | None

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.auto_pad = "NOTSET"
        self.dilations = None
        self.group = 1
        self.kernel_shape = None
        self.output_padding = None
        self.output_shape = None
        self.pads = None
        self.strides = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "auto_pad":
                self.auto_pad = attr.s.decode("UTF-8")
            elif attr.name == "dilations":
                self.dilations = meta.ONNXIntListAttribute(attr)
            elif attr.name == "group":
                self.group = attr.i
            elif attr.name == "kernel_shape":
                self.kernel_shape = meta.ONNXIntListAttribute(attr)
            elif attr.name == "output_padding":
                self.output_padding = meta.ONNXIntListAttribute(attr)
            elif attr.name == "output_shape":
                self.output_shape = meta.ONNXIntListAttribute(attr)
            elif attr.name == "pads":
                self.pads = meta.ONNXIntListAttribute(attr)
            elif attr.name == "strides":
                self.strides = meta.ONNXIntListAttribute(attr)
            else:
                logger.w(f"ONNX ConvTranspose attribute '{attr.name}' is not supported!")
