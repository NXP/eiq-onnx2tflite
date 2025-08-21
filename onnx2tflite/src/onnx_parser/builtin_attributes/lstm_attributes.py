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


class LSTM(meta.ONNXOperatorAttributes):
    activation_alpha: meta.ONNXFloatListAttribute | None
    activation_beta: meta.ONNXFloatListAttribute | None
    activations: meta.ONNXStringListAttribute | None
    clip: float | None
    direction: str
    hidden_size: int | None
    input_forget: int
    layout: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.activation_alpha = None
        self.activation_beta = None
        self.activations = None
        self.clip = None
        self.direction = "forward"
        self.hidden_size = None
        self.input_forget = 0
        self.layout = 0

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "activation_alpha":
                self.activation_alpha = meta.ONNXFloatListAttribute(attr)
            elif attr.name == "activation_beta":
                self.activation_beta = meta.ONNXFloatListAttribute(attr)
            elif attr.name == "activations":
                self.activations = meta.ONNXStringListAttribute(attr)
            elif attr.name == "clip":
                self.clip = attr.f
            elif attr.name == "direction":
                self.direction = attr.s.decode("utf-8")
            elif attr.name == "hidden_size":
                self.hidden_size = attr.i
            elif attr.name == "input_forget":
                self.input_forget = attr.i
            elif attr.name == "layout":
                self.layout = attr.i
            else:
                logger.w(f"ONNX LSTM attribute '{attr.name}' is not supported!")
