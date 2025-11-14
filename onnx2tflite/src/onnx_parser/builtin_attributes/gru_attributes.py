#
# Copyright 2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class GRU(meta.ONNXOperatorAttributes):
    activation_alpha: list[float] | None
    activation_beta: list[float] | None
    activations: list[str] | None
    clip: float | None
    direction: str
    hidden_size: int | None
    linear_before_reset: int
    layout: int

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self) -> None:
        self.activation_alpha = None
        self.activation_beta = None
        self.direction = "forward"
        self.activations = None  # Set later during _init_attributes
        self.clip = None
        self.layout = 0
        self.linear_before_reset = 0

    def _init_attributes(self) -> None:
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
            elif attr.name == "layout":
                self.layout = attr.i
            elif attr.name == "linear_before_reset":
                self.linear_before_reset = attr.i
            else:
                logger.w(f"ONNX GRU attribute '{attr.name}' is not supported!")

        # Attribute 'hidden_size' is required
        if not hasattr(self, "hidden_size"):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     "ONNX 'GRU' is missing required 'hidden_size' attribute!")

        if self.activations is None:
            if self.direction == "forward":
                self.activations = ["Sigmoid", "Tanh"]
            elif self.direction == "bidirectional":
                self.activations = ["Sigmoid", "Tanh", "Sigmoid", "Tanh"]
