#
# Copyright 2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from enum import Enum

from flatbuffers import flexbuffers

from onnx2tflite.src.tflite_generator.meta.meta import CustomOptions


class Activation(Enum):
    sigmoid = "Sigmoid"
    tanh = "Tanh"


class Direction(Enum):
    forward = "forward"
    bidirectional = "bidirectional"
    reverse = "reverse"


class OnnxGRU(CustomOptions):
    def __init__(
        self,
        hidden_size: int,
        clip: float,
        activations: tuple[Activation, ...],
        direction: Direction,
        linear_before_reset: int,
        layout: int = 0,
    ) -> None:
        # Replace the enums with their string values.
        activations_str = [act.value for act in activations]
        direction_str = direction.value

        custom_options_data = flexbuffers.Dumps({
            "activations": activations_str,
            "clip": clip,
            "direction": direction_str,
            "hidden_size": hidden_size,
            "layout": layout,
            "linear_before_reset": int(linear_before_reset)
        })

        super().__init__("OnnxGRU", custom_options_data)
