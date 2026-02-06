#
# Copyright 2024, 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain

import numpy as np
import onnx


@dataclass
class PreprocessingStep(ABC):
    model: onnx.ModelProto

    @property
    @abstractmethod
    def disabling_flag(self) -> str:
        # Return the preprocessing flag which disables this particular preprocessing step.
        pass

    @abstractmethod
    def run(self) -> None:
        # Execute the preprocessing step on the `self.model` ONNX model.
        pass

    def try_get_tensor_data(self, tensor: str) -> np.ndarray | None:
        for initializer in self.model.graph.initializer:
            if initializer.name == tensor:
                return onnx.numpy_helper.to_array(initializer)

        return None

    def create_tensor_for_data(self, data: np.ndarray, name: str | None = None) -> str:
        """Create an initializer (static tensor) for the given data. The name of the new tensor can be specified, but
        if it's not unique in the model, it will be modified to be unique. The final used name is returned.
        """
        name = self.validate_name(name)
        tensor = onnx.numpy_helper.from_array(data, name)
        self.add_initializer(tensor)

        return name

    def contains_quantization_nodes(self) -> bool:
        """Check if model contains quantization nodes ('QuantizeLinear' or 'DequantizeLinear').

        :return: True if mode contains at least one quantization node.
        """
        for node in self.model.graph.node:
            if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                return True

        return False

    def is_initializer(self, tensor_name: str) -> bool:
        """Check if tensor is initializer"""
        return any([initializer.name == tensor_name for initializer in self.model.graph.initializer])

    def add_initializer(self, initializer: onnx.TensorProto) -> None:
        """Add a `TensorProto` to the `graph.initializers`, with all necessary additional steps."""
        self.model.graph.initializer.append(initializer)
        if self.model.ir_version <= 3:
            # IR version <= 3 requires all initializers to be in the `graph.input` as well (according to comment).
            # https://github.com/onnx/onnx/blob/rel-1.15.0/onnx/checker.cc#L723-L728
            vi = onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims)
            self.model.graph.input.append(vi)

    def remove_initializer(self, tensor_name: str):
        """Remove initializer from graph or do nothing if initializer doesn't exist."""
        tensors = [initializer for initializer in self.model.graph.initializer if initializer.name == tensor_name]
        if len(tensors) == 0:
            return

        self.model.graph.initializer.remove(tensors[0])

    def get_value_info(self, tensor_name: str) -> onnx.ValueInfoProto | None:
        value_infos = [tensor for tensor in self.model.graph.value_info if tensor.name == tensor_name]

        return value_infos[0] if len(value_infos) > 0 else None

    def validate_name(self, tensor_name: str | None) -> str:
        tensor_name = tensor_name or "tensor"

        existing_names = [
            vi.name for vi in chain(self.model.graph.value_info, self.model.graph.input, self.model.graph.output)
        ]
        existing_names += [
            initializer.name for initializer in self.model.graph.initializer
        ]

        if tensor_name not in existing_names:
            return tensor_name

        # Add a number after the name to make it unique.
        n = 0
        while tensor_name + f"_{n}" in existing_names:
            n += 1
        return tensor_name + f"_{n}"

    def try_get_tensor_type(self, tensor: str) -> int | None:
        for vi in chain(self.model.graph.value_info, self.model.graph.input, self.model.graph.output):
            if vi.name == tensor:
                if hasattr(vi, "type_proto"):
                    return vi.type_proto.tensor_type
                if hasattr(vi, "type"):
                    return vi.type.tensor_type.elem_type

        for initializer in self.model.graph.initializer:
            if initializer.name == tensor:
                return initializer.data_type

        return None
