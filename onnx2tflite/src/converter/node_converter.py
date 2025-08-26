#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from abc import ABC, abstractmethod

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger, model_inspector
from onnx2tflite.src.conversion_context import ConversionContext
from onnx2tflite.src.converter.builder import model_builder
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, name_for_type


class NodeConverter(ABC):
    """Classes which implement conversion of ONNX operators to TFLite should inherit from this class and overwrite the
    'convert()' method.
    """

    context: ConversionContext

    onnx_supported_types: list[TensorType] | None = None  # List of data types supported by the ONNX documentation.
    tflite_supported_types: list[TensorType] | None = None  # List of types supported by the TFLite inference engine.
    verified_types: list[TensorType] | None = None  # List of data types verified by tests.

    def __init__(self, context: ConversionContext):
        self.context = context

    @abstractmethod
    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX operator in 'node' to TFLite. Resulting TFLite operators will be returned as a list.

            Classes which implement conversion for individual operators must overwrite this method.

        :param node: ONNX operator to convert.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of resulting TFLite operators.
        """

    @property
    @abstractmethod
    def node(self) -> str:
        """Name of the ONNX node which is handled by the child converter class."""

    @property
    def builder(self) -> model_builder.ModelBuilder:
        """Get instance of TFLite ModelBuilder from conversion context.
        :return: ModelBuilder instance.
        """
        return self.context.tflite_builder

    @property
    def inspector(self) -> model_inspector.ONNXModelInspector:
        """Get instance of ONNXModelInspector from conversion context.
        :return: ONNXModelBuilder instance.
        """
        return self.context.onnx_inspector

    def assert_type_allowed(self, _type: TensorType) -> None:
        """Check if the given type is supported by ONNX, TFLite and the converter based on the lists of types declared
        at the top of the class. Child classes must define these lists in order to use this method.
        """
        if _type not in (self.onnx_supported_types or ALL_TYPES):
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX operator `{self.node}` has type `{name_for_type(_type)}`, "
                                                     "which is not supported by the ONNX documentation.")

        if _type not in (self.tflite_supported_types or ALL_TYPES):
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, f"Conversion of the ONNX operator `{self.node}` with type "
                                                        f"`{name_for_type(_type)}`, is not possible.")

        if _type not in (self.verified_types or ALL_TYPES):
            logger.e(logger.Code.NOT_IMPLEMENTED, f"Conversion of the ONNX operator `{self.node}` with type "
                                                  f"`{name_for_type(_type)}`, is not supported.")
