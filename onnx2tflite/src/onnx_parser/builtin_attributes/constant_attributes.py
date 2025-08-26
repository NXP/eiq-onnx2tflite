#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#


from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser import onnx_tensor
from onnx2tflite.src.onnx_parser.meta import meta
from onnx2tflite.src.onnx_parser.onnx_tensor import SparseTensor, TensorProto


class Constant(meta.ONNXOperatorAttributes):
    # V1+
    value: TensorProto | None
    # V11+
    sparse_value: SparseTensor | None
    # V12+
    value_float: float | None
    value_floats: meta.ONNXFloatListAttribute | None
    value_int: int | None
    value_ints: meta.ONNXIntListAttribute | None
    value_string: str | None
    value_strings: meta.ONNXStringListAttribute | None

    # I see no need to distinguish between versions.

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "value":
                self.value = onnx_tensor.TensorProto(attr.t)
            elif attr.name == "sparse_value":
                self.sparse_value = onnx_tensor.SparseTensor(attr.sparse_tensor)
            elif attr.name == "value_float":
                self.value_float = attr.f
            elif attr.name == "value_floats":
                self.value_floats = meta.ONNXFloatListAttribute(attr)
            elif attr.name == "value_int":
                self.value_int = attr.i
            elif attr.name == "value_ints":
                self.value_ints = meta.ONNXIntListAttribute(attr)
            elif attr.name == "value_string":
                self.value_string = attr.s.decode("UTF-8")
            elif attr.name == "value_strings":
                self.value_strings = meta.ONNXStringListAttribute(attr)
            else:
                logger.e(logger.Code.UNSUPPORTED_OPERATOR_ATTRIBUTES,
                         f"ONNX Operator 'Constant' has attribute '{attr.name}' which is not yet supported!")

        # Exactly 1 of the operator attributes must be given!
        attrs = [hasattr(self, "value"), hasattr(self, "sparse_value"), hasattr(self, "value_int"),
                 hasattr(self, "value_ints"), hasattr(self, "value_float"), hasattr(self, "value_floats"),
                 hasattr(self, "value_string"), hasattr(self, "value_strings")]
        if sum(attrs) != 1:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX 'Constant' must have exactly 1 attribute set!")
