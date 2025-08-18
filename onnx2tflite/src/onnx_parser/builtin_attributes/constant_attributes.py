#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#


from typing import Iterable, Optional

import onnx

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.meta.meta as meta
from onnx2tflite.src.onnx_parser import onnx_tensor
from onnx2tflite.src.onnx_parser.onnx_tensor import TensorProto, SparseTensor


class Constant(meta.ONNXOperatorAttributes):
    # V1+
    value: Optional[TensorProto]
    # V11+
    sparse_value: Optional[SparseTensor]
    # V12+
    value_float: Optional[float]
    value_floats: Optional[meta.ONNXFloatListAttribute]
    value_int: Optional[int]
    value_ints: Optional[meta.ONNXIntListAttribute]
    value_string: Optional[str]
    value_strings: Optional[meta.ONNXStringListAttribute]

    # I see no need to distinguish between versions.

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _init_attributes(self):
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
