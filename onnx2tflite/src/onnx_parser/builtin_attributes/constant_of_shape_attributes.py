#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


from collections.abc import Iterable
from typing import Any

import onnx
from onnx import AttributeProto, TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser import onnx_tensor
from onnx2tflite.src.onnx_parser.meta import meta


class ConstantOfShape(meta.ONNXOperatorAttributes):
    value: Any

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.value = onnx_tensor.TensorProto(
            onnx.helper.make_tensor("", TensorProto.FLOAT, [1], [0.])
        )

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "value":
                # 'value' can have any type. It is stored in the 'attr.type'
                # Documentation states that it should be a 1 element tensor.
                # TODO Unresolved attribute reference TENSOR
                if attr.type == AttributeProto.AttributeType.TENSOR:
                    self.value = onnx_tensor.TensorProto(attr.t)

                else:
                    logger.e(logger.Code.NOT_IMPLEMENTED, f"ONNX ConstantOfShape attribute 'value' has type "
                                                          f"'{attr.type}'. This is not yet supported!")

            else:
                logger.w(f"ONNX ConstantOfShape attribute '{attr.name}' is not supported!")
