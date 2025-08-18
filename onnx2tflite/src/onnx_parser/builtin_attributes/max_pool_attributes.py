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


class MaxPool(meta.ONNXOperatorAttributes):
    auto_pad: str
    ceil_mode: int
    dilations: Optional[meta.ONNXIntListAttribute]
    kernel_shape: Optional[meta.ONNXIntListAttribute]
    pads: Optional[meta.ONNXIntListAttribute]
    storage_order: int  # Not necessary. Only has effect on the second output tensor, which cannot be converted anyway.
    strides: Optional[meta.ONNXIntListAttribute]

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.auto_pad = "NOTSET"
        self.ceil_mode = 0
        self.dilations = None
        self.kernel_shape = None
        self.pads = None
        self.storage_order = 0
        self.strides = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "auto_pad":
                self.auto_pad = attr.s.decode("UTF-8")
            elif attr.name == "ceil_mode":
                self.ceil_mode = attr.i
            elif attr.name == "dilations":
                self.dilations = meta.ONNXIntListAttribute(attr)
            elif attr.name == "kernel_shape":
                self.kernel_shape = meta.ONNXIntListAttribute(attr)
            elif attr.name == "pads":
                self.pads = meta.ONNXIntListAttribute(attr)
            elif attr.name == "storage_order":
                self.storage_order = attr.i
            elif attr.name == "strides":
                self.strides = meta.ONNXIntListAttribute(attr)
            else:
                logger.w(f"ONNX MaxPool attribute '{attr.name}' is not supported!")

        if self.kernel_shape is None:
            # 'kernel_shape' is a required attribute
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX MaxPool has no 'kernel_shape' attribute!")
