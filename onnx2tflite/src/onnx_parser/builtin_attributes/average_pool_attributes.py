#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#


from typing import Optional, Iterable

import onnx

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.meta.meta as meta


class AveragePool(meta.ONNXOperatorAttributes):
    auto_pad: str
    ceil_mode: int
    count_include_pad: int
    dilations: Optional[meta.ONNXIntListAttribute]
    kernel_shape: Optional[meta.ONNXIntListAttribute]
    pads: Optional[meta.ONNXIntListAttribute]
    strides: Optional[meta.ONNXIntListAttribute]

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.auto_pad = "NOTSET"
        self.ceil_mode = 0
        self.count_include_pad = 0
        self.dilations = None
        self.kernel_shape = None
        self.pads = None
        self.strides = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "auto_pad":
                self.auto_pad = attr.s.decode("UTF-8")
            elif attr.name == "ceil_mode":
                self.ceil_mode = attr.i
            elif attr.name == "count_include_pad":
                self.count_include_pad = attr.i
            elif attr.name == "dilations":
                self.dilations = meta.ONNXIntListAttribute(attr)
            elif attr.name == "kernel_shape":
                self.kernel_shape = meta.ONNXIntListAttribute(attr)
            elif attr.name == "pads":
                self.pads = meta.ONNXIntListAttribute(attr)
            elif attr.name == "strides":
                self.strides = meta.ONNXIntListAttribute(attr)
            else:
                logger.w(f"ONNX AveragePool attribute '{attr.name}' is not supported!")

        if self.kernel_shape is None:
            # Required attribute
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX AveragePool is missing the 'kernel_shape'!")
