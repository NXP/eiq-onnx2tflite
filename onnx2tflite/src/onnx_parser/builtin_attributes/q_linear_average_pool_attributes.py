#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class QLinearAveragePool(meta.ONNXOperatorAttributes):
    auto_pad: str
    ceil_mode: int
    channels_last: int
    count_include_pad: int
    kernel_shape: meta.ONNXIntListAttribute | None
    pads: meta.ONNXIntListAttribute | None
    strides: meta.ONNXIntListAttribute | None

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.auto_pad = "NOTSET"
        self.ceil_mode = 0
        self.channels_last = 0
        self.count_include_pad = 0
        self.kernel_shape = None
        self.pads = None
        self.strides = None

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "auto_pad":
                self.auto_pad = attr.s.decode("UTF-8")
            elif attr.name == "ceil_mode":
                self.ceil_mode = attr.i
            elif attr.name == "channels_last":
                self.channels_last = attr.i
            elif attr.name == "count_include_pad":
                self.count_include_pad = attr.i
            elif attr.name == "kernel_shape":
                self.kernel_shape = meta.ONNXIntListAttribute(attr)
            elif attr.name == "pads":
                self.pads = meta.ONNXIntListAttribute(attr)
            elif attr.name == "strides":
                self.strides = meta.ONNXIntListAttribute(attr)
            else:
                logger.w(f"ONNX QLinearAveragePool attribute '{attr.name}' is not supported!")

        if self.kernel_shape is None:
            # Required attribute
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     "ONNX QLinearAveragePool is missing the 'kernel_shape'!")

        if self.channels_last != 0:
            # The converter doesn't really consider this option right now.
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX QLinearAveragePool with channels_last = "
                                                  f"'{self.channels_last}' is not yet implemented.")
