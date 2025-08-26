#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import Optional

from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class Cast(meta.ONNXOperatorAttributes):
    saturate: int  # Only used for float8
    to: Optional[TensorProto.DataType.values]  # noqa: UP045

    def _default_values(self) -> None:
        self.saturate = 1
        self.to = None

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "saturate":
                self.saturate = attr.i
            elif attr.name == "to":
                try:
                    enum_name = TensorProto.DataType.Name(attr.i)
                    self.to = TensorProto.DataType.Value(enum_name)
                except ValueError as e:
                    logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                             "Failed to parse the `to` attribute of ONNX `Cast`.",
                             exception=e)
            else:
                logger.w(f"ONNX Cast attribute '{attr.name}' is not supported!")

        if self.to is None:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX `Cast` is missing the required attribute `to`.")
