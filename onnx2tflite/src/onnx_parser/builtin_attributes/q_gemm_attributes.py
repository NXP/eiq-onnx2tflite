#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


from collections.abc import Iterable

import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class QGemm(meta.ONNXOperatorAttributes):
    alpha: float
    transA: int  # noqa: N815
    transB: int  # noqa: N815

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self) -> None:
        # QGemm uses the same module to parse its attributes as regular Gemm. (line 6):
        # https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/quantization/quant_gemm.cc
        self.alpha = 1.0
        self.transA = 0
        self.transB = 0

    def _init_attributes(self) -> None:
        for attr in self._descriptor:
            if attr.name == "alpha":
                self.alpha = attr.f
            elif attr.name == "transA":
                self.transA = attr.i
            elif attr.name == "transB":
                self.transB = attr.i
            else:
                logger.w(f"ONNX QGemm attribute '{attr.name}' is not supported!")
