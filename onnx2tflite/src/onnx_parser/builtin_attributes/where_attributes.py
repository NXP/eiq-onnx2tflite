#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from collections.abc import Iterable

import onnx

from onnx2tflite.src.onnx_parser.meta import meta


class Where(meta.ONNXOperatorAttributes):
    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)
