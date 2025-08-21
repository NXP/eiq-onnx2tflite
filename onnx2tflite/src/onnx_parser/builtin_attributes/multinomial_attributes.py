#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class Multinomial(meta.ONNXOperatorAttributes):
    dtype: TensorProto.DataType
    sample_size: int
    seed: float | None

    def _default_values(self):
        self.dtype = TensorProto.INT32
        self.sample_size = 1
        self.seed = None

    def _init_attributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "dtype":
                    self.dtype = attr.i
                case "sample_size":
                    self.sample_size = attr.i
                case "seed":
                    self.seed = attr.f
                case _:
                    logger.w(f"ONNX `Multinomial` attribute '{attr.name}' is not supported!")
