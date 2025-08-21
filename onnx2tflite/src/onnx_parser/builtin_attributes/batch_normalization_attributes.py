#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""BatchNormalization

Representation of an ONNX 'BatchNormalization' operator. 
Initialized from a protobuf descriptor object.
"""

from collections.abc import Iterable

import numpy as np
import onnx

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser.meta import meta


class BatchNormalization(meta.ONNXOperatorAttributes):
    epsilon: float
    spatial: int

    # momentum: float  # Only used in training
    # training_mode: int  # Always '0' in our case as we only want to convert pre-trained models.

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _default_values(self):
        self.epsilon = np.float32(1e-5)
        self.spatial = 1

    def _init_attributes(self):
        for attr in self._descriptor:
            if attr.name == "epsilon":
                self.epsilon = attr.f

            elif attr.name == "momentum":
                # self.momentum = attr.f  # Only has effect during training
                pass

            elif attr.name == "spatial":
                self.spatial = attr.i

            elif attr.name == "training_mode":
                # self.training_mode = attr.i  # Always 0
                pass

            else:
                logger.w(f"ONNX BatchNormalization attribute '{attr.name}' is not supported.")
