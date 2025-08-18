#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from __future__ import annotations

from enum import Enum

from onnx2tflite.src import logger


class TensorFormat(Enum):
    CHANNELS_FIRST = 0

    CHANNELS_LAST = 10

    # The format of TFLite Conv3D weights tensor: [output_channels, input_channels, D, H, W]
    CONV_3D_WEIGHT_FORMAT = 11

    # Intermediate format between 'Transpose' and 'Reshape' ops when single dimension with value 1
    # is added/removed via reshaping
    RESHAPE_SINGLE_UNITARY_TRANSPOSITION = 12

    # The format of TFLite TransposeConv 2D weights tensor: [M/group, kH, kW, C]
    TRANSPOSE_CONV_2D_WEIGHT_FORMAT = 13

    # No special format (matrices, vectors, shapes etc.). All tensors with the FORMATLESS format MUST have EXACTLY
    #  the same shape and data in the TFLite model and in the ONNX model.
    FORMATLESS = 20

    NONE = 30  # Format has not been identified

    def is_channels_first(self) -> bool:
        return self == TensorFormat.CHANNELS_FIRST

    def is_channels_last(self) -> bool:
        return self == TensorFormat.CHANNELS_LAST

    def to_onnx(self) -> TensorFormat:
        """ Convert the tensor format from TFLite to ONNX. """
        if self == TensorFormat.CHANNELS_LAST:
            return TensorFormat.CHANNELS_FIRST

        logger.e(logger.Code.INTERNAL_ERROR,
                 f'TensorFormat.to_onnx(): Unexpected format `{self}`.')
