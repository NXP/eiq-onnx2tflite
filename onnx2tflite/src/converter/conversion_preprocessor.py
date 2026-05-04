#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from dataclasses import dataclass

import onnx

from onnx2tflite.src.converter.preprocessing_steps.duplicate_dequantize_linear_for_each_consumer import (
    DuplicateDequantizeLinearForEachConsumer,
)
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig


@dataclass
class ConversionPreprocessor:
    model: onnx.ModelProto
    conversion_config: ConversionConfig

    def preprocess(self) -> None:
        preprocessing_steps = []
        if self.conversion_config.duplicate_multiconsumer_dequantize_linear:
            preprocessing_steps.append(DuplicateDequantizeLinearForEachConsumer(self.model))

        for preprocessing_step in preprocessing_steps:
            try:
                preprocessing_step.run()

            except Exception as e:  # noqa: BLE001
                logger.e(
                    logger.Code.PREPROCESSING_ERROR,
                    "An unexpected error occurred during conversion preprocessing. Run the conversion again with the "
                    f"flag {logger.Style.cyan + preprocessing_step.disabling_flag() + logger.Style.end} to avoid "
                    "this.",
                    e,
                )
