#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from dataclasses import dataclass

import onnx

from onnx2quant.preprocessing_steps.replace_constant_with_static_tensor import ReplaceConstantWithStaticTensor
from onnx2quant.preprocessing_steps.replace_div_with_quantizable_mul import ReplaceDivWithQuantizableMul
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.src import logger


@dataclass
class Preprocessor:
    model: onnx.ModelProto
    quantization_config: QuantizationConfig

    def preprocess(self):
        preprocessing_steps = []
        if self.quantization_config.replace_constant_with_static_tensor:
            preprocessing_steps.append(ReplaceConstantWithStaticTensor(self.model))
        if self.quantization_config.replace_div_with_mul:
            preprocessing_steps.append(ReplaceDivWithQuantizableMul(self.model))

        for preprocessing_step in preprocessing_steps:
            try:
                preprocessing_step.run()

            except Exception as e:
                logger.e(logger.Code.PREPROCESSING_ERROR,
                         "An unexpected error occurred during preprocessing. Run the quantization again with the "
                         f"flag {logger.Style.cyan + preprocessing_step.disabling_flag() + logger.Style.end} to avoid "
                         "this.", e)
