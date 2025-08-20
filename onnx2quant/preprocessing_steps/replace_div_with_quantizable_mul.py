#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import onnx
from onnx import TensorProto

from onnx2quant.preprocessing_steps.base_preprocessing_step import PreprocessingStep
from onnx2tflite.src import logger


class ReplaceDivWithQuantizableMul(PreprocessingStep):
    """Replace `Div` nodes with `Mul` where possible. TFLite doesn't support int8 quantized `Div` so this preprocessing
    steps can avoid the need to run TFLite `Div` in float32.
    """

    def disabling_flag(self) -> str:
        return "--no-replace-div-with-mul"

    def run(self) -> None:
        num_affected_nodes = 0

        for node in self.model.graph.node:
            if node.op_type != "Div":
                continue

            if (data := self.try_get_tensor_data(node.input[1])) is None:
                # The second tensor is dynamic.
                continue

            static_tensor = node.input[1]

            if self.try_get_tensor_type(static_tensor) != TensorProto.FLOAT:
                # Conversion of `Mul` is only supported for float32, int32 and int64. For the int types, computing the
                #  reciprocal values would introduce errors.
                continue

            reciprocal_data = (1. / data).astype(data.dtype)

            # If the reciprocal data falls in a larger interval than the original data, quantizing it would cause a
            #  significant decrease in accuracy.
            # For example if the original data is [0.9, 0.95, 0.001], the reciprocal data would be [1.11, 1.05, 1000.0].
            #  Even in this trivial example the reciprocal tensor would lose a lot of precision due to the large
            #  interval to cover.

            original_range = data.max() - data.min()
            reciprocal_range = reciprocal_data.max() - reciprocal_data.min()
            if reciprocal_range > original_range:  # The tolerance can be adjusted.
                # Don't perform the replacement, to avoid large accuracy loss.
                continue

            # The `Div` can be replaced by `Mul`.
            node.op_type = "Mul"

            # Create a new tensor in case it is used elsewhere.
            node.input[1] = self.create_tensor_for_data(reciprocal_data, static_tensor)

            num_affected_nodes += 1

        # Make sure this optimization didn't break anything.
        onnx.checker.check_model(self.model)

        if num_affected_nodes != 0:
            logger.w("Replacing `Div` with `Mul` to improve quantization and inference speed. This may result in a "
                     "less accurate model. If you want to avoid this, run the quantization again with the flag "
                     f"{logger.Style.cyan + self.disabling_flag() + logger.Style.end}.")
