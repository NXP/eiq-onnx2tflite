#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import onnx

from onnx2quant.preprocessing_steps.base_preprocessing_step import PreprocessingStep
from onnx2tflite.src import logger


class ReplaceConstantWithStaticTensor(PreprocessingStep):
    """Remove `Constant` nodes and assign the static data directly to the output tensor. This can allow other
    preprocessing steps to improve the model.
    """

    def disabling_flag(self) -> str:
        return "--no-replace-constant-with-static-tensor"

    def run(self) -> None:
        to_remove = []
        for node in self.model.graph.node:
            if node.op_type != "Constant":
                continue

            new_output_tensor = None
            for attr in node.attribute:
                match attr.name:
                    case "value":
                        new_output_tensor = attr.t

                    case _:
                        logger.d(f"ReplaceConstantWithStaticTensor: attribute `{attr.name}` is not yet supported.")

            # Add the `output_tensor` to `initializers`.
            new_output_tensor.name = node.output[0]
            self.add_initializer(new_output_tensor)

            # Remove the original dynamic output of the `Constant` from the `value_info` collection.
            old_output_vi_list = [vi for vi in self.model.graph.value_info if vi.name == node.output[0]]
            if len(old_output_vi_list) != 0:
                self.model.graph.value_info.remove(old_output_vi_list[0])

            # Remove the `Constant` node.
            to_remove.append(node)

        for node in to_remove:
            self.model.graph.node.remove(node)

        # Make sure this optimization didn't break anything.
        onnx.checker.check_model(self.model)
