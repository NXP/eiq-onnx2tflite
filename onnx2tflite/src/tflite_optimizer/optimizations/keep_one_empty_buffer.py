#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.tflite_optimizer.optimizations.base_optimization import BaseOptimization


class KeepOneEmptyBuffer(BaseOptimization):

    def __call__(self) -> bool:
        """Create a single empty `Buffer` object and assign it to all tensors in the model that don't have static data.
        :return: True, if any tensors had their buffer changed. Otherwise, False.
        """
        made_changes = False
        empty_buffer = self._builder.get_first_empty_buffer()

        for t in self._builder.get_tensors().vector:
            if tensor_has_data(t):
                # The buffer of `t` is not empty.
                continue

            if t.tmp_buffer == empty_buffer:
                # Already optimized.
                continue

            if t.is_variable:
                # The data of the tensor will change at runtime, so it shouldn't share the buffer with other tensors.
                continue

            # It's safe to replace the buffer.
            t.tmp_buffer = empty_buffer
            made_changes = True

        return made_changes
