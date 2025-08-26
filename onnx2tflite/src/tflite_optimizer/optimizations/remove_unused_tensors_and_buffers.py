#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_optimizer.optimizations.base_optimization import BaseOptimization


class RemoveUnusedTensorsAndBuffers(BaseOptimization):

    def _get_used_tensors_and_buffers(self) -> (set[tflite_model.Tensor], set[tflite_model.Buffer]):
        """Get a set of all tensors used by the operators in the model, and a set of all buffers used by these tensors.
        """
        used_tensors = set()
        used_buffers = set()

        for op in self._builder.get_operators():
            for tensor in op.tmp_inputs + op.tmp_outputs:
                used_tensors.add(tensor)
                if tensor.tmp_buffer is not None:
                    used_buffers.add(tensor.tmp_buffer)

        return used_tensors, used_buffers

    def __call__(self) -> bool:
        """Remove all tensors and buffers from the model, that are not used.
        :return: True, if any tensors/buffers were removed. Otherwise, False.
        """
        used_tensors, used_buffers = self._get_used_tensors_and_buffers()

        made_changes = False
        model_inputs = self._builder.get_sub_graph().inputs.tmp_inputs
        to_remove = []
        for tensor in self._builder.get_tensors():
            if tensor not in used_tensors:
                if tensor in model_inputs:
                    # It is possible that an input tensor ended up not being used by any operators. But removing it from
                    #  the model would cause errors at runtime, so it must stay.
                    pass

                else:
                    to_remove.append(tensor)

        for tensor in to_remove:
            made_changes = True
            self._builder.get_tensors().remove(tensor)

        to_remove = []
        for buffer in self._builder.get_buffers().vector:
            if buffer not in used_buffers:
                to_remove.append(buffer)

        for buffer in to_remove:
            made_changes = True
            self._builder.get_buffers().remove(buffer)

        return made_changes
