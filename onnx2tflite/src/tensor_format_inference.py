#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

"""tensor_format_inference

This module contains a class, which can infer the format of all tensors in a given ONNX model.
It stores this format in the tensors '.tensor_format' attribute.
"""

import itertools

import numpy as np

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tensor_formatting import TensorFormat


class TensorFormatInference:
    model: onnx_model.ModelProto

    # A dictionary of ONNX operators, which always use channels first tensors. TODO Expand
    # The name of the operator is mapped to a dictionary, which holds the indices of input and output tensors, that
    # are always channels first.
    # noinspection SpellCheckingInspection
    onnx_ops_with_channels_first_tensors = {
        "AveragePool": {"inputs": [0], "outputs": [0]},
        "BatchNormalization": {"inputs": [0], "outputs": [0]},
        "Conv": {"inputs": [0, 1], "outputs": [0]},
        "ConvTranspose": {"inputs": [0], "outputs": [0]},
        "DepthToSpace": {"inputs": [0], "outputs": [0]},
        "GlobalAveragePool": {"inputs": [0], "outputs": [0]},
        "GlobalMaxPool": {"inputs": [0], "outputs": [0]},
        "InstanceNormalization": {"inputs": [0], "outputs": [0]},
        "LRN": {"inputs": [0], "outputs": [0]},
        "MaxPool": {"inputs": [0], "outputs": [0]},
        "QLinearConv": {"inputs": [0, 3], "outputs": [0]},

        # TODO Possibly consider the QLinear[Global]AveragePool.channels_last attribute?
        "QLinearGlobalAveragePool": {"inputs": [0], "outputs": [0]},
        "QLinearAveragePool": {"inputs": [0], "outputs": [0]},

        "Resize": {"inputs": [0], "outputs": [0]},
        "SpaceToDepth": {"inputs": [0], "outputs": [0]},
        "Upsample": {"inputs": [0], "outputs": [0]}
    }

    # A set of ONNX operators, which have the ability to change the format of their input or output tensors
    onnx_ops_that_can_change_tensor_format = {
        "Einsum",
        "Flatten",
        "Gather",
        "GatherND",
        "OneHot",
        "ReduceL2",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "Reshape",
        "Shape",
        "Squeeze",
        "Transpose",
        "Unsqueeze",

        # LSTM and RNN use their own format, but I believe it should be marked as FORMATLESS, because the main IO has
        #  the same shape as in the ONNX model.
        "LSTM",
        "RNN",
    }

    # A dictionary mapping a tensor name to an operator, which has the tensor among its outputs
    operator_for_output_tensor: dict[str, onnx_model.NodeProto]

    # A dictionary mapping a tensor name to the inferred tensor format
    inferred_tensor_formats: dict[str, TensorFormat]

    # Whether at least one tensor type was changed during last iteration of type inference
    _type_changed_during_last_run: bool

    def __init__(self, model: onnx_model.ModelProto):
        """'TensorFormatInference' can identify the formats of all tensors in an ONNX model with the
        'identify_tensor_formats()' method.

        :param model: ONNX ModelProto containing the model, which will have the format of its tensors inferred.
        """
        self.model = model
        self.inferred_tensor_formats = {}
        self.operator_for_output_tensor = self._initialize_output_tensor_to_operator_dictionary()
        self._type_changed_during_last_run = False

    def identify_tensor_formats(self) -> None:
        """Identify the format of all tensors in the ONNX model.
        Store the format in their '.tensor_format' attribute.
        """
        self._type_changed_during_last_run = True

        # Run repeatedly tensor type inference until there are no new assigned types.
        # Common model will require two runs. Model with individual intermediate tensors
        # consumed by multiple consumers will sometimes require 3 or more runs.
        while self._type_changed_during_last_run:
            self._type_changed_during_last_run = False

            for node in self.model.graph.nodes:
                self._infer_formats_of_node_tensors(node)

        self._commit_inferred_formats_to_tensors()

    def _infer_formats_of_node_tensors(self, node: onnx_model.NodeProto) -> None:
        """Identify the formats of all tensors used by 'node'.

        :param node: The ONNX NodeProto object, that will have the format of its tensors inferred.
        """
        if node.op_type in self.onnx_ops_with_channels_first_tensors:
            # Node uses channels first tensors
            self._handle_operator_which_uses_channels_first_tensors(node)

        elif node.op_type in self.onnx_ops_that_can_change_tensor_format:
            # 'node' is an operator which can change the format of its input or output tensors.

            if node.op_type == "Reshape":
                self._assign_format_to_tensor(node.inputs[1], TensorFormat.FORMATLESS)
                self._infer_format_based_on_io_ranks(node)

            elif node.op_type in ["Flatten", "Transpose", "Shape", "Unsqueeze", "OneHot", "Squeeze", "Einsum"]:
                self._assign_format_to_tensor(node.outputs[0], TensorFormat.FORMATLESS)

            elif node.op_type == "Gather":
                self._assign_format_to_tensor(node.inputs[1], TensorFormat.FORMATLESS)  # Indices are always formatless

                indices = self._get_tensor_data(node.inputs[1])
                if indices is not None:
                    if len(indices.shape) == 1:
                        # Gather preserves the number of dimensions -> propagate the format
                        self._match_formats_of_tensors(node.inputs[0], node.outputs[0])

                    else:
                        # Gather adds new dimensions -> the formats will have to be inferred later
                        self._assign_format_to_tensors([node.inputs[0], node.outputs[0]], TensorFormat.FORMATLESS)

                else:
                    # The 'indices' tensor is dynamic -> the formats will have to be inferred later
                    self._assign_format_to_tensors([node.inputs[0], node.outputs[0]], TensorFormat.FORMATLESS)

            elif node.op_type == "GatherND":
                # Any `channels_last` tensors just require extra `Transpose` ops to be added. Prefer `FORMATLESS` if
                #  possible.
                self._assign_format_to_tensors([node.inputs[0], node.inputs[1], node.outputs[0]],
                                               TensorFormat.FORMATLESS)


            elif node.op_type in ("LSTM", "RNN"):
                self._assign_format_to_tensors(list(node.outputs), TensorFormat.FORMATLESS)

            elif node.op_type in ["ReduceMean", "ReduceL2", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceSum"]:
                if len(node.inputs) > 1:
                    self._assign_format_to_tensor(node.inputs[1], TensorFormat.FORMATLESS)
                self._infer_format_based_on_io_ranks(node)

            else:
                logger.e(logger.Code.NOT_IMPLEMENTED, f"tensor_converter.identify_tensor_formats(): Operator "
                                                      f"'{node.op_type}' can change tensor format, but the format "
                                                      f"inference is not implemented!")

        else:
            # The node works independently of tensor formats.
            self._handle_operator_which_can_use_any_tensor_format(node)

    def _handle_operator_which_uses_channels_first_tensors(self, node: onnx_model.NodeProto) -> None:
        """Identify the formats of tensors used by an operator, which always uses channels first tensors.

        :param node: The operator with channels first tensors, represented by an ONNX NodeProto object.
        """
        if node.op_type == "BatchNormalization":
            # BatchNorm can use 2D inputs, which are formatless
            rank = self._get_tensor_rank_from_name(node.inputs[0])
            if rank is not None and rank <= 2:
                self._assign_format_to_tensors(node.inputs, TensorFormat.FORMATLESS)
                self._assign_format_to_tensors(node.outputs, TensorFormat.FORMATLESS)
                return

        for index, tensor_name in enumerate(node.inputs):
            if index in self.onnx_ops_with_channels_first_tensors[node.op_type]["inputs"]:
                # Tensor is channels first
                self._assign_format_to_tensor(tensor_name, TensorFormat.CHANNELS_FIRST)
                self._back_propagate_channels_first_format(tensor_name)

            else:
                # Tensor is formatless
                self._assign_format_to_tensor(tensor_name, TensorFormat.FORMATLESS)

        for index, tensor_name in enumerate(node.outputs):
            if index in self.onnx_ops_with_channels_first_tensors[node.op_type]["outputs"]:
                # Tensor is channels first
                self._assign_format_to_tensor(tensor_name, TensorFormat.CHANNELS_FIRST)
            else:
                # Tensor is formatless
                self._assign_format_to_tensor(tensor_name, TensorFormat.FORMATLESS)

    def _handle_operator_which_can_use_any_tensor_format(self, node: onnx_model.NodeProto) -> None:
        """Identify the formats of tensors used by an operator, which can use tensors in any format.

        :param node: The operator, represented by an ONNX NodeProto object.
        """
        if not self._node_uses_at_least_one_channels_first_tensor(node):
            # Nothing important is known about this node. Mark every tensor as formatless for now
            # Future back propagation can change it to channels first.
            for tensor_name in itertools.chain(node.inputs, node.outputs):
                self._assign_format_to_tensor(tensor_name, TensorFormat.FORMATLESS)

        else:
            # Node has at least 1 channels first tensor -> try to assign the format to all other valid tensors
            for tensor_name in itertools.chain(node.inputs, node.outputs):
                is_0d_to_2d = self._tensor_has_0_to_2_dimensions(tensor_name)

                if self._get_tensor_format_for_name(tensor_name).is_channels_first():
                    # Tensor is already channels first
                    continue

                if (is_0d_to_2d is not None) and is_0d_to_2d:
                    # 0D, 1D and 2D tensors are formatless
                    self._assign_format_to_tensor(tensor_name, TensorFormat.FORMATLESS)

                else:
                    # Mark this tensor as channels first
                    self._assign_format_to_tensor(tensor_name, TensorFormat.CHANNELS_FIRST)

                    # Propagate this change back
                    self._back_propagate_channels_first_format(tensor_name)

    def _back_propagate_channels_first_format(self, start_tensor_name: str) -> None:
        """Recursively call 'self.__infer_formats_of_node_tensors()' on previous operators. Starting with the
        operator, that has 'start_tensor_name' as its output.

        After a channels first format has been assigned to 'start_tensor_name', this change must be propagated
        to the operator, which produces this tensor. The formats of all tensors the operator uses will be
        recalculated, which can cause this function to be called again, with the other tensors as arguments.
        """
        if self._is_static_tensor(start_tensor_name):
            # Nowhere to propagate to
            return

        previous_op = self.operator_for_output_tensor.get(start_tensor_name, None)

        if previous_op is None:
            # Nowhere to propagate to
            return

        if previous_op.op_type in self.onnx_ops_that_can_change_tensor_format:
            # TODO For now propagation ends here. Theoretically if conversion of these operators is correct, everything
            #  should work. But this might potentially cause unnecessary Transpose ops in some rare cases.
            return

        self._infer_formats_of_node_tensors(previous_op)

    def _infer_format_based_on_io_ranks(self, node: onnx_model.NodeProto) -> None:
        """Determine the format of the output tensor of given operator.

        :param node: The Reshape/ReduceX operator represented as an ONNX NodeProto object.
        """
        main_input_rank = self._get_tensor_rank_from_name(node.inputs[0])
        main_output_rank = self._get_tensor_rank_from_name(node.outputs[0])

        if (main_input_rank is None) or (main_output_rank is None):
            # In most cases, this would still be fine. But we cannot identify the cases, when it wouldn't.
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"The shapes of tensors '{node.inputs[0]}' and '{node.outputs[0]}' around a '{node.op_type}' "
                     f"operator are not in the model. They are required for accurate conversion!")

        elif main_output_rank == main_input_rank:
            # Operator maintains the number of dimensions -> try to propagate the format.
            self._match_formats_of_tensors(node.inputs[0], node.outputs[0])

        else:
            # Either the op 'flattens' the tensor, so output is formatless, or it scales it up, in which case the
            # format is assumed to be 'FORMATLESS', and may be back propagated as channels first later.
            self._assign_format_to_tensor(node.outputs[0], TensorFormat.FORMATLESS)

    def _get_tensor_format_for_name(self, tensor_name: str) -> TensorFormat:
        """Return the inferred tensor format for a tensor with given name."""
        return self.inferred_tensor_formats.get(tensor_name, TensorFormat.NONE)

    def _get_tensor_rank_from_name(self, tensor_name: str) -> int | None:
        """Get the number of dimensions of a tensor with given name.

        :param tensor_name: Name of the tensor, to find the number of dimensions for.
        :return: The number of dimensions of the given tensor, or
                 None, if the tensor is not defined in the model.
        """
        # Search for the tensor in 'value_info', 'inputs' and 'outputs'
        for value_info in itertools.chain(self.model.graph.value_info, self.model.graph.inputs,
                                          self.model.graph.outputs):
            if value_info.name == tensor_name:
                if value_info.type.tensor_type is not None:
                    return len(value_info.type.tensor_type.shape.dims)

        # Search for the tensor in 'initializers'
        for tensor in self.model.graph.initializers:
            if tensor.name == tensor_name:
                return len(tensor.dims)

        return None

    def _tensor_has_0_to_2_dimensions(self, tensor_name: str) -> bool | None:
        """Determine if tensor 'tensor_name' has 0, 1 or 2 dimensions, or not.

        :return: True, if 'tensor_name' has 0, 1 or 2 dimensions.
                 None, if tensor doesn't have its shape specified in the model.
                 False, otherwise.
        """
        rank = self._get_tensor_rank_from_name(tensor_name)

        if rank is None:
            return None

        return 0 <= rank <= 2

    def _assign_format_to_tensor(self, tensor_name: str, tensor_format: TensorFormat) -> None:
        """Map 'tensor_format' to tensor with given name. If the tensor already had a channels first format, it will
         not be overwritten.

        :param tensor_name: Name of the tensor, to assign the tensor format to.
        :param tensor_format: The TensorFormat to assign to the given tensor.
        """
        previous_format = self.inferred_tensor_formats.get(tensor_name, tensor_format)
        if previous_format is not tensor_format:
            if previous_format.is_channels_first() and (not tensor_format.is_channels_first()):
                # If a tensor was once identified as channels first, it cannot be changed
                return

        if previous_format != tensor_format:
            self._type_changed_during_last_run = True

        self.inferred_tensor_formats[tensor_name] = tensor_format

    def _assign_format_to_tensors(self, tensor_names: list[str], tensor_format: TensorFormat) -> None:
        """Map given 'tensor_format' to tensors with their names in 'tensor_names'.

        :param tensor_names: A list of names of tensors, to assign the 'tensor_format' to.
        :param tensor_format: The TensorFormat, to assign to all tensors in 'tensor_names'.
        """
        for tensor_name in tensor_names:
            self._assign_format_to_tensor(tensor_name, tensor_format)

    def _match_formats_of_tensors(self, tensor_1: str, tensor_2: str) -> None:
        """If one of 'tensor_1' or 'tensor_2' is channels first, make the other channels first as well.
        If neither is channels first, make them both formatless.
        """
        format_1 = self._get_tensor_format_for_name(tensor_1)
        format_2 = self._get_tensor_format_for_name(tensor_2)

        if format_1.is_channels_first() or format_2.is_channels_first():
            # At least 1 is channels first
            if not format_1.is_channels_first():
                self._assign_format_to_tensor(tensor_1, TensorFormat.CHANNELS_FIRST)
            elif not format_2.is_channels_first():
                self._assign_format_to_tensor(tensor_2, TensorFormat.CHANNELS_FIRST)

        else:
            self._assign_format_to_tensor(tensor_1, TensorFormat.FORMATLESS)
            self._assign_format_to_tensor(tensor_2, TensorFormat.FORMATLESS)

    def _is_static_tensor(self, tensor_name: str) -> bool:
        """Determine if tensor with given name is static or not."""
        for static_tensor in self.model.graph.initializers:
            if static_tensor.name == tensor_name:
                return True

        return False

    def _get_tensor_data(self, tensor_name: str) -> np.ndarray | None:
        for tensor in self.model.graph.initializers:
            if tensor.name == tensor_name:
                # Found the tensor
                return tensor.data

        return None

    def _initialize_output_tensor_to_operator_dictionary(self) -> dict[str, onnx_model.NodeProto]:
        """Initialize the dictionary, which maps tensor names to operators, which have this tensor as an output.

        :return: The initialized dictionary.
        """
        output_tensor_to_operator = {}

        for node in self.model.graph.nodes:
            for output_tensor_name in node.outputs:
                output_tensor_to_operator[output_tensor_name] = node

        return output_tensor_to_operator

    def _node_uses_at_least_one_channels_first_tensor(self, node: onnx_model.NodeProto) -> bool:
        """Determine if given 'node' uses at least 1 tensor, which has been assigned a channels first format."""
        for tensor_name in itertools.chain(node.inputs, node.outputs):
            if self._get_tensor_format_for_name(tensor_name).is_channels_first():
                return True

        return False

    def _commit_inferred_formats_to_tensors(self) -> None:
        """Assign tensor formats in 'self.inferred_tensor_formats' to corresponding ONNX ValueInfoProto and TensorProto
        objects in the ONNX model.
        """
        tensors_without_inferred_format = set()

        for value_info in itertools.chain(self.model.graph.value_info, self.model.graph.inputs,
                                          self.model.graph.outputs):
            tensor_name = value_info.name
            if tensor_name in self.inferred_tensor_formats:
                value_info.tensor_format = self.inferred_tensor_formats[tensor_name]

            else:
                tensors_without_inferred_format.add(tensor_name)

        for tensor in self.model.graph.initializers:
            tensor_name = tensor.name
            if tensor_name in self.inferred_tensor_formats:
                tensor.tensor_format = self.inferred_tensor_formats[tensor_name]

            else:
                tensors_without_inferred_format.add(tensor.name)

        if len(tensors_without_inferred_format) != 0:
            logger.w(f"Tensors without inferred format: {''.join(tensors_without_inferred_format)}")
