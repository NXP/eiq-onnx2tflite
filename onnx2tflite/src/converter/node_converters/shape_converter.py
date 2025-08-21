#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import common, translator
from onnx2tflite.src.converter.conversion.translator import tf_lite_type_to_numpy
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import shape_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import gather_options, shape_options, slice_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, FLOATS, INTS


class ShapeConverter(NodeConverter):
    node = "Shape"

    onnx_supported_types = ALL_TYPES
    tflite_supported_types = ALL_TYPES
    verified_types = FLOATS + INTS + [TensorType.UINT8, TensorType.UINT32, TensorType.UINT64, TensorType.BOOL,
                                      TensorType.STRING]

    def _append_gather_operator(self, shape_op: tflite_model.Operator, ops_to_add: list[tflite_model.Operator]) -> None:
        """Append a 'Gather' operator after the 'shape_op' and add it to the 'ops_to_add'. The 'Gather' op will permute
             the output of 'shape_op' to a channels first shape.

        :param shape_op: A 'Shape' operator after which the 'Gather' operator will be created.
        :param ops_to_add: A list of operators that will be added to the model later.
        """
        input_rank = shape_op.tmp_inputs[0].rank

        gather_op = tflite_model.Operator(builtin_options=gather_options.Gather(0))
        ops_to_add.append(gather_op)

        gather_input = self.builder.duplicate_tensor(shape_op.tmp_outputs[0], "gather_input")
        to_onnx_format_perm = translator.create_channels_last_to_channels_first_permutation(input_rank)
        gather_indices = self.builder.create_tensor_for_data(to_onnx_format_perm, "permute_to_onnx_shape")

        gather_op.tmp_inputs = [gather_input, gather_indices]
        gather_op.tmp_outputs = [shape_op.tmp_outputs[0]]
        shape_op.tmp_outputs[0] = gather_input

    def _append_slice_operator(self, ops_to_add: list[tflite_model.Operator], begin_tensor: tflite_model.Tensor,
                               size_tensor: tflite_model.Tensor) -> None:
        """Create a 'Slice' operator after the last operator in 'ops_to_add' and add it to the list.

        :param ops_to_add: A list of operators that will be added to the model later.
        :param begin_tensor: The 'begin' operand of the TFLite Slice operator.
        :param size_tensor: The 'size' operand of the TFLite Slice operator.
        """
        previous_op = ops_to_add[-1]

        slice_op = tflite_model.Operator(builtin_options=slice_options.Slice())
        ops_to_add.append(slice_op)

        slice_input = self.builder.duplicate_tensor(previous_op.tmp_outputs[0], "slice_input")

        slice_op.tmp_inputs = [slice_input, begin_tensor, size_tensor]
        slice_op.tmp_outputs = [previous_op.tmp_outputs[0]]
        previous_op.tmp_outputs = [slice_input]

    # noinspection PyMethodMayBeStatic
    def _validate_index(self, index: int, default_value: int, rank: int) -> int:
        if index is None:
            index = default_value

        elif index < 0:
            index += rank

        return common.clamp(index, 0, rank)

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'Shape' operator to TFLite."""
        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX 'Shape' operator has unexpected number of inputs! Got'{len(t_op.tmp_inputs)}', expected '1'.")

        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        input_rank = t_op.tmp_inputs[0].rank
        output_type = t_op.tmp_outputs[0].type

        t_op.builtin_options = shape_options.Shape(output_type)

        ops_to_add = [t_op]

        if t_op.tmp_inputs[0].tensor_format.is_channels_last():
            # The input has a format but the output doesn't. Make sure the output is exactly the same as in the ONNX
            #  model. Instead of prepending Transpose, we can just append a Gather operator to permute the dimensions of
            #  the shape.
            self._append_gather_operator(t_op, ops_to_add)

        attrs = cast(shape_attributes.Shape, node.attributes)
        start = attrs.start
        end = attrs.end
        if start is None and end is None:
            # Basic case
            return ops_to_add

        # -- must create a 'Slice' operator after 'Shape' --

        # Validate the ONNX Shape attributes
        start = self._validate_index(start, 0, input_rank)
        end = self._validate_index(end, input_rank, input_rank)
        if end < start:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f"ONNX 'Shape' operator has 'end' < 'start' ('{end}' < '{start}')!")

        # Create the TFLite 'begin' tensor
        begin_tensor = self.builder.create_tensor_for_data(np.asarray([start], tf_lite_type_to_numpy(output_type)),
                                                           "begin")

        # Create the TFLite 'size' tensor
        size = end - start
        size_tensor = self.builder.create_tensor_for_data(np.asarray([size], tf_lite_type_to_numpy(output_type)),
                                                          "size")

        # Create the 'Slice' operator
        self._append_slice_operator(ops_to_add, begin_tensor, size_tensor)

        return ops_to_add
