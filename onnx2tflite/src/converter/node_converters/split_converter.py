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
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import split_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import split_options, split_v_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, INTS


# noinspection PyMethodMayBeStatic
class SplitConverter(NodeConverter):
    node = "Split"

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/split.cc#L87-L90
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8]
    verified_types = INTS + [TensorType.FLOAT32, TensorType.UINT8]

    def _validate_axis(self, axis: int, input_rank: int) -> int:
        """Make sure the 'axis' is valid for TFLite Split and return it.

        :param axis: The axis to validate.
        :param input_rank: The rank of the main input tensor of the Split operator.
        :return: A valid 'axis' parameter for the TFLite Split and SplitV operators.
        """
        original = axis
        if axis < 0:
            axis += input_rank

        if not (0 <= axis < input_rank):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f"ONNX 'Split' has invalid 'axis' attribute! ('{original}')")

        return axis

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX Split operator to TFLite.

        :param node: ONNX Split operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        if not (1 <= len(t_op.tmp_inputs) <= 2):
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX 'Split' has unexpected number of inputs! Got '{len(t_op.tmp_inputs)}', expected 1 or 2.")

        if node.version == 1:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX 'Split' version '1' is not supported.")

        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        o_split_attributes = cast(split_attributes.Split, node.attributes)

        main_input = t_op.tmp_inputs[0]
        rank = main_input.rank

        # Prepare the axis
        axis = self._validate_axis(o_split_attributes.axis, rank)
        if main_input.tensor_format.is_channels_last():
            # Change the 'axis' to match the input format
            axis = translator.create_channels_last_to_channels_first_permutation(rank)[axis]
        axis_tensor = self.builder.create_tensor_for_data(np.asarray([axis], np.int32), "split_dim_")

        if node.version < 13:
            # `split` is an operator attribute.

            if o_split_attributes.split is not None:
                # Create a tensor for the 'size_splits' of the TFLite SplitV operator
                size_splits = self.builder.create_tensor_for_data(np.asarray(o_split_attributes.split, np.int32),
                                                                  "size_splits_")
            else:
                size_splits = None

        else:
            # `split` is passed as input tensor.
            size_splits = try_get_input(t_op, 1)

        if o_split_attributes.num_outputs is not None:
            if size_splits is not None:
                # According to the documentation, 'num_outputs' and 'split' should not both be specified.
                logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                         "ONNX 'Split' has both 'num_outputs' and 'split' specified. Documentation prohibits this.")
            num_splits = o_split_attributes.num_outputs

            # ONNXRT: There are some issues with the ONNX definition and ONNX Runtime implementation here. ONNX states (and ONNX
            #  Runtime implements literally (onnxruntime/core/providers/cpu/tensor/split.cc#L89)), that if the input dim
            #  cannot be divided by num_splits exactly, all splits should be the same, and just the last one can be
            #  smaller.
            # For example if input dim == 10 and num_splits == 4, the splits are [3, 3, 3, 1]. But for example if the
            #  input dim == 6 and num splits == 4, the splits cannot obey the rules. In these cases, ONNX Runtime exits
            #  with error. The shape inference uses the same algorithm to compute the output shape, so if it is
            #  invalid, we can exit with error as well.
            output_splits = [output.shape.get(axis) for output in t_op.tmp_outputs]
            if sum(output_splits) != t_op.tmp_inputs[0].shape.get(axis) or any(split == 0 for split in output_splits):
                logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                         f"ONNX 'Split' with input shape '{t_op.tmp_inputs[0].shape.vector}', axis '{axis}' and "
                         f"num_outputs '{num_splits}' doesn't make sense.")

            # The 'splits' corresponding to this particular situation were computed as part of shape inference, so we
            #  can use them explicitly.
            size_splits = self.builder.create_tensor_for_data(np.asarray(output_splits, np.int32), "splits_")

        else:
            num_splits = len(t_op.tmp_outputs)

        if t_op.is_quantized_without_qdq():
            for output in t_op.tmp_outputs:
                propagate_quantization(main_input, output)

        if size_splits is None:
            # Convert to 'Split'
            t_op.builtin_options = split_options.Split(num_splits)
            t_op.tmp_inputs = [axis_tensor, main_input]

        else:
            # Convert to 'SplitV'
            t_op.builtin_options = split_v_options.SplitV(num_splits)
            t_op.tmp_inputs = [main_input, size_splits, axis_tensor]

        return [t_op]
