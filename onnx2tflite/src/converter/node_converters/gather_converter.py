#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import Any, cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import gather_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import gather_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, INTS


# noinspection PyMethodMayBeStatic
class GatherConverter(NodeConverter):
    node = "Gather"

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/gather.cc#L81-L99
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8, TensorType.BOOL, TensorType.STRING]
    verified_types = tflite_supported_types

    def _is_an_array_of_ints(self, array: Any) -> bool:
        """Determine if given 'array' is a numpy ndarray of integers.

        :param array: Object to check.
        :return: True, if 'array' is a numpy array with integer elements. False, otherwise.
        """
        return isinstance(array, np.ndarray) and array.dtype in {np.dtype(np.int32), np.dtype(np.int64)}

    def _validate_indices(self, indices: Any, dimension_size: int) -> np.ndarray:
        """Check the 'indices' of ONNX Gather and return corresponding indices validated and prepared for TFLite Gather.

        :param indices: A numpy array representing the 'indices' operand of the ONNX Gather operator.
        :param dimension_size: The size of the dimension, the Gather is applied to.
        :return: The indices for the TFLite Gather operator.
        """
        if not self._is_an_array_of_ints(indices):
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX 'Gather' has invalid indices tensor ('{indices}'). Only arrays of integers are supported.")

        original_shape = indices.shape

        new_indices = []
        for index in indices.flatten():
            if index < 0:
                index += dimension_size
            if not (0 <= index < dimension_size):
                logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX 'Gather' operator has invalid 'indices' tensor!")

            new_indices.append(index)

        new_indices = np.asarray(new_indices, np.int64).reshape(original_shape)

        return new_indices

    def _validate_axis(self, o_gather_attributes: gather_attributes.Gather, input_rank: int) -> int:
        """Make sure the 'axis' is valid for TFLite gather and return it.

        :param o_gather_attributes: Attributes of the ONNX Gather operator.
        :param input_rank: The rank of the input tensor of the Gather operator.
        :return: A valid 'axis' parameter for the TFLite Gather operator.
        """
        axis = o_gather_attributes.axis

        if axis < 0:
            axis += input_rank

        if not (0 <= axis < input_rank):
            # This code is currently unreachable, as the shape inference detects this before conversion.
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f"ONNX 'Gather' has invalid 'axis' attribute! ('{o_gather_attributes.axis}')")

        return axis

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX Gather operator to TFLite.

        :param node: ONNX Gather operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator
        :return: A list of TFLite operators, to add to the model.
        """
        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX 'Gather' has unexpected number of inputs! Got '{len(t_op.tmp_inputs)}', expected '2'.")

        o_gather_attributes = cast(gather_attributes.Gather, node.attributes)

        input_tensor = t_op.tmp_inputs[0]
        input_rank = input_tensor.rank

        output_tensor = t_op.tmp_outputs[0]
        output_rank = output_tensor.rank

        indices_tensor = t_op.tmp_inputs[1]

        self.assert_type_allowed(input_tensor.type)

        ops = OpsList(middle_op=t_op)

        if input_tensor.quantization is not None and output_tensor.quantization is None:
            # Non-QDQ model -> just propagate
            propagate_quantization(input_tensor, output_tensor)
        elif input_tensor.quantization is not None and output_tensor.quantization is not None:
            if input_tensor.quantization != output_tensor.quantization:
                # I/O q-params doesn't match -> external quantizer was used or removed Clip/Relu modified q-params.
                # We need to re-quantize output because Gather expects shared q-params for input and output.
                logger.w("Requantizing output of Gather operator. Internal quantizer can potentially avoid this.")
                scale = input_tensor.quantization.scale.vector
                zp = input_tensor.quantization.zero_point.vector
                ops.add_post(self.builder.create_quantize_operator_after(t_op, 0, input_tensor.type, scale, zp))

        axis = self._validate_axis(o_gather_attributes, input_rank)

        if input_tensor.tensor_format.is_channels_last() and output_tensor.tensor_format.is_channels_last():
            if input_rank == output_rank:
                # The axis refers to a channels first dimension. Update it to refer to a channels last dimension.
                axis = translator.create_channels_last_to_channels_first_permutation(input_rank)[axis]

            else:
                # The only way to convert this correctly is to insert a Transpose operator before, to make the input
                # channels first, and another Transpose after, to make the output channels last again.
                last_to_first_perm = translator.create_channels_last_to_channels_first_permutation(input_rank)
                ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, list(last_to_first_perm)))

                first_to_last_perm = translator.create_channels_first_to_channels_last_permutation(output_rank)
                transpose = self.builder.create_transpose_operator_after(t_op, 0, list(first_to_last_perm))
                ops.post_ops.insert(0, transpose)

        elif input_tensor.tensor_format.is_channels_last() and (not output_tensor.tensor_format.is_channels_last()):
            # The dimensions lose their meaning. Since the output must be the same as in the ONNX model -> insert a
            # Transpose in front to make the input channels first.
            last_to_first_perm = translator.create_channels_last_to_channels_first_permutation(input_rank)
            ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, list(last_to_first_perm)))

        elif (not input_tensor.tensor_format.is_channels_last()) and output_tensor.tensor_format.is_channels_last():
            # Dimensions gain meaning. Since the input is the same in both ONNX and TFLite, the axis must stay the same
            #  and a Transpose operator must be added after the gather to make the output channels last.
            first_to_last_perm = translator.create_channels_first_to_channels_last_permutation(output_rank)
            transpose = self.builder.create_transpose_operator_after(t_op, 0, list(first_to_last_perm))
            ops.post_ops.insert(0, transpose)

        else:
            # Both input and output are formatless -> nothing special needs to be done
            pass

        if not tensor_has_data(indices_tensor):
            # ONNX supports negative indices, but TFLite doesn't.

            if self.context.conversion_config.guarantee_non_negative_indices:
                # User guarantees, that the indices are not negative.
                pass

            else:
                # Conversion may be possible via `Less`, `Where`, `Add` and `Gather` operators.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX 'Gather' operator with dynamic 'indices' tensor is not yet implemented, "
                         "because ONNX may use negative indices, which TFLite doesn't support. " +
                         logger.Message.GUARANTEE_NON_NEGATIVE_INDICES)

        else:
            # Static indices. Make sure they are valid.
            dimension_size = t_op.tmp_inputs[0].shape.get(axis)
            indices_tensor.tmp_buffer.data = self._validate_indices(indices_tensor.tmp_buffer.data, dimension_size)

        t_op.builtin_options = gather_options.Gather(axis)

        return ops.flatten()
