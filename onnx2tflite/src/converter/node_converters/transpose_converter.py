#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from typing import cast

import numpy as np

import onnx2tflite.src.tflite_generator.builtin_options.transpose_options as tfl_transpose_options
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import transpose_attributes
from onnx2tflite.src.tflite_generator.custom_options.flex_transpose_options import FlexTranspose
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, INTS


class TransposeConverter(NodeConverter):
    node = 'Transpose'

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/transpose.cc#L147-L230
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8, TensorType.BOOL]
    verified_types = INTS + [TensorType.FLOAT32, TensorType.UINT8, TensorType.BOOL]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX 'Transpose' to TFLite 'Transpose'."""

        x = t_op.tmp_inputs[0]

        self.assert_type_allowed(x.type)

        input_format = x.tensor_format
        input_shape = x.shape.vector
        input_rank = len(input_shape)

        y = t_op.tmp_outputs[0]
        output_format = y.tensor_format
        output_shape = y.shape.vector

        if input_rank <= 6:
            t_op.builtin_options = tfl_transpose_options.Transpose()
        else:
            # Must use the `FlexTranspose`.

            if x.quantization is not None:
                # TFLite inference crashes badly. Couldn't figure out why. (Fatal Python error: Aborted)
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE, 'Conversion of ONNX `Transpose` with > 5 dimensions and '
                                                            'quantized data is not supported.')

            if not self.context.conversion_config.allow_select_ops:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         f'Conversion of ONNX `Transpose` with {input_rank} dimensions without Flex delegate is not '
                         'possible. ' + logger.Message.ALLOW_SELECT_OPS)

            t_op.custom_options = FlexTranspose()

        ops = OpsList(middle_op=t_op)

        if x.quantization is not None and y.quantization is None:
            # Non-QDQ model -> just propagate
            propagate_quantization(x, y)
        elif x.quantization is not None and y.quantization is not None:
            if x.quantization != y.quantization:
                # I/O q-params doesn't match -> external quantizer was used or removed Clip/Relu modified q-params.
                # We need to re-quantize output because Reshape expects shared q-params for input and output.
                logger.w("Requantizing output of Transpose operator. Internal quantizer can potentially avoid this.")
                scale = x.quantization.scale.vector
                zp = x.quantization.zero_point.vector
                ops.add_post(self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp))

        transpose_attrs = cast(transpose_attributes.Transpose, node.attributes)

        if (transpose_attrs.perm is None) or (len(transpose_attrs.perm) == 0):
            # By default, ONNX reverses the order of the dimensions
            perm = list(reversed(range(input_rank)))

        else:
            perm = transpose_attrs.perm.copy()
            if len(perm) != input_rank:
                logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                         "ONNX Transpose attribute 'perm' does not match the "
                         f"input rank! ('{len(perm)}' != '{input_rank}')")

        if input_format.is_channels_last() and (not output_format.is_channels_last()):
            # Dimensions lose their meaning -> the output must be the same in ONNX and TFLite, but input is not the
            #  same. Change the input to channels_first ( TFLite to ONNX perm + original perm )
            last_to_first_perm = translator.create_channels_last_to_channels_first_permutation(input_rank)
            perm = translator.combine_permutations(last_to_first_perm, perm)

        elif input_format.is_channels_last() and output_format.is_channels_last():
            # Input and output both have the channels last format. The permutation from the ONNX model expects channels
            # first. Modify the permutation ( TFLite to ONNX perm + original perm + ONNX to TFLite perm )
            last_to_first_perm = translator.create_channels_last_to_channels_first_permutation(input_rank)
            first_to_last_perm = translator.create_channels_first_to_channels_last_permutation(input_rank)
            perm = translator.combine_permutations(last_to_first_perm, perm)
            perm = translator.combine_permutations(perm, first_to_last_perm)

        elif (not input_format.is_channels_last()) and output_format.is_channels_last():
            # The dimensions gain meaning -> the output must be channels_last. The permutation creates a channels_first
            # tensor. Modify the permutation ( original perm + ONNX to TFLite perm )
            first_to_last_perm = translator.create_channels_first_to_channels_last_permutation(input_rank)
            perm = translator.combine_permutations(perm, first_to_last_perm)

        else:
            # Input and output are both format-less. Nothing needs to be done.
            pass

        # Verify that the permutation will create the expected output shape
        expected_output_shape = translator.apply_permutation_to(input_shape, perm)
        if expected_output_shape != output_shape:
            logger.w(
                f"convert_transpose: The Transpose operator with input shape '{input_shape}' and permutation '{perm}'"
                f" should produce output with shape '{output_shape}', but will produce '{expected_output_shape}'!")

        # Create the 'perm' tensor
        perm = np.asarray(perm, np.int32)
        perm_tensor = self.builder.create_tensor_for_data(perm, "perm_")

        t_op.tmp_inputs.append(perm_tensor)

        return ops.flatten()
