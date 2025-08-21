#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import einsum_attributes
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.custom_options import flex_einsum_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


class EinsumConverter(NodeConverter):
    node = "EinSum"

    onnx_supported_types = FLOATS + INTS + UINTS
    verified_types = [TensorType.FLOAT32, TensorType.FLOAT64, TensorType.INT32, TensorType.INT64]

    # noinspection PyMethodMayBeStatic
    def _validate_equation(self, equation: str) -> str:

        if "..." in equation:
            # ONNX allows this `ellipsis` in the equation. TFLite has no equivalent system. Not sure if conversion is
            #  possible.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     'Conversion of ONNX `Einsum` with ellipsis ("...") in the equation is not yet supported.')

        equation = equation.replace(" ", "")  # ONNX allows spaces, TF doesn't.

        if "->" not in equation:
            # ONNX allows default output shape. Taken from the ONNX doc:
            # "output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly
            #   once in the equation"

            used_dims = [char for char in equation if char != ","]

            # Remove those, which are used more than once.
            used_dims = [char for char in used_dims if used_dims.count(char) == 1]

            equation += "->" + "".join(sorted(used_dims))

        return equation

    def _handle_tensor_formats(self, t_op: tflite_model.Operator, ops: OpsList):
        for index, inpt in enumerate(t_op.tmp_inputs):
            if inpt.tensor_format.is_channels_last():
                # Transpose the input to channels_first to match the ONNX model.
                perm = translator.create_channels_last_to_channels_first_permutation(inpt.shape.len(), True)

                if tensor_has_data(inpt):
                    data = np.transpose(inpt.tmp_buffer.data, perm)
                    new_shape = translator.apply_permutation_to(inpt.shape.vector.copy(), perm)
                    logger.internal_assert(all([s1 == s2 for s1, s2 in zip(data.shape, new_shape, strict=False)]))
                    inpt.tmp_buffer.data = data
                    inpt.tensor_format = TensorFormat.CHANNELS_FIRST

                else:
                    ops.add_pre(self.builder.create_transpose_operator_before(t_op, index, perm))

        output = t_op.tmp_outputs[0]
        if output.tensor_format.is_channels_last():
            perm = translator.create_channels_first_to_channels_last_permutation(output.shape.len(), True)
            ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, perm))

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `Einsum` operator into the SELECT_TF_OP `FlexEinsum`."""
        if not self.context.conversion_config.allow_select_ops:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX `Einsum` without SELECT_TF_OPS is not possible. " +
                     logger.Message.ALLOW_SELECT_OPS)

        num_inputs = len(t_op.tmp_inputs)
        if num_inputs not in {1, 2}:
            # The `FlexEinsum` requires 1 or 2 operands in the equation.
            # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/core/util/einsum_op_util.cc#L42-L46
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"Conversion of ONNX `Einsum` with {num_inputs} inputs is not possible.")

        first_input = t_op.tmp_inputs[0]
        if any(i.type != first_input.type for i in t_op.tmp_inputs):
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX `Einsum` uses a combination of input types.")

        if first_input.type == TensorType.FLOAT16:
            # Seems that this type is not supported by the TensorFlow DataType
            #  (see translator.tflite_type_to_tensor_flow_data_type())
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX `Einsum` with input type FLOAT16 is not possible.")

        self.assert_type_allowed(first_input.type)

        ops = OpsList(middle_op=t_op)

        self._handle_tensor_formats(t_op, ops)

        attrs = cast(einsum_attributes.Einsum, node.attributes)
        equation = self._validate_equation(attrs.equation)

        t_op.custom_options = flex_einsum_options.Einsum(equation, num_inputs, first_input.type)

        return ops.flatten()
