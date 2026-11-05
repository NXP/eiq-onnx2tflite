#
# Copyright 2024,2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import logging

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import errors
from onnx2tflite.src.converter.builder.model_builder import ModelBuilder
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList, try_get_input
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.errors import ConversionError
from onnx2tflite.src.model_inspector import ONNXModelInspector
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import name_for_type

logger = logging.getLogger(__name__)


def convert_axes_from_attribute(t_op: tflite_model.Operator, builder: ModelBuilder, axes: list[int] | None) -> None:
    """Create an `axes` tensor and assign it as an input to the `t_op`, which is expected to represent an ONNX
    reduction operator.
    """
    x = t_op.tmp_inputs[0]
    rank = x.rank

    if axes is None:
        # Default axes -> reduce over all dimensions.
        axes = np.arange(rank).astype(np.int32)

    else:
        # Axes are initialized.
        axes = np.asarray(axes, np.int32)

    # TFLite has `axes` as input tensor -> create it.
    axes_tensor = builder.create_tensor_for_data(axes, "axes")
    t_op.tmp_inputs.append(axes_tensor)


def _axes_are_empty(t_op: tflite_model.Operator, inspector: ONNXModelInspector) -> bool:
    """Determine if the `axes` input (index 1) of `t_op` is omitted, or is a tensor with 0 elements."""
    axes_tensor = try_get_input(t_op, 1)
    if axes_tensor is None:
        return True

    if tensor_has_data(axes_tensor):
        return axes_tensor.tmp_buffer.data.size == 0

    if (axes_data := inspector.try_get_inferred_tensor_data(axes_tensor.name)) is not None:
        return axes_data.size == 0

    # Dynamic `axes` with a known static shape containing a 0.
    return 0 in axes_tensor.shape.vector


def is_noop_reduction(t_op: tflite_model.Operator, noop_with_empty_axes: int, inspector: ONNXModelInspector) -> bool:
    """Determine if the ONNX reduction operator represented by `t_op` does nothing.

    That is the case, when `noop_with_empty_axes` == 1 and the `axes` input (index 1) is either not specified, or it
     is an empty tensor (shape [0]). Both are treated as "empty axes" by ONNX.
    """
    return noop_with_empty_axes == 1 and _axes_are_empty(t_op, inspector)


def convert_noop_reduction(
    t_op: tflite_model.Operator, builder: ModelBuilder, inspector: ONNXModelInspector
) -> list[tflite_model.Operator]:
    """Convert an ONNX reduction operator, which does nothing, to a no-op TFLite operator.

    The operator is skipped if possible, or turned into an identity op. If the input and output have different
     quantization parameters (QDQ model), the builder turns it into a `Quantize` op to re-quantize the data.
    Use `is_noop_reduction()` to detect this case.
    """
    t_op.tmp_inputs[1:] = []  # Remove the (optional or empty) `axes` input.
    x = t_op.tmp_inputs[0]
    y = t_op.tmp_outputs[0]

    if builder.operator_can_be_skipped(t_op, inspector):
        builder.redirect_tensor(y, x)
        return []

    builder.turn_operator_to_identity(t_op)
    return [t_op]


def convert_axes_from_input_tensor(
    t_op: tflite_model.Operator,
    builder: ModelBuilder,
    inspector: ONNXModelInspector,
    ops: OpsList,
    op_type: str,
) -> None:
    """Verify the `axes` tensor (on input index 1) of the `t_op`, which is expected to represent an ONNX reduction
    operator.

    The case `noop_with_empty_axes` == 1 without `axes` must be resolved by the caller beforehand, using
     `is_noop_reduction()`.
    """
    x = t_op.tmp_inputs[0]
    rank = x.rank

    if axes_tensor := try_get_input(t_op, 1):
        # ONNX uses int64, while TFLite requires int32 for the `axes` tensor.
        if axes_tensor.type != TensorType.INT64:
            raise ConversionError(errors.Code.INVALID_ONNX_OPERATOR, f"ONNX `{op_type}` has `axes` of type `{name_for_type(axes_tensor.type)}`, instead of INT64.")

        # Try to get the inferred data for the `axes` input.
        if (axes_data := inspector.try_get_inferred_tensor_data(axes_tensor.name)) is not None:
            # The `axes` were inferred during shape inference.
            logger.debug(f"Using inferred data for the `axes` input tensor of ONNX `{op_type}`.")

            # Create a new tensor, in case the original `axes` tensor is used by multiple ops.
            axes_tensor = builder.create_tensor_for_data(axes_data.astype(np.int32), "axes")

        # Make sure the `axes` are int32.
        if tensor_has_data(axes_tensor):
            # Clone tensor if it has multiple consumers
            if inspector.get_number_of_onnx_consumers_safe(axes_tensor.name) != 1:
                # noinspection PyTypeChecker
                axes_tensor = builder.create_tensor_for_data(axes_tensor.tmp_buffer.data, "axes")

            # Cast the `axes` to int32 statically.
            axes_tensor.tmp_buffer.data = axes_tensor.tmp_buffer.data.astype(np.int32)
            axes_tensor.type = TensorType.INT32

        else:
            # The `axes` are dynamic and there is no inferred data for them. The shape inference is not possible in
            #  this case, so it must have been skipped. If the `axes` are empty at runtime, ONNX will reduce over
            #  all dimensions, whereas TFLite will not reduce at all. So the behavior is different, and it depends
            #  on runtime data. Conversion could be implemented by adding multiple extra operators.
            # I don't think that completely prohibiting the conversion here is ideal, since the issue arises only in
            #  an edge case, which is hopefully not very common. Just print a warning message for now.
            logger.warning(
                f"Conversion of ONNX `{op_type}` with a dynamic `axes` input will not be correct, if the `axes`"
                "are empty at runtime!"
            )

            # Insert a `Cast` op, to make the `axes` int32.
            cast_op = builder.create_cast_before(t_op, 1, TensorType.INT32)
            ops.add_pre(cast_op)

            # For future references. Following code only cares about the final axes tensor.
            axes_tensor = cast_op.tmp_outputs[0]

        # Assign the new `axes_tensor` to the ReduceX operator.
        t_op.tmp_inputs[1] = axes_tensor

    # No axes specified.

    else:
        # Default is to reduce all axes.
        axes_tensor = builder.create_tensor_for_data(np.arange(rank).astype(np.int32), "axes")

        t_op.tmp_inputs[1:] = []  # If the optional input was passed with name "", remove it.
        t_op.tmp_inputs.append(axes_tensor)


def ensure_reduce_transposition(builder, ops: OpsList) -> None:
    """Ensure transposition of ReduceX operator is defined correctly based on tensor format.
    New operators (Transpose) are added into "ops" collection when necessary.

    :param builder: ModelBuilder instance.
    :param ops: OpsList instance with operators related to currently converted ReduceX operator.
    """
    t_op = ops.middle_op
    input_tensor = t_op.tmp_inputs[0]
    input_rank = input_tensor.rank
    input_format = input_tensor.tensor_format
    output_tensor = t_op.tmp_outputs[0]
    output_rank = output_tensor.rank
    output_format = output_tensor.tensor_format

    if input_format.is_channels_last() and output_format.is_channels_last():
        to_onnx_perm = translator.create_channels_last_to_channels_first_permutation(input_rank)
        to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(output_rank, return_list=True)

        transpose_before = builder.create_transpose_operator_before(t_op, 0, to_onnx_perm)
        transpose_before.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST
        ops.add_pre(transpose_before)

        transpose_after = builder.create_transpose_operator_after(t_op, 0, to_tflite_perm)
        transpose_after.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST
        ops.post_ops.insert(0, transpose_after)

    elif input_format.is_channels_last() and not output_format.is_channels_last():
        # The dimensions of the tensor lose their meaning! Insert a transpose op, to change input to match ONNX.

        permutation = list(translator.create_channels_last_to_channels_first_permutation(input_rank))
        transpose = builder.create_transpose_operator_before(t_op, 0, permutation)
        transpose.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        ops.add_pre(transpose)

    elif not input_format.is_channels_last() and output_format.is_channels_last():
        # The ReduceX introduces format to the tensor
        # The ONNX ReduceX outputs a 'channels first' tensor. This has to stay the same, and then a Transpose operator
        # must be added, to change the tensor to 'channels last'.

        permutation = list(translator.create_channels_first_to_channels_last_permutation(output_rank))
        transpose = builder.create_transpose_operator_after(t_op, 0, permutation)
        transpose.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        ops.post_ops.insert(0, transpose)
