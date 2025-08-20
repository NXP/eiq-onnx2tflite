#
# Copyright 2023 Martin Pavella
# Copyright 2023-2025 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""translator

Module contains functions for context-free conversion of various
things from ONNX to TFLite.
"""

import math
from collections.abc import Collection, Sequence
from typing import Any

# TODO since Python 3.12 we can replace the deprecated package with typing.deprecated: https://peps.python.org/pep-0702/
import deprecated
import numpy as np
import onnx

import onnx2tflite.lib.tflite.Padding as tflPadding
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser import onnx_tensor
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import TensorFlowDataType


def permute_static_tensor(tensor: tflite_model.Tensor, perm: list[int]) -> None:
    """Take a static TFLite tensor and permute its shape and data according to the permutation in 'perm'.

    :param tensor: Static TFLite tensor to permute.
    :param perm: Permutation to apply to the tensor.
    """
    logger.internal_assert(tensor.tmp_buffer is not None, "permute_static_tensor: tensor is not static.")

    data = tensor.tmp_buffer.data
    data = np.transpose(data, perm)

    shape = apply_permutation_to(tensor.shape.vector.copy(), perm)
    logger.internal_assert(shape == list(data.shape), "permute_static_tensor: shapes do not match.")

    tensor.tmp_buffer.data = data
    tensor.shape = tflite_model.Shape(shape)


def get_tflite_tensor_shape_with_explicit_padding(tflite_shape: list[int], explicit_padding: list[list[int]]
                                                  ) -> list[int]:
    """Get the resulting shape of a tensor with shape 'tflite_shape' (in TFLite format), after 'explicit_padding' is
    applied to it.
    """
    if (len(tflite_shape) != len(explicit_padding)) or any([len(sub_list) != 2 for sub_list in explicit_padding]):
        logger.e(logger.Code.INTERNAL_ERROR,
                 f"Cannot apply padding '{explicit_padding}' to TFLite shape '{tflite_shape}'!")

    total_padding = [start + end for start, end in explicit_padding]  # Total padding for each dimension

    padded_shape = []
    for dimension, padding in zip(tflite_shape, total_padding, strict=False):
        if isinstance(dimension, int) and dimension > 0:
            padded_shape.append(dimension + padding)

        # Variable shape

        elif padding == 0:
            padded_shape.append(dimension)

        else:
            # Cannot add padding to a variable dimension.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Adding explicit padding to a variable sized tensor is not supported!")

    return padded_shape


def convert_tensor_format_to_tflite(tensor_format: TensorFormat) -> TensorFormat:
    """Convert the format of a tensor from ONNX to TFLite.
    :return: The tensor_format converted to TFLite.
    """
    if tensor_format is TensorFormat.CHANNELS_FIRST:
        return TensorFormat.CHANNELS_LAST

    if tensor_format not in (TensorFormat.FORMATLESS, TensorFormat.NONE):
        logger.d(f"translator.convert_tensor_format(): Got unexpected format '{tensor_format}'.")

    return tensor_format


def dims_to_channels_first(channels_last_dimensions: list[int]) -> list[int]:
    """Convert a list of ints which represent dimensions in the channels last (TFLite) format to the channels first
    (ONNX) format.
    """
    assert len(channels_last_dimensions) > 0, "Dimensions list is empty!"

    if len(channels_last_dimensions) == 1:
        return [0]

    res = list(channels_last_dimensions)

    res.insert(1, res.pop())  # Insert 'C' (last item) to index 1

    return res


def dims_to_channels_last(channels_first_dimensions: list[int]) -> list[int]:
    """Convert a list of ints which represent dimensions in the channels first (ONNX) format to the channels last
    (TFLite) format.
    """
    assert len(channels_first_dimensions) > 0, "Dimensions list is empty!"

    if len(channels_first_dimensions) == 1:
        return [0]

    res = list(channels_first_dimensions)

    res.append(res.pop(1))  # Move 'C' (idx 1) to the end

    return res


def collections_equal(col_a, col_b) -> bool:
    """Compare each individual element of both collections.
    They can be any combination of lists, tuples or numpy arrays.
    Return True if they are equal.
    """
    if len(col_a) != len(col_b):
        return False

    for (a, b) in zip(col_a, col_b, strict=False):
        if a != b:
            return False
    return True


@deprecated.deprecated(reason="This implementation of total required padding is deprecated, "
                              "use _tflite_compute_padding_with_offset")
def _calculate_total_required_padding(tflite_input_shape: list[int], tflite_output_shape: list[int],
                                      effective_kernel_shape: list[int], strides: list[int]) -> list[int]:
    """Calculate the total padding required for each dimension for a particular setting.
    :param tflite_input_shape: The TFLite (channels last) shape of the input tensor.
    :param tflite_output_shape: The TFLite (channels last) shape of the output tensor.
    :param effective_kernel_shape: The shape of the kernel including dilations.
    :param strides: The strides of the operation.

    :return: A list of ints representing the total required padding for each dimension for an operation with given
             inputs, outputs and attributes.
    """
    rank = len(effective_kernel_shape)
    if strides is None:
        strides = [1] * rank

    spatial_input_shape = tflite_input_shape[1:-1]  # The HW portion of the input
    spatial_output_shape = tflite_output_shape[1:-1]  # The HW portion of the output

    total_padding = [0] * rank

    for i in range(rank):
        total_padding[i] = (
                (spatial_output_shape[i] - 1) * strides[i]
                + effective_kernel_shape[i]
                - spatial_input_shape[i]
        )

    return total_padding


def _calculate_effective_kernel_shape(kernel_shape: list[int], dilations: list[int] | None) -> list[int]:
    """Calculate the reach of a kernel with respect to its shape and dilations.
    For example a [3, 3] kernel with dilations [2, 2] has effective shape of [5, 5].
    """
    if dilations is None:
        dilations = [1] * len(kernel_shape)

    return [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations, strict=False)]


def _same_upper_equals_same_lower(tflite_input_shape: list[int], tflite_output_shape: list[int],
                                  o_kernel_shape: list[int],
                                  o_strides: list[int] | None = None,
                                  o_dilations: list[int] | None = None) -> bool:
    """Determine if in a given particular setting, the values of the ONNX `auto_pads` attribute SAME_UPPER and
    SAME_LOWER represent the exact same padding.
    """
    padding, offset = tflite_compute_padding_with_offset(tflite_input_shape, o_kernel_shape, tflite_output_shape,
                                                         o_strides, o_dilations)

    # Only if offset for every dimension is 0, SAME_UPPER and SAME_LOWER will behave equally.
    return all([elt == 0 for elt in offset])


def _tflite_padding_compute_output_size(padding: tflPadding.Padding,
                                        tflite_spatial_input_shape: list[int],
                                        tflite_kernel_shape: list[int],
                                        strides: list[int] | None = None,
                                        dilations: list[int] | None = None) -> list[int]:
    """Calculates the output shape of the tensor with particular setting as tflite would. Implementation corresponds to
    tensorflow/lite/kernels/padding.h:ComputeOutSize()
    :param padding: TFLite Padding value - 'Same' or 'Valid'
    :param tflite_spatial_input_shape: input tensor shape
    :param tflite_kernel_shape: convolution kernel shape
    :param strides: strides (default is 1)
    :param dilations: dilation (default is 1)
    :return: Output shape of the tensor with particular padding settings
    """
    if strides is None:
        strides = [1] * len(tflite_kernel_shape)

    effective_kernel_shape = _calculate_effective_kernel_shape(tflite_kernel_shape, dilations)

    if padding == tflPadding.Padding.SAME:
        return [(in_shape + stride - 1) // stride for in_shape, stride in zip(tflite_spatial_input_shape, strides, strict=False)]
    if padding == tflPadding.Padding.VALID:
        return [(in_shape + stride - ef_kernel_shape) // stride
                for in_shape, stride, ef_kernel_shape in
                zip(tflite_spatial_input_shape, strides, effective_kernel_shape, strict=False)]


def tflite_compute_padding_with_offset(tflite_input_shape: list[int],
                                       tflite_kernel_shape: list[int],
                                       tflite_output_shape: list[int],
                                       strides: list[int] | None = None,
                                       dilations: list[int] | None = None) -> (list[int], list[int]):
    """Calculate padding and offset for each dimension for particular convolution setting as TFLite.
    Implementation corresponds to tensorflow/lite/kernels/padding.h:ComputePaddingWithOffset()
    :param tflite_input_shape: tensorflow lite input shape
    :param tflite_kernel_shape: tensorflow lite kernel shape
    :param tflite_output_shape: tensorflow lite output shape
    :param strides: stride setting, default is 1
    :param dilations: dilation setting, default is 1
    :return: (padding, offset) - padding and offset for each axis. Padding is added on beginning and end of the axis.
             Offset to be optionally added to end of the axis if odd.
    """
    if strides is None:
        strides = [1] * len(tflite_kernel_shape)

    spatial_input_shape = tflite_input_shape[1:-1]  # The spatial portion of the input
    spatial_output_shape = tflite_output_shape[1:-1]  # The spatial portion of the output

    effective_kernel_shape = _calculate_effective_kernel_shape(tflite_kernel_shape, dilations)

    total_padding = [(spatial_output - 1) * stride + effective_kernel - spatial_input
                     for spatial_output, stride, effective_kernel, spatial_input
                     in zip(spatial_output_shape, strides, effective_kernel_shape, spatial_input_shape, strict=False)]

    padding = [tp // 2 for tp in total_padding]
    offset = [tp % 2 for tp in total_padding]

    return padding, offset


def _is_same_padding(o_pads: list[int], tflite_input_shape: list[int], tflite_output_shape: list[int],
                     o_kernel_shape: list[int],
                     o_strides: list[int] | None = None, o_dilations: list[int] | None = None) -> bool:
    """Determine if given ONNX 'pads' padding can be represented exactly with the TFLite 'SAME' padding type.

    :param o_pads: ONNX 'pads' attribute.
    :param tflite_input_shape: The shape of the main input of the operator in TFLite format.
    :param tflite_output_shape: The shape of the main output of the operator in TFLite format.
    :param o_kernel_shape: ONNX 'kernel_shape' attribute.
    :param o_strides: ONNX 'strides' attribute. Can be omitted.
    :param o_dilations: ONNX 'dilations' attribute. Can be omitted.
    """
    if len(tflite_input_shape) == 0 or len(tflite_output_shape) == 0:
        logger.e(logger.Code.INVALID_TENSOR_SHAPE,
                 f"Cannot verify that padding '{o_pads}' can be represented as 'SAME' for input shape "
                 f"'{tflite_input_shape}' and output shape '{tflite_output_shape}'.")

    # Calculate if the output shape corresponds to Same padding setting in TFLite
    tflite_spatial_input_shape = tflite_input_shape[1:-1]
    tmp_spatial_output_shape = _tflite_padding_compute_output_size(tflPadding.Padding.SAME, tflite_spatial_input_shape,
                                                                   o_kernel_shape, o_strides, o_dilations)
    if tmp_spatial_output_shape != tflite_output_shape[1:-1]:
        return False

    # For every dimension, the padding is added to the start and end of the dimension.
    # TFLite padding 'SAME' tries to split it evenly, but in case of odd padding, 'SAME' adds the excess 1 at the end.
    # TFLite represents this in the offset. The offset is added to the end of particular dimension,
    # i.e. bottom for H dim, right for W dim and so on.
    # ONNX represents this in 'pads' as [x1_begin, x2_begin,... , x1_end, x2_end,...].
    padding, offset = tflite_compute_padding_with_offset(tflite_input_shape, o_kernel_shape, tflite_output_shape,
                                                         o_strides, o_dilations)
    start_padding = padding
    end_padding = [p + o for p, o in zip(padding, offset, strict=False)]
    effective_padding = start_padding + end_padding

    if effective_padding != o_pads:
        return False

    return True


def permutations_are_inverse(permutation1: Sequence[int], permutation2: Sequence[int]) -> bool:
    """Determine if given Transpose permutations are inverse of each other.
    i.e. when applied back to back, there will be no effect.

    Example:
      0 3 1 2
      0 2 3 1

    """
    if len(permutation1) != len(permutation2):
        logger.e(logger.Code.INTERNAL_ERROR, "translator.permutations_are_inverse(): permutations have different size!")

    for i, perm2 in enumerate(permutation2):
        if i != permutation1[perm2]:
            return False

    return True


def combine_permutations(permutation1: Sequence[int], permutation2: Sequence[int]) -> list[int]:
    """Combine 2 permutations into 1.

    :param permutation1: The first permutation to apply.
    :param permutation2:  The second permutation to apply.
    :return: The combined permutation.
    """
    if len(permutation1) != len(permutation2):
        logger.e(logger.Code.INTERNAL_ERROR, "translator.combine_permutations(): permutations have different size!")

    return [permutation1[perm2] for perm2 in permutation2]


def nhc_dimensions_to_nhwc(nhc_dimensions: list[int]) -> list[int]:
    """Convert a list of ints representing the shape of an NHC tensor to NHWC, where W = 1."""
    nhwc_dimensions = nhc_dimensions.copy()
    nhwc_dimensions.insert(2, 1)

    return nhwc_dimensions


def shape_from_numpy(numpy_array) -> tflite_model.Shape:
    """Return a 'Shape' object representing the shape of given 'numpy_array'."""
    dims = list(numpy_array.shape)
    return tflite_model.Shape(dims)


def onnx_explicit_padding_to_tflite(onnx_pads: list[int]) -> list[list[int]]:
    """Convert the attribute or input 'pads' of the ONNX 'Pad' operator to the 'paddings' input of the TFLite 'Pad'
     class of operators.

    This function does NOT take tensor formats into consideration.
    """
    start_padding = onnx_pads[:len(onnx_pads) // 2]  # Padding at the start of each dimension
    end_padding = onnx_pads[len(onnx_pads) // 2:]  # Padding at the end of each dimension

    return list(zip(start_padding, end_padding, strict=False))


def onnx_pads_to_tflite_explicit_padding(onnx_pads: list[int]) -> list[list[int]]:
    """Convert an ONNX attribute 'pads' of operators such as Conv, MaxPool or AveragePool, to a list of ints which is
    compatible with the TFLite 'Pad' operator.
    """
    tflite_padding = onnx_explicit_padding_to_tflite(onnx_pads)

    # TFLite also allows padding to the 'batch' and 'channels'. ONNX does not
    tflite_padding.insert(0, [0, 0])
    tflite_padding.append([0, 0])

    return tflite_padding


def _get_explicit_tflite_padding_for_same_lower(tflite_input_shape: list[int], tflite_output_shape: list[int],
                                                o_kernel_shape: list[int],
                                                o_strides: list[int] | None = None,
                                                o_dilations: list[int] | None = None) -> list[list[int]]:
    """Get the TFLite explicit padding required to represent ONNX 'SAME_LOWER' auto_pad for a particular setting.

    :param tflite_input_shape: TFLite (NHWC) shape of the input tensor of the operator.
    :param tflite_output_shape: TFLite (NHWC) shape of the output tensor of the operator.
    :param o_kernel_shape: ONNX 'kernel_shape' attribute.
    :param o_strides: Optional ONNX 'o_strides' attribute.
    :param o_dilations: Optional ONNX 'o_dilations' attribute.

    :return: A TFLite style explicit padding, compatible with the TFLite 'Pad' operator.
    """
    padding, offset = tflite_compute_padding_with_offset(tflite_input_shape, o_kernel_shape, tflite_output_shape,
                                                         o_strides, o_dilations)

    start_padding = [p + o for p, o in zip(padding, offset, strict=False)]  # In case of odd padding, the excess is added at the start
    end_padding = padding

    onnx_explicit_padding = start_padding + end_padding

    # Return explicit ONNX padding converted to TFLite padding
    return onnx_pads_to_tflite_explicit_padding(onnx_explicit_padding)


def convert_padding(o_auto_pad: str, o_pads: list[int],
                    tflite_input_shape: list[int], tflite_output_shape: list[int],
                    o_kernel_shape: list[int], o_strides: list[int] | None,
                    o_dilations: list[int] | None = None,
                    ) -> tuple[tflPadding.Padding, list[list[int]] | None]:
    """Convert ONNX operator attributes 'pads' and 'auto_pad' to TFLite.

    :param o_auto_pad: ONNX operator attribute 'auto_pad'
    :param o_pads: ONNX operator attribute 'pads'
    :param tflite_input_shape: The shape of the main input tensor in the TFLite format.
    :param tflite_output_shape: The shape of the main output tensor in the TFLite format.
    :param o_kernel_shape: ONNX operator attribute 'kernel_shape'
    :param o_strides: ONNX operator attribute 'strides'
    :param o_dilations: ONNX operator attribute 'dilations'

    :return: A tuple.
                The first element is the converted TFLite padding.
                The second is None, if conversion is finished. Or it is a list of ints representing the explicit
                padding in TFLite format (compatible with the 'Pad' operator), which needs to be provided by a
                'Pad' operator. Caller must add this operator using model_builder!
    """
    if o_auto_pad == "SAME_UPPER":
        return tflPadding.Padding.SAME, None

    if o_auto_pad == "SAME_LOWER":
        if _same_upper_equals_same_lower(tflite_input_shape, tflite_output_shape, o_kernel_shape, o_strides,
                                         o_dilations):
            return tflPadding.Padding.SAME, None

        logger.d("'SAME_LOWER' auto_pad cannot be exactly represented in TFLite as padding 'SAME' or 'VALID'. "
                 "Inserting an extra 'Pad' operator.")
        tflite_explicit_padding = _get_explicit_tflite_padding_for_same_lower(tflite_input_shape,
                                                                              tflite_output_shape,
                                                                              o_kernel_shape, o_strides,
                                                                              o_dilations)
        return tflPadding.Padding.VALID, tflite_explicit_padding

    if o_auto_pad == "VALID":
        return tflPadding.Padding.VALID, None

    # auto_pad is NOTSET -> use explicit padding
    if o_pads is None or all(val == 0 for val in o_pads):
        # No padding in any direction
        return tflPadding.Padding.VALID, None

    if _is_same_padding(o_pads, tflite_input_shape, tflite_output_shape, o_kernel_shape, o_strides, o_dilations):
        # Explicit padding can be represented with TFLite 'SAME' padding.
        return tflPadding.Padding.SAME, None

    # 'pads' cannot be converted directly. Return 'VALID' and the required explicit padding and caller must
    # implement conversion by adding a 'Pad' operator.

    logger.d("Explicit ONNX 'pads' cannot be represented directly as 'SAME' or 'VALID'. "
             "Inserting an extra 'Pad' operator.")

    # ONNX 'pads' uses different format than TFLite 'Pad' operator. Convert the explicit padding.
    tflite_explicit_padding = onnx_pads_to_tflite_explicit_padding(o_pads)

    return tflPadding.Padding.VALID, tflite_explicit_padding


def convert_tensor_data(o_tensor: onnx_tensor.TensorProto) -> np.ndarray:
    """Convert the data of a tensor from the 'NCHW' to 'NHWC' format."""
    assert np.all(np.equal(o_tensor.dims, list(o_tensor.data.shape)))

    if not o_tensor.tensor_format.is_channels_first():
        # 'data' does not need to be converted
        return o_tensor.data.copy()  # TODO Copy might be slow and not necessary

    # Product of all dimensions multiplied together (total size)
    size = math.prod(o_tensor.dims)
    if size != len(o_tensor.data.flatten()):
        logger.e(logger.Code.INVALID_TENSOR_SHAPE, f"Numpy array for tensor of shape '{o_tensor.dims}' should have "
                                                   f"'{size}' elements, but has '{len(o_tensor.data)}'!")

    # Assign 'data' its current shape
    o_tensor.data.shape = o_tensor.dims

    # "Move" the channels (index 1) to the end
    data = np.moveaxis(o_tensor.data, 1, -1)

    # Check if it worked
    nhwc_shape = dims_to_channels_last(o_tensor.dims)
    if not collections_equal(data.shape, nhwc_shape):
        logger.e(logger.Code.INVALID_TENSOR_SHAPE,
                 f"Failed to convert data from shape '{o_tensor.dims}'! Got '{data.shape}', expected '{nhwc_shape}'.")

    return data


def convert_data_to_channels_first(array: np.ndarray) -> np.ndarray:
    """Convert a numpy array representing the data of a tensor from the channels last format (TFLite), to channels
        first format (ONNX).

    :param array: Numpy array holding the tensor's data.
    :return: The transformed data.
    """
    if len(array.shape) < 3:
        logger.e(logger.Code.INTERNAL_ERROR,
                 f"translator.convert_data_to_channels_first(): 'array' only has '{len(array.shape)}' dimensions!")

    return np.moveaxis(array, -1, 1)  # Move last axis (C), to index 1


def convert_data_to_channels_last(array: np.ndarray) -> np.ndarray:
    """Convert a numpy array representing the data of a tensor from the channels first format (ONNX), to channels last
        format (TFLite).

    :param array: Numpy array holding the tensor's data.
    :return: The transformed data.
    """
    if len(array.shape) < 3:
        logger.e(logger.Code.INTERNAL_ERROR,
                 f"translator.convert_data_to_channels_last(): 'array' only has '{len(array.shape)}' dimensions!")

    return np.moveaxis(array, 1, -1)  # Move the second axis (C), to the end


def convert_shape(o_shape: onnx_tensor.TensorShapeProto, is_channels_first: bool) -> tflite_model.Shape:
    """Convert ONNX 'TensorShapeProto', to TFLite 'Shape'.
    :param o_shape: ONNX TensorShapeProto object, to be converted.
    :param is_channels_first: Indicates if the 'o_shape' is in the channels first format.
    """
    dims = [dim.value for dim in o_shape.dims]

    if (not is_channels_first) or (len(dims) == 0):
        return tflite_model.Shape(dims)

    return convert_onnx_dimensions_to_tflite_shape(dims)


def channels_first_shape_to_channels_last(channels_first_shape: tflite_model.Shape) -> tflite_model.Shape:
    """Create a channels last version of a channels first 'tflite_model.Shape' object."""
    dims = channels_first_shape.vector.copy()
    dims = dims_to_channels_last(dims)

    return tflite_model.Shape(dims)


def channels_last_shape_to_channels_first(nhwc_shape: tflite_model.Shape) -> tflite_model.Shape:
    """Create a channels first version of a channels last 'tflite_model.Shape' object."""
    dims = nhwc_shape.vector.copy()
    dims = dims_to_channels_first(dims)

    return tflite_model.Shape(dims)


def convert_onnx_dimensions_to_tflite_shape(o_dims: list[int]) -> tflite_model.Shape:
    """Convert list of ints representing the shape of an ONNX channels first Tensor to a TFLite 'Shape' object."""
    dims = [dim for dim in o_dims]  # Copy just in case

    dims = dims_to_channels_last(dims)

    return tflite_model.Shape(dims)


def create_channels_last_to_channels_first_permutation(rank: int, return_list: bool = False) -> np.ndarray | list[int]:
    """Return a numpy array with data that describes the permutation, which would change a tensor from the channels
    last (TFLite) format to the channels first (ONNX) format.

    This permutation is compatible with the TFLite `Transpose` operator.

    :param rank: The rank of the required permutation.
    :param return_list: If True, the function returns a list of ints. If False, a numpy array is returned.
    :return: A numpy array, or a list of ints, representing the desired permutation.
    """
    perm = dims_to_channels_first(list(range(rank)))

    if return_list:
        return perm
    return np.asarray(perm, np.int32)


def create_channels_first_to_channels_last_permutation(rank: int, return_list: bool = False) -> np.ndarray | list[int]:
    """Return a numpy array with data that describes the permutation, which would change a tensor from the channels
    first (ONNX) format to the channels last (TFLite) format.

    This permutation is compatible with the TFLite `Transpose` operator.

    :param rank: The rank of the required permutation.
    :param return_list: If True, the function returns a list of ints. If False, a numpy array is returned.
    :return: A numpy array, or a list of ints, representing the desired permutation.
    """
    perm = dims_to_channels_last(list(range(rank)))

    if return_list:
        return perm
    return np.asarray(perm, np.int32)


def create_axis_to_last_perm(axis, num_dims) -> np.ndarray:
    """Create a numpy array representing the transpose permutations needed, to
    make the 'axis' dimension, the last dimension.
    """
    dims = list(range(num_dims))

    if axis == num_dims - 1:
        return dims
    if axis >= num_dims or axis < 0:
        logger.e(logger.Code.INTERNAL_ERROR,
                 f"translator.create_axis_to_last_perm({axis},{num_dims}). Inputs don't make sense!")

    # Remember axis dimension
    axis_dim = dims[axis]

    # Move dimensions after 'axis' to the left
    dims[axis:-1] = dims[axis + 1:-1]

    # Add axis dimension to the end
    dims.append(axis_dim)

    return np.asarray(dims, np.int32)


def apply_permutation_to(target: list[Any], permutation: Collection[int]) -> list:
    """Permute a list according to a permutation. Uses the same permutation format as the TFLite Transpose operator.

    :param target: A list of any types, to permute. Must be same size as the permutation.
    :param permutation: The permutation to apply to the target.
    :return: Permuted list.
    """
    if len(target) != len(permutation):
        logger.e(logger.Code.INTERNAL_ERROR,
                 "translator.apply_permutation_to(): 'target' and 'permutation' have different length!")

    return [target[perm] for perm in permutation]


def create_inverse_permutation(permutation: list[int]) -> list[int]:
    """Create and return a permutation, that is the inverse of the given 'permutation' parameter.
        Uses the same permutation format as the TFLite Transpose operator.

    :param permutation: The permutation to create the inverse of.
    :return: Inverse permutation.
    """
    if set(permutation) != set(range(len(permutation))):
        # Irreversible permutation. For example [0, 1, 2, 2] (information is lost by applying permutation).
        logger.e(logger.Code.INTERNAL_ERROR, "translator.create_inverse_permutation(): permutation is not reversible!")

    return [permutation.index(perm) for perm in range(len(permutation))]


def get_max_value_for_type(dtype: np.dtype) -> any:
    """Return the maximum possible value for given numpy type."""
    if dtype.kind in ("i", "u"):
        return np.iinfo(dtype).max

    if dtype.kind == "f":
        return np.finfo(dtype).max

    logger.e(logger.Code.INTERNAL_ERROR, f"translator.get_max_value_for_type(): unexpected type {dtype.name}.")


def get_min_value_for_type(dtype: np.dtype) -> any:
    """Return the minimum possible value for given numpy type."""
    if dtype.kind in ("i", "u"):
        return np.iinfo(dtype).min

    if dtype.kind == "f":
        return np.finfo(dtype).min

    logger.e(logger.Code.INTERNAL_ERROR, f"translator.get_min_value_for_type(): unexpected type {dtype.name}.")


def convert_data_type(o_type: onnx.TensorProto.DataType) -> TensorType | int:
    """Convert ONNX DataType to TFLite TensorType"""
    if o_type == onnx.TensorProto.UNDEFINED:
        logger.e(logger.Code.CONVERSION_IMPOSSIBLE, "Cannot convert ONNX DataType 'UNDEFINED' to TFLite.")

    elif o_type == onnx.TensorProto.FLOAT:
        return TensorType.FLOAT32

    elif o_type == onnx.TensorProto.UINT8:
        return TensorType.UINT8

    elif o_type == onnx.TensorProto.INT8:
        return TensorType.INT8

    elif o_type == onnx.TensorProto.UINT16:
        return TensorType.UINT16

    elif o_type == onnx.TensorProto.INT16:
        return TensorType.INT16

    elif o_type == onnx.TensorProto.INT32:
        return TensorType.INT32

    elif o_type == onnx.TensorProto.INT64:
        return TensorType.INT64

    elif o_type == onnx.TensorProto.STRING:
        return TensorType.STRING

    elif o_type == onnx.TensorProto.BOOL:
        return TensorType.BOOL

    elif o_type == onnx.TensorProto.FLOAT16:
        return TensorType.FLOAT16

    elif o_type == onnx.TensorProto.DOUBLE:
        return TensorType.FLOAT64

    elif o_type == onnx.TensorProto.UINT32:
        return TensorType.UINT32

    elif o_type == onnx.TensorProto.UINT64:
        return TensorType.UINT64

    elif o_type == onnx.TensorProto.COMPLEX64:
        return TensorType.COMPLEX64

    elif o_type == onnx.TensorProto.COMPLEX128:
        return TensorType.COMPLEX128

    elif o_type == onnx.TensorProto.BFLOAT16:
        logger.e(logger.Code.CONVERSION_IMPOSSIBLE, "Cannot convert ONNX DataType 'BFLOAT16' to TFLite.")


def numpy_type_to_tf_lite(numpy_type: np.dtype) -> TensorType:
    """Convert the numpy data type to a corresponding TFLite 'TensorType'.

    :param numpy_type: Numpy dtype to convert.
    :return: Corresponding TFLite TensorType.
    """
    numpy_type = numpy_type.type

    if numpy_type == np.float32:
        return TensorType.FLOAT32

    if numpy_type == np.uint8:
        return TensorType.UINT8

    if numpy_type == np.int8:
        return TensorType.INT8

    if numpy_type == np.uint16:
        return TensorType.UINT16

    if numpy_type == np.int16:
        return TensorType.INT16

    if numpy_type == np.int32:
        return TensorType.INT32

    if numpy_type == np.int64:
        return TensorType.INT64

    if numpy_type == np.bytes_:
        return TensorType.STRING

    if numpy_type == np.bool_:
        return TensorType.BOOL

    if numpy_type == np.float16:
        return TensorType.FLOAT16

    if numpy_type == np.float64 or numpy_type == np.double:
        return TensorType.FLOAT64

    if numpy_type == np.uint32:
        return TensorType.UINT32

    if numpy_type == np.uint64:
        return TensorType.UINT64

    if numpy_type == np.complex64:
        return TensorType.COMPLEX64

    if numpy_type == np.complex128:
        return TensorType.COMPLEX128

    logger.e(logger.Code.CONVERSION_IMPOSSIBLE, f"Cannot convert numpy data type '{numpy_type}' to TFLite.")


def tf_lite_type_to_numpy(tfl_type: TensorType) -> np.ScalarType:
    """Convert TFLite TensorType to numpy dtype"""
    if tfl_type == TensorType.FLOAT32:
        return np.dtype(np.float32)

    if tfl_type == TensorType.UINT8:
        return np.dtype(np.uint8)

    if tfl_type == TensorType.INT8:
        return np.dtype(np.int8)

    if tfl_type == TensorType.UINT16:
        return np.dtype(np.uint16)

    if tfl_type == TensorType.INT16:
        return np.dtype(np.int16)

    if tfl_type == TensorType.INT32:
        return np.dtype(np.int32)

    if tfl_type == TensorType.INT64:
        return np.dtype(np.int64)

    if tfl_type == TensorType.STRING:
        return np.dtype(np.bytes_)

    if tfl_type == TensorType.BOOL:
        return np.dtype(np.bool_)

    if tfl_type == TensorType.FLOAT16:
        return np.dtype(np.float16)

    if tfl_type == TensorType.FLOAT64:
        return np.dtype(np.float64)

    if tfl_type == TensorType.UINT32:
        return np.dtype(np.uint32)

    if tfl_type == TensorType.UINT64:
        return np.dtype(np.uint64)

    if tfl_type == TensorType.COMPLEX64:
        return np.dtype(np.complex64)

    if tfl_type == TensorType.COMPLEX128:
        return np.dtype(np.complex128)

    logger.e(logger.Code.CONVERSION_IMPOSSIBLE, f"Cannot convert TFLite type '{tfl_type}' to numpy dtype.")


def tflite_type_to_tensor_flow_data_type(tfl_type: TensorType) -> TensorFlowDataType:
    """Convert TFLite TensorType to the internal type of TensorFlow."""
    match tfl_type:
        case TensorType.FLOAT16:
            # There seems to be no counterpart in the TF DataType.
            logger.e(logger.Code.INTERNAL_ERROR, "tflite_type_to_tensor_flow_data_type(): float16.")
        case TensorType.FLOAT32:
            return TensorFlowDataType.DT_FLOAT.value
        case TensorType.FLOAT64:
            return TensorFlowDataType.DT_DOUBLE.value

        case TensorType.INT4:
            return TensorFlowDataType.DT_INT4.value
        case TensorType.INT8:
            return TensorFlowDataType.DT_INT8.value
        case TensorType.INT16:
            return TensorFlowDataType.DT_INT16.value
        case TensorType.INT32:
            return TensorFlowDataType.DT_INT32.value
        case TensorType.INT64:
            return TensorFlowDataType.DT_INT64.value

        case TensorType.UINT8:
            return TensorFlowDataType.DT_UINT8.value
        case TensorType.UINT16:
            return TensorFlowDataType.DT_UINT16.value
        case TensorType.UINT32:
            return TensorFlowDataType.DT_UINT32.value
        case TensorType.UINT64:
            return TensorFlowDataType.DT_UINT64.value

        case TensorType.COMPLEX64:
            return TensorFlowDataType.DT_COMPLEX64.value
        case TensorType.COMPLEX128:
            return TensorFlowDataType.DT_COMPLEX128.value

        case TensorType.STRING:
            return TensorFlowDataType.DT_STRING.value

        case TensorType.BOOL:
            return TensorFlowDataType.DT_BOOL.value

        case TensorType.RESOURCE:
            return TensorFlowDataType.DT_RESOURCE.value
        case TensorType.VARIANT:
            return TensorFlowDataType.DT_VARIANT.value

        case _:
            # All TFLite types are covered. Must be an invalid type.
            logger.e(logger.Code.INTERNAL_ERROR,
                     f"tflite_type_to_tensor_flow_data_type(): invalid TFLite type `{tfl_type}`.")


def infer_kernel_shape(weight_tensor: tflite_model.Tensor) -> list[int]:
    """Returns the kernel shape inferred from the weight tensor.

    Weight tensors shape expected in TFlite Format, where the 0th index is output channels count, last is input channels
    count.
    """
    return weight_tensor.shape.vector[1:-1]
