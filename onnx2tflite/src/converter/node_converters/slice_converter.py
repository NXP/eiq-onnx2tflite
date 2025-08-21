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
from onnx2tflite.src.converter.conversion.common import OpsList, try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import slice_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import slice_options, strided_slice_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, INTS


# noinspection PyMethodMayBeStatic
class SliceConverter(NodeConverter):
    node = "Slice"

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/slice.cc#L229-L261
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8, TensorType.UINT32, TensorType.BOOL,
                                     TensorType.STRING]
    verified_types = INTS + [TensorType.FLOAT32, TensorType.UINT8, TensorType.UINT32, TensorType.BOOL,
                             TensorType.STRING]

    def _load_v_1_attributes(self, o_slice_attributes: slice_attributes.Slice, input_rank: int) -> \
            tuple[list[int], list[int], list[int], list[int]]:
        """Load the starts, ends, axes and steps attributes from the ONNX Slice attributes version 1.
    
        :param o_slice_attributes: ONNX Slice attributes v1.
        :return: starts, ends, axes and steps
        """
        starts = o_slice_attributes.starts
        if starts is None:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     "ONNX 'Slice' v1 has attribute 'starts' uninitialized!")

        ends = o_slice_attributes.ends
        if ends is None:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX 'Slice' v1 has attribute 'ends' uninitialized!")

        if o_slice_attributes.axes is not None:
            axes = o_slice_attributes.axes
        else:
            axes = list(range(input_rank))

        steps = [1] * len(starts)

        return starts, ends, axes, steps

    def _clamp_starts_and_ends(self, starts: list[int], ends: list[int], axes: list[int], steps: list[int],
                               input_shape: list[int]) -> None:
        """Clamp the ONNX Slice attributes 'starts' and 'ends' according to the documentation.
    
        :param starts: ONNX Slice attribute 'starts'
        :param ends: ONNX Slice attribute 'ends'
        :param axes: ONNX Slice attribute 'axes'
        :param steps: ONNX Slice attribute 'steps'
        :param input_shape: Shape of the main input tensor of the ONNX Slice
        """
        for i, step in enumerate(steps):
            if step < 0:
                # Negative direction
                starts[i] = common.clamp(starts[i], 0, input_shape[axes[i]] - 1)
                ends[i] = common.clamp(ends[i], -1, input_shape[axes[i]] - 1)

            else:
                # Positive direction
                starts[i] = common.clamp(starts[i], 0, input_shape[axes[i]])
                ends[i] = common.clamp(ends[i], 0, input_shape[axes[i]])

    def _validate_axes(self, axes: list[int], input_rank: int) -> list[int]:
        """Make sure the ONNX Sice 'axes' operand is valid and return the corresponding TFLite slice 'axes' operand.
    
        :param axes: ONNX Slice operand axes.
        :param input_rank: The rank of the main input tensor of the Slice operator.
        :return: The TFLite Slice operand 'axes'.
        """
        axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
        if not all(0 <= axis < input_rank for axis in axes):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     "ONNX Slice attribute 'axes' contains invalid indices!")
        if common.contains_duplicates(axes):
            # Behavior is undefined according to documentation
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX Slice attribute 'axes' contains duplicates!")

        return axes

    def _validate_starts_and_ends(self, starts: list[int], ends: list[int], axes: list[int],
                                  main_input_shape: list[int],
                                  steps: list[int]) -> None:
        # Replace negative indices in 'starts' and 'ends'
        for i, axis in enumerate(axes):
            if starts[i] < 0:
                starts[i] += main_input_shape[axis]
            if ends[i] < 0:
                ends[i] += main_input_shape[axis]

        # Make sure 'starts' and 'ends' are in the valid ranges
        self._clamp_starts_and_ends(starts, ends, axes, steps, main_input_shape)

    def _try_get_tensor_with_data(self, input_tensor) -> tflite_model.Tensor | None:
        if input_tensor is not None and not tensor_has_data(input_tensor):  # Tensor has buffer data
            inferred_data = self.inspector.try_get_inferred_tensor_data(input_tensor.name)
            if inferred_data is not None:
                # Tensor data inferred during shape inference, use it
                input_tensor = self.builder.create_tensor_for_data(inferred_data, name=input_tensor.name)

        return input_tensor

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX 'Slice' to TFLite `Slice`."""
        main_input = t_op.tmp_inputs[0]
        main_input_shape = main_input.shape.vector
        input_rank = len(main_input_shape)
        y = t_op.tmp_outputs[0]

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(main_input.type)

        ops = OpsList(middle_op=t_op)

        o_slice_attributes = cast(slice_attributes.Slice, node.attributes)

        if node.version < 10:
            # ONNX Slice v1
            starts, ends, axes, steps = self._load_v_1_attributes(o_slice_attributes, input_rank)

        else:
            # ONNX Slice v10+

            if not (3 <= len(t_op.tmp_inputs) <= 5):
                logger.e(logger.Code.INVALID_ONNX_OPERATOR, f"ONNX 'Slice' has unexpected number of "
                                                            f"inputs! Expected 3 to 5, got '{len(t_op.tmp_inputs)}'.")

            starts_tensor = self._try_get_tensor_with_data(t_op.tmp_inputs[1])
            ends_tensor = self._try_get_tensor_with_data(t_op.tmp_inputs[2])
            axes_tensor = self._try_get_tensor_with_data(try_get_input(t_op, 3))
            steps_tensor = self._try_get_tensor_with_data(try_get_input(t_op, 4))

            if any(t is not None and not tensor_has_data(t) for t in
                   [starts_tensor, ends_tensor, axes_tensor, steps_tensor]):
                # Some inputs are dynamic.
                # Conversion might be possible via combination with other operators. But that could be difficult.
                logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX Slice is only supported when"
                                                      " the starts, ends, axes and steps input tensors are static!")

            starts = list(starts_tensor.tmp_buffer.data)
            ends = list(ends_tensor.tmp_buffer.data)
            axes = list(axes_tensor.tmp_buffer.data) if axes_tensor is not None else list(range(input_rank))
            steps = list(steps_tensor.tmp_buffer.data if steps_tensor is not None else [1] * len(starts))

        # -- starts, ends, axes and steps are all initialized --

        if not (len(starts) == len(ends) == len(axes) == len(steps)):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX Slice attributes 'starts', 'ends', 'axes' and "
                                                                  "steps' don't have matching lengths!")

        axes = self._validate_axes(axes, input_rank)

        if main_input.tensor_format.is_channels_last():
            # Indices in axes point to channels first dimensions. Update them to refer to channels last dimensions.
            permutation = translator.create_channels_last_to_channels_first_permutation(input_rank)
            axes = [permutation[axis] for axis in axes]

        self._validate_starts_and_ends(starts, ends, axes, main_input_shape, steps)

        if any(step == 0 for step in steps):
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX Slice must have non-zero 'steps'.")
        elif all(step == 1 for step in steps):
            self._convert_to_slice(t_op, main_input, input_rank, starts, ends, axes)
        else:
            self._convert_to_strided_slice(t_op, main_input, input_rank, starts, ends, axes, steps)

        if t_op.is_quantized_without_qdq():
            propagate_quantization(main_input, y)

        elif t_op.is_qdq_quantized():
            if main_input.quantization != y.quantization:
                logger.w(
                    f"Re-quantizing input tensor '{main_input.name}' of Slice op to match output tensor's q-params. "
                    f"This can decrease accuracy of the model.")
                # Re-quantize input based on output's q-params
                scale = y.quantization.scale.vector
                zp = y.quantization.zero_point.vector
                ops.pre_ops.append(self.builder.create_quantize_operator_before(t_op, 0, main_input.type, scale, zp))

        return ops.flatten()

    def _convert_to_slice(self, t_op, main_input, input_rank, starts, ends, axes) -> None:
        # Prepare the TFLite parameters 'begin' and 'size'
        begin = [0] * input_rank  # By default, start the slice at 0
        size = main_input.shape.vector.copy()  # By default, end the slice at the end of the dimension

        for i, axis in enumerate(axes):
            begin[axis] = starts[i]
            size[axis] = ends[i] - starts[i]

            size[axis] = max(size[axis], 0)

        # Create the TFLite tensors
        begin_tensor = self.builder.create_tensor_for_data(np.asarray(begin, np.int32), "begin")
        size_tensor = self.builder.create_tensor_for_data(np.asarray(size, np.int32), "size")
        t_op.tmp_inputs = [main_input, begin_tensor, size_tensor]
        t_op.builtin_options = slice_options.Slice()

    def _convert_to_strided_slice(self, t_op, main_input, input_rank, starts, ends, axes, steps) -> None:
        tf_begin = [0] * input_rank  # By default, start slice from 0
        tf_end = main_input.shape.vector.copy()  # By default, end slice at the end of dimension
        tf_strides = [1] * input_rank  # By default, step by 1

        for i, axis in enumerate(axes):
            tf_begin[axis] = starts[i]
            tf_end[axis] = ends[i]
            tf_strides[axis] = steps[i]

            # TFLite cannot handle situation when we're iterating down
            # from positive values through 0, to negative values.
            # noinspection PyChainedComparisons
            if steps[i] < 0 and starts[i] >= 0 and ends[i] < 0:
                # Add negative offset of dimension size and make both 'begin' and 'end' negative
                tf_begin[axis] = starts[i] - main_input.shape.vector[axis]
                tf_end[axis] = ends[i] - main_input.shape.vector[axis]

        begin_tensor = self.builder.create_tensor_for_data(np.asarray(tf_begin, np.int32), "begin")
        end_tensor = self.builder.create_tensor_for_data(np.asarray(tf_end, np.int32), "ends")
        strides_tensor = self.builder.create_tensor_for_data(np.asarray(tf_strides, np.int32), "strides")

        t_op.tmp_inputs = [main_input, begin_tensor, end_tensor, strides_tensor]
        t_op.builtin_options = strided_slice_options.StridedSlice()
