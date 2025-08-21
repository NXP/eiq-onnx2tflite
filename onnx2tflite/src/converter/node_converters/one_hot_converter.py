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
from onnx2tflite.src.converter.conversion.translator import tf_lite_type_to_numpy
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import one_hot_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.one_hot_options import OneHot
from onnx2tflite.src.tflite_generator.meta.types import name_for_type


# noinspection PyMethodMayBeStatic
class OneHotConverter(NodeConverter):
    node = "OneHot"

    def _try_get_values(self, values: tflite_model.Tensor) -> (np.ndarray | None, np.ndarray | None):
        """Return the `off_value` and `on_value` from the `values` ONNX `OneHot` input tensor, if possible.
             Otherwise, return `None` for both.

        :param values: The `values` input tensor of the ONNX `OneHot` operator.
        :return: (`off_value`, `on_value`) or (None, None).
        """
        off_value = on_value = None
        np_type = tf_lite_type_to_numpy(values.type)

        if tensor_has_data(values):
            if values.tmp_buffer.data.size != 2:
                logger.e(logger.Code.INVALID_ONNX_MODEL,
                         f"ONNX `OneHot` has `values` input with {values.tmp_buffer.data.size} elements instead of 2.")

            off_value = np.asarray(values.tmp_buffer.data[0], np_type)
            on_value = np.asarray(values.tmp_buffer.data[1], np_type)

        elif (inferred_data := self.context.onnx_inspector.try_get_inferred_tensor_data(values.name)) is not None:
            size = inferred_data.size if isinstance(inferred_data, np.ndarray) else len(inferred_data)
            if size == 2:
                logger.i(f"Using statically inferred data for tensor {values.name} (`values` of a `OneHot` operator).")
                off_value = np.asarray(inferred_data[0], np_type)
                on_value = np.asarray(inferred_data[1], np_type)

        return off_value, on_value

    def _check_types(self, t_op: tflite_model.Operator) -> None:
        indices = t_op.tmp_inputs[0]
        depth = t_op.tmp_inputs[1]
        values = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        if indices.type not in {TensorType.INT32, TensorType.INT64}:
            # TFLite only supports these 2 types for the `indices` tensor.
            # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/one_hot.cc#L149-L150
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, "Conversion of ONNX `OneHot` with `indices` of type "
                                                        f"{name_for_type(indices.type)} is not possible.")

        if values.type != y.type:
            # ONNX documentation specifies:
            #  "The type of the output tensor is the same as the type of the ‘values’ input."
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX `OneHot` has mismatched `output` type and `values` type.")

        # ONNXRT: ONNX Runtime only support specific type combinations.
        # https://github.com/microsoft/onnxruntime/blob/v1.17.0/onnxruntime/core/providers/cpu/tensor/onehot.cc#L59-L69
        match (indices.type, y.type, depth.type):
            case (TensorType.INT64, TensorType.INT64, TensorType.INT64) | \
                 (TensorType.INT64, TensorType.FLOAT32, TensorType.INT64) | \
                 (TensorType.INT32, TensorType.FLOAT32, TensorType.INT32):
                # Supported by ORT and TFLite.
                # Ultimately, these are the only supported data type combinations.
                pass

            case (TensorType.INT64, TensorType.FLOAT32, TensorType.INT32):
                # Supported by ORT but TFLite inference crashes with `MemoryError: bad allocation`.
                # Required by a real model.
                if tensor_has_data(depth):
                    depth_value = depth.tmp_buffer.data.item()
                elif (depth_value := self.context.onnx_inspector.try_get_inferred_tensor_data(depth.name)) is not None:
                    logger.i(f"Using inferred data for `OneHot` input `depth` named `{depth.name}`.")

                if depth_value is not None:
                    # Create a INT64 copy (in case it is used by some other operator).
                    int64_depth = depth.tmp_buffer.data.astype(np.int64)
                    new_depth_tensor = self.builder.create_tensor_for_data(int64_depth, "depth")
                    t_op.tmp_inputs[1] = new_depth_tensor

                else:
                    # The depth is dynamic, prepend a `Cast` operator if support is needed.
                    logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX `OneHot` with a dynamic `depth` of type "
                                                          f"{depth.type} is not implemented.")

            case (TensorType.INT32, TensorType.FLOAT32, TensorType.FLOAT32) | \
                 (TensorType.INT64, TensorType.INT32, TensorType.FLOAT32) | \
                 (TensorType.INT64, TensorType.FLOAT32, TensorType.FLOAT32):
                # Supported by ORT but TFLite inference crashes with `MemoryError: bad allocation`.
                # Can be supported by recasting static tensors, or by adding `Cast` operators.

                # The error message terminology is taken from:
                # https://github.com/microsoft/onnxruntime/blob/v1.17.0/onnxruntime/core/providers/cpu/tensor/onehot.cc#L55
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         f"Conversion of ONNX `OneHot` with input type = {name_for_type(indices.type)}, output type = "
                         f"{name_for_type(y.type)} and depth type = {name_for_type(depth.type)} is not yet supported.")

            case _:
                # All cases supported by ONNX Runtime at time of writing should be checked in the code above.
                # This section of the code should only be reached if new type combinations are added by ORT.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         f"Conversion of ONNX `OneHot` with input type = {name_for_type(indices.type)}, output type = "
                         f"{name_for_type(y.type)} and depth type = {name_for_type(depth.type)} is not yet supported.")

    def _normalize_axis(self, axis: int, t_op: tflite_model.Operator) -> int:
        """TFLite requires the `axis` to be non-negative:
             https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/one_hot.cc#L151-L152
            This method assures the `axis` is in the valid range.

        :param axis: `axis` attribute of the ONNX `OneHot`.
        :param t_op: TFLite operator with IO corresponding to the ONNX operator.
        :return: The `axis` in valid range.
        """
        rank = t_op.tmp_inputs[0].rank

        if axis < 0:
            # The axis can take on the value of `rank`, which represents adding the dimension at the end.
            #  For example `-1` represents the value `rank`, instead of `rank-1` which is common for other operators.
            #  Therefore, `rank+1` must be added to the axis.
            axis += rank + 1

        if not (0 <= axis <= rank):  # As mentioned in the comment above, `axis` can take on the value of `rank`.
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     "ONNX `OneHot` has invalid value of the `axis` attribute.")

        return axis

    def _get_depth_value(self, depth: tflite_model.Tensor) -> int:
        """Get the `depth` value from the `depth` input tensor of the ONNX `OneHot` if possible.
        Otherwise, exit with error.
        """
        if tensor_has_data(depth):
            depth_val = depth.tmp_buffer.data.item()

        elif (depth_val := self.context.onnx_inspector.try_get_inferred_tensor_data(depth.name)) is not None:
            logger.i(f"Using inferred static data for `OneHot` input tensor `depth` named `{depth.name}`.")
            if isinstance(depth_val, np.ndarray):
                depth_val = depth_val.item()
            elif isinstance(depth_val, list):
                depth_val = depth_val[0]

        if depth_val is None:
            # Failed to get the data.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX `OneHot` with a dynamic `depth` input is not possible.")

        return depth_val

    def _validate_indices(self, t_op: tflite_model.Operator) -> tflite_model.Tensor:
        """TFLite doesn't support negative `indices`. Nothing crashes, just the output would not be correct.

        :param t_op: TFLite operator with IO corresponding to the ONNX `OneHot` operator.
        :return: The new and valid `indices` tensor.
        """
        indices = t_op.tmp_inputs[0]
        depth = t_op.tmp_inputs[1]

        if tensor_has_data(indices):
            data = indices.tmp_buffer.data

        elif (data := self.context.onnx_inspector.try_get_inferred_tensor_data(indices.name)) is not None:
            logger.i(f"Using inferred static data for `OneHot` input tensor `indices` named `{indices.name}`.")

        if data is None:
            # Cannot check for non-negative indices.
            if self.context.conversion_config.non_negative_indices:
                # User guarantees that the indices are not negative.
                pass

            else:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX `OneHot` with dynamic `indices` input is not possible, because it may "
                         " contain negative values, which isn't supported by TFLite. " +
                         logger.Message.GUARANTEE_NON_NEGATIVE_INDICES)

        elif any(val < 0 for val in data.flatten()):
            # Make all indices non-negative.
            # note: some values might still be negative, e.g. if depth is 2 and the index is -3, it will be -1. But that
            #  does not matter, because for ONNX, -3 is out of range, and -1 is out of range for TFLite.
            #  Therefore, the outputs will be identical.
            depth_val = self._get_depth_value(depth)
            data = np.asarray([val if val >= 0 else val + depth_val for val in data.flatten()], data.dtype).reshape(
                indices.shape.vector)

            if tensor_has_data(indices):
                # Replace the static tensor data.
                indices.tmp_buffer.data = data
            else:
                # Create a new tensor for the `indices`, in case inferred data was used and the `indices` are dynamic.
                indices = self.builder.create_tensor_for_data(data, "indices")

        return indices

    def _handle_tensor_formats(self, t_op: tflite_model.Operator, axis: int, ops: OpsList) -> int:
        """Handle combinations of input and output tensor formats, by adding `Transpose` operators or by modifying the
        `axis`.
        """
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]
        rank = x.rank

        if x.tensor_format.is_channels_last() and y.tensor_format.is_channels_last():
            # If the `axis` refers to the batch or channels, we need to insert a `Transpose` before and after.
            # Otherwise, it suffices to modify the axis to refer to `channels_last` dimensions.

            if axis in {0, 1}:
                # First permute the input to `channels_first`, then do the `OneHot` and after that permute the output to
                #  `channels_last`.
                to_onnx_perm = translator.create_channels_last_to_channels_first_permutation(rank)
                to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(rank + 1, True)
                ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, to_onnx_perm))
                ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, to_tflite_perm))

            else:
                perm = translator.create_channels_last_to_channels_first_permutation(rank + 1, True)
                axis = perm[axis]

        elif x.tensor_format.is_channels_last():  # y is not channels last
            # Permute the input to `channels_first`.
            to_onnx_perm = translator.create_channels_last_to_channels_first_permutation(rank)
            ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, to_onnx_perm))

        elif y.tensor_format.is_channels_last():  # x is not channels last
            # Permute the output to `channels_last`.
            to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(rank + 1, True)
            ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, to_tflite_perm))

        else:  # Both are formatless
            # Nothing needs to be done.
            pass

        return axis

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `OneHot` operator into TFLite `OneHot`."""
        if len(t_op.tmp_inputs) != 3:
            # ONNX `OneHot` has 3 mandatory input tensors.
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `OneHot` has unexpected number of inputs ({len(t_op.tmp_inputs)}).")

        indices = self._validate_indices(t_op)
        depth = t_op.tmp_inputs[1]
        values = t_op.tmp_inputs[2]

        ops = OpsList(middle_op=t_op)

        # ONNX `values` contains 2 elements [off_value, on_value].
        # TFLite instead uses 2 one-element tensors `on_value` and `off_value`.
        off_value, on_value = self._try_get_values(values)
        if off_value is None or on_value is None:
            # The `values` tensor is dynamic. TODO Add a `Split` operator.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX `OneHot` with a dynamic `values` input is not yet supported.")

        else:
            on_value_tensor = self.builder.create_tensor_for_data(on_value, "on_value")
            off_value_tensor = self.builder.create_tensor_for_data(off_value, "off_value")

        # noinspection PyUnboundLocalVariable
        t_op.tmp_inputs = [indices, depth, on_value_tensor, off_value_tensor]

        attrs = cast(one_hot_attributes.OneHot, node.attributes)
        axis = self._normalize_axis(attrs.axis, t_op)

        axis = self._handle_tensor_formats(t_op, axis, ops)

        self._check_types(t_op)

        t_op.builtin_options = OneHot(axis)

        return ops.flatten()
