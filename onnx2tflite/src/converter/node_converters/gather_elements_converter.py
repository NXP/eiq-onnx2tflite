#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import errors, messages
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.conversion.translator import tf_lite_type_to_numpy
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.errors import ConversionError
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import gather_elements_attributes
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import concatenation_options, gather_nd_options, reshape_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, INTS


class GatherElementsConverter(NodeConverter):
    node = "GatherElements"

    onnx_supported_types = ALL_TYPES
    # `GatherElements` is decomposed into a TFLite `GatherND`.
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/gather_nd.cc#L138-L169
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8, TensorType.STRING, TensorType.BOOL]
    verified_types = INTS + [TensorType.FLOAT32, TensorType.UINT8, TensorType.STRING, TensorType.BOOL]

    def _make_channels_first(self, t_op: tflite_model.Operator, ops: OpsList) -> None:
        """Turn any `channels_last` `data`/output tensors of `t_op` back to `channels_first`.

        This ensures that all shapes and the `axis` match the original ONNX model. The `indices` tensor is always
        `formatless` (see `tensor_format_inference.py`), so it never needs to be handled here.
        """
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if x.tensor_format.is_channels_last():
            to_onnx_perm = translator.create_channels_last_to_channels_first_permutation(x.rank)
            ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, to_onnx_perm))
            t_op.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        if y.tensor_format.is_channels_last():
            to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(y.rank, return_list=True)
            ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, to_tflite_perm))
            t_op.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX `GatherElements` operator to TFLite.

        `GatherElements` gathers a single element along `axis` for every position of the `indices` tensor. The output
        has the same shape as `indices`, which has the same rank as `data`. For `axis = a`:

            y[i_0, ..., i_{n-1}] = data[i_0, ..., indices[i_0, ..., i_{n-1}], ..., i_{n-1}]

        TFLite has no direct equivalent, so this is expressed as a `GatherND`, which reads `data` at an explicit
        coordinate vector for every output position. Each such vector is the output position's own multi-dimensional
        index, except that the coordinate on `axis` is taken from the `indices` tensor. The coordinates for the
        non-gather axes form a static mesh grid that depends only on the `indices` shape (not on the index values),
        so the same decomposition works for both static and dynamic `indices`.
        """
        if len(t_op.tmp_inputs) != 2:
            raise ConversionError(errors.Code.INVALID_ONNX_MODEL, f"ONNX `GatherElements` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `2`.")

        x = t_op.tmp_inputs[0]
        rank = x.rank
        indices = t_op.tmp_inputs[1]
        y = t_op.tmp_outputs[0]

        attrs = cast(gather_elements_attributes.GatherElements, node.attributes)
        axis = self._normalize_axis(attrs, rank)

        self.assert_type_allowed(x.type)

        if t_op.is_quantized_without_qdq():
            propagate_quantization(x, y)

        ops = OpsList(middle_op=t_op)

        # Make all IO tensors `channels_first` so that the shapes and `axis` match the original ONNX model.
        self._make_channels_first(t_op, ops)

        indices_shape = indices.shape.vector
        dim_size = x.shape[axis]

        # A static mesh grid of coordinates for every axis. `grids[j]` holds, for each position of the output, its
        # index along axis `j`. This depends only on the `indices` shape, not on the index values.
        grids = np.indices(indices_shape)

        if tensor_has_data(indices):
            # Static `indices`. Build the whole coordinate tensor with numpy and use a single `GatherND`.
            indices_data = indices.tmp_buffer.data

            # TFLite `GatherND` only supports non-negative indices. ONNX allows negative ones.
            normalized = indices_data.copy()
            negative_mask = normalized < 0
            normalized[negative_mask] += dim_size
            if np.any(normalized < 0) or np.any(normalized >= dim_size):
                raise ConversionError(errors.Code.INVALID_ONNX_MODEL, "ONNX `GatherElements` operator has invalid `indices` tensor!")

            # TFLite `GatherND` requires full coordinate vector for each output position.
            # For the gather axis substitute `indices`, for every other axis keep the
            # "where am I" grid.
            coordinates = [normalized if j == axis else grids[j] for j in range(rank)]

            full_indices = np.stack(coordinates, axis=-1).astype(np.int64)
            t_op.tmp_inputs[1] = self.builder.create_tensor_for_data(full_indices, "indices")

        else:
            # Dynamic `indices`. Build the coordinate tensor at runtime with `Concat`. Only the `axis` coordinate is
            #  dynamic, the rest are static constants.
            if not self.context.conversion_config.guarantee_non_negative_indices:
                raise ConversionError(errors.Code.CONVERSION_IMPOSSIBLE, "Conversion of ONNX `GatherElements` with dynamic `indices` is not possible because they may "
                    "contain negative values, which is not supported by TFLite. "
                    + messages.Message.GUARANTEE_NON_NEGATIVE_INDICES)

            numpy_type = tf_lite_type_to_numpy(indices.type)

            concat_inputs = []
            for j in range(rank):
                if j == axis:
                    # Reshape the dynamic `indices` to append a trailing dimension of size 1.
                    reshaped = self.builder.create_empty_tensor(
                        "gather_elements_indices", indices.type, [*indices_shape, 1]
                    )
                    reshape_op = tflite_model.Operator(builtin_options=reshape_options.Reshape([*indices_shape, 1]))
                    reshape_op.tmp_inputs = [indices]
                    reshape_op.tmp_outputs = [reshaped]
                    ops.add_pre(reshape_op)

                    concat_inputs.append(reshaped)
                else:
                    grid_data = grids[j].reshape([*indices_shape, 1]).astype(numpy_type)
                    concat_inputs.append(self.builder.create_tensor_for_data(grid_data, "gather_elements_grid"))

            full_indices = self.builder.create_empty_tensor(
                "gather_elements_coordinates", indices.type, [*indices_shape, rank]
            )

            # Concatenate the coordinates along the new trailing dimension (index `rank`).
            concat_op = tflite_model.Operator(builtin_options=concatenation_options.Concatenation(rank))
            concat_op.tmp_inputs = concat_inputs
            concat_op.tmp_outputs = [full_indices]
            ops.add_pre(concat_op)

            t_op.tmp_inputs = [x, full_indices]

        t_op.builtin_options = gather_nd_options.GatherND()

        return ops.flatten()

    # noinspection PyMethodMayBeStatic
    def _normalize_axis(self, attrs: gather_elements_attributes.GatherElements, rank: int) -> int:
        axis = attrs.axis
        if axis < 0:
            axis += rank

        if not (0 <= axis < rank):
            raise ConversionError(errors.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, f"ONNX `GatherElements` has invalid `axis` attribute! (`{axis}`)")
        return axis
