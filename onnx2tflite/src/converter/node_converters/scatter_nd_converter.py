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
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import (
    propagate_quantization,
    quantize_static_float_tensor,
    re_quantize_static_tensor,
)
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import scatter_nd_attributes
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import scatter_nd_options
from onnx2tflite.src.tflite_generator.builtin_options.select_v2_options import SelectV2
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES


class ScatterNDConverter(NodeConverter):
    node = "ScatterND"

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/comparisons.cc#L173-L212
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/scatter_nd.cc#L152-L176
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.BOOL, TensorType.INT8, TensorType.INT32,
                              TensorType.INT64]
    # With int8 and uint8 TFLite inference crashes badly (Fatal Python error: Aborted) if the tensors are not quantized.
    verified_types = [TensorType.FLOAT32, TensorType.BOOL, TensorType.INT32, TensorType.INT64]

    def _get_mask(self, t_op: tflite_model.Operator, ops: list[tflite_model.Operator]) -> tflite_model.Tensor:
        """Get a tensor which will contain `True` in places specified by the `indices` of the ONNX ScatterND, and
        `False` everywhere else, at runtime.
        """
        x = t_op.tmp_inputs[0]
        indices_tensor = t_op.tmp_inputs[1]
        updates_tensor = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        if not updates_tensor.shape.is_well_defined():
            # The shape must be known in order to compute the mask.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, "Conversion of `ONNX` ScatterND with `updates` operand with an "
                                                        "unknown shape is not possible.")

        # Create TFLite operator, which will compute the mask at runtime.
        output_mask_tensor = self.builder.duplicate_tensor(x, empty_buffer=True)
        true_tensor = self.builder.create_tensor_for_data(np.ones(updates_tensor.shape.vector, np.bool_), "true")
        shape_tensor = self.builder.create_tensor_for_data(np.array(y.shape.vector, np.int32), "shape")

        scatter_nd = tflite_model.Operator(builtin_options=scatter_nd_options.ScatterND())
        scatter_nd.tmp_inputs = [indices_tensor, true_tensor, shape_tensor]
        scatter_nd.tmp_outputs = [output_mask_tensor]

        ops.append(scatter_nd)

        return output_mask_tensor

    def _get_updated_values(self, t_op: tflite_model.Operator, ops: list[tflite_model.Operator]) -> tflite_model.Tensor:
        """Get a tensor, which will contain the `updates` of the ONNX `ScatterND` at the corresponding indices.
        The other values will be 0.

        The tensor may or may not contain static data! If it's dynamic, the operators which produce it will be added
         into `ops`.
        """
        x = t_op.tmp_inputs[0]
        indices_tensor = t_op.tmp_inputs[1]
        updates_tensor = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        if tensor_has_data(indices_tensor):
            indices = indices_tensor.tmp_buffer.data
        else:
            indices = self.inspector.try_get_inferred_tensor_data(indices_tensor.name)

        if tensor_has_data(updates_tensor):
            updates = updates_tensor.tmp_buffer.data
        else:
            updates = self.inspector.try_get_inferred_tensor_data(updates_tensor.name)

        if x.tensor_format.is_channels_last():
            if indices is not None and updates is not None:
                # The conversion is not simple. The `indices` are an ND tensor, for example: [[1, 0, 2], [0, 2, 3]].
                #  The values in the innermost dimension are indices to the main input tensor (with shape for example
                #  [2, 3, 4, 5]). In this example, the indices refer to 2 slices, both of size `5`. The problem is that
                #  if the input is not formatless, so the TFLite input would have shape [2, 4, 5, 3]. The indices would
                #  now have to be something like [[1, -1, 2, 0], [0, -1, 3, 2]] (where `-1` refers to the entire
                #  dimension), which is not possible. The only way to convert this would be to create a new element
                #  in the `indices` for every valid dimension value in place of the `-1`, and the `updates` would
                #  have to be reshaped and permuted accordingly.
                # It seems that real models use `ScatterND` only with formatless data. Therefore, conversion is not
                #  yet implemented.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX `ScatterND` with channels first inputs is not yet supported.")

            else:
                # The indices are dynamic. The preprocessing described in the comment above would be too complicated to
                #  implement by adding extra operators.
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX `ScatterND` with dynamic channels first inputs is not supported.")

        # Create a TFLite operator which will compute the tensor at runtime.
        updated_values = self.builder.duplicate_tensor(x, empty_buffer=True)
        shape_tensor = self.builder.create_tensor_for_data(np.array(y.shape.vector, np.int32), "shape")

        scatter_nd = tflite_model.Operator(builtin_options=scatter_nd_options.ScatterND())
        scatter_nd.tmp_inputs = [indices_tensor, updates_tensor, shape_tensor]
        scatter_nd.tmp_outputs = [updated_values]

        if x.quantization is not None:
            scale = x.quantization.scale.vector
            zp = x.quantization.zero_point.vector

            if tensor_has_data(updates_tensor):
                if updates_tensor.quantization is None:
                    # Updates tensor not quantized and data available -> quantize
                    scatter_nd.tmp_inputs[1] = quantize_static_float_tensor(self.builder, updates_tensor, x.type, scale, zp)
                elif x.quantization != updates_tensor.quantization:
                    # Updates tensor is quantized but quantization params doesn't match -> re-quantize
                    scatter_nd.tmp_inputs[1] = re_quantize_static_tensor(self.builder, updates_tensor, x.type, scale, zp)
                    logger.w("Requantizing 'updates' tensor of ScatterND operator. onnx2quant quantizer can potentially"
                             " avoid this.")
            elif updates_tensor.quantization is None or x.quantization != updates_tensor.quantization:
                # 'updates' tensor is quantized and data not available -> prepend with Quantize
                logger.w("Requantizing 'updates' tensor of ScatterND operator. onnx2quant quantizer can potentially"
                         " avoid this.")

                ops.append(self.builder.create_quantize_operator_before(scatter_nd, 1, x.type, scale, zp))

        ops.append(scatter_nd)

        return updated_values

    def _ensure_tensor_is_same_as_in_onnx_model(self, t_op: tflite_model.Operator, input_index: int,
                                                ops: list[tflite_model.Operator]) -> None:
        """If the input tensor of `t_op` on index `input_index` is `channels last`, this method will turn it to
        `channels first` to match the ONNX model. Static tensor is permuted statically and for dynamic tensors, a
        `Transpose` operator is prepended and it is added to `ops`.
        """
        tensor = t_op.tmp_inputs[input_index]
        if not tensor.tensor_format.is_channels_last():
            return

        to_onnx_perm = translator.create_channels_last_to_channels_first_permutation(tensor.rank, True)

        if tensor_has_data(tensor):
            # Permute the static tensor.
            tensor = self.builder.duplicate_tensor(tensor)
            translator.permute_static_tensor(tensor, to_onnx_perm)

            t_op.tmp_inputs[input_index] = tensor

        else:
            # Prepend a `Transpose` operator.
            ops.append(self.builder.create_transpose_operator_before(t_op, input_index, to_onnx_perm))

        t_op.tmp_inputs[input_index].tensor_format = TensorFormat.CHANNELS_FIRST

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `ScatterND` operator into TFLite.

        The 'ScatterND' in TFLite behaves differently from the ONNX variant.
         - ONNX ScatterND updates its input tensor on specific indices with some given values.
         - TFLite ScatterND creates a new tensor, that will have the given values on specified indices, but all
            other values wil be 0.
            https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/internal/reference/reference_ops.h#L690

        The ONNX ScatterND behavior can be represented in TFLite as:

              indices   true   output shape      indices  updates  output shape
                 │        │         │               │        │         │
                 │  ┌─────▼─────┐   │               │  ┌─────▼─────┐   │
                 └──► ScatterND │◄──┘               └──► ScatterND │◄──┘
                    └─────┬─────┘                      └─────┬─────┘
                          │                 ┌────────────────┘
                          │  (mask)    ┌────▼─────┐
                          └────────────► SelectV2 ◄─────── x
                                       └────┬─────┘
                                            │
                                            ▼
                                            y

        This is similar to what MLIR does, just more efficient.
        If the `indices` and/or `updates` are static, some operators can be omitted. This is however not implemented
         in order to avoid having to add a dependency on `tensorflow`, which is the only library with the
         `scatter_nd` function.

        An alternative is to use the `ScatterNdUpdate`, which is one of the SELECT_TF_OPS.

        If the ONNX `ScatterND` has attribute `reduction` != 'none', convert to one of the following SELECT_TF_OPS:
            - ScatterNDAdd
            - ScatterNDMax
            - ScatterNDMin
        """
        attrs = cast(scatter_nd_attributes.ScatterND, node.attributes)
        if attrs.reduction != "none":
            # Other values must be converter to Flex delegate operators. See comment above.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     f"Conversion of ONNX `ScatterND` with `reduction={attrs.reduction}` is not yet supported.")

        if len(t_op.tmp_inputs) != 3:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `ScatterND` has {len(t_op.tmp_inputs)} inputs instead of 3.")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if t_op.is_quantized_without_qdq():
            propagate_quantization(x, y)

            updates_tensor = t_op.tmp_inputs[2]
            if updates_tensor.quantization is None:
                propagate_quantization(x, updates_tensor)

        elif not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        if not y.shape.is_well_defined():
            # TFLite ScatterND takes the output shape as an input operand. So it must be known.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX `ScatterND` with an unknown output shape is not possible.")

        pre_ops = []
        ops = []
        post_ops = []

        # If any inputs are `channels last`, turn them to `channels first` to match the ONNX models. Otherwise, the
        #  shapes would not match and the `indices` would refer to different data.
        for i, _ in enumerate(t_op.tmp_inputs):
            self._ensure_tensor_is_same_as_in_onnx_model(t_op, i, pre_ops)

        if y.tensor_format.is_channels_last():
            # Make the output `channels first` as well.
            to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(y.rank, True)
            post_ops.append(self.builder.create_transpose_operator_after(t_op, 0, to_tflite_perm))
            t_op.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        x = t_op.tmp_inputs[0]
        indices_tensor = t_op.tmp_inputs[1]
        y = t_op.tmp_outputs[0]

        # Prepare the `indices` for TFLite. It requires strictly non-negative values of type int32.
        #  ONNX allows negative indices and requires int64.
        if tensor_has_data(indices_tensor):
            # Do it statically.

            indices_data = indices_tensor.tmp_buffer.data

            # TFLite doesn't support negative indices. Re-compute them to be strictly >= 0.
            if any(elt < 0 for elt in indices_data.flat):
                innermost_dim = indices_tensor.shape[-1]
                indices_data = indices_data.reshape([-1, innermost_dim])  # Flatten all dimensions except the last one.
                if len(indices_data.shape) != 2:
                    # This shouldn't happen.
                    logger.e(logger.Code.NOT_IMPLEMENTED,
                             "Conversion of ONNX `ScatterND` with negative `indices` is not yet supported.")

                # `indices_data` is now a 2D matrix where the rows are vectors of indices to `x`.
                #  Iterate over the vectors and for each vector, normalize the index to range [0, dim_size - 1].
                new_indices_data = []
                for vector in indices_data:
                    new_vector = []
                    for i, elt in enumerate(vector):
                        if elt < 0:
                            elt += x.shape[i]
                            if elt < 0:
                                logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX `ScatterND` has invalid `indices`.")

                        new_vector.append(elt)

                    new_indices_data.append(new_vector)

                indices_data = np.array(new_indices_data, np.int64).reshape(indices_data.shape)
                indices_tensor = self.builder.create_tensor_for_data(indices_data, "indices")
                t_op.tmp_inputs[1] = indices_tensor

            indices_tensor.tmp_buffer.data = indices_tensor.tmp_buffer.data.astype(np.int32)
            indices_tensor.type = TensorType.INT32

        else:
            # Dynamic cast.
            if not self.context.conversion_config.guarantee_non_negative_indices:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX `ScatterND` with dynamic `indices` is not supported, because they may "
                         "contain negative values, which is not supported by TFLite. " +
                         logger.Message.GUARANTEE_NON_NEGATIVE_INDICES)

            ops.append(self.builder.create_cast_before(t_op, 1, TensorType.INT32))

        # Create a boolean tensor marking where the `updates` will be placed into the input tensor.
        mask_tensor = self._get_mask(t_op, ops)

        # Create a tenor holding the `updates` on indices specified by the ONNX `ScatterND`. The other values will be 0.
        updated_values = self._get_updated_values(t_op, ops)

        # Create a `Select` operator which will chose a value either from `x` or from `updated_values`, based on the
        #  `mask_tensor`.
        select = tflite_model.Operator(builtin_options=SelectV2())
        select.tmp_inputs = [mask_tensor, updated_values, x]
        select.tmp_outputs = [y]

        ops.append(select)

        return pre_ops + ops + post_ops
