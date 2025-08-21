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
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import arg_max_attributes
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import arg_max_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS, name_for_type


class ArgMaxConverter(NodeConverter):
    node = "ArgMax"

    onnx_supported_types = FLOATS + INTS + UINTS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/arg_min_max.cc#L100-L114
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT32, TensorType.BOOL]
    verified_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `ArgMax`` operator into TFLite `ArgMax` + potential `Reshape` and `Transpose` operators."""
        ops = OpsList(middle_op=t_op)

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `ArgMax` has unexpected number of inputs ({len(t_op.tmp_inputs)}).")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]
        self.assert_type_allowed(x.type)
        if y.type != TensorType.INT64:
            # ONNX only allows `int64` output.
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `ArgMax` has output type `{name_for_type(y.type)}` instead of `INT64`.")

        attrs = cast(arg_max_attributes.ArgMax, node.attributes)

        if attrs.select_last_index != 0:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"Conversion of ONNX `ArgMax` with `select_last_index={attrs.select_last_index}` is not possible.")

        if attrs.keepdims == 1:
            # TFLite always removes the reduced dimension. When `keepdims == 1`, ONNX leaves the dimension with size 1.
            # Append a `Reshape` operator to match the ONNX output shape.

            onnx_output_shape = y.shape.vector.copy()

            # Specify the correct TFLite output shape.
            tflite_corrected_axis = attrs.axis
            if x.tensor_format.is_channels_last():
                perm = translator.create_channels_last_to_channels_first_permutation(x.rank)
                tflite_corrected_axis = perm[tflite_corrected_axis]

            tflite_output_shape = y.shape.vector.copy()
            tflite_output_shape.pop(tflite_corrected_axis)
            y.shape = tflite_model.Shape(tflite_output_shape)

            # Add the `Reshape`, which will match the ONNX shape.
            ops.add_post(self.builder.create_reshape_after(t_op, 0, onnx_output_shape))

            y = t_op.tmp_outputs[0]  # The output tensor got changed. Update the `y` variable for future use.

        axis = attrs.axis
        if x.tensor_format.is_channels_last():
            # The `axis` refers to an axis of a channels first tensor. If the `axis` refers to a spatial dimension, it
            #  can be statically modified to work with a channels last tensor.
            # If it refers to `batch` or `channels`, a `Transpose` operator must be added before and after the `ArgMax`.
            normalized_axis = axis if axis >= 0 else axis + x.rank
            if normalized_axis in [0, 1]:
                # Prepend the `Transpose` to make the input `channels first`.
                to_onnx_perm = translator.create_channels_last_to_channels_first_permutation(x.rank)
                ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, to_onnx_perm))
                t_op.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

                # Append the `Transpose` to make the output `channels last` again.
                to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(y.rank, True)
                ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, to_tflite_perm))

                # The output of the `Transpose` is now `channels last`, but the output of the `ArgMax` is
                #  `channels first` because of the first `Transpose`.
                t_op.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

            else:
                # Just modify the `axis`.
                perm = translator.create_channels_last_to_channels_first_permutation(x.rank)
                axis = perm[axis]

        # TFLite uses a `dim` input tensor, which can represent the ONNX attribute `axis`. Create the tensor.
        dim_tensor = self.builder.create_tensor_for_data(np.array([axis], np.int32), "axis")
        t_op.tmp_inputs.append(dim_tensor)

        t_op.builtin_options = arg_max_options.ArgMax(TensorType.INT64)  # ONNX always has `int64` output.

        return ops.flatten()
