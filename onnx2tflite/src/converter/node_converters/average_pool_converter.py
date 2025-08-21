#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from typing import cast

import numpy as np

import onnx2tflite.src.onnx_parser.builtin_attributes.average_pool_attributes as onnx_average_pool_attributes
from onnx2tflite.lib.tflite.Padding import Padding
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import common, translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import average_pool_2d_options as tfl_average_pool_2d_options
from onnx2tflite.src.tflite_generator.builtin_options import reshape_options as tfl_reshape_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class AveragePoolConverter(NodeConverter):
    node = "AveragePool"

    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/pooling.cc#L390-L407
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT64, TensorType.INT8, TensorType.INT16]
    onnx_supported_types = FLOATS + [TensorType.INT8]
    verified_types = [TensorType.FLOAT32]  # INT8 not supported by ORT

    def _convert_1d_average_pool(self, node: onnx_model.NodeProto,
                                 t_op: tflite_model.Operator) -> [tflite_model.Operator]:
        """Convert the ONNX 'AveragePool' operator with a 1D kernel to TFLite 'AveragePool2D'.
        TFLite doesn't support 1D AveragePool, but this behaviour can be represented using
               Reshape -> AveragePool2D -> Reshape.
        The first reshape introduces a 4th dimension with size 1. The second Reshape removes the temporary
         dimension.
        """
        attrs = cast(onnx_average_pool_attributes.AveragePool, node.attributes)

        for dim in t_op.tmp_inputs[0].shape.vector:
            if (not isinstance(dim, int)) or dim < 0:
                # Dynamic shapes make it difficult to use the Reshape operators.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of 1D ONNX AveragePool with a dynamic shape is not yet supported.")

        # -- Calculate the shapes for equivalent 2D AveragePool --
        reshape1_output_shape = translator.nhc_dimensions_to_nhwc(t_op.tmp_inputs[0].shape.vector)
        reshape2_input_shape = translator.nhc_dimensions_to_nhwc(t_op.tmp_outputs[0].shape.vector)

        # -- Generate tensors taking part in the conversion --
        reshape1_input = t_op.tmp_inputs[0]

        reshape1_output = self.builder.duplicate_tensor(reshape1_input, name_suffix="_4D_")
        reshape1_output.shape = tflite_model.Shape(reshape1_output_shape)

        reshape2_input = self.builder.duplicate_tensor(reshape1_input, name_suffix="_4D_")
        reshape2_input.shape = tflite_model.Shape(reshape2_input_shape)

        reshape2_output = t_op.tmp_outputs[0]

        # -- Create the new operators --
        reshape1 = tflite_model.Operator(builtin_options=tfl_reshape_options.Reshape(reshape1_output_shape))
        reshape1.tmp_inputs = [reshape1_input]
        reshape1.tmp_outputs = [reshape1_output]

        reshape2 = tflite_model.Operator(builtin_options=tfl_reshape_options.Reshape(reshape2_output.shape.vector))
        reshape2.tmp_inputs = [reshape2_input]
        reshape2.tmp_outputs = [reshape2_output]

        # Connect the AveragePool with the Reshape operators
        t_op.tmp_inputs = [reshape1_output]
        t_op.tmp_outputs = [reshape2_input]

        # Extend all ONNX attributes to 2D
        common.extend_1d_dilations_to_2d(attrs.dilations)
        common.extend_1d_pads_to_2d(attrs.pads)
        common.extend_1d_strides_to_2d(attrs.strides)
        common.extend_1d_kernel_shape_to_2d(attrs.kernel_shape)

        # Convert the now 2D AveragePool
        converted_average_pool_ops = self._convert_2d_average_pool(node, t_op)

        return [reshape1] + converted_average_pool_ops + [reshape2]

    def _convert_2d_average_pool(self, node: onnx_model.NodeProto,
                                 t_op: tflite_model.Operator) -> [tflite_model.Operator]:
        """Convert the ONNX 'AveragePool' operator with a 2D kernel to TFLite 'AveragePool2D'."""
        attrs = cast(onnx_average_pool_attributes.AveragePool, node.attributes)

        if attrs.dilations is not None:
            if any([dilation != 1 for dilation in attrs.dilations]):
                # TFLite AveragePool doesn't support dilations. TODO Convert to DepthwiseConv2D, which has dilations.
                # ONNX dilations were only added in v19. Also onnx.checker still doesn't allow dilations even for v19.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         f"Conversion of ONNX AveragePool with dilations '{attrs.dilations}' is not yet implemented.")

        if attrs.ceil_mode == 1:
            # TFLite always uses 'floor' to round, so the output shape may be different from ONNX.
            # TODO Calculate the output shape, to see if this attribute even makes a difference.
            # TODO Prepending a 'Pad' operator should work, but only if 'count_include_pad' is 1.
            if attrs.count_include_pad == 1:
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX AveragePool with 'ceil_mode' = 1 is not yet supported.")

            else:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE, "Conversion of ONNX AveragePool with 'ceil_mode' = 1 and "
                                                            "'count_include_pad' = 0 is not possible.")

        t_op.builtin_options = tfl_average_pool_2d_options.AveragePool2D()
        ops = OpsList(middle_op=t_op)

        common.assign_2d_strides(t_op.builtin_options, attrs.strides)

        t_op.builtin_options.filter_h = attrs.kernel_shape[0]
        t_op.builtin_options.filter_w = attrs.kernel_shape[1]

        # Convert the padding
        t_op.builtin_options.padding, explicit_padding = translator.convert_padding(attrs.auto_pad, attrs.pads,
                                                                                    t_op.tmp_inputs[0].shape.vector,
                                                                                    t_op.tmp_outputs[0].shape.vector,
                                                                                    attrs.kernel_shape, attrs.strides)
        if explicit_padding is not None:
            # Need to prepend a 'Pad' operator, which adds 0s. But these will be included in the computation!
            if attrs.count_include_pad == 0:
                # The 0s must NOT be included in the computation of the average value.
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX AveragePool with 'count_include_pad' = 0 and a specific combination of "
                         "input shape, 'kernel_shape', 'strides', 'dilations' and padding is not possible!")

            ops.add_pre(self.builder.create_pad_operator_before(t_op, 0, explicit_padding))

        elif t_op.builtin_options.padding == Padding.SAME:
            # SAME padding is used. TFLite doesn't include the padding 0s in the computation.

            if attrs.count_include_pad == 1:
                # Add the padding via a `Pad` operator, so that the 0s are taking part in the computation.

                x = t_op.tmp_inputs[0]
                y = t_op.tmp_outputs[0]

                # Calculate the required padding.
                padding, offset = translator.tflite_compute_padding_with_offset(x.shape.vector, attrs.kernel_shape,
                                                                                y.shape.vector, attrs.strides,
                                                                                attrs.dilations)
                start_padding = padding
                end_padding = [p + o for p, o in zip(padding, offset, strict=False)]
                onnx_padding = start_padding + end_padding
                tflite_padding = translator.onnx_pads_to_tflite_explicit_padding(onnx_padding)

                # Set the `AveragePool` padding to VALID (no padding) and prepend a `Pad` operator.
                t_op.builtin_options.padding = Padding.VALID

                np_type = translator.tf_lite_type_to_numpy(x.type)
                ops.add_pre(self.builder.create_pad_operator_before(t_op, 0, tflite_padding, np.array([0], np_type)))

        return ops.flatten()

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> [tflite_model.Operator]:
        """Convert the ONNX 'AveragePool' operator to TFLite 'AveragePool2D' and 'Reshape' operators."""
        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(t_op.tmp_inputs[0].type)
        elif t_op.is_quantized_without_qdq():
            # ONNX doesn't support (U)INT8. Leave this check in case the support is added in the future.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX `AveragePool` with a quantized input is not supported.")

        attrs = cast(onnx_average_pool_attributes.AveragePool, node.attributes)

        kernel_rank = len(attrs.kernel_shape)
        if kernel_rank == 1:
            return self._convert_1d_average_pool(node, t_op)

        if kernel_rank == 2:
            return self._convert_2d_average_pool(node, t_op)

        num_ones = attrs.kernel_shape.count(1)
        if kernel_rank - num_ones <= 2:
            # TODO Enough dimensions are '1', so the input can be reshaped to 4D and a AveragePool2D can be applied.
            #  Not sure if this is a realistic scenario and worth putting time into.
            logger.e(logger.Code.NOT_IMPLEMENTED, f"Conversion of ONNX AveragePool with kernel shape "
                                                  f"'{attrs.kernel_shape}' is not yet implemented.")

        else:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"Conversion of ONNX AveragePool with kernel shape '{attrs.kernel_shape}' is not possible!")
