#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from typing import cast

import numpy as np

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.builtin_attributes.max_pool_attributes as onnx_max_pool_attributes
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.converter.conversion import translator, common
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator.builtin_options import (
    max_pool_2d_options as tfl_max_pool_2d_options,
    reshape_options as tfl_reshape_options
)
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class MaxPoolConverter(NodeConverter):
    node = 'MaxPool'

    onnx_supported_types = FLOATS + [TensorType.UINT8, TensorType.INT8]
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/pooling.cc#L420-L440
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT16]
    verified_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8]

    def _convert_1d_max_pool(self, o_mp: onnx_max_pool_attributes.MaxPool, t_op: tflite_model.Operator) -> OpsList:
        """ Convert the ONNX 'MaxPool' operator with a 1D kernel to TFLite 'MaxPool2D'.
             TFLite doesn't support 1D MaxPool, but this behaviour can be represented using
                    Reshape -> MaxPool2D -> Reshape.
             The first reshape introduces a 4th dimension with size 1. The second Reshape removes the temporary
              dimension.

        """

        for dim in t_op.tmp_inputs[0].shape.vector:
            if (not isinstance(dim, int)) or dim < 0:
                # Dynamic shapes make it difficult to use the Reshape operators.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of 1D ONNX MaxPool with a dynamic shape is not yet supported.")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if t_op.is_quantized_without_qdq():
            # Non-QDQ model -> just propagate
            propagate_quantization(x, y)

        # Calculate the shapes for equivalent 2D MaxPool
        reshape_pre_output_shape = translator.nhc_dimensions_to_nhwc(t_op.tmp_inputs[0].shape.vector)
        reshape_post_input_shape = translator.nhc_dimensions_to_nhwc(t_op.tmp_outputs[0].shape.vector)

        final_output_shape = t_op.tmp_outputs[0].shape.vector.copy()

        # Generate tensors taking part in the conversion
        reshape_pre_input = t_op.tmp_inputs[0]

        reshape_pre_output = self.builder.duplicate_tensor(reshape_pre_input, name_suffix="_4D_")
        reshape_pre_output.shape = tflite_model.Shape(reshape_pre_output_shape)

        reshape_post_input = self.builder.duplicate_tensor(t_op.tmp_outputs[0], name_suffix="_4D_")
        reshape_post_input.shape = tflite_model.Shape(reshape_post_input_shape)

        reshape_post_output = t_op.tmp_outputs[0]

        # ----------------------- Create the new operators -----------------------
        reshape_pre = tflite_model.Operator(builtin_options=tfl_reshape_options.Reshape(reshape_pre_output_shape))
        reshape_pre.tmp_inputs = [reshape_pre_input]
        reshape_pre.tmp_outputs = [reshape_pre_output]

        reshape_post = tflite_model.Operator(builtin_options=tfl_reshape_options.Reshape(final_output_shape))
        reshape_post.tmp_inputs = [reshape_post_input]
        reshape_post.tmp_outputs = [reshape_post_output]

        # Assign the new input and output of the MaxPool
        t_op.tmp_inputs = [reshape_pre_output]
        t_op.tmp_outputs = [reshape_post_input]

        # Extend all ONNX attributes to 2D
        common.extend_1d_dilations_to_2d(o_mp.dilations)
        common.extend_1d_pads_to_2d(o_mp.pads)
        common.extend_1d_strides_to_2d(o_mp.strides)
        common.extend_1d_kernel_shape_to_2d(o_mp.kernel_shape)

        # Convert the now 2D MaxPool
        converted_max_pool_ops = self._convert_2d_max_pool(o_mp, t_op)
        converted_max_pool_ops.pre_ops.insert(0, reshape_pre)
        converted_max_pool_ops.add_post(reshape_post)

        return converted_max_pool_ops

    # noinspection PyMethodMayBeStatic
    def _get_pad_constant_value(self, input_type: TensorType) -> np.ndarray:
        """
        Get scalar NumPy array with constant value used as constant value for 'Pad' operator.

        :param input_type: Input tensor type.
        :return: Scalar array with single minimum value of given type.
        """

        match input_type:
            case TensorType.INT8:
                return np.asarray([np.iinfo(np.int8).min], dtype=np.int8)
            case TensorType.UINT8:
                return np.asarray([np.iinfo(np.uint8).min], dtype=np.uint8)
            case TensorType.FLOAT32:
                return np.asarray([np.finfo(np.float32).min], dtype=np.float32)
            case _:
                logger.e(logger.Code.INVALID_TYPE, f"Unexpected input type for MaxPool operator.")

    def _convert_2d_max_pool(self, o_mp: onnx_max_pool_attributes.MaxPool, t_op: tflite_model.Operator) -> OpsList:
        """ Convert the ONNX 'MaxPool' operator with a 2D kernel to TFLite 'MaxPool2D'. """

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if o_mp.dilations is not None:
            if any(dilation != 1 for dilation in o_mp.dilations):
                # TFLite MaxPool2D doesn't support dilations.
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         f"MaxPool dilations '{o_mp.dilations}' cannot be converted to TFLite!")

        if o_mp.ceil_mode == 1:
            # TFLite always uses 'floor' to round, so the output shape may be different from ONNX.
            # Conversion should be possible by inserting a 'Pad' operator before the MaxPool.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX MaxPool with 'ceil_mode' == 1 is not yet supported.")

        if len(t_op.tmp_outputs) == 2:
            # The 'Indices' tensor is the second output. TFLite doesn't provide such functionality.
            # Right now, there is no simple way to check if the second output is actually used by other operators.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX MaxPool with a second 'Indices' output tensor is not possible.")

        ops = OpsList(middle_op=t_op)

        if t_op.is_quantized_without_qdq():
            # Non-QDQ model -> just propagate
            propagate_quantization(x, y)

        elif t_op.is_qdq_quantized() and x.quantization != y.quantization:
            # I/O q-params doesn't match -> external quantizer was used. We need to re-quantize output
            # because MaxPool expects shared q-params for input and output.
            logger.w("Requantizing output of MaxPool operator. This can be avoided by using internal quantizer.")
            scale = x.quantization.scale.vector
            zp = x.quantization.zero_point.vector
            ops.add_post(self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp))

        t_op.builtin_options = tfl_max_pool_2d_options.MaxPool2D()

        common.assign_2d_strides(t_op.builtin_options, o_mp.strides)

        t_op.builtin_options.filter_h = o_mp.kernel_shape[0]
        t_op.builtin_options.filter_w = o_mp.kernel_shape[1]

        # Convert the padding
        t_op.builtin_options.padding, explicit_padding = translator.convert_padding(o_mp.auto_pad, o_mp.pads,
                                                                                    x.shape.vector,
                                                                                    y.shape.vector,
                                                                                    o_mp.kernel_shape, o_mp.strides)
        if explicit_padding is not None:
            pad_op = self.builder.create_pad_operator_before(t_op, 0, explicit_padding,
                                                             constant_value=self._get_pad_constant_value(x.type))
            ops.add_pre(pad_op)

        return ops

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX 'MaxPool' operator to TFLite 'MaxPool2D' and 'Reshape' operators. """

        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        attrs = cast(onnx_max_pool_attributes, node.attributes)

        kernel_rank = len(attrs.kernel_shape)
        if kernel_rank == 1:
            return self._convert_1d_max_pool(attrs, t_op).flatten()

        elif kernel_rank == 2:
            return self._convert_2d_max_pool(attrs, t_op).flatten()

        else:
            num_ones = attrs.kernel_shape.count(1)
            if kernel_rank - num_ones <= 2:
                # Enough dimensions are '1', so the input can be reshaped to 4D and a MaxPool2D can be applied.
                # Not sure if this is a realistic scenario and worth putting time into.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         f"Conversion of ONNX MaxPool with kernel shape '{attrs.kernel_shape}'"
                         f" is not yet implemented.")

            else:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         f"Conversion of ONNX MaxPool with kernel shape '{attrs.kernel_shape}' is not possible!")
