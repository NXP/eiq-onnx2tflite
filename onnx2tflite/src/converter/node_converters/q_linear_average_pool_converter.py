#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.Padding import Padding
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.builder import model_builder
from onnx2tflite.src.converter.conversion import common, translator
from onnx2tflite.src.converter.conversion.common import try_get_input
from onnx2tflite.src.converter.conversion.translator import tf_lite_type_to_numpy
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import (
    calculate_uint_to_int_re_quantization_zero_point,
    is_per_channel_quantized,
    set_quantization_parameters_to_tensor,
)
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import q_linear_average_pool_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import average_pool_2d_options as tfl_average_pool_2d_options
from onnx2tflite.src.tflite_generator.builtin_options import reshape_options as tfl_reshape_options
from onnx2tflite.src.tflite_generator.meta.types import name_for_type


# noinspection PyMethodMayBeStatic
class QLinearAveragePoolConverter(NodeConverter):
    node = "QLinearAveragePool"

    def _check_ceil_mode(self, q_ap_attributes: q_linear_average_pool_attributes.QLinearAveragePool) -> None:
        """Check if the ONNX QLinearAveragePool is convertible according to its 'ceil_mode' attribute. If not, exit
             with appropriate error message.

        :param q_ap_attributes: Attributes of the ONNX QLinearAveragePool operator.
        """
        if q_ap_attributes.ceil_mode == 1:
            # TFLite always uses 'floor' to round, so the output shape may be different from ONNX.
            # TODO Calculate the output shape, to see if this attribute even makes a difference.
            # TODO Prepending a 'Pad' operator should work, but only if 'count_include_pad' is 1.
            if q_ap_attributes.count_include_pad == 1:
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX QLinearAveragePool with 'ceil_mode' = 1 is not yet supported.")

            else:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX QLinearAveragePool with 'ceil_mode' = 1 and "
                         "'count_include_pad' = 0 is not possible.")

    def _handle_padding(self, t_op: tflite_model.Operator,
                        q_ap_attributes: q_linear_average_pool_attributes.QLinearAveragePool,
                        builder: model_builder, input_zero_point: [int], input_type: TensorType,
                        ops_to_add: list[tflite_model.Operator]) -> None:
        """Convert the padding of the ONNX QLinearAveragePool operator according to its attributes. Insert any extra
             necessary operators into the 'ops_to_add' list. If the padding cannot be converted, exit with appropriate
             error.

        :param t_op: TFLite AveragePool operator.
        :param q_ap_attributes: Attributes of the ONNX QLinearAveragePool operator.
        :param builder: ModelBuilder object.
        :param input_zero_point: Zero point quantization parameter of the input tensor.
        :param input_type: Data type of the input tensor.
        :param ops_to_add: A list of operators, which will be added to the model. This function may only add to it.
        """
        t_op.builtin_options.padding, explicit_padding = translator.convert_padding(
            q_ap_attributes.auto_pad, q_ap_attributes.pads, t_op.tmp_inputs[0].shape.vector,
            t_op.tmp_outputs[0].shape.vector, q_ap_attributes.kernel_shape, q_ap_attributes.strides
        )
        if explicit_padding is not None:
            # Need to prepend a 'Pad' operator, which adds 0s. But these will be included in the computation!
            if q_ap_attributes.count_include_pad == 0:
                # The 0s must NOT be included in the computation of the average value.
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX QLinearAveragePool with 'count_include_pad' = 0 and a specific combination"
                         " of input shape, 'kernel_shape', 'strides' and padding is not possible!")

            # Pad with '0' for the input type and quantization. Earlier check ensures per-tensor quantization.
            zero = self._get_zero_for_quantization(input_zero_point[0], input_type)
            ops_to_add.insert(0, builder.create_pad_operator_before(t_op, 0, explicit_padding, zero))

        elif t_op.builtin_options.padding == Padding.SAME:
            # SAME padding is used. TFLite doesn't include the padding 0s in the computation.
            if q_ap_attributes.count_include_pad == 1:
                # TODO ONNX includes the 0s. We can prepend a 'Pad' operator to add the padding explicitly and use
                #  'VALID'. Seems like an extremely rare case, so not sure if it is worth putting time into right now.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX QLinearAveragePool with 'count_include_pad' == 1 is not yet supported.")

    def _get_zero_for_quantization(self, zp: int, data_type: TensorType) -> np.ndarray:
        """Get a value for 0 given the quantization parameters and type.

        :param zp: Zero point quantization parameter.
        :param data_type: TFLite data type of the 'zero'.
        :return: A numpy array holding one 'zero' value.
        """
        if type(zp) is not int:
            logger.e(logger.Code.INTERNAL_ERROR,
                     "convert_q_linear_average_pool._get_zero_for_quantization_parameters(): "
                     "Zero point must be a scalar.")

        np_type = translator.tf_lite_type_to_numpy(data_type)
        return np.array([zp], np_type)

    def _convert_1d_q_linear_average_pool(self, q_ap_attributes: q_linear_average_pool_attributes.QLinearAveragePool,
                                          t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'QLinearAveragePool' operator with a 1D kernel to TFLite 'AveragePool2D'.
         TFLite doesn't support 1D AveragePool, but this behaviour can be represented using
                Reshape -> AveragePool2D -> Reshape.
         The first reshape introduces a 4th dimension with size 1. The second Reshape removes the temporary
          dimension.

        :param q_ap_attributes: Attributes of the ONNX QLinearAveragePool operator.
        :param t_op: A TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        if not t_op.tmp_inputs[0].shape.is_well_defined():
            # Dynamic shapes make it difficult to use the Reshape operators.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of 1D ONNX QLinearAveragePool with a dynamic shape is not yet supported.")

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
        t_op.tmp_inputs[0] = reshape1_output
        t_op.tmp_outputs = [reshape2_input]

        # Extend all ONNX attributes to 2D
        common.extend_1d_pads_to_2d(q_ap_attributes.pads)
        common.extend_1d_strides_to_2d(q_ap_attributes.strides)
        common.extend_1d_kernel_shape_to_2d(q_ap_attributes.kernel_shape)

        # Convert the now 2D AveragePool
        converted_average_pool_ops = self._convert_2d_q_linear_average_pool(q_ap_attributes, t_op)

        return [reshape1] + converted_average_pool_ops + [reshape2]

    def _convert_2d_q_linear_average_pool(self, q_ap_attributes: q_linear_average_pool_attributes.QLinearAveragePool,
                                          t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'QLinearAveragePool' operator with a 2D kernel to TFLite 'AveragePool2D'.

        :param q_ap_attributes: Attributes of the ONNX QLinearAveragePool operator.
        :param t_op: A TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        self._check_ceil_mode(q_ap_attributes)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        ops_to_add = [t_op]

        x_zp = x.quantization.zero_point.vector
        x_scale = x.quantization.scale.vector
        y_zp = y.quantization.zero_point.vector

        # Make sure the input and output tensors are INT8.
        if x.type == TensorType.UINT8:
            # Prepend a Quantize operators to make the input and output INT8.
            x_zp = calculate_uint_to_int_re_quantization_zero_point(1, x_zp)
            q_op = self.builder.create_quantize_operator_before(t_op, 0, TensorType.INT8, None, [x_zp.item()])
            ops_to_add.insert(0, q_op)

            y_zp = calculate_uint_to_int_re_quantization_zero_point(1, y_zp)
            q_op = self.builder.create_quantize_operator_after(t_op, 0, TensorType.INT8, None, [y_zp.item()])
            ops_to_add.append(q_op)

        if x.quantization != y.quantization:
            # TFLite requires the same quantization for the input and output tensor. Append a Quantize operator after
            #  to ensure this.
            q_op = self.builder.create_quantize_operator_after(t_op, 0, t_op.tmp_outputs[0].type, x_scale, x_zp)
            index_of_top = ops_to_add.index(t_op)
            ops_to_add.insert(index_of_top + 1, q_op)

        t_op.builtin_options = tfl_average_pool_2d_options.AveragePool2D()

        # Remove the extra inputs that TFLite doesn't use.
        t_op.tmp_inputs = [t_op.tmp_inputs[0]]

        common.assign_2d_strides(t_op.builtin_options, q_ap_attributes.strides)

        t_op.builtin_options.filter_h = q_ap_attributes.kernel_shape[0]
        t_op.builtin_options.filter_w = q_ap_attributes.kernel_shape[1]

        # Convert the padding
        self._handle_padding(t_op, q_ap_attributes, self.builder, x_zp, x.type, ops_to_add)

        return ops_to_add

    def _assign_q_params_to_tensors(self, t_op) -> None:
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        x_scale_tensor = t_op.tmp_inputs[1]
        x_zp_tensor = try_get_input(t_op, 2)
        y_scale_tensor = t_op.tmp_inputs[3]
        y_zp_tensor = try_get_input(t_op, 4)

        # Check for dynamic quantization parameters.
        for param_tensor in [x_scale_tensor, x_zp_tensor, y_scale_tensor, y_zp_tensor]:
            if param_tensor is not None and not tensor_has_data(param_tensor):
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX QLinearAveragePool with dynamic quantization parameters is not possible.")

        numpy_zp_type = tf_lite_type_to_numpy(x.type)

        # Quantization parameters
        x_scale = x_scale_tensor.tmp_buffer.data
        x_zp = x_zp_tensor.tmp_buffer.data if x_zp_tensor is not None else np.zeros(x_scale.shape, numpy_zp_type)
        y_scale = y_scale_tensor.tmp_buffer.data
        y_zp = y_zp_tensor.tmp_buffer.data if y_zp_tensor is not None else np.zeros(y_scale.shape, numpy_zp_type)

        if is_per_channel_quantized(x_scale, x_zp):
            # TFLite quantized AveragePool doesn't support per-channel quantization.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX QLinearAveragePool with per-channel quantization is not possible.")

        # Assign the quantization parameters to the tensors.
        set_quantization_parameters_to_tensor(x, x_scale, x_zp)
        set_quantization_parameters_to_tensor(y, y_scale, y_zp)

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX Runtime 'QLinearAveragePool' operator to TFLite 'AveragePool2D' and potential 'Reshape'
             operators.

        :param node: ONNX Runtime QLinearAveragePool operator.
        :param t_op: A TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        if not (4 <= len(t_op.tmp_inputs) <= 5):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"ONNX QLinearAveragePool has {len(t_op.tmp_inputs)} inputs, where 4 or 5 was expected.")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        q_ap_attributes = cast(q_linear_average_pool_attributes.QLinearAveragePool, node.attributes)

        # Input and output types must be the same
        if not (x.type == y.type):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     "ONNX QLinearAveragePool has input and output tensors with different data type!")

        # ONNX only supports UINT8 and INT8. (onnxruntime/contrib_ops/cpu/quantization/qlinear_pool.cc#L507-L513)
        if x.type not in {TensorType.INT8, TensorType.UINT8}:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"ONNX QLinearAveragePool only supports INT8 and UINT8 data types. Got '{name_for_type(x.type)}'.")

        self._assign_q_params_to_tensors(t_op)

        kernel_rank = len(q_ap_attributes.kernel_shape)
        if kernel_rank == 1:
            return self._convert_1d_q_linear_average_pool(q_ap_attributes, t_op)

        if kernel_rank == 2:
            return self._convert_2d_q_linear_average_pool(q_ap_attributes, t_op)

        num_ones = q_ap_attributes.kernel_shape.count(1)
        if kernel_rank - num_ones <= 2:
            # TODO Enough dimensions are '1', so the input can be reshaped to 4D and a AveragePool2D can be applied.
            #  Not sure if this is a realistic scenario and worth putting time into.
            logger.e(logger.Code.NOT_IMPLEMENTED, f"Conversion of ONNX QLinearAveragePool with kernel shape "
                                                  f"'{q_ap_attributes.kernel_shape}' is not yet implemented.")

        else:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"Conversion of ONNX QLinearAveragePool with kernel shape `{q_ap_attributes.kernel_shape}` is "
                     "not possible!")
