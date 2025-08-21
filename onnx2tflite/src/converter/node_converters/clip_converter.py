#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import try_get_input
from onnx2tflite.src.converter.conversion.translator import (
    get_max_value_for_type,
    get_min_value_for_type,
    tf_lite_type_to_numpy,
)
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import (
    propagate_quantization,
    quantize_static_float_tensor,
    set_quantization_parameters_to_tensor,
)
from onnx2tflite.src.converter.tensor_utils import all_tensors_are_static, tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import clip_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import maximum_options, minimum_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


# noinspection PyMethodMayBeStatic
class ClipConverter(NodeConverter):
    node = "Clip"
    onnx_supported_types = FLOATS + INTS + UINTS
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8] + INTS
    verified_types = [TensorType.FLOAT32, TensorType.UINT8] + INTS

    def _return_as_activation_function(self, t_op: tflite_model.Operator,
                                       builtin_operator: BuiltinOperator) -> tflite_model.Operator:
        """Turn 't_op' into a TFLite operator identified by 'builtin_operator'.

        :param t_op: TFLite operator, that will be turned into 'builtin_operator'.
        :param builtin_operator: BuiltinOperator enum value, which specifies the resulting operator.
        :return: The resulting TFLite operator.
        """
        t_op.builtin_options = None
        t_op.opcode_index = self.builder.op_code_index_for_op_type(builtin_operator)
        t_op.tmp_inputs = [t_op.tmp_inputs[0]]  # Remove the extra inputs
        return t_op

    def _try_convert_as_relu(self, min_tensor: tflite_model.Tensor, max_tensor: tflite_model.Tensor,
                             t_op: tflite_model.Operator) -> list[tflite_model.Operator] | None:
        """Try to convert the 'Clip' into a Relu type operator. If successful, return a list of TFLite operators to add to
             the model. Otherwise, return None.

        :param min_tensor: TFLite tensor which is the second input of the ONNX Clip operator.
        :param max_tensor: TFLite tensor which is the third input of the ONNX Clip operator.
        :param t_op: TFLite operator with IO corresponding to the ONNX Clip operator.
        :return: A list of resulting TFLite operators, or None if the conversion into Relu is not possible.
        """
        if not all_tensors_are_static(min_tensor, max_tensor):
            return None

        min_val = min_tensor.tmp_buffer.data.item()
        max_val = max_tensor.tmp_buffer.data.item()

        x = t_op.tmp_inputs[0]

        if x.quantization is not None:
            # ONNX Clip clips based on the 'raw' int values in the tensors. TFLite activation functions clip based on
            #  the actual quantized values. Therefore, this optimization is only possible if the zero point is 0 and
            #  scale is 1.

            if x.quantization.is_per_channel():
                # Only per tensor quantized Clip can be optimized as Relu, if the clipping values match the single
                #  quantization parameters.
                return None

            # Compute the values 0, 1, -1 and 6 represented using the particular quantization parameters.
            scale = x.quantization.scale.get(0)
            zero = x.quantization.zero_point.get(0)
            one = round((1. / scale) + zero)
            n_one = round((-1. / scale) + zero)
            six = round((6. / scale) + zero)

        else:
            # The 'Clip' is not quantized.
            if x.type == TensorType.UINT8:
                # For some reason, TFLite Relu with UINT8 outputs all 0. Convert as Maximum + Minimum.
                return None

            zero = 0.
            one = 1.
            n_one = -1.
            six = 6.

        upper_limit = get_max_value_for_type(tf_lite_type_to_numpy(x.type))

        if min_val == zero and max_val == upper_limit:
            # Optimization as basic Relu is not exactly equivalent. In ONNX, this would be a 'Clip' with just 2 inputs
            #  (without the max). If an input is omitted, ORT uses the minimum and maximum values for the given type.
            #  https://github.com/microsoft/onnxruntime/blob/d00adb7989635a4046d896fab6358ad4c7b695db/onnxruntime/core/providers/cpu/math/clip.cc#L104-L105
            # This means that in case of float32, +inf will be turned to 3.4028237e38. If the 'Clip' is used to remove
            #  the +inf, conversion into 'Relu' would not be accurate.
            # For now, convert into Relu anyway.
            if x.type == TensorType.FLOAT32:
                logger.i(
                    "Converting 'Clip' operator into Relu. This may cause some +inf values to stay unchanged, whereas"
                    " in ONNX, they would be clipped to 3.4028237e38.")

            return [self._return_as_activation_function(t_op, BuiltinOperator.RELU)]

        if min_val == zero and max_val == six:
            return [self._return_as_activation_function(t_op, BuiltinOperator.RELU6)]

        if min_val == n_one and max_val == one:
            return [self._return_as_activation_function(t_op, BuiltinOperator.RELU_N1_TO_1)]

        if min_val == zero and max_val == one:
            return [self._return_as_activation_function(t_op, BuiltinOperator.RELU_0_TO_1)]

        return None

    def _quantize_min_max_tensors(self, x, min_tensor, max_tensor) -> (tflite_model.Tensor, tflite_model.Tensor):
        """Return the quantized `min_tensor` and `max_tensor`."""
        if x.quantization is not None:
            # Check type of min/max tensors and quantize them if necessary
            scale = x.quantization.scale.vector
            zp = x.quantization.zero_point.vector

            if min_tensor.quantization is None:
                if min_tensor.type == x.type:
                    # QDQ model with undefined min tensor
                    set_quantization_parameters_to_tensor(min_tensor, np.array(scale).astype(np.float32), np.array(zp))
                elif tensor_has_data(min_tensor):
                    # QDQ model with static min tensor
                    min_tensor = quantize_static_float_tensor(self.builder, min_tensor, x.type, scale, zp)
                else:
                    logger.e(logger.Code.NOT_IMPLEMENTED,
                             "ONNX operator Clip with dynamic 'min' tensor and quantized by non-internal "
                             "quantizer is currently not supported. Quantize model with internal quantizer.")

            if max_tensor.quantization is None:
                if max_tensor.type == x.type:
                    # QDQ model with undefined max tensor
                    set_quantization_parameters_to_tensor(
                        max_tensor, np.array(scale).astype(np.float32), np.array(zp))
                elif tensor_has_data(max_tensor):
                    # QDQ model with static max tensor
                    max_tensor = quantize_static_float_tensor(self.builder, max_tensor, x.type, scale, zp)
                else:
                    logger.e(logger.Code.NOT_IMPLEMENTED,
                             "ONNX operator Clip with dynamic 'max' tensor and quantized by non-internal "
                             "quantizer is currently not supported. Quantize model with internal quantizer.")

        return min_tensor, max_tensor

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX Clip operator into TFLite.

        There is no 'Clip' in TFLite. If the 'Clip' is representing an activation function such as Relu6..., we can
         convert it directly to that. Otherwise, we chain the Maximum and Minimum operators together. Maximum outputs
         the larger value from its 2 inputs (effectively clipping to a lower limit). Minimum outputs the smaller value
         from its 2 inputs (effectively clipping to an upper limit).
        """
        o_clip = cast(clip_attributes.Clip, node.attributes)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        is_qdq_model = False

        if x.quantization is not None and y.quantization is None:
            # Non-QDQ model
            propagate_quantization(x, y)
        elif x.quantization is not None and y.quantization is not None:
            is_qdq_model = True

        self.assert_type_allowed(x.type)

        np_type = translator.tf_lite_type_to_numpy(x.type)

        # Get the minimum and maximum clipping values as tensors.
        if node.version < 11:
            # V6 -> uses attributes
            if len(t_op.tmp_inputs) != 1:
                logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX Clip v6 has invalid number of inputs.")

            min_tensor = self.builder.create_tensor_for_data(np.array([o_clip.min], np.float32), "min")
            max_tensor = self.builder.create_tensor_for_data(np.array([o_clip.max], np.float32), "max")

        else:
            # V11+ -> uses input tensors
            if not (1 <= len(t_op.tmp_inputs) <= 3):
                logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX Clip v11+ has invalid number of inputs.")

            if (min_tensor := try_get_input(t_op, 1)) is None:
                min_tensor = self.builder.create_tensor_for_data(np.array([get_min_value_for_type(np_type)], np_type),
                                                                 "min")

            if (max_tensor := try_get_input(t_op, 2)) is None:
                max_tensor = self.builder.create_tensor_for_data(np.array([get_max_value_for_type(np_type)], np_type),
                                                                 "max")

            # All tensors should have the same type if it's not QDQ model (min/max tensors can be float)
            if not is_qdq_model and not all(t.type == x.type for t in [x, min_tensor, max_tensor]):
                logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX Clip uses multiple types.")

        # -- min_tensor and max_tensor are initialized, and may or may not contain static data. --

        min_tensor, max_tensor = self._quantize_min_max_tensors(x, min_tensor, max_tensor)
        if t_op.is_quantized_without_qdq():
            # Propagate the quantization parameters, in case they are ever useful in the future.
            propagate_quantization(x, y)

        # Try to convert the 'Clip' into a Relu activation function.
        if ops_to_add := self._try_convert_as_relu(min_tensor, max_tensor, t_op):
            # Success
            return ops_to_add

        # -- Couldn't convert to an activation function. Convert into a Minimum, Maximum pair. --

        # This order (Maximum -> Minimum) mimics the internal behavior of ONNX Runtime. Therefore, if the clipping
        #  values are 'invalid' (e.g. Clip(min=10, max=5)), the output will be the same after conversion. (ORT supports
        #  this, and so does the converted TFLite model).

        ops_to_add = []

        if not (tensor_has_data(min_tensor) and min_tensor.tmp_buffer.data.item() == get_min_value_for_type(np_type)):
            # A Maximum operator needs to be added.
            max_op = tflite_model.Operator(builtin_options=maximum_options.Maximum())
            max_op.tmp_inputs = [x, min_tensor]
            max_op.tmp_outputs = [y]

            ops_to_add.append(max_op)

        if not (tensor_has_data(max_tensor) and max_tensor.tmp_buffer.data.item() == get_max_value_for_type(np_type)):
            # A Minimum operator needs to be added.
            min_op = tflite_model.Operator(builtin_options=minimum_options.Minimum())
            if len(ops_to_add) == 1:
                # A Maximum op will come before the Minimum -> create an in-between tensor.
                input_tensor = self.builder.duplicate_tensor(x)
                ops_to_add[0].tmp_outputs[0] = input_tensor

            else:
                input_tensor = x

            min_op.tmp_inputs = [input_tensor, max_tensor]
            min_op.tmp_outputs = [y]

            ops_to_add.append(min_op)

        if x.quantization is not None and x.quantization != y.quantization and len(ops_to_add) > 0:
            # I/O q-params doesn't match -> external quantizer was used or removed Clip/Relu modified q-params.
            # We need to re-quantize output because Minimum/Maximum expects shared q-params for input and output.
            logger.w("Requantizing output of Clip operator. Internal quantizer can potentially avoid this.")
            scale = x.quantization.scale.vector
            zp = x.quantization.zero_point.vector
            ops_to_add.append(self.builder.create_quantize_operator_after(ops_to_add[-1], 0, x.type, scale, zp))

        if len(ops_to_add) == 0:
            # The clip is doing nothing.

            if self.builder.operator_can_be_skipped(t_op, self.inspector):
                self.builder.redirect_tensor(y, x)

            else:
                t_op.tmp_inputs = [x]  # Specify main input.
                self.builder.turn_operator_to_identity(t_op)
                ops_to_add = [t_op]

        return ops_to_add
