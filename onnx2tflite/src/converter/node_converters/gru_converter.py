#
# Copyright 2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList, try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import (
    quantize_static_float_tensor,
    set_quantization_parameters_to_tensor,
)
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import gru_attributes
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.custom_options.onnx_gru_options import Activation, Direction, OnnxGRU


class GRUConverter(NodeConverter):
    node = "GRU"

    # noinspection PyMethodMayBeStatic
    def check_quantization(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator):
        if not t_op.is_qdq_quantized():
            logger.e(logger.Code.NOT_IMPLEMENTED, "Only quantized version of GRU is currently implemented.")

        error_message_base = ("GRU wasn't quantized by onnx2quant but by some third party quantizer, because {reason}. "
                              "This is not supported yet.")

        x = t_op.tmp_inputs[0]
        w = t_op.tmp_inputs[1]
        r = t_op.tmp_inputs[2]

        y = t_op.tmp_outputs[0]
        y_h = t_op.tmp_outputs[1]

        if x.quantization is None:
            message = error_message_base.format(reason="input 'x' is not quantized")
            logger.e(logger.Code.NOT_IMPLEMENTED, message)
        if w.quantization is None:
            message = error_message_base.format(reason="input 'w' is not quantized")
            logger.e(logger.Code.NOT_IMPLEMENTED, message)
        if r.quantization is None:
            message = error_message_base.format(reason="input 'r' is not quantized")
            logger.e(logger.Code.NOT_IMPLEMENTED, message)

        if (bias := try_get_input(t_op, 3)) is not None:
            if bias.quantization is not None:
                message = error_message_base.format(reason="input 'b' is quantized")
                logger.e(logger.Code.NOT_IMPLEMENTED, message)

        if (initial_h := try_get_input(t_op, 5)) is not None:
            if initial_h.quantization is not None:
                message = error_message_base.format(reason="input 'initial_h' is quantized")
                logger.e(logger.Code.NOT_IMPLEMENTED, message)

        assert y.quantization is not None

        if y_h.quantization is not None:
            message = error_message_base.format(reason="output 'y_h' is quantized")
            logger.e(logger.Code.NOT_IMPLEMENTED, message)

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        attrs = cast(gru_attributes.GRU, node.attributes)

        if attrs.direction not in [Direction.forward.value, Direction.bidirectional.value]:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Only 'forward'/'bidirectional' value of GRU attribute "
                                                  "'direction' is currently implemented.")

        if attrs.activation_alpha is not None:
            logger.e(logger.Code.NOT_IMPLEMENTED, "GRU attribute 'activation_alpha' isn't implemented yet.")

        if attrs.activation_beta is not None:
            logger.e(logger.Code.NOT_IMPLEMENTED, "GRU attribute 'activation_beta' isn't implemented yet.")

        if attrs.layout != 0:
            logger.e(logger.Code.NOT_IMPLEMENTED, "GRU operator currently supports only layout=0. Other layout "
                                                  "values are not implemented yet.")
        if not t_op.is_qdq_quantized():
            logger.e(logger.Code.NOT_IMPLEMENTED, "Only quantized version of GRU is currently implemented.")

        self.check_quantization(node, t_op)

        x = t_op.tmp_inputs[0]
        w = t_op.tmp_inputs[1]
        r = t_op.tmp_inputs[2]

        y = t_op.tmp_outputs[0]
        y_h = t_op.tmp_outputs[1]

        if self.inspector.get_ops_with_input_tensor(y_h.name) or self.inspector.is_output_of_model(y_h.name):
            logger.e(logger.Code.NOT_IMPLEMENTED, "GRU operator with output tensor 'y_h' consumed by other "
                                                  "operators or used as an output of the model is currently not "
                                                  "implemented.")

        if not tensor_has_data(w):
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX GRU with a dynamic weight tensor 'w' is "
                                                  "not supported yet.")

        if not tensor_has_data(r):
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX GRU with a dynamic weight tensor 'r' is "
                                                  "not supported yet.")

        if (bias := try_get_input(t_op, 3)) is None:
            bias = self.builder.create_null_tensor()
        else:
            if not tensor_has_data(bias):
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE, "Conversion of ONNX GRU with a dynamic bias tensor is "
                                                            "not supported.")

            scale_w_b = np.array(w.quantization.scale.vector) * x.quantization.scale.vector[0]
            scale_w_r = np.array(r.quantization.scale.vector) * y.quantization.scale.vector[0]

            scale = np.concat((scale_w_b, scale_w_r))
            zp = np.concat((w.quantization.zero_point.vector, r.quantization.zero_point.vector))
            bias = quantize_static_float_tensor(self.builder, bias, TensorType.INT32, scale, zp, quantized_dimension=1)

        if (sequence_length := try_get_input(t_op, 4)) is None:
            sequence_length = self.builder.create_null_tensor()
        else:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX GRU with 'sequence_length' tensor is "
                                                  "not supported yet.")

        if (initial_h := try_get_input(t_op, 5)) is None:
            initial_h = self.builder.create_null_tensor()
        else:
            scale = y.quantization.scale.vector
            zp = y.quantization.zero_point.vector

            initial_h = quantize_static_float_tensor(self.builder, initial_h, y.type, scale, zp)

        hidden_size = r.shape[-1]

        t_op.custom_options = OnnxGRU(
            hidden_size,
            clip=attrs.clip,
            activations=tuple([Activation(act) for act in attrs.activations]),
            direction=Direction(attrs.direction),
            linear_before_reset=attrs.linear_before_reset,
        )

        t_op.tmp_inputs = [x, w, r, bias, sequence_length, initial_h]

        y_h_scale = np.array(y.quantization.scale.vector).astype(np.float32)
        y_h_zp = np.array(y.quantization.zero_point.vector)
        set_quantization_parameters_to_tensor(y_h, y_h_scale, y_h_zp)
        y_h.type = y.type

        ops = OpsList(middle_op=t_op)

        if x.tensor_format.is_channels_last():
            # Prepend a `Transpose` operator to make the 'x' input `channels_first`.
            reverse_perm = translator.create_channels_last_to_channels_first_permutation(x.rank, return_list=True)
            ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, reverse_perm))
            t_op.tmp_inputs[0].tensor_format = TensorFormat.FORMATLESS

        if y.tensor_format.is_channels_last():
            # Append a `Transpose` operator to make the 'y' output `channels_first`.
            reverse_perm = translator.create_channels_first_to_channels_last_permutation(y.rank, return_list=True)
            ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, reverse_perm))
            t_op.tmp_outputs[0].tensor_format = TensorFormat.FORMATLESS

        return ops.flatten()
