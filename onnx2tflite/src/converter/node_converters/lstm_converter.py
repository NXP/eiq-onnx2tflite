#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.ActivationFunctionType import ActivationFunctionType
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converters.shared.recurrent_utils import check_sequence_lens, \
    get_activation_function_for_name, ensure_correct_tensor_formatting
from onnx2tflite.src.converter.conversion.common import try_get_input, OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data, all_tensors_are_static
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import lstm_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import unidirectional_sequence_lstm_options, \
    bidirectional_sequence_lstm_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS

# Dictionary mapping names of activation functions to corresponding TFLite ActivationFunctionType enum values.
act_fun_for_name = {
    'Tanh': ActivationFunctionType.TANH,
    'Relu': ActivationFunctionType.RELU
}


def _get_activation_function_for_name(name: str) -> ActivationFunctionType:
    """ Return the TFLite ActivationFunctionType corresponding to given 'name'. """
    if act_fun := act_fun_for_name.get(name, None):
        return act_fun

    # Couldn't find a corresponding activation function
    logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
             f"Conversion of ONNX LSTM with activation function '{name}' is not possible.")


def _validate_activations(activations: list[str] | None, bidirectional: bool = False) -> list[str]:
    """ Make sure the 'activations' attribute on ONNX LSTM can be converted to TFLite. If not, exit with error.

        When it comes to activation functions inside the LSTM, ONNX allows the user to specify all of them.
         In TFLite, the 'input', 'forget', and 'output' gates always use the 'Sigmoid' activation function.
         https://github.com/tensorflow/tensorflow/blob/a3fb2743707b6d36297dd53fb41b96aa01ae677d/tensorflow/lite/kernels/lstm_eval.cc#L938 (also lines 951 and 979).
        For the 'cell update gate' and for the final output, TFLite uses the 'fused_activation_function' attribute.
         https://github.com/tensorflow/tensorflow/blob/a3fb2743707b6d36297dd53fb41b96aa01ae677d/tensorflow/lite/kernels/lstm_eval.cc#L963C15-L963C25 (also line 984).

    :param activations: The 'activations' attribute of the ONNX LSTM operator, or None.
    :param bidirectional: If True, the 'activations' attribute is meant for bidirectional LSTM. Forward LSTM otherwise.
    :return: A valid 'activations' attribute.
    """

    activations_len = 6 if bidirectional else 3

    if activations is None:
        # Return the default activation functions.
        return ['Sigmoid', 'Tanh', 'Tanh'] * 2 if bidirectional else ['Sigmoid', 'Tanh', 'Tanh']

    if len(activations) != activations_len:
        logger.e(logger.Code.INVALID_ONNX_MODEL,
                 f"ONNX LSTM 'activations' attribute has '{len(activations)}' elements instead of {activations_len}.")

    f_is_sigmoid = activations[0] == 'Sigmoid'
    if bidirectional:
        f_is_sigmoid = f_is_sigmoid and activations[3] == 'Sigmoid'
    if not f_is_sigmoid:
        logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                 f"Conversion of ONNX LSTM is only possible when the activation function 'f' is a Sigmoid function.")

    g_and_h_are_the_same = activations[1] == activations[2]
    if bidirectional:
        g_and_h_are_the_same = g_and_h_are_the_same and activations[4] == activations[5] == activations[1]
    if not g_and_h_are_the_same:
        logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                 "Conversion of ONNX LSTM with different activations functions 'h' and 'g' is not possible.")

    g_h_supported = activations[1] in act_fun_for_name.keys()
    if not g_h_supported:
        logger.e(logger.Code.CONVERSION_IMPOSSIBLE, f"Conversion of ONNX LSTM with '{activations[1]}' as the  'h' "
                                                    f"and 'g' activations functions is not possible.")

    return activations


def _get_clip_values(onnx_clip_attribute: float | None) -> tuple[float, float]:
    """ Convert the 'clip' attribute of the ONNX LSTM to TFLite 'cell_clip' and 'proj_clip' attributes.
        It from reading the documentation, TFLite LSTM kernel implementation and from some experiments, it appears that
         ONNX and TFLite handle the clipping differently. I'm not 100% sure that conversion is not possible, but it
         seems that way.

        TFLite seems to clip the new value for each state.
         https://github.com/tensorflow/tensorflow/blob/a3fb2743707b6d36297dd53fb41b96aa01ae677d/tensorflow/lite/kernels/lstm_eval.cc#L261

        ONNX does the same, and also clips the inputs of all activation functions.
         https://github.com/microsoft/onnxruntime/blob/d7aebf9ea8a4a651088384f219292bae9062439b/onnxruntime/test/providers/cpu/rnn/LSTM.py#L223

    :param onnx_clip_attribute: The 'clip' attribute of the ONNX LSTM operator.
    :return: The 'cell_clip' and 'proj_clip' attributes of the TFLite LSTM operators.
    """

    if onnx_clip_attribute is not None and onnx_clip_attribute != float('inf'):
        # Possibly we could allow the clip to be > 100 or something, if the user just put a large number, instead of
        #  omitting the attribute.
        logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                 "Conversion of ONNX LSTM with a 'clip' attribute is not possible.")

    return float('inf'), float('inf')  # No clipping.


class LSTMConverter(NodeConverter):
    node = 'LSTM'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/unidirectional_sequence_lstm.cc#L1363-L1502
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/bidirectional_sequence_lstm.cc#L1177-L1371
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8]
    verified_types = [TensorType.FLOAT32]

    def _get_1_directional_peephole_weights(self, t_op: tflite_model.Operator) -> \
            tuple[tflite_model.Tensor, tflite_model.Tensor, tflite_model.Tensor]:
        """ Get the peephole to input, forget and output weights from the 't_op' operator.
             The LSTM must be either 'forward' or 'backward'.

        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX LSTM.
        :return: 3 TFLite tensors: peephole_to_input, peephole_to_forget, peephole_to_output.
        """
        if peephole := try_get_input(t_op, 7):
            if not tensor_has_data(peephole):
                # Add a Split operator. (not sure if this would ever happen)
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX LSTM with a dynamic operand 'P' is not yet supported.")

            peephole_to_input_data, peephole_to_output_data, peephole_to_forget_data = np.split(
                peephole.tmp_buffer.data.squeeze(0), 3
            )
            peephole_to_input = self.builder.create_tensor_for_data(peephole_to_input_data,
                                                                    'peephole_to_input')
            peephole_to_forget = self.builder.create_tensor_for_data(peephole_to_forget_data,
                                                                     'peephole_to_forget')
            peephole_to_output = self.builder.create_tensor_for_data(peephole_to_output_data,
                                                                     'peephole_to_output')
        else:
            # No peephole specified -> use null tensors.
            peephole_to_input = self.builder.create_null_tensor('peephole_to_input')
            peephole_to_forget = self.builder.create_null_tensor('peephole_to_forget')
            peephole_to_output = self.builder.create_null_tensor('peephole_to_output')

        return peephole_to_input, peephole_to_forget, peephole_to_output

    def _get_1_directional_biases(self, t_op: tflite_model.Operator, hidden_size: int) -> \
            tuple[tflite_model.Tensor, tflite_model.Tensor, tflite_model.Tensor, tflite_model.Tensor]:
        """ Get the bias tensors from the 't_op' operator representing an ONNX LSTM.
             The LSTM must be either 'forward' or 'backward'.

        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX LSTM.
        :param hidden_size: The 'hidden size' parameter of the LSTM.
        :return: 4 TFLite tensors: input_bias, forget_bias, output_bias, cell_bias.
        """
        if input_bias := try_get_input(t_op, 3):
            # ONNXRT:  ONNX 'B' (bias) has shape [1, hidden_size * 8] and consists of 2 parts.
            #  The first one [1. hidden_size * 4] is Wb, the second one is Rb with the same shape.
            #  TFLite supports biases via the 4 '*_bias' operands of shape [hidden_size].
            # Reading through the ONNX Runtime implementation, I found that Wb and Rb were always used together as a
            #  'fused bias' named 'bias_WR*' of size hidden_size.
            #  https://github.com/microsoft/onnxruntime/blob/d7aebf9ea8a4a651088384f219292bae9062439b/onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc#L186-L189
            # Conversion is possible by simply adding the Wb and Rb together.

            if not tensor_has_data(input_bias):
                # Prepend 'Split', 'Add' and 'Split' operators. (not sure if this would ever happen)
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX LSTM with a dynamic 'B' input tensor is not yet supported.")

            w_b, r_b = np.split(np.squeeze(input_bias.tmp_buffer.data, 0), 2)
            bias = w_b + r_b

            input_bias_data, output_bias_data, forget_bias_data, cell_bias_data = np.split(bias, 4)
            input_bias = self.builder.create_tensor_for_data(input_bias_data, 'input_bias')
            forget_bias = self.builder.create_tensor_for_data(forget_bias_data, 'forget_bias')
            output_bias = self.builder.create_tensor_for_data(output_bias_data, 'output_bias')
            cell_bias = self.builder.create_tensor_for_data(cell_bias_data, 'cell_bias')

        else:
            # No biases specified, set them all to 0. Using null tensors didn't work.
            input_bias = self.builder.create_zeros_tensor([hidden_size], "input_bias",
                                                          np.dtype('float32'), True)
            forget_bias = self.builder.create_zeros_tensor([hidden_size], "forget_bias",
                                                           np.dtype('float32'), True)
            cell_bias = self.builder.create_zeros_tensor([hidden_size], "output_bias",
                                                         np.dtype('float32'), True)
            output_bias = self.builder.create_zeros_tensor([hidden_size], "cell_bias",
                                                           np.dtype('float32'), True)

        return input_bias, forget_bias, cell_bias, output_bias

    def _convert_forward_lstm(self, o_lstm: lstm_attributes.LSTM, t_op: tflite_model.Operator,
                              ops: OpsList) -> list[tflite_model.Operator]:
        """ Convert ONNX LSTM with 'forward' direction attribute to TFLite UnidirectionalSequenceLSTM.

        :param o_lstm: Attributes of the ONNX LSTM operator.
        :param t_op: TFLite operators with inputs and outputs corresponding to the ONNX operator.
        :param ops: OpsList with operators that are the result of the conversion. May already contain some operators.
        :return: A list of TFLite operators to add to the model.
        """

        # Convert to TFLite 'UnidirectionalSequenceLSTM'.
        t_op.builtin_options = unidirectional_sequence_lstm_options.UnidirectionalSequenceLSTM(
            *_get_clip_values(o_lstm.clip), time_major=True)

        o_lstm.activations = _validate_activations(o_lstm.activations)

        # Use the 'fused_activation_function' attribute to represent the output and cell update activation functions.
        t_op.builtin_options.fused_activation_function = _get_activation_function_for_name(o_lstm.activations[1])

        x = t_op.tmp_inputs[0]
        w = t_op.tmp_inputs[1]
        r = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        if not all_tensors_are_static(w, r):
            # Add a 'Split' operator. (not sure if this would ever happen)
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX LSTM with dynamic weights is not yet supported.")

        # Determine the constants from tensor shapes.
        seq_length = x.shape.get(0)
        batch_size = x.shape.get(1)
        # input_size = x.shape.get(2)
        hidden_size = y.shape.get(-1)
        num_directions = y.shape.get(1)

        logger.internal_assert(num_directions == 1, "Forward LSTM: 'num_directions' != 1.")

        check_sequence_lens(t_op, seq_length)

        w_data = w.tmp_buffer.data.squeeze(0)  # The shape is [4 * hidden_size, input_size]
        r_data = r.tmp_buffer.data.squeeze(0)  # The shape is [4 * hidden_size, hidden_size]

        # Split the weights and recurrent weights based on which connections they effect.
        input_to_input, input_to_output, input_to_forget, input_to_cell = np.split(w_data, 4)
        recurrent_to_input, recurrent_to_output, recurrent_to_forget, recurrent_to_cell = np.split(r_data, 4)

        # Peephole connections.
        peephole_to_input, peephole_to_forget, peephole_to_output = self._get_1_directional_peephole_weights(t_op)

        # Bias tensors.
        input_bias, forget_bias, cell_bias, output_bias = self._get_1_directional_biases(t_op, hidden_size)

        # Projection. ONNX doesn't have an equivalent system. For no effect, use null tensors. Alternatively, we
        #  can set the weights to an identity matrix and the bias to 0.
        projection_weights = self.builder.create_null_tensor('projection_weights')
        projection_bias = self.builder.create_null_tensor('projection_bias')

        # Output state variable tensor.
        output_state = self.builder.create_empty_tensor("output_state", TensorType.FLOAT32,
                                                        [batch_size * hidden_size])
        output_state.is_variable = True

        # Cell state variable tensor.
        cell_state = self.builder.create_empty_tensor("cell_state", TensorType.FLOAT32,
                                                      [batch_size * hidden_size])
        cell_state.is_variable = True

        t_op.tmp_inputs = [
            x,
            self.builder.create_tensor_for_data(input_to_input, "input_to_input"),
            self.builder.create_tensor_for_data(input_to_forget, "input_to_forget"),
            self.builder.create_tensor_for_data(input_to_cell, "input_to_cell"),
            self.builder.create_tensor_for_data(input_to_output, "input_to_output"),
            self.builder.create_tensor_for_data(recurrent_to_input, "recurrent_to_input"),
            self.builder.create_tensor_for_data(recurrent_to_forget, "recurrent_to_forget"),
            self.builder.create_tensor_for_data(recurrent_to_cell, "recurrent_to_cell"),
            self.builder.create_tensor_for_data(recurrent_to_output, "recurrent_to_output"),
            peephole_to_input, peephole_to_forget, peephole_to_output,
            input_bias, forget_bias, cell_bias, output_bias,
            projection_weights, projection_bias,
            output_state, cell_state
        ]

        # ONNX output has shape [seq_length, num_directions, batch_size, hidden_size]. TFLite UnidirectionalSequenceLSTM
        #  omits the 'num_directions', because it is always 1. Append a 'Reshape' operator after, to match the output
        #  shape.
        t_op.tmp_outputs[0].shape = tflite_model.Shape([seq_length, batch_size, hidden_size])

        reshape_op = self.builder.create_reshape_after(t_op, 0, [seq_length, 1, batch_size, hidden_size])

        # The Reshape must come before any other operators, added after the LSTM.
        ops.post_ops.insert(0, reshape_op)

        return ops.flatten()

    def _convert_bidirectional_lstm(self, o_lstm: lstm_attributes.LSTM, t_op: tflite_model.Operator,
                                    ops: OpsList) -> list[tflite_model.Operator]:
        """ Convert ONNX LSTM with 'bidirectional' direction attribute to TFLite BidirectionalSequenceLSTM.

        :param o_lstm: Attributes of the ONNX LSTM operator.
        :param t_op: TFLite operators with inputs and outputs corresponding to the ONNX operator.
        :param ops: OpsList with operators that are the result of the conversion. May already contain some operators.
        :return: A list of TFLite operators to add to the model.
        """
        # Convert to TFLite 'BidirectionalSequenceLSTM'.
        t_op.builtin_options = bidirectional_sequence_lstm_options.BidirectionalSequenceLSTM(
            *_get_clip_values(o_lstm.clip), time_major=True, merge_outputs=True)

        o_lstm.activations = _validate_activations(o_lstm.activations, bidirectional=True)

        # Use the 'fused_activation_function' attribute to represent the output and cell update activation functions.
        t_op.builtin_options.fused_activation_function = get_activation_function_for_name(o_lstm.activations[1])

        x = t_op.tmp_inputs[0]
        w = t_op.tmp_inputs[1]
        r = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        if not all_tensors_are_static(w, r):
            # Add a 'Split' operator. (not sure if this would ever happen)
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX LSTM with dynamic weights is not yet supported.")

        # Determine the constants from tensor shapes.
        seq_length = x.shape.get(0)
        batch_size = x.shape.get(1)
        # input_size = x.shape.get(2)
        hidden_size = y.shape.get(-1)
        num_directions = y.shape.get(1)

        logger.internal_assert(num_directions == 2, "Bidirectional LSTM: 'num_directions' != 2.")

        check_sequence_lens(t_op, seq_length)

        w_data = w.tmp_buffer.data  # The shape is [2, 4 * hidden_size, input_size]
        r_data = r.tmp_buffer.data  # The shape is [2, 4 * hidden_size, hidden_size]

        # Split the weights and recurrent weights based on which connections they effect.
        # fw -> forward, bw -> backward
        fw_w, bw_w = np.split(w_data, 2)
        fw_r, bw_r = np.split(r_data, 2)

        # i_2_i means input_to_input, f -> forget, o -> output, c -> cell ...
        fw_i_2_i, fw_i_2_o, fw_i_2_f, fw_i_2_c = np.split(fw_w.squeeze(), 4)
        bw_i_2_i, bw_i_2_o, bw_i_2_f, bw_i_2_c = np.split(bw_w.squeeze(), 4)
        fw_r_2_i, fw_r_2_o, fw_r_2_f, fw_r_2_c = np.split(fw_r.squeeze(), 4)
        bw_r_2_i, bw_r_2_o, bw_r_2_f, bw_r_2_c = np.split(bw_r.squeeze(), 4)

        # Peephole connections.
        if peephole := try_get_input(t_op, 7):
            if not tensor_has_data(peephole):
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX LSTM with a dynamic operand 'P' is not yet supported.")

            fw_peephole, bw_peephole = np.split(peephole.tmp_buffer.data, 2)
            fw_p_2_i_data, fw_p_2_o_data, fw_p_2_f_data = np.split(fw_peephole.squeeze(), 3)
            bw_p_2_i_data, bw_p_2_o_data, bw_p_2_f_data = np.split(bw_peephole.squeeze(), 3)
            fw_p_2_i = self.builder.create_tensor_for_data(fw_p_2_i_data, 'fw_peephole_to_input')
            fw_p_2_f = self.builder.create_tensor_for_data(fw_p_2_f_data, 'fw_peephole_to_forget')
            fw_p_2_o = self.builder.create_tensor_for_data(fw_p_2_o_data, 'fw_peephole_to_output')
            bw_p_2_i = self.builder.create_tensor_for_data(bw_p_2_i_data, 'bw_peephole_to_input')
            bw_p_2_f = self.builder.create_tensor_for_data(bw_p_2_f_data, 'bw_peephole_to_forget')
            bw_p_2_o = self.builder.create_tensor_for_data(bw_p_2_o_data, 'bw_peephole_to_output')
        else:
            # No peephole specified -> use null tensors
            fw_p_2_i = self.builder.create_null_tensor('fw_p_2_i')
            fw_p_2_f = self.builder.create_null_tensor('fw_p_2_f')
            fw_p_2_o = self.builder.create_null_tensor('fw_p_2_o')
            bw_p_2_i = self.builder.create_null_tensor('bw_p_2_i')
            bw_p_2_f = self.builder.create_null_tensor('bw_p_2_f')
            bw_p_2_o = self.builder.create_null_tensor('bw_p_2_o')

        # Bias tensors.
        if input_bias := try_get_input(t_op, 3):
            # See comments in the bias handling section of '_convert_forward_lstm()' for explanation.

            if not tensor_has_data(input_bias):
                # Prepend 'Split', 'Add' and 'Split' operators. (not sure if this would ever happen)
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX LSTM with a dynamic 'B' input tensor is not yet supported.")

            # 'B' has shape [2, 8 * hidden_size]
            fw_b, bw_b = np.split(input_bias.tmp_buffer.data, 2)

            fw_w_b, fw_r_b = np.split(np.squeeze(fw_b), 2)
            bw_w_b, bw_r_b = np.split(np.squeeze(bw_b), 2)

            fw_bias = fw_w_b + fw_r_b
            bw_bias = bw_w_b + bw_r_b

            # Meaning of names, for example: fw_i -> forward_input, bw_c -> backward_cell ...
            fw_i_bias_data, fw_o_bias_data, fw_f_bias_data, fw_c_bias_data = np.split(fw_bias, 4)
            bw_i_bias_data, bw_o_bias_data, bw_f_bias_data, bw_c_bias_data = np.split(bw_bias, 4)

            fw_i_bias = self.builder.create_tensor_for_data(fw_i_bias_data, 'fw_input_bias')
            fw_o_bias = self.builder.create_tensor_for_data(fw_o_bias_data, 'fw_output_bias')
            fw_f_bias = self.builder.create_tensor_for_data(fw_f_bias_data, 'fw_forget_bias')
            fw_c_bias = self.builder.create_tensor_for_data(fw_c_bias_data, 'fw_cell_bias')
            bw_i_bias = self.builder.create_tensor_for_data(bw_i_bias_data, 'bw_input_bias')
            bw_o_bias = self.builder.create_tensor_for_data(bw_o_bias_data, 'bw_output_bias')
            bw_f_bias = self.builder.create_tensor_for_data(bw_f_bias_data, 'bw_forget_bias')
            bw_c_bias = self.builder.create_tensor_for_data(bw_c_bias_data, 'bw_cell_bias')

        else:
            # No biases specified, set them all to 0. (the bias tensors are not optional)
            fw_i_bias = self.builder.create_zeros_tensor([hidden_size], "zeros", np.dtype('float32'), True)
            fw_o_bias = self.builder.create_zeros_tensor([hidden_size], "zeros", np.dtype('float32'), True)
            fw_f_bias = self.builder.create_zeros_tensor([hidden_size], "zeros", np.dtype('float32'), True)
            fw_c_bias = self.builder.create_zeros_tensor([hidden_size], "zeros", np.dtype('float32'), True)
            bw_i_bias = self.builder.create_zeros_tensor([hidden_size], "zeros", np.dtype('float32'), True)
            bw_o_bias = self.builder.create_zeros_tensor([hidden_size], "zeros", np.dtype('float32'), True)
            bw_f_bias = self.builder.create_zeros_tensor([hidden_size], "zeros", np.dtype('float32'), True)
            bw_c_bias = self.builder.create_zeros_tensor([hidden_size], "zeros", np.dtype('float32'), True)

        # Projection. ONNX doesn't have an equivalent system. For no effect, use null tensors. Alternatively, we can
        #  set the weights to an identity matrix and the bias to 0.
        projection_weights = self.builder.create_null_tensor('projection_weights')
        projection_bias = self.builder.create_null_tensor('projection_bias')

        fw_output_state = self.builder.create_empty_tensor("fw_output_state", TensorType.FLOAT32,
                                                           [batch_size * hidden_size])
        fw_output_state.is_variable = True
        fw_cell_state = self.builder.create_empty_tensor("fw_cell_state", TensorType.FLOAT32,
                                                         [batch_size * hidden_size])
        fw_cell_state.is_variable = True
        bw_output_state = self.builder.create_empty_tensor("fw_output_state", TensorType.FLOAT32,
                                                           [batch_size * hidden_size])
        bw_output_state.is_variable = True
        bw_cell_state = self.builder.create_empty_tensor("fw_cell_state", TensorType.FLOAT32,
                                                         [batch_size * hidden_size])
        bw_cell_state.is_variable = True

        null_tensor = self.builder.create_null_tensor()

        t_op.tmp_inputs = [
            x,
            self.builder.create_tensor_for_data(fw_i_2_i, "fw_input_to_input"),
            self.builder.create_tensor_for_data(fw_i_2_f, "fw_input_to_forget"),
            self.builder.create_tensor_for_data(fw_i_2_c, "fw_input_to_cell"),
            self.builder.create_tensor_for_data(fw_i_2_o, "fw_input_to_output"),
            self.builder.create_tensor_for_data(fw_r_2_i, "fw_recurrent_to_input"),
            self.builder.create_tensor_for_data(fw_r_2_f, "fw_recurrent_to_forget"),
            self.builder.create_tensor_for_data(fw_r_2_c, "fw_recurrent_to_cell"),
            self.builder.create_tensor_for_data(fw_r_2_o, "fw_recurrent_to_output"),
            fw_p_2_i, fw_p_2_f, fw_p_2_o,
            fw_i_bias, fw_f_bias, fw_c_bias, fw_o_bias,
            projection_weights, projection_bias,

            self.builder.create_tensor_for_data(bw_i_2_i, "bw_input_to_input"),
            self.builder.create_tensor_for_data(bw_i_2_f, "bw_input_to_forget"),
            self.builder.create_tensor_for_data(bw_i_2_c, "bw_input_to_cell"),
            self.builder.create_tensor_for_data(bw_i_2_o, "bw_input_to_output"),
            self.builder.create_tensor_for_data(bw_r_2_i, "bw_recurrent_to_input"),
            self.builder.create_tensor_for_data(bw_r_2_f, "bw_recurrent_to_forget"),
            self.builder.create_tensor_for_data(bw_r_2_c, "bw_recurrent_to_cell"),
            self.builder.create_tensor_for_data(bw_r_2_o, "bw_recurrent_to_output"),
            bw_p_2_i, bw_p_2_f, bw_p_2_o,
            bw_i_bias, bw_f_bias, bw_c_bias, bw_o_bias,
            projection_weights, projection_bias,

            fw_output_state, fw_cell_state,
            bw_output_state, bw_cell_state,

            # Auxiliary inputs. Not necessary for conversion.
            null_tensor, null_tensor, null_tensor, null_tensor, null_tensor, null_tensor, null_tensor, null_tensor,
            null_tensor
        ]

        # TFLite BidirectionalSequenceLSTM has output shape [seq_len, batch_size, 2 * hidden_size]
        # ONNX output has shape [seq_length, num_directions (2), batch_size, hidden_size].
        t_op.tmp_outputs[0].shape = tflite_model.Shape([seq_length, batch_size, 2 * hidden_size])

        # Modify the TFLite output tensor, to match the ONNX output. There are at least 2 ways to do this.
        #  1.) Add a Reshape and Transpose operators after.
        #  2.) Set 'merge_outputs = False' and add Reshape and Concat operators.
        # Since there are some optimizations for Transpose operators already implemented, I decided to go with the first
        #  option. If you believe some different solution is better, please let me know.
        reshape_op = self.builder.create_reshape_after(t_op, 0, [seq_length, batch_size, 2, hidden_size])
        transpose_op = self.builder.create_transpose_operator_after(reshape_op, 0, [0, 2, 1, 3])

        # The Reshape and Transpose must come before any other operators added after the LSTM.
        ops.post_ops = [reshape_op, transpose_op] + ops.post_ops

        return ops.flatten()

    def _convert_reverse_lstm(self, o_lstm: lstm_attributes.LSTM, t_op: tflite_model.Operator,
                              ops: OpsList) -> list[tflite_model.Operator]:
        """ Convert ONNX LSTM with 'reverse' direction attribute to TFLite BidirectionalSequenceLSTM.
             There is no direct equivalent to 'reverse' LSTM in TFLite. But we can use the BidirectionalSequenceLSTM and
             just use the backward part of the operator. We can do this by setting all 'forward' weights and biases to
             null or fill them with 0s, when the TFLite kernel requires not null tensors.

        :param o_lstm: Attributes of the ONNX LSTM operator.
        :param t_op: TFLite operators with inputs and outputs corresponding to the ONNX operator.
        :param ops: OpsList with operators that are the result of the conversion. May already contain some operators.
        :return: A list of TFLite operators to add to the model.
        """

        # Convert to TFLite 'BidirectionalSequenceLSTM'.
        t_op.builtin_options = bidirectional_sequence_lstm_options.BidirectionalSequenceLSTM(
            *_get_clip_values(o_lstm.clip), time_major=True, merge_outputs=False)

        # Using 'merge_outputs = False' and we only need the second output. (First output is the forward part, second is
        #  backward.)
        # TODO Be careful when adding support for other ONNX outputs!
        main_output = t_op.tmp_outputs[0]
        dummy_output = self.builder.create_empty_tensor('dummy_', TensorType.FLOAT32)
        t_op.tmp_outputs = [dummy_output, main_output]

        o_lstm.activations = _validate_activations(o_lstm.activations)

        # Use the 'fused_activation_function' attribute to represent the output and cell update activation functions.
        t_op.builtin_options.fused_activation_function = get_activation_function_for_name(o_lstm.activations[1])

        x = t_op.tmp_inputs[0]
        w = t_op.tmp_inputs[1]
        r = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[1]

        if not all_tensors_are_static(w, r):
            # Add a 'Split' operator. (not sure if this would ever happen)
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX LSTM with dynamic weights is not yet supported.")

        # Determine the constants from tensor shapes.
        seq_length = x.shape.get(0)
        batch_size = x.shape.get(1)
        input_size = x.shape.get(2)
        hidden_size = y.shape.get(-1)
        num_directions = y.shape.get(1)

        logger.internal_assert(num_directions == 1, "Reverse LSTM: 'num_directions' != 1.")

        check_sequence_lens(t_op, seq_length)

        w_data = w.tmp_buffer.data.squeeze(0)  # The shape is [4 * hidden_size, input_size]
        r_data = r.tmp_buffer.data.squeeze(0)  # The shape is [4 * hidden_size, hidden_size]

        # Split the weights and recurrent weights based on which connections they effect.
        input_to_input, input_to_output, input_to_forget, input_to_cell = np.split(w_data, 4)
        recurrent_to_input, recurrent_to_output, recurrent_to_forget, recurrent_to_cell = np.split(r_data, 4)

        # Peephole connections.
        peephole_to_input, peephole_to_forget, peephole_to_output = self._get_1_directional_peephole_weights(t_op)

        # Bias tensors.
        input_bias, forget_bias, cell_bias, output_bias = self._get_1_directional_biases(t_op, hidden_size)

        # Projection. ONNX doesn't have an equivalent system. For no effect, use null tensors. Alternatively, we
        #  could set the weights to an identity matrix and the bias to 0.
        projection_weights = self.builder.create_null_tensor('projection_weights')
        projection_bias = self.builder.create_null_tensor('projection_bias')

        # Output state variable tensor.
        output_state = self.builder.create_empty_tensor("output_state", TensorType.FLOAT32,
                                                        [batch_size * hidden_size])
        output_state.is_variable = True

        # Cell state variable tensor.
        cell_state = self.builder.create_empty_tensor("cell_state", TensorType.FLOAT32,
                                                      [batch_size * hidden_size])
        cell_state.is_variable = True

        # Forward state (will always be 0)
        fw_state = self.builder.create_empty_tensor("forward_state", TensorType.FLOAT32,
                                                    [batch_size * hidden_size])
        fw_state.is_variable = True

        null_tensor = self.builder.create_null_tensor()
        float_dtype = np.dtype('float32')

        t_op.tmp_inputs = [
            x,

            # Forward W  (only the first one can be null)
            null_tensor,
            self.builder.create_zeros_tensor([hidden_size, input_size], 'wb', float_dtype),
            self.builder.create_zeros_tensor([hidden_size, input_size], 'wc', float_dtype),
            self.builder.create_zeros_tensor([hidden_size, input_size], 'wd', float_dtype),

            # Forward R  (only the first one can be null)
            null_tensor,
            self.builder.create_zeros_tensor([hidden_size, hidden_size], 'b', float_dtype),
            self.builder.create_zeros_tensor([hidden_size, hidden_size], 'c', float_dtype),
            self.builder.create_zeros_tensor([hidden_size, hidden_size], 'd', float_dtype),

            null_tensor, null_tensor, null_tensor,  # Forward peephole

            # Forward Bias (only the first one can be null)
            null_tensor,
            self.builder.create_zeros_tensor([hidden_size], 'ba', float_dtype),
            self.builder.create_zeros_tensor([hidden_size], 'ba', float_dtype),
            self.builder.create_zeros_tensor([hidden_size], 'ba', float_dtype),

            null_tensor, null_tensor,  # Forward projection

            # Backward weights
            self.builder.create_tensor_for_data(input_to_input, "input_to_input"),
            self.builder.create_tensor_for_data(input_to_forget, "input_to_forget"),
            self.builder.create_tensor_for_data(input_to_cell, "input_to_cell"),
            self.builder.create_tensor_for_data(input_to_output, "input_to_output"),
            self.builder.create_tensor_for_data(recurrent_to_input, "recurrent_to_input"),
            self.builder.create_tensor_for_data(recurrent_to_forget, "recurrent_to_forget"),
            self.builder.create_tensor_for_data(recurrent_to_cell, "recurrent_to_cell"),
            self.builder.create_tensor_for_data(recurrent_to_output, "recurrent_to_output"),
            peephole_to_input, peephole_to_forget, peephole_to_output,
            input_bias, forget_bias, cell_bias, output_bias,
            projection_weights, projection_bias,

            fw_state, fw_state,  # Forward state tensors (dummy)

            output_state, cell_state,  # Backward state tensors

            # Auxiliary inputs
            null_tensor, null_tensor, null_tensor, null_tensor, null_tensor, null_tensor, null_tensor, null_tensor,
            null_tensor,
        ]

        # ONNX output has shape [seq_length, num_directions (1), batch_size, hidden_size]. TFLite
        #  BidirectionalSequenceLSTM with 'merge_outputs=False' omits the 'num_directions', because it is always 1.
        #  Append a 'Reshape' operator after, to match the ONNX output shape.
        t_op.tmp_outputs[1].shape = tflite_model.Shape([seq_length, batch_size, hidden_size])

        reshape_op = self.builder.create_reshape_after(t_op, 1, [seq_length, 1, batch_size, hidden_size])

        # The Reshape must come before any other operators added after the LSTM, otherwise shapes won't match.
        ops.post_ops.insert(0, reshape_op)

        return ops.flatten()

    def convert(self, lstm_node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX 'LSTM' operator to TFLite.

        :param lstm_node: ONNX LSTM operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """

        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        ops = OpsList(middle_op=t_op)
        o_lstm = cast(lstm_attributes.LSTM, lstm_node.attributes)

        if o_lstm.input_forget != 0:
            # Combine the calculations with the input gate and forget gate together.
            # ONNX does this: 'f = 1.0 - i'.
            #  https://github.com/microsoft/onnxruntime/blob/d7aebf9ea8a4a651088384f219292bae9062439b/onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc#L513
            # TFLite however computes this differently: 'i = 1.0 - f'. If 'input_to_input' weights is null.
            #  https://github.com/tensorflow/tensorflow/blob/a3fb2743707b6d36297dd53fb41b96aa01ae677d/tensorflow/lite/kernels/lstm_eval.cc#L912
            #  https://github.com/tensorflow/tensorflow/blob/a3fb2743707b6d36297dd53fb41b96aa01ae677d/tensorflow/lite/kernels/lstm_eval.cc#L263
            #
            # The 'input_forget' attribute effects the internal computation, not the weights themselves. In ONNX, first
            #  the input activations are computed 'i', and then instead of computing the forget activations, '1.0 - i'
            #  is used.
            # So I don't think conversion is possible.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX LSTM with 'input_forget' != 0 is not possible.")

        if initial_h := try_get_input(t_op, 5):
            if tensor_has_data(initial_h):
                initial_h_data = initial_h.tmp_buffer.data

            else:
                initial_h_data = self.context.onnx_inspector.try_get_inferred_tensor_data(initial_h.name)
                if initial_h_data is not None:
                    logger.i(f"Using inferred tensor data to assume that the LSTM 'initial_h' input tensor named "
                             f"'{initial_h.name}' will always contain only '0' values during runtime. If this is not "
                             f"the case, converted TFLite model may produce incorrect results.")

            if initial_h_data is None or any(val != 0.0 for val in np.asarray(initial_h_data).flatten()):
                # The initial values for the 'hidden' are specified and are not zero. TFLite has the
                #  'input_activation_state' input tensor, which could be initially set to 'initial_h', but it is a
                #  'variable' tensor and TFLite doesn't support variable tensors with initial static data right now.
                # Therefore, I believe conversion is not possible.
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX LSTM with 'initial_h' input is not possible.")

        if initial_c := try_get_input(t_op, 6):
            if tensor_has_data(initial_c):
                initial_c_data = initial_c.tmp_buffer.data

            else:
                initial_c_data = self.context.onnx_inspector.try_get_inferred_tensor_data(initial_c.name)
                if initial_c_data is not None:
                    logger.i(f"Using inferred tensor data to assume that the LSTM 'initial_c' input tensor named "
                             f"'{initial_c.name}' will always contain only '0' values during runtime. If this is not "
                             f"the case, converted TFLite model may produce incorrect results.")

            if initial_c_data is None or any(val != 0.0 for val in np.asarray(initial_c_data).flatten()):
                # See the comment above, for 'initial_h'. This time the TFLite 'input_cell_state' operand is relevant.
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX LSTM with 'initial_c' input is not possible.")

        if len(t_op.tmp_outputs) != 1:
            # TODO At least 'Y_h' output might be possible to compute by appending a 'Gather' operator.

            # If the extra outputs are not used, conversion can proceed.
            for output_tensor in t_op.tmp_outputs[1:]:
                if self.inspector.get_number_of_onnx_consumers_safe(output_tensor.name) != 0:
                    logger.e(logger.Code.NOT_IMPLEMENTED,
                             "Conversion of ONNX LSTM with more than 1 output is not yet supported.")

            # Remove the unused outputs.
            t_op.tmp_outputs[1:] = []

        if o_lstm.layout != 0:
            # ORT doesn't support this.
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"Conversion of ONNX LSTM with layout = '{o_lstm.layout}' is not supported.")

        ensure_correct_tensor_formatting(t_op, self.builder, ops)

        if o_lstm.direction == 'forward':
            return self._convert_forward_lstm(o_lstm, t_op, ops)

        elif o_lstm.direction == 'bidirectional':
            return self._convert_bidirectional_lstm(o_lstm, t_op, ops)

        elif o_lstm.direction == 'reverse':
            return self._convert_reverse_lstm(o_lstm, t_op, ops)

        else:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f"ONNX LSTM has unexpected value of the direction attribute ('{o_lstm.direction}').")
