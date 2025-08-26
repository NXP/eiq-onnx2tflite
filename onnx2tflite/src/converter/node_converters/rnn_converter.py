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
from onnx2tflite.src.converter.conversion.common import OpsList, try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.node_converters.shared.recurrent_utils import (
    check_sequence_lens,
    ensure_correct_tensor_formatting,
    get_activation_function_for_name,
)
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import rnn_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import (
    bidirectional_sequence_rnn_options,
    unidirectional_sequence_rnn_options,
)
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class RNNConverter(NodeConverter):
    node = "RNN"

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/unidirectional_sequence_rnn.cc#L95
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/bidirectional_sequence_rnn.cc#L145
    tflite_supported_types = [TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32]

    def _convert_forward_rnn(self, o_rnn: rnn_attributes.RNN, t_op: tflite_model.Operator,
                             ops: OpsList) -> list[tflite_model.Operator]:
        """Convert ONNX RNN with 'forward' direction attribute to TFLite UnidirectionalSequenceRNN.

        :param o_rnn: Attributes of the ONNX RNN operator.
        :param t_op: TFLite operators with inputs and outputs corresponding to the ONNX operator.
        :param ops: OpsList with operators that are the result of the conversion. May already contain some operators.
        :return: A list of TFLite operators to add to the model.
        """
        # Convert to TFLite 'UnidirectionalSequenceRNN'.
        t_op.builtin_options = unidirectional_sequence_rnn_options.UnidirectionalSequenceRNN(
            time_major=(o_rnn.layout == 0))

        # Use the 'fused_activation_function' attribute to represent the output and cell update activation functions.
        if o_rnn.activations is None:
            t_op.builtin_options.fused_activation_function = ActivationFunctionType.TANH
        else:
            if len(o_rnn.activations) != 1:
                logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                         "ONNX forward RNN has invalid 'activations' attribute.")
            t_op.builtin_options.fused_activation_function = get_activation_function_for_name(o_rnn.activations[0])

        x = t_op.tmp_inputs[0]
        w = t_op.tmp_inputs[1]
        r = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        # Determine the constants from tensor shapes.
        seq_length = x.shape.get(0)
        batch_size = x.shape.get(1)
        # input_size = x.shape.get(2)
        hidden_size = y.shape.get(-1)
        num_directions = y.shape.get(1)

        logger.internal_assert(num_directions == 1, "Forward RNN: 'num_directions' != 1.")

        # Make sure the 'sequence_lens' operand is convertible.
        check_sequence_lens(t_op, seq_length)

        if tensor_has_data(w):
            # ONNX has an extra leading 1 in the shape -> remove it.
            w.shape = tflite_model.Shape(w.shape.vector[1:])
            w.tmp_buffer.data.squeeze(0)
        else:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX RNN with a dynamic W is not yet supported.")

        if tensor_has_data(r):
            # ONNX has an extra leading 1 in the shape -> remove it.
            r.shape = tflite_model.Shape(r.shape.vector[1:])
            r.tmp_buffer.data.squeeze(0)
        else:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX RNN with a dynamic R is not yet supported.")

        if b := try_get_input(t_op, 3):
            # The bias in ONNX has shape [num_directions (1), 2 * hidden_size]. TFLite requires [hidden_size].
            # The ONNX bias is a concatenation of W bias and R bias. For inference purposes, these can simply be added
            #  together as a single bias vector in TFLite.
            if tensor_has_data(b):
                w_bias, r_bias = np.split(b.tmp_buffer.data.squeeze(0), 2)
                bias_data = w_bias + r_bias
                b.tmp_buffer.data = bias_data
                b.shape = tflite_model.Shape(list(bias_data.shape))

            else:
                logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX RNN with a dynamic B is not yet supported.")

        else:
            b = self.builder.create_zeros_tensor([hidden_size], "B", np.dtype("float32"), can_reuse=True)

        # Hidden state variable tensor.
        hidden_state = self.builder.create_empty_tensor("hidden_state", TensorType.FLOAT32, [batch_size, hidden_size])
        hidden_state.is_variable = True

        t_op.tmp_inputs = [x, w, r, b, hidden_state]

        # ONNX output has shape [seq_length, num_directions, batch_size, hidden_size]. TFLite UnidirectionalSequenceRNN
        #  omits the 'num_directions', because it is always 1. Append a 'Reshape' operator after, to match the output
        #  shape.
        t_op.tmp_outputs[0].shape = tflite_model.Shape([seq_length, batch_size, hidden_size])

        reshape_op = self.builder.create_reshape_after(t_op, 0, [seq_length, 1, batch_size, hidden_size])

        # The Reshape must come before any other operators, added after the RNN.
        ops.post_ops.insert(0, reshape_op)

        return ops.flatten()

    def _convert_reverse_rnn(self, o_rnn: rnn_attributes.RNN, t_op: tflite_model.Operator,
                             ops: OpsList) -> list[tflite_model.Operator]:
        """Convert ONNX RNN with 'reverse' direction attribute to TFLite BidirectionalSequenceRNN.

        :param o_rnn: Attributes of the ONNX RNN operator.
        :param t_op: TFLite operators with inputs and outputs corresponding to the ONNX operator.
        :param ops: OpsList with operators that are the result of the conversion. May already contain some operators.
        :return: A list of TFLite operators to add to the model.
        """
        # Convert to TFLite 'BidirectionalSequenceRNN'. Set 'merge_outputs = False' and use only the 2nd output tensor.
        # TODO Be careful when adding support for the second ONNX RNN output!
        logger.internal_assert(len(t_op.tmp_outputs) == 1, "Reverse RNN with multiple outputs is not properly handled!")
        t_op.builtin_options = bidirectional_sequence_rnn_options.BidirectionalSequenceRNN(
            time_major=(o_rnn.layout == 0),
            merge_outputs=False)

        # Use the 'fused_activation_function' attribute to represent the output and cell update activation functions.
        if o_rnn.activations is None:
            t_op.builtin_options.fused_activation_function = ActivationFunctionType.TANH
        else:
            if len(o_rnn.activations) != 1:
                logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                         "ONNX reverse RNN has invalid 'activations' attribute.")
            t_op.builtin_options.fused_activation_function = get_activation_function_for_name(o_rnn.activations[0])

        x = t_op.tmp_inputs[0]
        w = t_op.tmp_inputs[1]
        r = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        # Determine the constants from tensor shapes.
        seq_length = x.shape.get(0)
        batch_size = x.shape.get(1)
        # input_size = x.shape.get(2)
        hidden_size = y.shape.get(-1)
        num_directions = y.shape.get(1)

        logger.internal_assert(num_directions == 1, "Reverse RNN: 'num_directions' != 1.")

        # Make sure the 'sequence_lens' operand is convertible.
        check_sequence_lens(t_op, seq_length)

        if tensor_has_data(w):
            # ONNX has an extra leading 1 in the shape -> remove it.
            w.shape = tflite_model.Shape(w.shape.vector[1:])
            w.tmp_buffer.data.squeeze(0)
        else:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX RNN with a dynamic W is not yet supported.")

        if tensor_has_data(r):
            # ONNX has an extra leading 1 in the shape -> remove it.
            r.shape = tflite_model.Shape(r.shape.vector[1:])
            r.tmp_buffer.data.squeeze(0)
        else:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX RNN with a dynamic R is not yet supported.")

        if b := try_get_input(t_op, 3):
            # The bias in ONNX has shape [num_directions (1), 2 * hidden_size]. TFLite requires [hidden_size].
            # The ONNX bias is a concatenation of W bias and R bias. For inference purposes, these can simply be added
            #  together as a single bias vector in TFLite.
            if tensor_has_data(b):
                w_bias, r_bias = np.split(b.tmp_buffer.data.squeeze(0), 2)
                bias_data = w_bias + r_bias
                b.tmp_buffer.data = bias_data
                b.shape = tflite_model.Shape(list(bias_data.shape))

            else:
                logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX RNN with a dynamic B is not yet supported.")

        else:
            b = self.builder.create_zeros_tensor([hidden_size], "B", np.dtype("float32"), can_reuse=True)

        # Hidden state variable tensors.
        hidden_state = self.builder.create_empty_tensor("hidden_state", TensorType.FLOAT32, [batch_size, hidden_size])
        hidden_state.is_variable = True

        fw_dummy_state = self.builder.create_empty_tensor("dummy_state", TensorType.FLOAT32, [batch_size, hidden_size])
        fw_dummy_state.is_variable = True

        null_tensor = self.builder.create_null_tensor()

        # The TFLite BidirectionalSequenceRNN inputs are listed here:
        # https://github.com/tensorflow/tensorflow/blob/1af2ea61f9b9c887409945849988685559970330/tensorflow/lite/kernels/bidirectional_sequence_rnn.cc#L43-L62
        t_op.tmp_inputs = [
            x,

            # Forward weights -> set all to 0
            self.builder.create_zeros_tensor(w.shape.vector.copy(), "fw_w", np.dtype("float32"), True),
            self.builder.create_zeros_tensor(r.shape.vector.copy(), "fw_r", np.dtype("float32"), True),
            self.builder.create_zeros_tensor([hidden_size], "fw_b", np.dtype("float32"), True),
            fw_dummy_state,

            # Reverse weights
            w, r, b, hidden_state,

            # Auxiliary inputs (not important)
            null_tensor, null_tensor, null_tensor
        ]

        # We only want the second output of the TFLite BidirectionalSequenceRNN -> create a dummy tensor for the first.
        dummy_output = self.builder.create_empty_tensor("dummy", TensorType.FLOAT32)
        t_op.tmp_outputs = [dummy_output, y]

        # ONNX output has shape [seq_length, num_directions, batch_size, hidden_size]. TFLite BidirectionalSequenceRNN
        #  with merge_outputs = False omits the 'num_directions', because it is always 1.
        #  Append a 'Reshape' operator after, to match the output shape.
        t_op.tmp_outputs[1].shape = tflite_model.Shape([seq_length, batch_size, hidden_size])

        reshape_op = self.builder.create_reshape_after(t_op, 1, [seq_length, 1, batch_size, hidden_size])

        # The Reshape must come before any other operators, added after the RNN.
        ops.post_ops.insert(0, reshape_op)

        return ops.flatten()

    def _convert_bidirectional_rnn(self, o_rnn: rnn_attributes.RNN, t_op: tflite_model.Operator,
                                   ops: OpsList) -> list[tflite_model.Operator]:
        """Convert ONNX RNN with 'bidirectional' direction attribute to TFLite BidirectionalSequenceRNN.

        :param o_rnn: Attributes of the ONNX RNN operator.
        :param t_op: TFLite operators with inputs and outputs corresponding to the ONNX operator.
        :param ops: OpsList with operators that are the result of the conversion. May already contain some operators.
        :return: A list of TFLite operators to add to the model.
        """
        # Convert to TFLite 'BidirectionalSequenceRNN'.
        t_op.builtin_options = bidirectional_sequence_rnn_options.BidirectionalSequenceRNN(
            time_major=(o_rnn.layout == 0),
            merge_outputs=True)

        # Use the 'fused_activation_function' attribute to represent the output and cell update activation functions.
        if o_rnn.activations is None:
            t_op.builtin_options.fused_activation_function = ActivationFunctionType.TANH
        else:
            if len(o_rnn.activations) != 2:
                logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                         "ONNX bidirectional RNN has invalid 'activations' attribute.")

            if o_rnn.activations[0] != o_rnn.activations[1]:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         f"Conversion ONNX bidirectional RNN with 'activations' = {o_rnn.activations} is not possible.")

            t_op.builtin_options.fused_activation_function = get_activation_function_for_name(o_rnn.activations[0])

        x = t_op.tmp_inputs[0]
        w = t_op.tmp_inputs[1]
        r = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        # Determine the constants from tensor shapes.
        seq_length = x.shape.get(0)
        batch_size = x.shape.get(1)
        # input_size = x.shape.get(2)
        hidden_size = y.shape.get(-1)
        num_directions = y.shape.get(1)

        logger.internal_assert(num_directions == 2, "Bidirectional RNN: 'num_directions' != 2.")

        # Make sure the 'sequence_lens' operand is convertible.
        check_sequence_lens(t_op, seq_length)

        if tensor_has_data(w):
            # ONNX 'W' has the shape [num_directions (2), hidden_size, input_size]. TFLite uses 2 separate tensors of
            #  shape [hidden_size, input_size].
            fw_w_data, bw_w_data = np.split(w.tmp_buffer.data, 2)
            fw_w = self.builder.create_tensor_for_data(fw_w_data.squeeze(0), "fw_w_")
            bw_w = self.builder.create_tensor_for_data(bw_w_data.squeeze(0), "bw_w_")
        else:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX RNN with a dynamic W is not yet supported.")

        if tensor_has_data(r):
            # ONNX 'W' has the shape [num_directions (2), hidden_size, hidden_size]. TFLite uses 2 separate tensors of
            #  shape [hidden_size, hidden_size].
            fw_r_data, bw_r_data = np.split(r.tmp_buffer.data, 2)
            fw_r = self.builder.create_tensor_for_data(fw_r_data.squeeze(0), "fw_r_")
            bw_r = self.builder.create_tensor_for_data(bw_r_data.squeeze(0), "bw_r_")
        else:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX RNN with a dynamic R is not yet supported.")

        if b := try_get_input(t_op, 3):
            # The bias in ONNX has shape [num_directions (2), 2 * hidden_size]. TFLite requires 2 biases of shape
            #  [hidden_size].
            # The ONNX bias is a concatenation of W bias and R bias. For inference purposes, these can simply be added
            #  together as a single bias vector in TFLite.
            if tensor_has_data(b):
                fw_b_data, bw_b_data = np.split(b.tmp_buffer.data, 2)  # Split into the forward and backward parts

                # Split into the W and R biases
                fw_b_w_data, fw_b_r_data = np.split(fw_b_data.squeeze(0), 2)
                bw_b_w_data, bw_b_r_data = np.split(bw_b_data.squeeze(0), 2)

                # Add the W and R biases together, to get the TFLite biases.
                fw_b_data = fw_b_w_data + fw_b_r_data
                bw_b_data = bw_b_w_data + bw_b_r_data

                fw_b = self.builder.create_tensor_for_data(fw_b_data, "fw_b_")
                bw_b = self.builder.create_tensor_for_data(bw_b_data, "bw_b_")

            else:
                logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX RNN with a dynamic B is not yet supported.")

        else:
            fw_b = self.builder.create_zeros_tensor([hidden_size], "B", np.dtype("float32"),
                                                    can_reuse=True)
            bw_b = fw_b

        # Hidden state variable tensors.
        fw_state = self.builder.create_empty_tensor("fw_hidden_state", TensorType.FLOAT32,
                                                    [batch_size, hidden_size])
        fw_state.is_variable = True

        bw_state = self.builder.create_empty_tensor("bw_hidden_state", TensorType.FLOAT32,
                                                    [batch_size, hidden_size])
        bw_state.is_variable = True

        null_tensor = self.builder.create_null_tensor()

        # The TFLite BidirectionalSequenceRNN inputs are listed here:
        # https://github.com/tensorflow/tensorflow/blob/1af2ea61f9b9c887409945849988685559970330/tensorflow/lite/kernels/bidirectional_sequence_rnn.cc#L43-L62
        # noinspection PyUnboundLocalVariable
        t_op.tmp_inputs = [
            x,

            # Forward weights
            fw_w, fw_r, fw_b, fw_state,

            # Reverse weights
            bw_w, bw_r, bw_b, bw_state,

            # Auxiliary inputs (not important)
            null_tensor, null_tensor, null_tensor
        ]

        # TFLite BidirectionalSequenceRNN has output shape [seq_len, batch_size, 2 * hidden_size]
        # ONNX output has shape [seq_length, num_directions (2), batch_size, hidden_size].
        t_op.tmp_outputs[0].shape = tflite_model.Shape([seq_length, batch_size, 2 * hidden_size])

        # Modify the TFLite output tensor, to match the ONNX output. There are at least 2 ways to do this.
        #  1.) Add a Reshape and Transpose operators after.
        #  2.) Set 'merge_outputs = False' and add Reshape and Concat operators.
        # Since there are some optimizations for Transpose operators already implemented, I decided to go with the first
        #  option. If you believe some different solution is better, please let me know.
        reshape_op = self.builder.create_reshape_after(t_op, 0, [seq_length, batch_size, 2, hidden_size])
        transpose_op = self.builder.create_transpose_operator_after(reshape_op, 0, [0, 2, 1, 3])

        # The Reshape and Transpose must come before any other operators added after the RNN.
        ops.post_ops = [reshape_op, transpose_op] + ops.post_ops

        return ops.flatten()

    def convert(self, rnn_node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'RNN' operator to TFLite 'UnidirectionalSequenceRNN` or `BidirectionalSequenceRNN`."""
        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        ops = OpsList(middle_op=t_op)
        o_rnn = cast(rnn_attributes.RNN, rnn_node.attributes)

        if initial_h := try_get_input(t_op, 5):
            if tensor_has_data(initial_h):
                initial_h_data = initial_h.tmp_buffer.data

            else:
                initial_h_data = self.context.onnx_inspector.try_get_inferred_tensor_data(initial_h.name)
                if initial_h_data is not None:
                    logger.i(f"Using inferred tensor data to assume that the RNN 'initial_h' input tensor named "
                             f"'{initial_h.name}' will always contain only '0' values during runtime. If this is not "
                             f"the case, converted TFLite model may produce incorrect results.")

            if initial_h_data is None or any(val != 0.0 for val in np.asarray(initial_h_data).flatten()):
                # The initial values for the 'hidden' are specified and are not zero. TFLite RNN has the
                #  'hidden_state' input tensor, which could be initially set to 'initial_h', but it is a
                #  'variable' tensor and TFLite doesn't support variable tensors with initial static data right now.
                # Therefore, I believe conversion is not possible.
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX RNN with 'initial_h' input is not possible.")

        if len(t_op.tmp_outputs) != 1:
            # TODO 'Y_h' output might be possible to compute by appending a 'Gather' operator.

            # If the extra outputs are not used, conversion can proceed.
            for output_tensor in t_op.tmp_outputs[1:]:
                if self.inspector.get_number_of_onnx_consumers_safe(output_tensor.name) != 0:
                    logger.e(logger.Code.NOT_IMPLEMENTED,
                             "Conversion of ONNX RNN with more than 1 output is not yet supported.")

            # Remove the unused outputs.
            t_op.tmp_outputs[1:] = []

        if o_rnn.layout != 0:
            # ORT doesn't support this.
            # https://github.com/microsoft/onnxruntime/blob/435e19953ea54115124fd637a67a87681a7fc8eb/onnxruntime/core/providers/cpu/rnn/rnn.h#L45
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"Conversion of ONNX RNN with layout = '{o_rnn.layout}' is not supported.")

        if o_rnn.clip is not None:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX RNN with a 'clip' attribute is not possible.")

        ensure_correct_tensor_formatting(t_op, self.builder, ops)

        if o_rnn.direction == "forward":
            return self._convert_forward_rnn(o_rnn, t_op, ops)

        if o_rnn.direction == "bidirectional":
            return self._convert_bidirectional_rnn(o_rnn, t_op, ops)

        if o_rnn.direction == "reverse":
            return self._convert_reverse_rnn(o_rnn, t_op, ops)

        logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                 f"ONNX RNN has unexpected value of the direction attribute ('{o_rnn.direction}').")
