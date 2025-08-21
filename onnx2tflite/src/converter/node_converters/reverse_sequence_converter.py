#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import reverse_sequence_attributes
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import reverse_sequence_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES


class ReverseSequenceConverter(NodeConverter):
    node = "ReverseSequence"

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/reverse_sequence.cc#L136-L158
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT16, TensorType.INT32,
                              TensorType.INT64]
    verified_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT16, TensorType.INT32, TensorType.INT64]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `ReverseSequence` operator into TFLite `ReverseSequence`."""
        if len(t_op.tmp_inputs) != 2 or len(t_op.tmp_outputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     "ONNX `ReverseSequence` has invalid number of input and output tensors.")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]
        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX `ReverseSequence` has different input and output types.")

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        elif t_op.is_quantized_without_qdq():
            propagate_quantization(x, y)

        ops = OpsList(middle_op=t_op)

        if x.tensor_format.is_channels_last():
            # Surround the `ReverseSequence` with `Transpose` operators to make the IO channels first.
            to_onnx_perm = translator.create_channels_last_to_channels_first_permutation(x.rank)
            to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(y.rank, True)

            ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, to_onnx_perm))
            ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, to_tflite_perm))

            t_op.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST
            t_op.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        attrs = cast(reverse_sequence_attributes.ReverseSequence, node.attributes)

        time_axis, batch_axis = attrs.time_axis, attrs.batch_axis
        if (time_axis, batch_axis) not in [(0, 1), (1, 0)]:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `ReverseSequence` has invalid `time_axis` ({time_axis}) or `batch_axis` ({batch_axis}).")

        # When `sequence_lens` contains `0`, ONNX will just fill that output slice with `0`s, instead of taking it
        #  to mean that a sequence of length `0` should be reversed. TFLite does nothing in these cases, which is
        #  expected. (it does the same for `0` as it does for `1`)
        sequence_lens_tensor = t_op.tmp_inputs[1]
        if tensor_has_data(sequence_lens_tensor):
            if any(val == 0 for val in sequence_lens_tensor.tmp_buffer.data.flat):
                logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX `ReverseSequence` with a `sequence_lens` "
                                                      "input which contains the value `0` is not supported.")
        else:
            # We could add extra operators which would set the corresponding values to `0` if needed.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX `ReverseSequence` with a dynamic `sequence_lens` input is not supported.")

        t_op.builtin_options = reverse_sequence_options.ReverseSequence(time_axis, batch_axis)

        return ops.flatten()
