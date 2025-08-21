#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


import numpy as np

from onnx2tflite.lib.tflite import BuiltinOperator as tflBuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import broadcast_to_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, FLOATS, INTS


class ExpandConverter(NodeConverter):
    node = "Expand"

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/broadcast_to.cc#L102
    tflite_supported_types = ALL_TYPES.copy()
    tflite_supported_types.remove(TensorType.STRING)
    verified_types = INTS + FLOATS + [TensorType.UINT8, TensorType.UINT32, TensorType.UINT64, TensorType.BOOL]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert 'Expand' operator into TFLite 'BroadcastArgs' and 'BroadcastTo'. 'BroadcastArgs' is added
        because 'BroadcastTo' expects 'output shape after broadcasting' as an input parameter, whereas 'Expand'
        expects 'broadcasted shape' and output shape is computed internally.
        """
        x = t_op.tmp_inputs[0]
        new_shape = t_op.tmp_inputs[1]
        output = t_op.tmp_outputs[0]

        self.assert_type_allowed(x.type)

        if t_op.is_quantized_without_qdq():
            # Non-QDQ model -> just propagate
            propagate_quantization(x, output)

        if new_shape.type != TensorType.INT64:
            logger.e(logger.Code.INVALID_TYPE, f"Input 'Shape' must be 'INT64' data type. Got '{new_shape.type}'.")

        if tensor_has_data(new_shape):
            # Try to broadcast shape and check if broadcasting has an effect on output shape
            broadcasted_shape = np.broadcast_shapes(x.shape.vector, new_shape.tmp_buffer.data)
            output_shape_same_as_input = np.array_equal(x.shape.vector, broadcasted_shape)

            if output_shape_same_as_input and self.builder.operator_can_be_skipped(t_op, self.inspector):
                logger.i("Skipping operator 'Expand' because 'new_shape' input has no effect on output shape.")
                self.builder.redirect_tensor(t_op.tmp_outputs[0], t_op.tmp_inputs[0])
                return []

        if any(input_tensor.tensor_format.is_channels_last() for input_tensor in t_op.tmp_inputs):
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "ONNX Operator 'Expand' uses shape broadcasting for its input tensors."
                     " This requires explicit support by the converter and has not yet been "
                     "implemented!")

        ops = OpsList()

        x_shape_tensor = self.builder.create_tensor_for_data(np.array(x.shape.vector, np.int64), "x_shape")

        broadcasted_shape = [max(new_shape.rank, x.rank)]
        broadcast_output_tensor = self.builder.create_empty_tensor("broadcasted_shape", TensorType.INT64,
                                                                   broadcasted_shape)

        # Prepend 'BroadcastArgs' operator to compute the output shape (result of broadcasting 'x.shape' and
        #  'new_shape.value')
        broadcast_args = tflite_model.Operator(
            opcode_index=self.builder.op_code_index_for_op_type(tflBuiltinOperator.BuiltinOperator.BROADCAST_ARGS)
        )
        broadcast_args.tmp_inputs = [x_shape_tensor, new_shape]
        broadcast_args.tmp_outputs = [broadcast_output_tensor]

        ops.add_pre(broadcast_args)

        broadcast_to = tflite_model.Operator(builtin_options=broadcast_to_options.BroadcastTo())
        broadcast_to.tmp_inputs = [x, broadcast_output_tensor]
        broadcast_to.tmp_outputs = [output]
        broadcast_to.tmp_version = 2

        ops.middle_op = broadcast_to

        if x.quantization is not None and output.quantization is not None and x.quantization != output.quantization:
            # I/O q-params doesn't match -> external quantizer was used or removed Clip/Relu modified q-params.
            # We need to re-quantize output because BroadcastTo expects shared q-params for input and output.
            logger.w("Requantizing output of Expand operator. Internal quantizer can potentially avoid this.")
            scale = x.quantization.scale.vector
            zp = x.quantization.zero_point.vector
            ops.add_post(self.builder.create_quantize_operator_after(broadcast_to, 0, x.type, scale, zp))

        return ops.flatten()
