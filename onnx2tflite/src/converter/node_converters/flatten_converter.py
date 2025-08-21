#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import onnx2tflite.src.tflite_generator.builtin_options.reshape_options as tfl_reshape_options
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes.flatten_attributes import Flatten
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, FLOATS, INTS


class FlattenConverter(NodeConverter):
    node = "Flatten"

    onnx_supported_types = ALL_TYPES
    tflite_supported_types = ALL_TYPES
    verified_types = FLOATS + INTS + [TensorType.UINT8, TensorType.UINT32, TensorType.UINT64, TensorType.BOOL,
                                      TensorType.STRING]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert: Flatten -> Reshape | Transpose + Reshape"""
        attrs = cast(Flatten, node.attributes)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        self.assert_type_allowed(x.type)

        axis = attrs.axis
        rank = len(x.shape.vector)

        ops = OpsList(middle_op=t_op)

        if axis < -rank or axis > rank:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f"ONNX Flatten attribute 'axis' ({axis}) must be in range [{-rank}, {rank}]!!")

        # Ensure flatten is done in N[C]HW
        if x.tensor_format.is_channels_last():
            permutation = translator.create_channels_last_to_channels_first_permutation(rank).tolist()
            transpose = self.builder.create_transpose_operator_before(t_op, 0, permutation)
            transpose.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST
            ops.add_pre(transpose)

        if t_op.is_quantized_without_qdq():
            # Non-QDQ model -> just propagate
            propagate_quantization(x, y)

        elif t_op.is_qdq_quantized() and x.quantization != y.quantization:
            # I/O q-params doesn't match -> external quantizer was used or removed Clip/Relu modified q-params.
            # We need to re-quantize output because Reshape expects shared q-params for input and output.
            logger.w("Requantizing output of Flatten operator. Internal quantizer can potentially avoid this.")
            scale = x.quantization.scale.vector
            zp = x.quantization.zero_point.vector
            ops.add_post(self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp))

        t_reshape = tfl_reshape_options.Reshape(y.shape.vector)
        t_op.builtin_options = t_reshape

        return ops.flatten()
