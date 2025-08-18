#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.converter.node_converters.shared.reshape_transposition import ensure_reshape_transposition
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import unsqueeze_attributes
from onnx2tflite.src.tflite_generator.builtin_options import reshape_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, FLOATS, INTS


class UnsqueezeConverter(NodeConverter):
    node = 'Unsqueeze'

    onnx_supported_types = ALL_TYPES
    tflite_supported_types = ALL_TYPES
    verified_types = FLOATS + INTS + [TensorType.UINT8, TensorType.UINT32, TensorType.UINT64, TensorType.STRING,
                                      TensorType.BOOL]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX `Unsqueeze` to TFLite `Reshape`. """
        attrs = cast(unsqueeze_attributes.Unsqueeze, node.attributes)

        if attrs.axes is not None and len(t_op.tmp_inputs) > 1:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     "Input 'axes' provided both as input tensor and attribute for 'Unsqueeze' operator.")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        self.assert_type_allowed(x.type)

        ops = OpsList(middle_op=t_op)

        if t_op.is_quantized_without_qdq():
            # Non-QDQ model -> just propagate
            propagate_quantization(x, y)

        elif t_op.is_qdq_quantized():
            if x.quantization != y.quantization:
                # I/O q-params doesn't match -> external quantizer was used or removed Clip/Relu modified q-params.
                # We need to re-quantize output because Reshape expects shared q-params for input and output.
                logger.w("Requantizing output of Unsqueeze operator. Internal quantizer can potentially avoid this.")
                scale = x.quantization.scale.vector
                zp = x.quantization.zero_point.vector
                ops.add_post(self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp))

        new_shape = ensure_reshape_transposition(self.builder, ops)

        if len(new_shape) > 8:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"ONNX Reshape output tensor has '{len(new_shape)}' dimensions TFLite only supports up to 8!")

        # The new shape can be represented using operators parameters. No need for the 'axes' input tensor -> remove it
        if len(t_op.tmp_inputs) != 1:
            t_op.tmp_inputs.pop()

        # Create the TFLite Reshape with the new shape
        t_op.builtin_options = reshape_options.Reshape(new_shape)

        return ops.flatten()
