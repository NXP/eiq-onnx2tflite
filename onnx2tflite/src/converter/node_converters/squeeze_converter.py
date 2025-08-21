#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList, try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.node_converters.shared.reshape_transposition import ensure_reshape_transposition
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import squeeze_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import reshape_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, FLOATS, INTS


class SqueezeConverter(NodeConverter):
    node = "Squeeze"

    onnx_supported_types = ALL_TYPES
    tflite_supported_types = ALL_TYPES
    verified_types = FLOATS + INTS + [TensorType.UINT8, TensorType.UINT32, TensorType.UINT64, TensorType.STRING,
                                      TensorType.BOOL]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        ops = OpsList(middle_op=t_op)

        self.assert_type_allowed(x.type)

        if t_op.is_quantized_without_qdq():
            # Non-QDQ model -> just propagate
            propagate_quantization(x, y)

        elif t_op.is_qdq_quantized():
            if x.quantization != y.quantization:
                # I/O q-params doesn't match -> external quantizer was used or removed Clip/Relu modified q-params.
                # We need to re-quantize output because Reshape expects shared q-params for input and output.
                logger.w("Re-quantizing output of Squeeze operator. Internal quantizer can potentially avoid this.")
                scale = x.quantization.scale.vector
                zp = x.quantization.zero_point.vector
                ops.add_post(self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp))

        o_attrs = cast(squeeze_attributes.Squeeze, node.attributes)

        axes_tensor = try_get_input(t_op, 1)

        if o_attrs.axes is not None and axes_tensor is not None:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     "Input 'axes' provided both as input tensor and attribute for 'Squeeze' operator.")

        new_shape = ensure_reshape_transposition(self.builder, ops)

        if len(new_shape) > 8:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"ONNX Reshape output tensor has '{len(new_shape)}' dimensions TFLite only supports up to 8!")

        # The new shape can be represented using operators parameters. No need for the 'axes' input tensor -> remove it
        t_op.tmp_inputs[1:] = []

        # Create the TFLite Reshape with the new shape
        t_op.builtin_options = reshape_options.Reshape(new_shape)

        return ops.flatten()
