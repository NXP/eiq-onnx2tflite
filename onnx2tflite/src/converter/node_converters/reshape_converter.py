#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

import onnx2tflite.src.tflite_generator.builtin_options.reshape_options as tfl_reshape_options
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.node_converters.shared.reshape_transposition import ensure_reshape_transposition
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, FLOATS, INTS


class ReshapeConverter(NodeConverter):
    node = "Reshape"

    onnx_supported_types = ALL_TYPES
    tflite_supported_types = ALL_TYPES
    verified_types = FLOATS + INTS + [TensorType.UINT8, TensorType.UINT32, TensorType.UINT64, TensorType.STRING,
                                      TensorType.BOOL]

    # noinspection PyMethodMayBeStatic
    def _safe_compute_flat_size(self, shape: list[int | str]) -> int:
        """Compute the flat size of a tensor with given shape. Strings and negative dimensions are treated as '1'.

        :param shape: Shape of the tensor. Can include integers and strings.
        :return: The flat size of the tensor.
        """
        flat_size = 1
        for dim in shape:
            if isinstance(dim, int) and dim > 1:
                flat_size *= dim

        return flat_size

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX 'Reshape' to TFLite 'Reshape'."""
        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX: Reshape operator has unexpected number of inputs! "
                                                     f"Got '{len(t_op.tmp_inputs)}', expected '2'.")

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
                logger.w("Requantizing output of Reshape operator. Internal quantizer can potentially avoid this.")
                scale = x.quantization.scale.vector
                zp = x.quantization.zero_point.vector
                ops.add_post(self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp))

        new_shape = ensure_reshape_transposition(self.builder, ops)

        if len(new_shape) > 8:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"ONNX Reshape output tensor has '{len(new_shape)}' dimensions TFLite only supports up to 8!")

        # The new shape can be represented using operators parameters. No need for the input tensor -> remove it
        t_op.tmp_inputs.pop()

        # Create the TFLite Reshape with the new shape
        t_op.builtin_options = tfl_reshape_options.Reshape(new_shape)

        flat_input_size = self._safe_compute_flat_size(x.shape.vector)
        flat_output_size = self._safe_compute_flat_size(y.shape.vector)

        if flat_input_size != flat_output_size:
            # Doesn't necessarily indicate an error in conversion. For example [-1, 3, 12, 12] can be reshaped to
            #  [-1, 12].
            logger.w(f"convert_reshape: Flat size of the input tensor '{x.name}' is not the same as "
                     f"the flat size of the output tensor '{y.name}'.")

        return ops.flatten()
