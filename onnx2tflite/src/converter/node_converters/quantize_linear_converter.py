#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter import quantization_utils
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import set_quantization_parameters_to_tensor
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import quantize_linear_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import quantize_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, name_for_type

QuantizeLinearAttrs = quantize_linear_attributes.QuantizeLinear


# noinspection PyMethodMayBeStatic
class QuantizeLinearConverter(NodeConverter):
    node = "QuantizeLinear"

    onnx_supported_types = FLOATS + [TensorType.INT32]
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/quantize.cc#L167-L341
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT16, TensorType.INT32]
    verified_types = [TensorType.FLOAT32]

    def _extract_quant_params(self, ql_attributes: QuantizeLinearAttrs, t_op: tflite_model.Operator):
        if len(t_op.tmp_inputs) not in {2, 3}:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX QuantizeLinear has '{len(t_op.tmp_inputs)}' inputs!")

        rank = t_op.tmp_inputs[0].rank
        scale = t_op.tmp_inputs[1].tmp_buffer.data
        if (zp_tensor := try_get_input(t_op, 2)) is not None:
            zp = zp_tensor.tmp_buffer.data

        else:
            # Implicit zero point
            zp = np.zeros(scale.shape, np.uint8)  # Default according to the documentation

        if scale is None or zp is None:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX 'QuantizeLinear' with dynamic quantization parameters is not possible.")

        if not quantization_utils.is_quantization_valid(scale, zp):
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     "ONNX `QuantizeLinear` has invalid quantization parameters.")

        # Set up the quantized dimension (axis in ONNX file).
        # The axis is ignored for per-tensor quantization [https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html]
        # We use the quantized_dimension = 0 as default in the converter, hence for per-tensor set the
        #  quantized_dimension to 0.
        if quantization_utils.is_per_channel_quantized(scale, zp):
            quantized_dimension = ql_attributes.axis

            # Normalize to positive number
            if quantized_dimension < 0:
                quantized_dimension += rank

            if not (0 <= quantized_dimension < rank):
                logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                         f"ONNX QuantizeLinear has invalid 'axis'. '{ql_attributes.axis}' "
                         f"must be in range [-{rank}, {rank - 1}]!")

            if t_op.tmp_inputs[0].tensor_format.is_channels_last():
                # Convert the quantized dimension index from ONNX to TFLite format
                quantized_dimension = translator.create_channels_last_to_channels_first_permutation(rank)[
                    quantized_dimension]
        else:
            quantized_dimension = 0

        return scale, zp, quantized_dimension

    def convert_into_tensor(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator
                            ) -> list[tflite_model.Operator]:
        """Convert quantization parameters (scale & zero point) of ONNX operator 'QuantizeLinear'
        into its input tensor and skip the operator. Operators converted by this function are part
        of QDQ cluster of some float operator.

        :param node: ONNX 'QuantizeLinear' operator
        :param t_op: TFLite operators with inputs and outputs corresponding to the ONNX operator.
        :return: Empty list of new operators added to the graph.
        """
        if len(t_op.tmp_outputs) != 1:
            logger.e(logger.Code.NOT_IMPLEMENTED, "'QuantizeLinear' operators that are part of QDQ cluster can "
                                                  "currently have only one output. Use extra option 'DedicatedQDQPair'"
                                                  " when quantizing the model with ONNX quantizer.")

        ql_attributes = cast(QuantizeLinearAttrs, node.attributes)

        scale, zero_point, quantized_dimension = self._extract_quant_params(ql_attributes, t_op)

        # Add the QuantizationParameters to the input tensor
        set_quantization_parameters_to_tensor(t_op.tmp_inputs[0], scale, zero_point, quantized_dimension)

        # Assign the type of quantized tensor (int/uint) to float tensor (input) so we pass
        # tensor similarity check (same type and shape) when redirecting.
        t_op.tmp_inputs[0].type = t_op.tmp_outputs[0].type
        self.builder.redirect_tensor(t_op.tmp_outputs[0], t_op.tmp_inputs[0])

        return []

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'QuantizeLinear' to TFLite 'Quantize'.

        :param node: ONNX QuantizeLinear operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        output_type = t_op.tmp_outputs[0].type
        if output_type not in {TensorType.INT8, TensorType.UINT8}:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX QuantizeLinear with output type "
                                                  f"'{name_for_type(output_type)}' is not yet supported.")

        ql_attributes = cast(QuantizeLinearAttrs, node.attributes)

        if ql_attributes.block_size != 0:
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Attribute 'block_size' of ONNX operator 'QuantizeLinear' is not supported yet.")

        if ql_attributes.output_dtype != 0:
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Attribute 'output_dtype' of ONNX operator 'QuantizeLinear' is not supported yet.")

        if ql_attributes.precision != 0:
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Attribute 'precision' of ONNX operator 'QuantizeLinear' is not supported yet.")

        if ql_attributes.saturate != 1:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, f"Conversion of ONNX operator 'QuantizeLinear' with attribute "
                                                        f"'saturate' = '{ql_attributes.saturate}' is not possible!")

        scale, zero_point, quantized_dimension = self._extract_quant_params(ql_attributes, t_op)

        # Add the QuantizationParameters to the output tensor
        set_quantization_parameters_to_tensor(t_op.tmp_outputs[0], scale, zero_point, quantized_dimension)

        # Remove the extra input tensors, which were used in the ONNX model.
        t_op.tmp_inputs = [t_op.tmp_inputs[0]]
        t_op.builtin_options = quantize_options.Quantize()

        return [t_op]
