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
from onnx2tflite.src.converter.quantization_utils import dequantize, set_quantization_parameters_to_tensor
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import dequantize_linear_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import dequantize_options

DequantizeLinearAttrs = dequantize_linear_attributes.DequantizeLinear


class DequantizeLinearConverter(NodeConverter):
    node = "DequantizeLinear"

    # noinspection PyMethodMayBeStatic
    def _extract_quant_params(
        self, o_dequantize_linear: DequantizeLinearAttrs, t_op, allow_int32_input
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if not (2 <= len(t_op.tmp_inputs) <= 3):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX 'DequantizeLinear' has invalid number of inputs. "
                                                        f"Expected 2 or 3, got '{len(t_op.tmp_inputs)}'.")

        input_tensor = t_op.tmp_inputs[0]
        scale_tensor = t_op.tmp_inputs[1]
        zero_point_tensor = try_get_input(t_op, 2)

        dynamic_scale = not tensor_has_data(scale_tensor)
        dynamic_zero_point = not tensor_has_data(zero_point_tensor) if zero_point_tensor else False
        if dynamic_scale or dynamic_zero_point:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, "Conversion of ONNX 'DequantizeLinear' is only possible "
                                                        "when the quantization parameters are static!")

        supported_input_types = {TensorType.INT8, TensorType.UINT8}

        if allow_int32_input:
            supported_input_types.add(TensorType.INT32)

        if input_tensor.type not in supported_input_types:
            logger.e(logger.Code.INVALID_TYPE, "ONNX DequantizeLinear supports only INT8 and UINT8 input data "
                                               f"types. Got '{input_tensor.type}'.")

        input_scale_data = scale_tensor.tmp_buffer.data
        if zero_point_tensor:
            input_zero_point_data = zero_point_tensor.tmp_buffer.data
        else:
            input_zero_point_data = np.zeros(scale_tensor.shape.vector,
                                             translator.tf_lite_type_to_numpy(input_tensor.type))

        if not quantization_utils.is_quantization_valid(input_scale_data, input_zero_point_data):
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     "Invalid quantization parameter: scale.size() != zero_point.size())")

        # Set up the quantized dimension (axis in ONNX file).
        # The axis is ignored for per-tensor quantization [https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html]
        # We use the quantized_dimension = 0 as default in the converter, hence for per-tensor set the
        # quantized_dimension to 0.
        if quantization_utils.is_per_channel_quantized(input_scale_data, input_zero_point_data):
            rank = input_tensor.rank
            quantized_dimension = o_dequantize_linear.axis

            if quantized_dimension < 0:
                quantized_dimension += rank

            if input_tensor.tensor_format.is_channels_last():
                # Convert the quantized dimension index from ONNX to TFLite format
                quantized_dimension = translator.create_channels_last_to_channels_first_permutation(rank)[
                    quantized_dimension]
        else:
            quantized_dimension = 0

        return input_scale_data, input_zero_point_data, quantized_dimension

    def _convert_quantized_bias_to_float_tensor(self, o_dequantize_linear, t_op) -> None:
        scale, zero_point, quantized_dimension = self._extract_quant_params(o_dequantize_linear, t_op,
                                                                            allow_int32_input=True)

        x = t_op.tmp_inputs[0]
        x.tmp_buffer.data = dequantize(x.tmp_buffer.data, scale, zero_point)
        x.quantization = None
        x.type = TensorType.FLOAT32

        self.builder.redirect_tensor(t_op.tmp_outputs[0], x)

    def convert_into_tensor(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[
        tflite_model.Operator]:
        """Convert quantization parameters (scale & zero point) of ONNX operator 'DequantizeLinear'
        into its input tensor and skip the operator. Operators converted by this function are part
        of QDQ cluster of some float operator.

        :param node: ONNX DequantizeLinear operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: Empty list of new operators added to the graph.
        """
        y = t_op.tmp_outputs[0]

        dql_attributes = cast(DequantizeLinearAttrs, node.attributes)

        scale, zero_point, quantized_dim = self._extract_quant_params(dql_attributes, t_op, allow_int32_input=True)

        consumers = self.context.onnx_inspector.get_ops_with_input_tensor(y.name)
        is_bias = t_op.tmp_inputs[2].type == TensorType.INT32
        if len(consumers) > 1 and not is_bias:
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     f"'DequantizeLinear' operators that are part of QDQ cluster and don't represent bias tensors,"
                     f" can currently have only one consumer. Output tensor '{y.name}' is consumed by multiple nodes. "
                     f"Use extra option 'DedicatedQDQPair' when quantizing the model with ONNX quantizer.")

        set_quantization_parameters_to_tensor(t_op.tmp_inputs[0], scale, zero_point, quantized_dim)

        # Assign the type of quantized tensor (int/uint) to float tensor (output) so we pass
        # tensor similarity check (same type and shape) when redirecting.
        t_op.tmp_outputs[0].type = t_op.tmp_inputs[0].type
        self.builder.redirect_tensor(t_op.tmp_outputs[0], t_op.tmp_inputs[0])

        return []

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX operator 'DequantizeLinear' to TFLite 'Dequantize'.

        :param node: ONNX DequantizeLinear operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        dql_attributes = cast(DequantizeLinearAttrs, node.attributes)

        if dql_attributes.output_dtype != 0:
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Attribute 'output_dtype' of ONNX operator 'DequantizeLinear' is not supported yet.")

        if dql_attributes.block_size != 0:
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Attribute 'block_size' of ONNX operator 'DequantizeLinear' is not supported yet.")

        if t_op.tmp_inputs[0].type == TensorType.INT32:
            # Quantized bias tensor but corresponding operator (Conv, Gemm) not run as quantized (running
            # conversion of QDQ model with QDQ recognition turned off OR QDQ cluster not matching
            # criteria) -> dequantize to FLOAT to make model runnable
            self._convert_quantized_bias_to_float_tensor(dql_attributes, t_op)
            return []

        scale, zero_point, quantized_dimension = self._extract_quant_params(dql_attributes, t_op,
                                                                            allow_int32_input=False)

        set_quantization_parameters_to_tensor(t_op.tmp_inputs[0], scale, zero_point, quantized_dimension)

        # Assign the operator its TFLite options, inputs and outputs
        t_op.builtin_options = dequantize_options.Dequantize()
        t_op.tmp_inputs = [t_op.tmp_inputs[0]]
        t_op.tmp_outputs = [t_op.tmp_outputs[0]]

        return [t_op]
