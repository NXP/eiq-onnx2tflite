#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import List, cast

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import cast_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import cast_options
from onnx2tflite.src.tflite_generator.meta.types import name_for_type


class CastConverter(NodeConverter):
    node = 'Cast'

    # noinspection PyMethodMayBeStatic
    def _check_input_type(self, from_type: TensorType):
        if from_type == TensorType.UINT16:
            # For some reason, I couldn't run the TFLite model with uint16 input. Some weird error was raised when
            # 'SetTensor' was called. I couldn't find any test, where we used uint16 input tensors.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     'Conversion of ONNX `Cast` with input type `UINT16` is not yet implemented.')

        if from_type in {TensorType.UINT64, TensorType.STRING}:
            # Unsupported by TFLite.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f'Conversion of ONNX `Cast` with input type `{name_for_type(from_type)}` is not possible.')

        if from_type in {TensorType.COMPLEX64, TensorType.COMPLEX128}:
            # ONNXRT: Unsupported by ONNX/ORT.
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f'Conversion of ONNX `Cast` with input type `{name_for_type(from_type)}` is not supported.')

        if from_type not in {TensorType.UINT8, TensorType.UINT32,
                             TensorType.INT8, TensorType.INT16, TensorType.INT32, TensorType.INT64,
                             TensorType.FLOAT16, TensorType.FLOAT32, TensorType.FLOAT64, TensorType.BOOL}:
            # Only these types are verified.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     f'Conversion of ONNX `Cast` with input type `{name_for_type(from_type)}` is not implemented.')

    # noinspection PyMethodMayBeStatic
    def _check_output_type(self, to_type: TensorType):
        if to_type in {TensorType.UINT64, TensorType.STRING}:
            # Not supported by TFLite
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f'Conversion of ONNX `Cast` with output type `{name_for_type(to_type)}` is not possible.')

        if to_type in {TensorType.COMPLEX64, TensorType.COMPLEX128}:
            # ONNXRT: Not supported by ONNX/ORT
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f'Conversion of ONNX `Cast` with output type `{name_for_type(to_type)}` is not supported.')

        if to_type not in {TensorType.UINT8, TensorType.UINT16, TensorType.UINT32,
                           TensorType.INT8, TensorType.INT16, TensorType.INT32, TensorType.INT64,
                           TensorType.FLOAT16, TensorType.FLOAT32, TensorType.FLOAT64, TensorType.BOOL}:
            # Only these types are verified.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     f'Conversion of ONNX `Cast` with output type `{name_for_type(to_type)}` is not implemented.')

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert ONNX Cast operator into TFLite.

        :param node: ONNX NodeProto representing the Cast operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators to add to the model.
        """

        if len(t_op.tmp_inputs) != 1 or len(t_op.tmp_outputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `Cast` has invalid number of input and output tensors.')

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        o_cast = cast(cast_attributes.Cast, node.attributes)

        from_type = x.type
        to_type = translator.convert_data_type(o_cast.to)

        self._check_input_type(from_type)
        self._check_output_type(to_type)

        if t_op.is_quantized_without_qdq():
            if x.type == y.type:
                # The type stays the same. Knowing the quantization parameters later on could be useful to us, so
                #  propagate them to the output tensor.
                propagate_quantization(x, y)

            else:
                # It doesn't make sense to propagate the quantization parameters to the output, because they may be
                #  invalid (e.g. int8 -> uint8 with zero point -1).
                # It makes sense to do nothing, because the `Cast` doesn't care that the input is quantized, and simply
                #  converts the raw data to a different type. This behavior is the same in ONNX and TFLite.
                logger.w(
                    'ONNX `Cast` has a quantized input, which is not standard usage. Output model might not be valid.')

        if y.type != to_type:
            # This probably indicates an error in shape inference.
            logger.d("ONNX 'Cast' has output type different from the 'to' attribute.")

            y.type = to_type

        t_op.builtin_options = cast_options.Cast(from_type, to_type)

        return [t_op]
