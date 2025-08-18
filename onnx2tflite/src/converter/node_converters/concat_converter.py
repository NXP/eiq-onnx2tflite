#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List, cast

import onnx2tflite.src.onnx_parser.builtin_attributes.concat_attributes as concat_attributes
import onnx2tflite.src.tflite_generator.builtin_options.concatenation_options as tfl_concatenation_options
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.quantization_utils import propagate_quantization, \
    re_quantize_static_tensor, quantize_static_float_tensor
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import UINTS, INTS, FLOATS


class ConcatConverter(NodeConverter):
    node = 'Concat'

    onnx_supported_types = FLOATS + INTS + UINTS + [TensorType.BOOL, TensorType.COMPLEX64, TensorType.COMPLEX128,
                                                    TensorType.STRING]
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/concatenation.cc#L140-L144
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8, TensorType.UINT32, TensorType.BOOL]
    verified_types = tflite_supported_types

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        self.assert_type_allowed(t_op.tmp_outputs[0].type)

        attrs = cast(concat_attributes, node.attributes)

        axis = attrs.axis
        rank = len(t_op.tmp_inputs[0].shape.vector)

        if axis < -rank or axis > rank - 1:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f"ONNX attribute 'axis' ({axis}) must be in range [{-rank}, {rank - 1}]!!")

        if axis < 0:  # convert negative index to positive
            axis += rank

        ops = OpsList(middle_op=t_op)

        y = t_op.tmp_outputs[0]

        if t_op.tmp_inputs[0].quantization is not None:
            # TODO(Lukas) This might be dangerous when:
            # 1. First tensor is non-quantized and static + second tensor is quantized
            # 2. Range of first tensor is smaller than range of others -> which one to choose?
            if y.quantization is None:
                logger.w(f"Propagating q-params from tensor '{t_op.tmp_inputs[0].name}' to output tensor of Concat op. "
                         "This can decrease accuracy of the model.")
                propagate_quantization(t_op.tmp_inputs[0], y)

            scale = y.quantization.scale.vector
            zp = y.quantization.zero_point.vector

            # Re-quantize input tensors if they don't match with output quantization. This applies only to INT8
            # because UINT8 quantization with different q-params is supported in TFLite.
            for idx, input_tensor in enumerate(t_op.tmp_inputs):
                if tensor_has_data(input_tensor):
                    if input_tensor.quantization is None:
                        # Tensor wasn't quantized because it is constant (our QDQ quantizer was used) -> quantize the
                        #  data.
                        logger.w(
                            f"Quantizing tensor '{input_tensor.name}' using q-params of Concat's op output tensor.")
                        input_tensor = quantize_static_float_tensor(self.builder, input_tensor, y.type, scale, zp)
                        t_op.tmp_inputs[idx] = input_tensor

                    elif input_tensor.quantization != y.quantization and input_tensor.type == TensorType.INT8:
                        # Tensor's q-params doesn't match with output ones and type is INT8 -> re-quantize data
                        logger.w(f"Re-quantizing tensor '{input_tensor.name}' to match output tensor's q-params of "
                                 "Concat op.")
                        input_tensor = re_quantize_static_tensor(self.builder, input_tensor, y.type, scale, zp)
                        t_op.tmp_inputs[idx] = input_tensor
                else:
                    # TFLite support re-quantization for UINT8, so apply only to INT8
                    if input_tensor.quantization != y.quantization and input_tensor.type == TensorType.INT8:
                        ops.add_pre(self.builder.create_quantize_operator_before(t_op, idx, y.type, scale, zp))
                        logger.w(f"Re-quantizing tensor '{input_tensor.name}' to match output tensor's q-params of "
                                 "Concat op.")

        if t_op.tmp_inputs[0].tensor_format.is_channels_last():
            axis = translator.create_channels_last_to_channels_first_permutation(rank)[axis]

        t_op.builtin_options = tfl_concatenation_options.Concatenation(axis)

        return ops.flatten()
