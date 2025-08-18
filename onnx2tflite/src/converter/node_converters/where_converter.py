#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.quantization_utils import propagate_quantization, \
    re_quantize_static_tensor
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import select_v2_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, INTS


class WhereConverter(NodeConverter):
    node = 'Where'

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/select.cc#L151-L183
    tflite_supported_types = INTS + [TensorType.UINT8, TensorType.UINT32, TensorType.BOOL, TensorType.FLOAT32]
    verified_types = [TensorType.INT32, TensorType.INT64, TensorType.UINT8, TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert ONNX 'Where' operator into TFLite 'SelectV2'. """

        if len(t_op.tmp_inputs) != 3:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX 'Where' has unexpected number of inputs. "
                                                     f"Got '{len(t_op.tmp_inputs)}', expected '3'.")

        condition = t_op.tmp_inputs[0]
        x = t_op.tmp_inputs[1]
        y = t_op.tmp_inputs[2]
        out = t_op.tmp_outputs[0]

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        if t_op.is_quantized_without_qdq():
            logger.w(
                f"Propagating quantization params from first input tensor ('{x.name}') to output tensor ('{out.name}')"
                " in 'Where' operator. This can negatively affect accuracy of the output.")
            propagate_quantization(x, out)

        ops = OpsList(middle_op=t_op)

        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, f"ONNX operator 'Where' has inputs with different data types!")

        if condition.type != TensorType.BOOL:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, f"Type of condition tensor in 'Where' operator is not bool!")

        ops.pre_ops.extend(self.builder.ensure_correct_broadcasting(t_op, out))

        if t_op.is_qdq_quantized():
            # Verify that q-params of IO tensors match
            scale = out.quantization.scale.vector
            zp = out.quantization.zero_point.vector

            for idx, input_tensor in enumerate(t_op.tmp_inputs):
                if idx == 0:  # Skip condition tensor
                    continue

                if input_tensor.quantization != out.quantization:
                    # Tensor's q-params doesn't match with output ones
                    if tensor_has_data(input_tensor):
                        logger.w(
                            f"Requantizing tensor '{input_tensor.name}' to match q-params of 'Where' output tensor.")
                        input_tensor = re_quantize_static_tensor(self.builder, input_tensor, y.type, scale, zp)
                        t_op.tmp_inputs[idx] = input_tensor
                    else:
                        logger.w(f"Re-quantizing input tensor '{input_tensor.name}' of 'Where' op to satisfy TFLite's "
                                 "q-param requirements. This can decrease accuracy of the model.")
                        ops.pre_ops.append(self.builder.create_quantize_operator_before(t_op, idx, x.type, scale, zp))

        t_op.builtin_options = select_v2_options.SelectV2()

        return ops.flatten()
