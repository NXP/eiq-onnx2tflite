#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.abs_options import Abs
from onnx2tflite.src.tflite_generator.meta.types import INTS, FLOATS, UINTS


class AbsConverter(NodeConverter):
    node = 'Abs'
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT8, TensorType.INT16, TensorType.INT32]
    onnx_supported_types = FLOATS + INTS + UINTS
    verified_types = [TensorType.INT16, TensorType.INT32, TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX `Abs` operator to TFLite `Abs`. """

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `Abs` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `1`.')

        if t_op.is_quantized_without_qdq():
            # ONNX `Abs` computes on the INT data as if it wasn't quantized. TFLite cannot do that, it always takes the
            # quantization parameters into account, so the output would be different.
            # Conversion may be possible by adding some operator which could remove the quantization parameters.
            logger.e(logger.Code.NOT_IMPLEMENTED, 'Conversion of ONNX `Abs` with a quantized input is not supported.')

        ops = OpsList(middle_op=t_op)

        x = t_op.tmp_inputs[0]
        if x.type == TensorType.INT64:
            # Required by real model. Cast to int32.
            if not self.context.conversion_config.cast_int64_to_int32:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         'Direct conversion of ONNX `Abs` with type INT64 is not possible due to a TFLite '
                         'limitation.' + logger.Message.CAST_INT64_TO_INT32)

            # Recast the input.
            if tensor_has_data(x):
                # Create a new tensor in case it is used by some other operator as well.
                t_op.tmp_inputs[0] = self.builder.create_tensor_for_data(x.tmp_buffer.data.astype(np.int32), x.name)

            else:
                ops.add_pre(self.builder.create_cast_before(t_op, 0, TensorType.INT32))

            # Re-cast the output.
            ops.add_post(self.builder.create_cast_after(t_op, 0, TensorType.INT32))

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        if x.type == TensorType.INT8 and x.quantization is None:
            # Int8 is only allowed for quantized data.
            # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/elementwise.cc#L142-L144
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f'Conversion of ONNX `Abs` with input type `INT8` without quantization is not possible.')

        t_op.builtin_options = Abs()

        return ops.flatten()
