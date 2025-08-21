#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import div_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


class DivConverter(NodeConverter):
    node = "Div"

    onnx_supported_types = FLOATS + INTS + UINTS
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32, TensorType.INT32]

    def _cast_from_int64_to_int32(self, ops, t_op):
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_inputs[1]

        if tensor_has_data(x):
            x.tmp_buffer.data = x.tmp_buffer.data.astype(np.int32)
            x.type = TensorType.INT32
        else:
            ops.add_pre(self.builder.create_cast_before(t_op, 0, TensorType.INT32))

        if tensor_has_data(y):
            y.tmp_buffer.data = y.tmp_buffer.data.astype(np.int32)
            y.type = TensorType.INT32
        else:
            ops.add_pre(self.builder.create_cast_before(t_op, 1, TensorType.INT32))

        ops.add_post(self.builder.create_cast_after(t_op, 0, TensorType.INT32))

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'Div' operator to TFLite 'Div'."""
        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, f"ONNX operator 'Div' has '{len(t_op.tmp_inputs)}' inputs!")

        input_a = t_op.tmp_inputs[0]
        input_b = t_op.tmp_inputs[1]

        if input_a.type != input_b.type:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX operator 'Div' has inputs with different data types!")

        ops = OpsList(middle_op=t_op)
        ops.add_pre(self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0]))

        if t_op.is_quantized_without_qdq():
            # ONNXRT: INT8 and UINT8 are not supported by ONNX Runtime, so conversion cannot be verified.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     f"Conversion of ONNX `Div` with quantized inputs of type {input_a.type} is not supported.")

        if input_a.type == TensorType.INT64:
            if not self.context.conversion_config.cast_int64_to_int32:
                logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                         "TFLite operator 'Div' doesn't support type INT64. " + logger.Message.CAST_INT64_TO_INT32)
            else:
                self._cast_from_int64_to_int32(ops, t_op)

        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        t_op.builtin_options = div_options.Div()

        return ops.flatten()
