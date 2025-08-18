#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import equal_options
from onnx2tflite.src.tflite_generator.meta.types import name_for_type, FLOATS, INTS, UINTS


class EqualConverter(NodeConverter):
    node = 'Equal'

    onnx_supported_types = FLOATS + INTS + UINTS + [TensorType.BOOL, TensorType.STRING]
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/comparisons.cc#L173-L212
    tflite_supported_types = INTS + [TensorType.BOOL, TensorType.FLOAT32, TensorType.UINT8, TensorType.STRING]
    verified_types = [TensorType.INT32, TensorType.INT64, TensorType.BOOL, TensorType.FLOAT32, TensorType.STRING]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX 'Equal' operator to TFLite 'Equal'. """

        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX 'Equal' has unexpected number of inputs. "
                                                     f"Got '{len(t_op.tmp_inputs)}', expected '2'.")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_inputs[1]

        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX 'Equal' has mismatched input data types!")

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        if x.type in {TensorType.INT8, TensorType.UINT8}:
            # ONNXRT: Not supported by ONNX/ORT.
            if x.quantization is None:
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         f'Conversion of ONNX `Equal` with input type `{name_for_type(x.type)}` is not supported.')
            else:
                # The input is quantized. ONNX Runtime doesn't support int8/uint8 regardless of quantization.
                # Using the QDQ quantizer will result in an intentionally invalid ONNX model, which can be converted to
                #  TFLite and run without issues.
                # Non QDQ quantized `Equal` is not supported by ONNX Runtime and is not supported by the converter.
                #  Since it is impossible to detect if the Equal was QDQ quantized or not right now (because output has
                #  type bool), there is no way to catch the potential invalid use. But it shouldn't be common at all.
                pass

        reshape_and_transpose_ops = self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0])

        t_op.builtin_options = equal_options.Equal()

        return reshape_and_transpose_ops + [t_op]
