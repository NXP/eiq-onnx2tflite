#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import neg_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS


class NegConverter(NodeConverter):
    node = "Neg"

    onnx_supported_types = FLOATS + INTS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/neg.cc#L54-L76
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]
    verified_types = [TensorType.FLOAT32, TensorType.INT32, TensorType.INT64]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX `Neg` operator to TFLite `Neg`."""
        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `Neg` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `1`.")

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(t_op.tmp_inputs[0].type)

        t_op.builtin_options = neg_options.Neg()

        return [t_op]
