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
from onnx2tflite.src.tflite_generator.builtin_options import hard_swish_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class HardSwishConverter(NodeConverter):
    node = "HardSwish"

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/activations.cc#L805-L850
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT8, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX `HardSwish` operator to TFLite `HardSwish`.

        :param node: ONNX `HardSwish` operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `HardSwish` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `1`.")

        x = t_op.tmp_inputs[0]
        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        t_op.builtin_options = hard_swish_options.HardSwish()

        return [t_op]
