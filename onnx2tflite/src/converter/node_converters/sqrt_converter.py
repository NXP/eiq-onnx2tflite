#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class SqrtConverter(NodeConverter):
    node = 'Sqrt'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/elementwise.cc#L374
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/elementwise.cc#L250-L253
    tflite_supported_types = [TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX `Sqrt` to TFLite `Sqrt`.

            The TFLite variant has no builtin options in the library, just set it to `None` and use the opcode_index
             directly.
        """

        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        t_op.builtin_options = None
        t_op.opcode_index = self.builder.op_code_index_for_op_type(BuiltinOperator.SQRT)

        return [t_op]
