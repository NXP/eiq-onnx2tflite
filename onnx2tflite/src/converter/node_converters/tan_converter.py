#
# Copyright 2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import div_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class TanConverter(NodeConverter):
    node = "Tan"

    # Sin + Cos support only floats
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/elementwise.cc#L504
    onnx_supported_types = FLOATS
    tflite_supported_types = [TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `Tan` operator into TFLite `Sin`, `Cos` and `Div`."""
        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `Tan` has unexpected number of inputs ({len(t_op.tmp_inputs)}).")

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        cos_output = self.builder.duplicate_tensor(x, name_suffix="_cos_out")
        sin_output = self.builder.duplicate_tensor(x, name_suffix="_sin_out")

        sin_op = tflite_model.Operator()
        sin_op.opcode_index = self.builder.op_code_index_for_op_type(BuiltinOperator.SIN)
        sin_op.tmp_inputs = [x]
        sin_op.tmp_outputs = [sin_output]

        cos_op = tflite_model.Operator()
        cos_op.opcode_index = self.builder.op_code_index_for_op_type(BuiltinOperator.COS)
        cos_op.tmp_inputs = [x]
        cos_op.tmp_outputs = [cos_output]

        div_op = tflite_model.Operator(builtin_options=div_options.Div())
        div_op.tmp_inputs = [sin_output, cos_output]
        div_op.tmp_outputs = [t_op.tmp_outputs[0]]

        return [sin_op, cos_op, div_op]
