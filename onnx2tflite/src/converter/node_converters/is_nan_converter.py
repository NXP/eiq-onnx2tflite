#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import equal_options, logical_not_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class IsNaNConverter(NodeConverter):
    node = "IsNaN"

    onnx_supported_types = FLOATS
    tflite_supported_types = [TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `IsNaN` to TFLite `Equal` + `LogicalNot`.

        Equal(x, x) -> produces False for NaNs, True otherwise.
        LogicalNot(Equal) -> flips the result -> True for NaNs.

        :param node: ONNX `IsNaN` operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        is_equal = self.builder.duplicate_tensor(x)

        equal = tflite_model.Operator(builtin_options=equal_options.Equal())
        equal.tmp_inputs = [x, x]
        equal.tmp_outputs = [is_equal]

        logical_not = tflite_model.Operator(builtin_options=logical_not_options.LogicalNot())
        logical_not.tmp_inputs = [is_equal]
        logical_not.tmp_outputs = [y]

        return [equal, logical_not]
