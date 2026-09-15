#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#


import numpy as np

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS
from onnx2tflite.src import errors
from onnx2tflite.src.errors import ConversionError


class AtanConverter(NodeConverter):
    node = "Atan"

    # TFLite has no `Atan` operator, so it is emitted as `Atan2(x, 1)`.
    # `Atan2` supports floats only.
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/atan2.cc
    onnx_supported_types = FLOATS
    tflite_supported_types = [TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `Atan` operator into TFLite `Atan2`, using `atan(x) == atan2(x, 1)`."""
        if len(t_op.tmp_inputs) != 1:
            raise ConversionError(errors.Code.INVALID_ONNX_MODEL, f"ONNX `Atan` has unexpected number of inputs ({len(t_op.tmp_inputs)}).")

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        # TFLite `Atan2` requires both operands to have the same shape (its kernel does not
        # broadcast), so the constant `1` tensor is created with `x`'s full shape.
        ones = self.builder.create_tensor_for_data(np.ones(x.shape.vector, dtype=np.float32), "atan_ones")

        t_op.builtin_options = None
        t_op.opcode_index = self.builder.op_code_index_for_op_type(BuiltinOperator.ATAN2)
        t_op.tmp_inputs = [x, ones]

        return [t_op]
