#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import elu_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class EluConverter(NodeConverter):
    node = "Elu"

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.18.1/tensorflow/lite/kernels/activations.cc#L1503-L1522
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT8]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `Elu`` operator into TFLite `Elu`."""

        x = t_op.tmp_inputs[0]

        if t_op.is_qdq_quantized():
            # TFLite `ELU` has no UINT8 quantized kernel. Only INT8 is supported.
            if x.type != TensorType.INT8:
                logger.e(
                    logger.Code.CONVERSION_IMPOSSIBLE,
                    f"Quantized ONNX `Elu` is only supported for INT8, got `{x.type}`.",
                )
        else:
            self.assert_type_allowed(x.type)

        attrs = cast(elu_attributes.Elu, node.attributes)

        if not np.allclose(attrs.alpha, 1.0):
            # TFLite only uses `alpha=1.0`. Other values could be converted into multiple operators, but it doesn't seem
            #  like a common use-case.
            logger.e(
                logger.Code.NOT_IMPLEMENTED, f"Conversion of ONNX `Elu` with `alpha={attrs.alpha}` is not supported."
            )

        t_op.opcode_index = self.builder.op_code_index_for_op_type(BuiltinOperator.ELU)

        return [t_op]
