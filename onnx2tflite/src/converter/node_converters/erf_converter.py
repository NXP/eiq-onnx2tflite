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
from onnx2tflite.src.tflite_generator.custom_options.flex_erf_options import Erf
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


class ErfConverter(NodeConverter):
    node = "Erf"

    onnx_supported_types = FLOATS + INTS + UINTS
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `Erf` to TFLite `FlexErf`, which requires the Flex delegate and is part of `SELECT_TF_OPS`."""
        if not self.context.conversion_config.allow_select_ops:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX `Erf` without SELECT_TF_OPS is not possible. " +
                     logger.Message.ALLOW_SELECT_OPS)

        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        # Custom options are generated in FlexBuffers format (similar to FlatBuffers but without schema definition).
        # Options contain serialized name of the operator and operator attributes. Specification for Erf operator:
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/compat/ops_history_v2/Erf.pbtxt
        t_op.custom_options = Erf()

        return [t_op]
