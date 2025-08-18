#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List, cast

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import space_to_depth_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.space_to_depth_options import SpaceToDepth
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES


class SpaceToDepthConverter(NodeConverter):
    node = 'SpaceToDepth'
    tflite_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/space_to_depth.cc#L99-L139
    onnx_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT32, TensorType.INT64]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX `SpaceToDepth` operator to TFLite `SpaceToDepth`. """

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `SpaceToDepth` has unexpected number of inputs ({len(t_op.tmp_inputs)}).')

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        attrs = cast(space_to_depth_attributes.SpaceToDepth, node.attributes)
        t_op.builtin_options = SpaceToDepth(attrs.block_size)

        return [t_op]
