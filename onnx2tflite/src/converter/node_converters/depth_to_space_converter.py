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
from onnx2tflite.src.onnx_parser.builtin_attributes import depth_to_space_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.depth_to_space_options import DepthToSpace
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES


class DepthToSpaceConverter(NodeConverter):
    node = 'DepthToSpace'
    tflite_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/depth_to_space.cc#L103-L143
    onnx_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT32, TensorType.INT64]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX `DepthToSpace` operator to TFLite `DepthToSpace`. """

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `DepthToSpace` has unexpected number of inputs ({len(t_op.tmp_inputs)}).')

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        attrs = cast(depth_to_space_attributes.DepthToSpace, node.attributes)

        if attrs.mode != 'DCR':
            # TFLite `DepthToSpace` always uses the `DCR` mode. Other modes would have to be converted into expanded
            #  form: Reshape -> Transpose -> Reshape, as defined in the ONNX documentation for other modes.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     f'Conversion of ONNX `DepthToSpace` with `mode={attrs.mode}` is not yet supported.')

        t_op.builtin_options = DepthToSpace(attrs.block_size)

        return [t_op]
