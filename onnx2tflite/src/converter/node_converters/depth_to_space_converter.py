#
# Copyright 2024,2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import depth_to_space_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import reshape_options, transpose_options
from onnx2tflite.src.tflite_generator.builtin_options.depth_to_space_options import DepthToSpace
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES


class DepthToSpaceConverter(NodeConverter):
    node = "DepthToSpace"
    tflite_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/depth_to_space.cc#L103-L143
    onnx_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT32, TensorType.INT64]
    verified_types = [TensorType.FLOAT32]

    def _convert_crd(self, attrs, t_op) -> list[tflite_model.Operator]:
        # Based on https://onnx.ai/onnx/operators/onnx__DepthToSpace.html#id2 but adapted for
        # NHWC tensor format.
        # tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
        # tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
        # y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])

        block_size = attrs.block_size

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        b, h, w, c = x.shape.vector
        c_new = c // (block_size ** 2)

        # Reshape #1
        pre_reshape_new_shape = [b, h, w, c_new, block_size, block_size]

        reshape_pre_output = self.builder.duplicate_tensor(x, x.name + "_pre_reshaped", empty_buffer=True)
        reshape_pre_output.shape = tflite_model.Shape(pre_reshape_new_shape)

        reshape_pre_op = tflite_model.Operator(builtin_options=reshape_options.Reshape(pre_reshape_new_shape))
        reshape_pre_op.tmp_inputs = [x]
        reshape_pre_op.tmp_outputs = [reshape_pre_output]

        # Transpose
        transpose_perm = [0, 1, 4, 2, 5, 3]

        perm = np.asarray(transpose_perm, np.int32)
        perm_tensor = self.builder.create_tensor_for_data(perm, "perm_")

        transpose_output_shape = translator.apply_permutation_to(pre_reshape_new_shape, transpose_perm)
        transpose_output = self.builder.duplicate_tensor(x, x.name + "_transposed", empty_buffer=True)
        transpose_output.shape = tflite_model.Shape(transpose_output_shape)

        transpose_op = tflite_model.Operator(builtin_options=transpose_options.Transpose())
        transpose_op.tmp_inputs = [reshape_pre_output, perm_tensor]
        transpose_op.tmp_outputs = [transpose_output]

        # Reshape #2
        reshape_post_op = tflite_model.Operator(builtin_options=reshape_options.Reshape(y.shape.vector))
        reshape_post_op.tmp_inputs = [transpose_output]
        reshape_post_op.tmp_outputs = [y]

        return [reshape_pre_op, transpose_op, reshape_post_op]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX `DepthToSpace` operator to TFLite `DepthToSpace`."""
        if len(t_op.tmp_inputs) != 1:
            logger.e(
                logger.Code.INVALID_ONNX_MODEL,
                f"ONNX `DepthToSpace` has unexpected number of inputs ({len(t_op.tmp_inputs)}).",
            )

        x = t_op.tmp_inputs[0]
        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        attrs = cast(depth_to_space_attributes.DepthToSpace, node.attributes)

        if attrs.mode == "CRD":
            return self._convert_crd(attrs, t_op)

        t_op.builtin_options = DepthToSpace(attrs.block_size)

        return [t_op]
