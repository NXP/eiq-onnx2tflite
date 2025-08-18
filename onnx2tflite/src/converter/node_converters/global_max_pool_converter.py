#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np

import onnx2tflite.lib.tflite.Padding as tflPadding
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator.builtin_options import max_pool_2d_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class GlobalMaxPoolConverter(NodeConverter):
    node = 'GlobalMaxPool'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/pooling.cc#L420-L440
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT16]
    verified_types = [TensorType.FLOAT32]

    def _convert_2d_global_max_pool(self, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the 2D ONNX `GlobalMaxPool` operator to TFLite `MaxPool2D`. """

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        ops_list = OpsList(middle_op=t_op)

        # Input and output types must be the same.
        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     "ONNX `GlobalMaxPool` has input and output tensors with different types.")

        if t_op.is_qdq_quantized():
            if x.quantization != y.quantization:
                # Quantization parameters of IO tensors don't match. Append re-quantization to output.
                scale = x.quantization.scale.vector
                zp = x.quantization.zero_point.vector
                ops_list.add_post(self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp))

        else:
            self.assert_type_allowed(x.type)

        input_height, input_width = x.shape[1:3]

        # Create the MaxPool2D
        t_op.builtin_options = max_pool_2d_options.MaxPool2D(
            tflPadding.Padding.VALID, filter_w=input_width, filter_h=input_height
        )

        return ops_list.flatten()

    def _convert_1d_global_max_pool(self, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX GlobalMaxPool operator with a 1D kernel, to TFLite. """
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        # Calculate the shapes, that the Reshape operators will use.
        new_shape_1 = x.shape.vector.copy()
        new_shape_1.insert(2, 1)
        reshape1 = self.builder.create_reshape_before(t_op, 0, new_shape_1)

        reshape_2_input_shape = new_shape_1.copy()  # Shape of the GlobalMaxPool output tensor.
        reshape_2_input_shape[1] = 1

        n, c = x.shape[0], x.shape[-1]
        y.shape = tflite_model.Shape([n, 1, 1, c])

        reshape2 = self.builder.create_reshape_after(t_op, 0, [n, 1, c])

        # Convert the now 2D GlobalMaxPool
        max_pool = self._convert_2d_global_max_pool(t_op)

        return [reshape1] + max_pool + [reshape2]

    def _convert_more_than_2d_global_max_pool(self, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX GlobalMaxPool operator with a kernel of at least 3 dimensions, to TFLite. """

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        # Calculate the shapes, that the Reshape operators will use.
        new_shape_1 = x.shape.vector.copy()
        new_shape_1[2:-1] = [np.prod(new_shape_1[2:-1]).item()]  # Combine multiple dimensions into 1
        reshape1 = self.builder.create_reshape_before(t_op, 0, new_shape_1)

        new_shape_2 = y.shape.vector.copy()

        n, c = x.shape[0], x.shape[-1]
        y.shape = tflite_model.Shape([n, 1, 1, c])

        reshape2 = self.builder.create_reshape_after(t_op, 0, new_shape_2)

        # Convert the now 2D GlobalMaxPool
        max_pool = self._convert_2d_global_max_pool(t_op)

        return [reshape1] + max_pool + [reshape2]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX `GlobalMaxPool` operator to TFLite `MaxPool2D` + potential `Reshape` operators. """

        rank = t_op.tmp_inputs[0].rank

        if rank == 3:
            # 1D kernel
            return self._convert_1d_global_max_pool(t_op)

        elif rank == 4:
            # 2D kernel
            return self._convert_2d_global_max_pool(t_op)

        elif rank >= 5:
            # kernel with at least 3 dimensions
            return self._convert_more_than_2d_global_max_pool(t_op)

        else:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"ONNX `GlobalMaxPool` has unexpected number of dimensions (`{rank}`).")
