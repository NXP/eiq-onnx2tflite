#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import cum_sum_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import cum_sum_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, name_for_type


class CumSumConverter(NodeConverter):
    node = "CumSum"

    onnx_supported_types = FLOATS + [TensorType.INT32, TensorType.INT64, TensorType.UINT32, TensorType.UINT64]
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/cumsum.cc#L71-L88
    tflite_supported_types = [TensorType.INT32, TensorType.INT64, TensorType.FLOAT32]
    verified_types = [TensorType.INT32, TensorType.INT64, TensorType.FLOAT32]

    def _check_axis_type(self, t_op: tflite_model.Operator) -> None:
        axis = t_op.tmp_inputs[1]

        if axis.type == TensorType.INT32:
            # All is good.
            pass

        elif axis.type == TensorType.INT64:
            # Re-cast to int32.
            if tensor_has_data(axis):
                # Static re-cast. Create a new tensor with the re-cast data, in case it was used by some other operator.
                int32_axis = axis.tmp_buffer.data.astype(np.int32)
                new_axis_tensor = self.builder.create_tensor_for_data(int32_axis, "axis")
                t_op.tmp_inputs[1] = new_axis_tensor

            else:
                # Prepend a `Cast` operator.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX `CumSum` with a dynamic INT64 `axis` is not yet supported.")

        else:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `CumSum` has unexpected `axis` type ({name_for_type(axis.type)}).")

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `CumSum`` operator into TFLite `CumSum`."""
        ops = OpsList(middle_op=t_op)

        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `CumSum` has unexpected number of inputs ({len(t_op.tmp_inputs)}).")

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        self._check_axis_type(t_op)

        if x.tensor_format.is_channels_last():
            # The `axis` refers to an axis of a channels_first tensors. Modify it to work with a channels_last tensor.
            axis_tensor = t_op.tmp_inputs[1]
            if tensor_has_data(axis_tensor):
                axis = axis_tensor.tmp_buffer.data.item()
                perm = translator.create_channels_last_to_channels_first_permutation(x.rank)
                axis = perm[axis]

                # Create a new `axis` tensor, in case it is used by some other operator as well.
                new_axis = self.builder.create_tensor_for_data(np.array([axis], np.int32), "axis")
                t_op.tmp_inputs[1] = new_axis

            else:
                logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX `CumSum` with a channels first input and a "
                                                      "dynamic `axis` is not supported.")

        attrs = cast(cum_sum_attributes.CumSum, node.attributes)
        t_op.builtin_options = cum_sum_options.CumSum(bool(attrs.exclusive), bool(attrs.reverse))

        return ops.flatten()
