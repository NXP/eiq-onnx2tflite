#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np
from onnx import TensorProto as onnxType

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model, onnx_tensor
from onnx2tflite.src.onnx_parser.builtin_attributes.constant_of_shape_attributes import ConstantOfShape
from onnx2tflite.src.onnx_parser.onnx_tensor import TensorProto
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import broadcast_to_options, gather_options
from onnx2tflite.src.tflite_generator.meta.types import name_for_type


class ConstantOfShapeConverter(NodeConverter):
    node = "ConstantOfShape"

    def _validate_value_tensor(self, value: TensorProto) -> tflite_model.Tensor:
        """Make sure the 'value' is a valid 1 element ONNX tensor and return a corresponding TFLite tensor."""
        # ONNX documentation says the 'value' should always be a tensor.
        if not isinstance(value, onnx_tensor.TensorProto):
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX ConstantOfShape is only implemented when the "
                                                  "'value' attribute is a tensor!")

        if value.data.size != 1:
            # ONNX Runtime only supports 1 element tensors and documentation also suggests a 1D tensor.
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX ConstantOfShape attribute 'value' should only have '1' "
                                                     f"element! Got '{value.data.size()}'!")

        if value.data_type not in {onnxType.BOOL, onnxType.DOUBLE, onnxType.FLOAT, onnxType.FLOAT16,
                                   onnxType.INT16, onnxType.INT32, onnxType.INT64, onnxType.INT8,
                                   onnxType.UINT16, onnxType.UINT32, onnxType.UINT64, onnxType.UINT8}:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX ConstantOfShape has 'value' tensor of type '{value.data_type}'"
                     " which is not allowed in the documentation!")

        return self.builder.create_tensor_for_data(value.data, "value")

    def _prepend_gather_operator(self, broadcast_to_op: tflite_model.Operator, ops_to_add: list[tflite_model.Operator]):
        """Create a TFLite 'Gather' operator in front the 'broadcast_to_op' operator and add it to 'ops_to_add'.
        The 'Gather' will permute the input data from representing the shape of a channels first tensor, to a shape of
        a channels last tensor.
        """
        output_rank = broadcast_to_op.tmp_outputs[0].rank
        if output_rank == 0:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, "Conversion of ONNX 'ConstantOfShape' with a channels first "
                                                        "output and a dynamic 'shape' with unknown rank is not possible!")

        to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(output_rank)
        gather_output = self.builder.duplicate_tensor(broadcast_to_op.tmp_inputs[1], "channels_last_shape")
        gather_indices = self.builder.create_tensor_for_data(to_tflite_perm, "to_channels_last_perm")

        gather_op = tflite_model.Operator(builtin_options=gather_options.Gather(0))
        gather_op.tmp_inputs = [broadcast_to_op.tmp_inputs[1], gather_indices]
        gather_op.tmp_outputs = [gather_output]

        broadcast_to_op.tmp_inputs[1] = gather_output

        ops_to_add.insert(0, gather_op)

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX ConstantOfShape operator to TFLite BroadcastTo."""
        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX ConstantOfShape has unexpected number of operators. "
                                                     f"Got '{len(t_op.tmp_inputs)}', expected '1'.")

        shape_tensor = t_op.tmp_inputs[0]
        if shape_tensor.type != TensorType.INT64:
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX operator ConstantOfShape has input 'shape' tensor of type "
                                                     f"'{name_for_type(shape_tensor.type)}' instead of the expected 'INT64'!")

        attrs = cast(ConstantOfShape, node.attributes)

        output_tensor = t_op.tmp_outputs[0]
        ops_to_add = [t_op]

        # TODO If the 'shape' operand is static, the output tensor can be statically added to the model during
        #  conversion. This would make the model larger, but inference might be faster. Perhaps add the option for user
        #  to decide

        value_tensor = self._validate_value_tensor(attrs.value)
        t_op.tmp_inputs = [value_tensor, shape_tensor]

        if output_tensor.tensor_format.is_channels_last():
            if tensor_has_data(shape_tensor):
                # The original output was channels first. Statically convert the shape to channels last.
                channels_last_shape = translator.dims_to_channels_last(list(shape_tensor.tmp_buffer.data))
                shape_tensor.tmp_buffer.data = np.asarray(channels_last_shape, np.int64)

            else:
                # Need to prepend a 'Gather' operator to change the input shape to channels last
                self._prepend_gather_operator(t_op, ops_to_add)

        t_op.builtin_options = broadcast_to_options.BroadcastTo()
        t_op.tmp_version = 2  # Version 1 is not supported by the inference engine

        return ops_to_add
