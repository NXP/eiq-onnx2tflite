#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from typing import cast

import numpy as np
from onnx import TensorProto

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes.constant_attributes import Constant
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import transpose_options
from onnx2tflite.src.tflite_generator.meta.types import INTS, ALL_TYPES


class ConstantConverter(NodeConverter):
    node = 'Constant'

    onnx_supported_types = ALL_TYPES
    tflite_supported_types = INTS + [TensorType.UINT8, TensorType.FLOAT32]  # Supported by TFLite `Transpose`.
    verified_types = tflite_supported_types

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert the ONNX Constant operator to TFLite.
            Since the operator simply produces a constant value, there is no need to create a TFLite counterpart. Just
             create a static tensor in the model. Since the tensor is already in the model, just add data to it.
        """
        attrs = cast(Constant, node.attributes)

        output = t_op.tmp_outputs[0]

        if output.tmp_buffer is None:
            output.tmp_buffer = self.builder.build_empty_buffer()

        if hasattr(attrs, "value"):
            if attrs.value.data_type in [TensorProto.COMPLEX64, TensorProto.COMPLEX128]:
                # ONNXRT: Limitation
                logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                         "ONNX 'Constant' produces output with a complex type which is "
                         "not supported by the ONNX Runtime.")

            if attrs.value.data_type == TensorProto.STRING:
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "ONNX 'Constant' produces output of string type, which is not yet supported.")

            # Other types work just fine
            output.tmp_buffer.data = np.asarray(attrs.value.data)

        elif hasattr(attrs, "value_int"):
            output.tmp_buffer.data = np.int64([attrs.value_int])
        elif hasattr(attrs, "value_float"):
            output.tmp_buffer.data = np.float32([attrs.value_float])
        elif hasattr(attrs, "value_str"):
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX 'Constant' which produces a string tensor is not "
                                                  "yet supported.")

        elif hasattr(attrs, "value_ints"):
            output.tmp_buffer.data = np.asarray(attrs.value_ints, np.int64)
        elif hasattr(attrs, "value_floats"):
            output.tmp_buffer.data = np.asarray(attrs.value_floats, np.float32)
        elif hasattr(attrs, "value_strings"):
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX 'Constant' which produces a string tensor is not "
                                                  "yet supported.")

        elif hasattr(attrs, "sparse_value"):
            # TODO Implementing support for TFLite sparse tensors is not a priority right now.
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX 'Constant' which produces a sparse tensor is not "
                                                  "yet supported.")

        if self.builder.operator_produces_graph_output(t_op):
            # The operator cannot just be skipped. Return a Transpose operator, which does nothing. (rare use-case)
            self.assert_type_allowed(t_op.tmp_outputs[0].type)

            t_op.builtin_options = transpose_options.Transpose()

            identity = np.asarray(range(output.rank), np.int32)
            identity_tensor = self.builder.create_tensor_for_data(identity, "identity")

            new_input = self.builder.duplicate_tensor(output, name_suffix='_static_')

            # Now 'new_input' will hold the static data and 'output' will be computed. Remove static data from 'output'.
            output.tmp_buffer.data = np.array([])

            t_op.tmp_inputs = [new_input, identity_tensor]
            t_op.tmp_outputs = [output]

            return [t_op]

        return []
