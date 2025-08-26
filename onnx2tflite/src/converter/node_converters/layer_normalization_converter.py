#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import try_get_input, uses_shape_broadcasting
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import layer_normalization_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import (
    add_options,
    mean_options,
    mul_options,
    square_options,
    sub_options,
)
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS


# noinspection PyMethodMayBeStatic
class LayerNormalizationConverter(NodeConverter):
    node = "LayerNormalization"

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/reduce.cc#L525-L548
    tflite_supported_types = INTS + [TensorType.UINT8, TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32]

    def _axes_from_axis(self, axis: int, input_tensor: tflite_model.Tensor) -> np.ndarray:
        """Return all axes that are >= axis, up to the rank of the input tensor.

        :param axis: Index of a dimension of the input tensor.
        :param input_tensor: Input tensor of a LayerNormalization operator.
        :return: A list of indices of dimensions, that are higher than or equal to 'axis'.
        """
        rank = input_tensor.rank

        if not (-rank <= axis <= rank):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, f"ONNX LayerNormalization attribute 'axis' has value "
                                                                  f"'{axis}', which is outside of the allowed range "
                                                                  f"['{-rank}', '{rank}']!")

        if axis < 0:
            axis += rank

        axes = list(range(rank))[axis:]

        if input_tensor.tensor_format.is_channels_last():
            # Original axes refer to an ONNX shape. Update them to refer to the same dimensions but for a TFLite shape
            tmp_permutation = translator.create_channels_last_to_channels_first_permutation(rank)
            axes = [tmp_permutation[axis] for axis in axes]

        return np.asarray(axes, np.int32)

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'LayerNormalization' operator to TFLite.

            There is no corresponding TFLite operator to the ONNX LayerNormalization. The only way to convert it is to
             use multiple simple operators, that carry out the internal computation of LayerNormalization, which is:

                d = x - mean(x, axis)
                y = (d / sqrt(mean(d**2, axis) + epsilon))) * scale + b

        :param node: ONNX LayerNormalization operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator
        :return: A list of TFLite operators, to add to the model.
        """
        if len(t_op.tmp_inputs) not in {2, 3}:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX 'LayerNormalization' has unexpected number of inputs! "
                                                     f"Got '{len(t_op.tmp_inputs)}', expected '2' or '3'.")

        # Assign the operands
        x = t_op.tmp_inputs[0]
        scale = t_op.tmp_inputs[1]
        b = try_get_input(t_op, 2)

        y = t_op.tmp_outputs[0]

        self.assert_type_allowed(x.type)

        layer_normalization = cast(layer_normalization_attributes.LayerNormalization, node.attributes)

        epsilon = np.asarray([layer_normalization.epsilon], np.float32)
        axes = self._axes_from_axis(layer_normalization.axis, x)

        # Create the tensors taking part in the computation. Names are based on the ONNX documentation.
        axes_tensor = self.builder.create_tensor_for_data(axes, "axes")
        epsilon_tensor = self.builder.create_tensor_for_data(epsilon, "epsilon")
        d = self.builder.duplicate_tensor(x, "D")
        d_squared = self.builder.duplicate_tensor(x, "D squared")
        var = self.builder.duplicate_tensor(x, "var")
        var_eps = self.builder.duplicate_tensor(x, "var_eps")
        normalized = self.builder.duplicate_tensor(x, "normalized")
        normalized_scaled = self.builder.duplicate_tensor(x, "normalized_scaled")

        if len(t_op.tmp_outputs) >= 2:
            # The operator also outputs the 'Mean' tensor
            mean = t_op.tmp_outputs[1]
        else:
            mean = self.builder.duplicate_tensor(x, "mean")

        if len(t_op.tmp_outputs) == 3:
            # The operator also outputs the 'InvStdDev ' tensor
            inv_std_dev = t_op.tmp_outputs[2]
        else:
            inv_std_dev = self.builder.duplicate_tensor(x, "inv_std_dev")

        # Create the operators
        mean_1_op = tflite_model.Operator(builtin_options=mean_options.Mean(True))
        mean_1_op.tmp_inputs = [x, axes_tensor]
        mean_1_op.tmp_outputs = [mean]

        sub_op = tflite_model.Operator(builtin_options=sub_options.Sub())
        sub_op.tmp_inputs = [x, mean]
        sub_op.tmp_outputs = [d]

        square_op = tflite_model.Operator(builtin_options=square_options.Square())
        square_op.tmp_inputs = [d]
        square_op.tmp_outputs = [d_squared]

        mean_2_op = tflite_model.Operator(builtin_options=mean_options.Mean(True))
        mean_2_op.tmp_inputs = [d_squared, axes_tensor]
        mean_2_op.tmp_outputs = [var]

        add_1_op = tflite_model.Operator(builtin_options=add_options.Add())
        add_1_op.tmp_inputs = [var, epsilon_tensor]
        add_1_op.tmp_outputs = [var_eps]

        rsqrt_op = tflite_model.Operator(opcode_index=self.builder.op_code_index_for_op_type(BuiltinOperator.RSQRT))
        rsqrt_op.tmp_inputs = [var_eps]
        rsqrt_op.tmp_outputs = [inv_std_dev]

        mul_1_op = tflite_model.Operator(builtin_options=mul_options.Mul())
        mul_1_op.tmp_inputs = [d, inv_std_dev]
        mul_1_op.tmp_outputs = [normalized]

        mul_2_op = tflite_model.Operator(builtin_options=mul_options.Mul())
        mul_2_op.tmp_inputs = [normalized, scale]
        mul_2_op.tmp_outputs = [normalized_scaled]

        if x.tensor_format.is_channels_last() and uses_shape_broadcasting(mul_2_op):
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX 'LayerNormalization' with channels first tensors "
                                                  "and shape broadcasting is not yet supported!")

        ops_to_add = [mean_1_op, sub_op, square_op, mean_2_op, add_1_op, rsqrt_op, mul_1_op, mul_2_op]

        if b is not None:
            # Add the optional bias
            add_2_op = tflite_model.Operator(builtin_options=add_options.Add())
            add_2_op.tmp_inputs = [normalized_scaled, b]
            add_2_op.tmp_outputs = [y]
            ops_to_add.append(add_2_op)

            if x.tensor_format.is_channels_last() and uses_shape_broadcasting(add_2_op):
                logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX 'LayerNormalization' with channels first "
                                                      "tensors and shape broadcasting is not yet supported!")
        else:
            # Make sure the last operator produces the original output tensor, that other operators are connected to.
            mul_2_op.tmp_outputs = [y]

        # Because the LayerNormalization has to be represented using so many operators, there is sometimes a relatively
        #  large error. Notify the user of this.
        logger.i(f"ONNX `LayerNormalization` is converted into {len(ops_to_add)} operators, which can sometimes "
                 "introduce a numerical error.")

        return ops_to_add
