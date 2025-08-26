#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes.batch_normalization_attributes import BatchNormalization
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import add_options, mul_options, sub_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS


# noinspection PyMethodMayBeStatic
class BatchNormalizationConverter(NodeConverter):
    node = "BatchNormalization"

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/reduce.cc#L525-L546
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def _convert_batch_normalization_with_static_operands(self, attrs: BatchNormalization,
                                                          t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX BatchNormalization with static operands to TFLite.

        The equation computed by BatchNormalization can be rewritten as:

            y = x * (scale / sqrt(var + eps)) + (bias - mean * scale / sqrt(var + eps))

        where `scale / sqrt(var + eps)` and `bias - mean * scale / sqrt(var + eps)` are static tensors, provided the
        ONNX operands are static as well.
        """
        x = t_op.tmp_inputs[0]
        scale = t_op.tmp_inputs[1].tmp_buffer.data
        bias_tensor = t_op.tmp_inputs[2]
        mean = t_op.tmp_inputs[3].tmp_buffer.data
        var = t_op.tmp_inputs[4].tmp_buffer.data

        rank = x.rank
        if rank == 1:
            # ONNXRT: BatchNormalization v9+ should support 1D input according to the doc, but ORT crashes in this case.
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX BatchNormalization with a 1D main input is not "
                                                  "implemented, because ONNX Runtime doesn't support it!")

        # Calculate the static portion of the expression
        tmp = scale / np.sqrt(var + attrs.epsilon)
        bias_tensor.tmp_buffer.data = bias_tensor.tmp_buffer.data - mean * tmp

        # Create the tensors taking part in the equation
        multiplication_factor = self.builder.create_tensor_for_data(tmp, "BatchNorm_multiplication_factor")
        fraction = self.builder.duplicate_tensor(x, "BatchNorm_fraction")

        # Create 'Mul' operator, to multiply the input with the static tensor `(scale / sqrt(var + eps))`
        mul = tflite_model.Operator(builtin_options=mul_options.Mul())
        mul.tmp_inputs = [x, multiplication_factor]
        mul.tmp_outputs = [fraction]

        # Create 'Add' operator to add 'bias' to the previous result
        add = tflite_model.Operator(builtin_options=add_options.Add())
        add.tmp_inputs = [fraction, bias_tensor]
        add.tmp_outputs = [t_op.tmp_outputs[0]]

        return [mul, add]

    def _convert_batch_normalization_with_dynamic_operands(self, attrs: BatchNormalization,
                                                           t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX BatchNormalization with dynamic operands to TFLite.

        BatchNormalization computes the following equation when training mode is off:

            y = scale * (x - mean) / sqrt(var + eps) + bias

        This function assumes that at least 1 operand of the BatchNormalization is dynamic. If some of the operands
        are static, it might be possible to statically precompute some values and reduce the number of added operators.
        However, as this is probably not a very common scenario, such an optimization is not yet implemented.
        """
        x, scale, bias, mean, var = t_op.tmp_inputs
        y = t_op.tmp_outputs[0]

        # Create tensors taking part in the equation
        epsilon = self.builder.create_tensor_for_data(np.asarray([attrs.epsilon], np.float32), "eps")
        var_eps = self.builder.duplicate_tensor(x)
        inv_sqrt = self.builder.duplicate_tensor(x)
        normalized_x = self.builder.duplicate_tensor(x)
        scaled_x = self.builder.duplicate_tensor(x)
        fraction = self.builder.duplicate_tensor(x)

        var_eps.shape = tflite_model.Shape(var.shape.vector.copy())
        inv_sqrt.shape = tflite_model.Shape(var.shape.vector.copy())

        # --  var + eps  --
        add_1 = tflite_model.Operator(builtin_options=add_options.Add())
        add_1.tmp_inputs = [var, epsilon]
        add_1.tmp_outputs = [var_eps]

        # --  1 / sqrt(var + eps)  --
        rsqrt = tflite_model.Operator(opcode_index=self.builder.op_code_index_for_op_type(BuiltinOperator.RSQRT))
        rsqrt.tmp_inputs = [var_eps]
        rsqrt.tmp_outputs = [inv_sqrt]

        # --  x - mean  --
        sub = tflite_model.Operator(builtin_options=sub_options.Sub())
        sub.tmp_inputs = [x, mean]
        sub.tmp_outputs = [normalized_x]

        # --  scale * (x - mean)  --
        mul_1 = tflite_model.Operator(builtin_options=mul_options.Mul())
        mul_1.tmp_inputs = [scale, normalized_x]
        mul_1.tmp_outputs = [scaled_x]

        # --  scale * (x - mean) / sqrt(var + eps)  --
        mul_2 = tflite_model.Operator(builtin_options=mul_options.Mul())
        mul_2.tmp_inputs = [scaled_x, inv_sqrt]
        mul_2.tmp_outputs = [fraction]

        # --  scale * (x - mean) / sqrt(var + eps) + bias  --
        add_2 = tflite_model.Operator(builtin_options=add_options.Add())
        add_2.tmp_inputs = [fraction, bias]
        add_2.tmp_outputs = [y]

        return [add_1, rsqrt, sub, mul_1, mul_2, add_2]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX BatchNormalization to TFLite.

        TFLite doesn't have a corresponding operator.
        Since we don't need to worry about what BatchNormalization does during training it can be represented via
        multiple simpler operators.
        """
        self.assert_type_allowed(t_op.tmp_inputs[0].type)
        if t_op.is_quantized_without_qdq():
            # ONNX doesn't support (U)INT8. Leave this check in case the support is added in the future.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX `AveragePool` with a quantized input is not supported.")

        attrs = cast(BatchNormalization, node.attributes)

        if len(t_op.tmp_inputs) != 5:
            # This case is caught by the shape inference anyway
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX BatchNormalization has '{len(t_op.tmp_inputs)}' inputs instead of 5!")

        if len(t_op.tmp_outputs) != 1:
            # This case is caught by the shape inference anyway
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX BatchNormalization has '{len(t_op.tmp_outputs)}' outputs. "
                                                     f"Only 1 output should be used after training is complete!")

        additional_ops = []
        if attrs.spatial == 0:
            # The operands should have shape (C, H, W,...).

            expected_operand_rank = t_op.tmp_inputs[0].rank - 1
            if any(operand.rank != expected_operand_rank for operand in t_op.tmp_inputs[1:]):

                # Some models (e.g. ResNet-101-DUC-7.onnx) have spatial=0 and operands with shape (C).
                # According to the documentation, this shouldn't happen. But the inference still works.
                if any(operand.rank != 1 for operand in t_op.tmp_inputs[1:]):
                    logger.e(logger.Code.INVALID_ONNX_MODEL,
                             f"ONNX BatchNormalization with 'spatial=0' should have all "
                             f"operands with rank '{expected_operand_rank}' or '1'!")

                else:
                    # All operands have rank 1. The 'builder.ensure_correct_broadcasting()' must NOT be called.
                    logger.w("ONNX BatchNormalization with 'spatial=0' should have operands with "
                             f"rank '{expected_operand_rank}'!")

            else:
                # Make sure the operands can be correctly broadcasted with the input.
                additional_ops = self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0])

        # All operands should be 1D
        elif any(operand.rank != 1 for operand in t_op.tmp_inputs[1:]):
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX BatchNormalization should only have 1D operands!")

        # Convert the operator
        if all(tensor_has_data(tensor) for tensor in t_op.tmp_inputs[1:]):
            # All operands are static. The BatchNormalization can be converted to just 2 operators.
            return self._convert_batch_normalization_with_static_operands(attrs, t_op)

        # At least 1 operand is dynamic.
        return additional_ops + self._convert_batch_normalization_with_dynamic_operands(attrs, t_op)
