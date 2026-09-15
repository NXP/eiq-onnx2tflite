#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np
from onnx import TensorProto
from onnxruntime.quantization import quant_utils

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import errors
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import (
    compute_scale_zero_point,
    get_symmetric_zero_point_for_type,
    quantize_static_float_tensor,
)
from onnx2tflite.src.errors import ConversionError
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import hard_sigmoid_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import add_options, minimum_options, mul_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS


class HardSigmoidConverter(NodeConverter):
    node = "HardSigmoid"

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/maximum_minimum.cc#L176-L252
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `HardSigmoid` operator into TFLite.

            The HardSigmoid function is :

                Max(0 ,Minimum(1, alpha * x + beta))

            which can be represented in TFLite as:

                x -> Mul(alpha) -> Add(beta) -> Min(1) -> Relu

                An alternative to Min(1) is ReluN1To1, if it has better support on the boards.

                !! If the conversion strategy is changed, update the optimization in
                    `combine_hard_sigmoid_and_mul_to_hard_swish.py` accordingly. !!


            In the future, it may be possible to convert the HardSigmoid into:

                x -> Mul(alpha) -> Add(beta) -> Relu0To1

             However, right now it is not supported on the boards and there is no Relu0To1 in the TFLite schema, so it
              might be a custom operator.

        :param node: ONNX NodeProto representing the `HardSigmoid` operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators to add to the model.
        """
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        hs_attributes = cast(hard_sigmoid_attributes.HardSigmoid, node.attributes)

        if t_op.is_qdq_quantized():
            return self._convert_quantized(x, y, hs_attributes)

        self.assert_type_allowed(x.type)

        return self._convert_float(x, y, hs_attributes)

    def _convert_float(
        self,
        x: tflite_model.Tensor,
        y: tflite_model.Tensor,
        hs_attributes: hard_sigmoid_attributes.HardSigmoid,
    ) -> list[tflite_model.Operator]:
        ops = []

        # ---- Create a `Mul` operator ----
        if hs_attributes.alpha == 1.0:
            # Multiplying by 1. No need for a `Mul` operator.
            mul_output = x
        else:
            # Create a `Mul` operator
            mul_output = self.builder.duplicate_tensor(x, empty_buffer=True)

            mul_op = tflite_model.Operator(builtin_options=mul_options.Mul())
            mul_op.tmp_inputs = [
                x,
                self.builder.create_tensor_for_data(np.array([hs_attributes.alpha], np.float32), "alpha"),
            ]
            mul_op.tmp_outputs = [mul_output]

            ops.append(mul_op)

        # ---- Create ad `Add` operator ----
        if hs_attributes.beta == 0.0:
            # Adding 0. No need for an `Add` operator.
            add_output = mul_output
        else:
            # Create an `Add` operator
            add_output = self.builder.duplicate_tensor(x, empty_buffer=True)

            add_op = tflite_model.Operator(builtin_options=add_options.Add())
            add_op.tmp_inputs = [
                mul_output,
                self.builder.create_tensor_for_data(np.array([hs_attributes.beta], np.float32), "beta"),
            ]
            add_op.tmp_outputs = [add_output]

            ops.append(add_op)

        # ---- Create a `Minimum` operator ----
        # Alternatively we can use ReluN1To1, if it has better support by the boards.
        min_output = self.builder.duplicate_tensor(x, empty_buffer=True)

        min_op = tflite_model.Operator(builtin_options=minimum_options.Minimum())
        min_op.tmp_inputs = [add_output, self.builder.create_tensor_for_data(np.array([1.0], np.float32), "one")]
        min_op.tmp_outputs = [min_output]

        ops.append(min_op)

        # ---- Create a `Relu` operator ----
        relu_op = tflite_model.Operator(opcode_index=self.builder.op_code_index_for_op_type(BuiltinOperator.RELU))
        relu_op.tmp_inputs = [min_output]
        relu_op.tmp_outputs = [y]

        ops.append(relu_op)

        return ops

    def _convert_quantized(
        self,
        x: tflite_model.Tensor,
        y: tflite_model.Tensor,
        hs_attributes: hard_sigmoid_attributes.HardSigmoid,
    ) -> list[tflite_model.Operator]:
        """Convert a QDQ-quantized `HardSigmoid`.

        The decomposition (`Mul -> Add -> Minimum -> Relu`) is kept, but the added constant tensors must be
         quantized to the same type as the (quantized) input, and the intermediate tensors must be assigned valid
         quantization parameters so every TFLite op in the chain has legal quantized inputs/outputs.
        """
        if x.quantization is None or y.quantization is None:
            raise ConversionError(errors.Code.INVALID_ONNX_MODEL, "Quantized `HardSigmoid` is missing input/output quantization.")

        if x.type not in (TensorType.INT8, TensorType.UINT8):
            raise ConversionError(errors.Code.CONVERSION_IMPOSSIBLE, f"Quantized `HardSigmoid` supports only INT8/UINT8, got `{x.type}`.")

        alpha = hs_attributes.alpha
        beta = hs_attributes.beta

        if alpha < 0.0 or beta < 0.0:
            # Quantizing a constant with a negative value would require a negative scale, which is invalid. A negative
            #  alpha/beta could be handled via a dedicated Mul/Add operand quantization, but it is not a common case.
            raise ConversionError(errors.Code.NOT_IMPLEMENTED, "Quantized `HardSigmoid` with negative `alpha`/`beta` is not supported.")

        # Compute the float range of the (quantized) input from its quantization parameters.
        x_min, x_max = self._quantized_tensor_range(x)

        ops = []

        # ---- Create a `Mul` operator (multiply by `alpha`) ----
        if alpha == 1.0:
            mul_output = x
            mul_min, mul_max = x_min, x_max
        else:
            mul_min, mul_max = sorted((alpha * x_min, alpha * x_max))
            mul_output = self._duplicate_with_quantization(x, mul_min, mul_max)

            alpha_tensor = self.builder.create_tensor_for_data(np.array([alpha], np.float32), "alpha")
            alpha_tensor = quantize_static_float_tensor(
                self.builder, alpha_tensor, x.type, [alpha], [get_symmetric_zero_point_for_type(x.type)]
            )

            mul_op = tflite_model.Operator(builtin_options=mul_options.Mul())
            mul_op.tmp_inputs = [x, alpha_tensor]
            mul_op.tmp_outputs = [mul_output]

            ops.append(mul_op)

        # ---- Create an `Add` operator (add `beta`) ----
        if beta == 0.0:
            add_output = mul_output
            add_min, add_max = mul_min, mul_max
        else:
            add_min, add_max = mul_min + beta, mul_max + beta
            add_output = self._duplicate_with_quantization(x, add_min, add_max)

            beta_tensor = self.builder.create_tensor_for_data(np.array([beta], np.float32), "beta")
            beta_tensor = quantize_static_float_tensor(
                self.builder, beta_tensor, x.type, [beta], [get_symmetric_zero_point_for_type(x.type)]
            )

            add_op = tflite_model.Operator(builtin_options=add_options.Add())
            add_op.tmp_inputs = [mul_output, beta_tensor]
            add_op.tmp_outputs = [add_output]

            ops.append(add_op)

        # ---- Create a `Minimum` operator (clamp to <= 1) ----
        # The TFLite `Minimum` kernel requires all its input/output tensors to share the same quantization
        #  parameters. We derive a single scale/zero-point from a range covering the operator's input values, the
        #  constant `1.0`, and `0.0` (so the value can also be represented for the following `Relu`), and assign it
        #  to both `Minimum` inputs and its output.
        shared_min = min(add_min, 0.0)
        shared_max = max(add_max, 1.0)
        min_scale, min_zero_point = compute_scale_zero_point(x.type, shared_min, shared_max)

        # The `Minimum`'s first input is the output of the preceding op (`Add`/`Mul`), or `x` when both `alpha` and
        #  `beta` are neutral. When it is a tensor we created, override its quantization to match the shared params.
        #  When it is `x`, its params are fixed by the QDQ cluster, so use those as the shared params instead.
        if add_output is x:
            min_scale = x.quantization.scale.vector[0]
            min_zero_point = x.quantization.zero_point.vector[0]
        else:
            add_output.quantization = tflite_model.Quantization(
                scale=tflite_model.Scale([min_scale]),
                zero_point=tflite_model.ZeroPoint([min_zero_point]),
            )

        one_tensor = self.builder.create_tensor_for_data(np.array([1.0], np.float32), "one")
        one_tensor = quantize_static_float_tensor(self.builder, one_tensor, x.type, [min_scale], [min_zero_point])

        min_output = self.builder.duplicate_tensor(x, empty_buffer=True)
        min_output.quantization = tflite_model.Quantization(
            scale=tflite_model.Scale([min_scale]),
            zero_point=tflite_model.ZeroPoint([min_zero_point]),
        )

        min_op = tflite_model.Operator(builtin_options=minimum_options.Minimum())
        min_op.tmp_inputs = [add_output, one_tensor]
        min_op.tmp_outputs = [min_output]

        ops.append(min_op)

        # ---- Create a `Relu` operator (clamp to >= 0) ----
        # The output `y` already carries its quantization parameters from the surrounding QDQ cluster, which satisfies
        #  the TFLite requirement of a quantized `Relu` output.
        relu_op = tflite_model.Operator(opcode_index=self.builder.op_code_index_for_op_type(BuiltinOperator.RELU))
        relu_op.tmp_inputs = [min_output]
        relu_op.tmp_outputs = [y]

        ops.append(relu_op)

        return ops

    @staticmethod
    def _quantized_tensor_range(tensor: tflite_model.Tensor) -> tuple[float, float]:
        """Compute the represented float min/max range of a per-tensor quantized tensor."""
        scale = tensor.quantization.scale.vector
        zero_point = tensor.quantization.zero_point.vector

        if len(scale) != 1 or len(zero_point) != 1:
            raise ConversionError(errors.Code.NOT_IMPLEMENTED, "Quantized `HardSigmoid` supports only per-tensor quantization.")

        q_type = TensorProto.INT8 if tensor.type == TensorType.INT8 else TensorProto.UINT8
        q_min, q_max = quant_utils.get_qmin_qmax_for_qType(q_type, reduce_range=False, symmetric=False)

        # Cast to Python `int` before subtracting. The quantization vectors hold NumPy integers (e.g. `int8`), and
        #  `q_max - zero_point` would overflow in the narrow integer type (e.g. `127 - int8(-128)` wraps to `-1`).
        zp = int(zero_point[0])
        r_min = scale[0] * (int(q_min) - zp)
        r_max = scale[0] * (int(q_max) - zp)

        return r_min, r_max

    def _duplicate_with_quantization(
        self, template: tflite_model.Tensor, r_min: float, r_max: float
    ) -> tflite_model.Tensor:
        """Duplicate `template` (keeping its type) and assign quantization parameters for the `[r_min, r_max]` range."""
        tensor = self.builder.duplicate_tensor(template, empty_buffer=True)

        scale, zero_point = compute_scale_zero_point(template.type, r_min, r_max)

        tensor.quantization = tflite_model.Quantization(
            scale=tflite_model.Scale([scale]),
            zero_point=tflite_model.ZeroPoint([zero_point]),
        )

        return tensor
