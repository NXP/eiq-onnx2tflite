#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.node_converters.shared import reduce_utils
from onnx2tflite.src.converter.conversion.common import OpsList, try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes.reduce_l2_attributes import ReduceL2
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import cast_options, square_options, sum_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, name_for_type


class ReduceL2Converter(NodeConverter):
    node = 'ReduceL2'

    onnx_supported_types = FLOATS + [TensorType.INT32, TensorType.INT64, TensorType.UINT32, TensorType.UINT64]
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/elementwise.cc#L353-L374
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/elementwise.cc#L250-L253
    tflite_supported_types = [TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32]

    def _convert_v_13(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX ReduceL2 version 1 / 11 / 13 to TFLites `Square` + `Sum` + `Sqrt`.

            The `axes` is in the form of an attribute.
        """

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]
        rank = x.rank

        reduce_l2_attributes = cast(ReduceL2, node.attributes)

        if reduce_l2_attributes.axes is None:
            # Default axes -> reduce over all dimensions.

            # TFLite has `axes` as input tensor -> create it.
            axes_tensor = self.builder.create_tensor_for_data(np.arange(rank).astype(np.int32), 'axes')
            t_op.tmp_inputs.append(axes_tensor)

        else:
            # Axes are initialized.
            axes = reduce_l2_attributes.axes

            # TFLite requires an `axes` tensor -> create it.
            axes_tensor = self.builder.create_tensor_for_data(np.asarray(axes, np.int32), 'axes')
            t_op.tmp_inputs.append(axes_tensor)

        square_out = self.builder.duplicate_tensor(x, name_suffix="_squared")

        square_op = tflite_model.Operator(builtin_options=square_options.Square())
        square_op.tmp_inputs = [x]
        square_op.tmp_outputs = [square_out]

        sum_out = self.builder.duplicate_tensor(y, "x_summed")

        sum_op = tflite_model.Operator(builtin_options=sum_options.Sum(bool(reduce_l2_attributes.keepdims)))
        sum_op.tmp_inputs = [square_out, axes_tensor]
        sum_op.tmp_outputs = [sum_out]

        sqrt_op = tflite_model.Operator(opcode_index=self.builder.op_code_index_for_op_type(BuiltinOperator.SQRT))
        sqrt_op.tmp_inputs = [sum_out]
        sqrt_op.tmp_outputs = [y]

        ops = OpsList(pre_ops=[square_op], middle_op=sum_op, post_ops=[sqrt_op])

        reduce_utils.ensure_reduce_transposition(self.builder, ops)

        return ops.flatten()

    def _convert_v_18(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX ReduceL2 version 18+ to TFLite's `Square` + `Sum` + `Sqrt`.

            The `axes` is an optional input tensor.
        """
        reduce_l2_attributes = cast(ReduceL2, node.attributes)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        rank = x.rank
        axes_cast_ops = []

        if axes_tensor := try_get_input(t_op, 1):

            # ONNX uses int64, while TFLite requires int32 for the `axes` tensor.
            if axes_tensor.type != TensorType.INT64:
                logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                         f'ONNX `Cast` has `axes` of type `{name_for_type(axes_tensor.type)}`, instead of int64.')

            # Try to get the inferred data for the `axes` input.
            if (axes_data := self.inspector.try_get_inferred_tensor_data(axes_tensor.name)) is not None:
                # The `axes` were inferred during shape inference.
                logger.d('Using inferred data for the `axes` input tensor of ONNX `ReduceL2`.')

                # Create a new tensor, in case the original `axes` tensor is used by multiple ops.
                axes_tensor = self.builder.create_tensor_for_data(axes_data.astype(np.int32), 'axes')

            # Make sure the `axes` are int32.
            if tensor_has_data(axes_tensor):
                # Cast the `axes` to int32 statically.
                axes_tensor.tmp_buffer.data = axes_tensor.tmp_buffer.data.astype(np.int32)
                axes_tensor.type = TensorType.INT32
            else:
                # The `axes` are dynamic and there is no inferred data for them. The shape inference is not possible in
                #  this case, so it must have been skipped. If the `axes` are empty at runtime, ONNX will reduce over
                #  all dimensions, whereas TFLite will not reduce at all. So the behavior is different, and it depends
                #  on runtime data. Conversion could be implemented by adding multiple extra operators.
                # I don't thing that completely prohibiting the conversion here is ideal, since the issue arises only in
                #  an edge case, which is hopefully not very common. Just print a warning message for now.
                logger.w(
                    f'Conversion of ONNX `ReduceL2` with a dynamic `axes` input will not be correct, if the `axes`'
                    'are empty at runtime!')

                # Create a `Cast` op, to make the `axes` int32.
                cast_op = tflite_model.Operator(
                    builtin_options=cast_options.Cast(TensorType.INT64, TensorType.INT32))
                new_axes = self.builder.duplicate_tensor(axes_tensor)
                new_axes.type = TensorType.INT32
                cast_op.tmp_inputs = [axes_tensor]
                cast_op.tmp_outputs = [new_axes]

                axes_tensor = new_axes  # For future references. Following code only cares about the final axes tensor.

                axes_cast_ops.append(cast_op)

        else:
            # No axes specified.

            if reduce_l2_attributes.noop_with_empty_axes == 1:
                # ONNXRT: According to the documentation, the operator should do nothing in this situation. But that's not
                #  what happens in ONNX Runtime. ORT seems to simply ignore the `noop_with_empty_axes` attribute.
                # For now, exit with error. If later ORT adds support for this attribute, simply uncomment the
                #  following code.
                # TODO https://github.com/microsoft/onnxruntime/issues/19147

                # if self.builder.operator_can_be_skipped(t_op, self.inspector):
                #     # Skip the operator.
                #     self.builder.redirect_tensor(t_op.tmp_outputs[0], t_op.tmp_inputs[0])
                #     return []
                #
                # else:
                #     # Return an operator which does nothing.
                #     self.builder.turn_operator_to_identity(t_op)
                #     return [t_op]

                logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                         'ONNX `ReduceL2` has `noop_with_empty_axes` == 1 and the `axes` are not specified, which'
                         ' indicates that the operator should do nothing. This is however not supported by ONNX'
                         ' Runtime, and therefore the conversion is also not supported.')

            else:
                # Default is to reduce all axes.
                axes_tensor = self.builder.create_tensor_for_data(np.arange(rank).astype(np.int32), 'axes')

        square_output = self.builder.duplicate_tensor(x, name_suffix="_squared")

        square_op = tflite_model.Operator(builtin_options=square_options.Square())
        square_op.tmp_inputs = [x]
        square_op.tmp_outputs = [square_output]

        sum_output = self.builder.duplicate_tensor(y, "x_summed")

        sum_op = tflite_model.Operator()
        sum_op.builtin_options = sum_options.Sum(bool(reduce_l2_attributes.keepdims))
        sum_op.tmp_inputs = [square_output, axes_tensor]
        sum_op.tmp_outputs = [sum_output]

        sqrt_op = tflite_model.Operator(opcode_index=self.builder.op_code_index_for_op_type(BuiltinOperator.SQRT))
        sqrt_op.tmp_inputs = [sum_output]
        sqrt_op.tmp_outputs = [y]

        ops = OpsList(pre_ops=[square_op], middle_op=sum_op, post_ops=[sqrt_op])
        ops.add_pre(axes_cast_ops)

        reduce_utils.ensure_reduce_transposition(self.builder, ops)

        return ops.flatten()

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX `ReduceL2` operator into TFLite's `Square` + `Sum` + `Sqrt`. """

        if not (1 <= len(t_op.tmp_inputs) <= 2):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `ReduceL2` has unexpected number of inputs.')

        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        if node.version < 18:
            # Version 1 / 11 / 13 -> axes are passed as attribute.
            return self._convert_v_13(node, t_op)

        else:
            # Version 18+ -> axes are an optional input tensor.
            return self._convert_v_18(node, t_op)
