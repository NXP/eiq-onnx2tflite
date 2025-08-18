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
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes.instance_normalization_attributes import InstanceNormalization
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.add_options import Add
from onnx2tflite.src.tflite_generator.builtin_options.mean_options import Mean
from onnx2tflite.src.tflite_generator.builtin_options.mul_options import Mul
from onnx2tflite.src.tflite_generator.builtin_options.squared_difference_options import SquaredDifference
from onnx2tflite.src.tflite_generator.builtin_options.sub_options import Sub
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS


class InstanceNormalizationConverter(NodeConverter):
    node = 'InstanceNormalization'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/reduce.cc#L525-L548
    tflite_supported_types = INTS + [TensorType.UINT8, TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX `InstanceNormalization` operator into TFLite.

            There is no 'InstanceNormalization' in TFLite. The ONNX operator carries out the operation

                y = scale * (x - mean) / sqrt(variance + epsilon) + B

            This can be represented in TFLite as:

                                      X
                           ┌──────────┼───────────┐
                           │       ┌──▼───┐       │
                           │       │ Mean │       │
                           │       └─┬──┬─┘       │
                           │         │  └─────┐   │
                      ┌────▼─────────▼────┐  ┌▼───▼┐
                      │ SquaredDifference │  │ Sub │
                      └─────────┬─────────┘  └──┬──┘
                            ┌───▼──┐         ┌──▼──┐
                            │ Mean │         │ Mul ◄───── Scale
                            └───┬──┘         └──┬──┘
                             ┌──▼──┐            │
                Epsilon ─────► Add │            │
                             └──┬──┘            │
                            ┌───▼───┐           │
                            │ RSqrt │           │
                            └───┬───┘           │
                                └─────┐   ┌─────┘
                                     ┌▼───▼┐
                                     │ Mul │
                                     └──┬──┘
                                     ┌──▼──┐
                                     │ Add ◄───── B
                                     └──┬──┘
                                        ▼
                                        Y
        """

        if len(t_op.tmp_inputs) != 3:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `InstanceNormalization` has {len(t_op.tmp_inputs)} inputs instead of 3.')

        x = t_op.tmp_inputs[0]
        scale = t_op.tmp_inputs[1]
        b = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        if x.shape.len() < 3:
            # The input must have at least 3 dimensions.
            logger.e(logger.Code.INVALID_ONNX_MODEL, f'ONNX `InstanceNormalization` has main input with {x.shape.len()}'
                                                     ' dimensions. At least 3 are expected.')

        # The ONNX `InstanceNormalization` uses `channels_first` tensors. This should be recognized by the converter in
        #  the `tensor_format_inference.py`.
        # Since the `TensorConverter` would have already converted the format to `channels_last` before this point, we
        #  have to check for `channels_last` here.
        logger.internal_assert(x.tensor_format.is_channels_last(),
                               'InstanceNormalization should use `channels_first` in the `tensor_format_inference.py`')

        self.assert_type_allowed(x.type)

        attrs = cast(InstanceNormalization, node.attributes)
        axes = list(range(1, x.shape.len() - 1))  # The spatial dimensions. [1, 2] for N, H, W, C.

        ops = []

        # ---- Mean(x) ----
        mean_1_out = self.builder.duplicate_tensor(x, name_suffix='_mean')
        axes_1 = self.builder.create_tensor_for_data(np.array(axes, np.int32), 'axes')

        mean_1 = tflite_model.Operator(builtin_options=Mean(True))
        mean_1.tmp_inputs = [x, axes_1]
        mean_1.tmp_outputs = [mean_1_out]

        ops.append(mean_1)

        # ---- SquaredDifference(x, mean_1_out) ----
        squared_diff_out = self.builder.duplicate_tensor(mean_1_out)

        squared_diff = tflite_model.Operator(builtin_options=SquaredDifference())
        squared_diff.tmp_inputs = [x, mean_1_out]
        squared_diff.tmp_outputs = [squared_diff_out]

        ops.append(squared_diff)

        # ---- Mean(squared_diff_out) ----  (compute the variance)
        mean_2_out = self.builder.duplicate_tensor(squared_diff_out, name_suffix='_mean')
        axes_2 = self.builder.create_tensor_for_data(np.array(axes, np.int32), 'axes')

        mean_2 = tflite_model.Operator(builtin_options=Mean(True))
        mean_2.tmp_inputs = [squared_diff_out, axes_2]
        mean_2.tmp_outputs = [mean_2_out]

        ops.append(mean_2)

        # ---- Add(mean_2_out, epsilon) ----  (variance + epsilon)
        add_1_out = self.builder.duplicate_tensor(mean_2_out, name_suffix='_plus_epsilon')
        epsilon = self.builder.create_tensor_for_data(np.array([attrs.epsilon], np.float32), 'epsilon')

        add_1 = tflite_model.Operator(builtin_options=Add())
        add_1.tmp_inputs = [mean_2_out, epsilon]
        add_1.tmp_outputs = [add_1_out]

        ops.append(add_1)

        # ---- RSqrt(add_1_out) ----  (1 / sqrt(variance + epsilon))
        rsqrt_out = self.builder.duplicate_tensor(add_1_out, name_suffix='_reverse_square_root')

        rsqrt = tflite_model.Operator(builtin_options=None,
                                      opcode_index=self.builder.op_code_index_for_op_type(BuiltinOperator.RSQRT))
        rsqrt.tmp_inputs = [add_1_out]
        rsqrt.tmp_outputs = [rsqrt_out]

        ops.append(rsqrt)

        # ---- Sub(x, mean_1_out) ----  (x - mean(x))
        sub_out = self.builder.duplicate_tensor(x, name_suffix='_minus_mean')

        sub = tflite_model.Operator(builtin_options=Sub())
        sub.tmp_inputs = [x, mean_1_out]
        sub.tmp_outputs = [sub_out]

        ops.append(sub)

        # ---- Mul(sub_out, scale) ----  (scale * (x - mean))
        mul_1_out = self.builder.duplicate_tensor(sub_out, name_suffix='_scaled')

        mul_1 = tflite_model.Operator(builtin_options=Mul())
        mul_1.tmp_inputs = [sub_out, scale]
        mul_1.tmp_outputs = [mul_1_out]

        ops.append(mul_1)

        # ---- Mul(mul_1_out, rsqrt_out) ----  (scale * (x - mean) / sqrt(variance + epsilon))
        mul_2_out = self.builder.duplicate_tensor(x, name_suffix='_normalized')

        mul_2 = tflite_model.Operator(builtin_options=Mul())
        mul_2.tmp_inputs = [mul_1_out, rsqrt_out]
        mul_2.tmp_outputs = [mul_2_out]

        ops.append(mul_2)

        # ---- Add(mul_2_out, b) ----  (scale * (x - mean) / sqrt(variance + epsilon) + B)
        add_2 = tflite_model.Operator(builtin_options=Add())
        add_2.tmp_inputs = [mul_2_out, b]
        add_2.tmp_outputs = [y]

        ops.append(add_2)

        return ops
