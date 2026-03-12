#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.src.tflite_generator.builtin_options import reshape_options
from onnx2tflite.src.tflite_optimizer.optimizations.base_optimization import BaseOptimization
from onnx2tflite.src.tflite_optimizer.pattern_matcher import Op, PatternMatcher
from onnx2tflite.src.tflite_optimizer.tensor_rules import (
    TensorHasData,
)


class ReplaceInputInvariantTransposeWithReshape(BaseOptimization):
    """Replace Transpose with input invariant permutation by Reshape.

          │  'x'                                                      │  'x'
    ┌─────▼─────┐                                               ┌─────▼─────┐
    │ Transpose ◄───── invariant permutation       ─────►       │  Reshape  │
    └─────┬─────┘                                               └─────┬─────┘
          │  'y'                                                      │  'y'
    """

    @staticmethod
    def _is_tensor_invariant_permutation(input_shape: list[int], permutation: list[int] | np.ndarray) -> bool:
        """Detect tensor invariant permutations.

        Example:
            Input shape [1,20,20,1] and permutation [0,3,1,2] are tensor invariant, because we're
            effectively moving dimensions of size 1 (dim 3), which doesn't change the tensor layout.
        """

        def input_dim_is_not_one(index):
            return input_shape[index] != 1

        new_permutation = list(filter(input_dim_is_not_one, permutation))

        return new_permutation == sorted(new_permutation)

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            pattern=[Op(["Transpose"], ["x", "perm"], ["y"])],
            tensor_rules=[TensorHasData("perm")]
        )

        transpose_replaced = False

        for [transpose], tensor_map, input_to_ops, _ in matcher.match_patterns():
            x = tensor_map["x"]
            y = tensor_map["y"]

            permutation = tensor_map["perm"].tmp_buffer.data
            if not self._is_tensor_invariant_permutation(x.shape.vector, permutation):
                continue

            transpose.builtin_options = reshape_options.Reshape(y.shape.vector.copy())
            transpose.opcode_index = self._builder.op_code_index_for_op_type(BuiltinOperator.RESHAPE)
            transpose.tmp_inputs = [x]

            transpose_replaced = True

        return transpose_replaced
