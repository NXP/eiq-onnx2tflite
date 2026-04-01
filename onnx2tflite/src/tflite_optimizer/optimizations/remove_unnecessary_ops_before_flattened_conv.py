#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src.tflite_optimizer.optimizations.base_optimization import BaseOptimization
from onnx2tflite.src.tflite_optimizer.pattern_matcher import Op, PatternMatcher
from onnx2tflite.src.tflite_optimizer.tensor_rules import (
    TensorHasStaticValue,
    TensorsHaveOneConsumer, TensorHasRank, TensorHasDimensionOfSize, TensorIsNotModelOutput,
)


class RemoveUnnecessaryOpsBeforeFlattenedConv(BaseOptimization):
    """
    Remove unnecessary Reshape and Transpose operations before a flattened Conv operation.

          │ 2D                               │
     ┌────▼────┐                             │
     │Transpose◄─── perm=[1,0]               │
     └────┬────┘                             │
          │ 2D                               │
     ┌────▼────┐            ───────►    ┌────▼────┐
     │ Reshape │                        │ Reshape │
     └────┬────┘                        └────┬────┘
          │ 4D with shape (1,1,1,C)          │ 4D with same shape
     ┌────▼────┐                        ┌────▼────┐
     │ Conv2D  │                        │ Conv2D  ◄─── Weights transposed
     └────┬────┘                        └────┬────┘
          │                                  │
          ▼                                  ▼

    """
    def __call__(self):
        matcher = PatternMatcher(
            self._builder,
            [
                Op(["Transpose"], ["x", "perm"], ["transpose_out"]),
                Op(["Reshape"], ["transpose_out"], ["reshape_out"]),
                Op(["Conv2D"], ["reshape_out", "w", ...], ["y"]),
            ],
            [
                TensorsHaveOneConsumer(["x", "transpose_out", "reshape_out"]),

                # Transpose IO tensors should not be model outputs
                TensorIsNotModelOutput("x"),
                TensorIsNotModelOutput("transpose_out"),
                TensorIsNotModelOutput("reshape_out"),

                # Transpose only switches 2D dimensions
                TensorHasRank("x", 2),
                TensorHasRank("transpose_out", 2),
                TensorHasStaticValue("perm", [1, 0]),

                # Conv input in format (1, 1, 1, N)
                TensorHasRank("reshape_out", 4),
                TensorHasDimensionOfSize("reshape_out", 0, 1),
                TensorHasDimensionOfSize("reshape_out", 1, 1),
                TensorHasDimensionOfSize("reshape_out", 2, 1),
            ]
        )

        to_remove = []
        for (transpose, reshape, _), tensor_map, _, _ in matcher.match_patterns():
            x = tensor_map["x"]
            w = tensor_map["w"]
            transpose_out = tensor_map["transpose_out"]

            w_data = w.tmp_buffer.data
            original_w_shape = list(w_data.shape)

            # Transpose Conv's weight so we can remove unnecessary Transpose op before it
            w_data = w_data.reshape(original_w_shape[:3] + transpose_out.shape.vector)
            w_data = w_data.transpose([0, 1, 2, 4, 3])
            w_data = w_data.reshape(original_w_shape)
            w.tmp_buffer.data = w_data

            to_remove.append(transpose)

            reshape.tmp_inputs[0] = x

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0
