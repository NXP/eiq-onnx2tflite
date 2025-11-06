#
# Copyright 2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src.tflite_generator.builtin_options import reshape_options
from onnx2tflite.src.tflite_optimizer.optimizations.base_optimization import BaseOptimization
from onnx2tflite.src.tflite_optimizer.pattern_matcher import PatternMatcher, Op
from onnx2tflite.src.tflite_optimizer.tensor_rules import TensorsHaveOneConsumer, TensorHasStaticValue, \
    TensorIsChannelsLast, TensorDimensionsMatch, RuleOr


class RemoveCancellingTransposes(BaseOptimization):
    """
    Optimization that removes cancelling Transpose operations around a Reshape:

               │  (4D)
        ┌──────▼──────┐
        │  Transpose  │ ◄─  perm=[0, 3, 1, 2]
        └──────┬──────┘
               │  (4D)                                                            │  (4D)
         ┌─────▼─────┐                                                      ┌─────▼─────┐
         │  Reshape  │  ◄─ Batch & Channel dim not changed     ─────►       │  Reshape  │
         └─────┬─────┘                                                      └─────┬─────┘
               │  (4D/3D)                                                         │  (4D/3D)
        ┌──────▼──────┐                                                           ▼
        │  Transpose  │ ◄─  perm=[0, 2, 3, 1] | [0, 2, 1]
        └──────┬──────┘
               │  (4D/3D)
               ▼
    """

    def __call__(self):
        matcher = PatternMatcher(
            self._builder,
            [
                Op(["Transpose"], ["x", "perm1"], ["a"]),
                Op(["Reshape"], ["a"], ["b"]),
                Op(["Transpose"], ["b", "perm2"], ["y"])
            ],
            [
                TensorsHaveOneConsumer(["a", "b"]),
                TensorHasStaticValue("perm1", [0, 3, 1, 2]),
                RuleOr(
                    TensorHasStaticValue("perm2", [0, 2, 1]),
                    TensorHasStaticValue("perm2", [0, 2, 3, 1]),
                ),
                TensorDimensionsMatch("a", 0, "b", 0), # Reshape preserves batch dimension
                TensorDimensionsMatch("a", 1, "b", 1), # Reshape preserves channel dimension
            ]
        )

        to_remove = []
        for (transpose_1, reshape, transpose_2), tensor_map, _, _ in matcher.match_patterns():
            x, y = tensor_map["x"], tensor_map["y"]

            reshape.builtin_options = reshape_options.Reshape(y.shape.vector.copy())
            reshape.tmp_inputs[0] = x
            reshape.tmp_outputs[0] = y

            to_remove.append(transpose_1)
            to_remove.append(transpose_2)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0