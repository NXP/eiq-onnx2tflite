#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src.tflite_optimizer.optimizations.base_optimization import BaseOptimization
from onnx2tflite.src.tflite_optimizer.pattern_matcher import MultipleSameOps, Op, PatternMatcher
from onnx2tflite.src.tflite_optimizer.tensor_rules import (
    RuleOr,
    TensorIsNotModelOutput,
    TensorIsNotQuantized,
    TensorsAreNotQuantized,
    TensorsHaveSameType,
)


class FuseCastOperators(BaseOptimization):
    """Remove some `Cast` operators in the following pattern.

         │  'x'
      ┌──▼───┐
      │ Cast │
      └──┬───┘                                           │  'x'
       ┌─┴─── ... ──────┐  'y'        ─────►          ┌──┴── ... ─────┐   ('y' is not in the model anymore)
    ┌──▼───┐         ┌──▼───┐                      ┌──▼───┐        ┌──▼───┐
    │ Cast │  ...    │ Cast │                      │ Cast │  ...   │ Cast │
    └──┬───┘         └──┬───┘                      └──┬───┘        └──┬───┘
       │                │  'z'                        │               │  'z'
    """

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [
                Op(["Cast"], outputs=["y"]),
                MultipleSameOps(["Cast"], ["y", ...])  # Only `Cast` ops can use `y`.
            ],
            [
                TensorIsNotModelOutput("y"),
                TensorIsNotQuantized("y")
            ]
        )

        to_remove = []
        for [leading_cast, following_cast_ops], _, _, _ in matcher.match_patterns():
            # Remove the leading cast.
            for cast in following_cast_ops:
                cast.tmp_inputs[0] = leading_cast.tmp_inputs[0]

            to_remove.append(leading_cast)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0


class RemoveCastOperatorsWithNoEffect(BaseOptimization):
    """Remove operators that match the following pattern.

       │  'x'
    ┌──▼───┐
    │ Cast │
    └──┬───┘
       │  'y'  (same type as 'x')
    """

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [
                Op(["Cast"], ["x", ...], ["y"])
            ],
            [
                TensorsHaveSameType(["x", "y"]),
                TensorsAreNotQuantized(["x", "y"]),
                RuleOr(
                    TensorIsNotModelOutput("x"),
                    TensorIsNotModelOutput("y")
                    # If both 'x' and 'y' are model outputs, the `Cast` cannot be removed. If the op was removed, its
                    #  input and output would be combined into 1 tensor, which would have to represent 2 model outputs
                    #  with 2 different names, which is not possible.
                )
            ])

        to_remove = []
        for [cast], tensor_map, input_to_ops, _ in matcher.match_patterns():
            if not self._builder.operator_can_be_skipped(cast):
                continue

            x = tensor_map["x"]
            y = tensor_map["y"]
            model_outputs = self._builder.get_sub_graph().outputs.tmp_outputs

            # Replace `y` with `x` in the inputs of all following operators.
            following_ops = input_to_ops.get(y.name, [])
            for op in following_ops:
                while y in op.tmp_inputs:
                    input_idx = op.tmp_inputs.index(y)
                    op.tmp_inputs[input_idx] = x

            if y in model_outputs:
                # Replace the output as well.
                while y in model_outputs:
                    idx = model_outputs.index(y)
                    model_outputs[idx] = x

                self._builder.swap_tensor_names(x, y)

            to_remove.append(cast)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0
