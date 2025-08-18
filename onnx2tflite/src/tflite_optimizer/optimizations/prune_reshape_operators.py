#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src.tflite_optimizer.optimizations.base_optimization import BaseOptimization
from onnx2tflite.src.tflite_optimizer.pattern_matcher import MultipleSameOps, Op, PatternMatcher
from onnx2tflite.src.tflite_optimizer.tensor_rules import RuleOr, TensorIsNotModelOutput, TensorsHaveSameShape


class FuseReshapeOperators(BaseOptimization):
    """ Remove some `Reshape` operator in the following pattern.

              │  'x'
         ┌────▼────┐
         │ Reshape │
         └────┬────┘                                              │  'x'
          ┌───┴─── ... ───────┐  'y'        ─────►            ┌───┴─── ... ───────┐   ('y' is not in the model anymore)
     ┌────▼────┐         ┌────▼────┐                     ┌────▼────┐         ┌────▼────┐
     │ Reshape │   ...   │ Reshape │                     │ Reshape │   ...   │ Reshape │
     └────┬────┘         └────┬────┘                     └────┬────┘         └────┬────┘
          │                   │  'z'                          │                   │  'z'
     """

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [
                Op(['Reshape'], outputs=['y']),
                MultipleSameOps(['Reshape'], ['y', ...])  # Nothing other than `Reshape` ops can use `y`.
            ],
            [
                TensorIsNotModelOutput('y')
            ]
        )

        to_remove = []
        for [leading_reshape, following_reshapes], _, _, _ in matcher.match_patterns():
            # Remove the leading reshape.
            for r in following_reshapes:
                r.tmp_inputs[0] = leading_reshape.tmp_inputs[0]

            to_remove.append(leading_reshape)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0


class RemoveReshapeOperatorsWithNoEffect(BaseOptimization):
    """ Remove operators that match the following pattern.

                    │  'x'
               ┌────▼────┐
               │ Reshape │
               └────┬────┘
                    │  'y'  (same shape as 'x')
    """

    def __call__(self) -> bool:
        matcher = PatternMatcher(
            self._builder,
            [
                Op(['Reshape'], ['x', ...], ['y'])
            ],
            [
                TensorsHaveSameShape(['x', 'y']),
                RuleOr(
                    TensorIsNotModelOutput('x'),
                    TensorIsNotModelOutput('y')
                    # If both 'x' and 'y' are model outputs, the `Reshape` cannot be removed. If the op was removed, its
                    #  input and output would be combined into 1 tensor, which would have to represent 2 model outputs
                    #  with 2 different names, which is not possible.
                )
            ])

        to_remove = []
        for [reshape], tensor_map, input_to_ops, _ in matcher.match_patterns():
            if not self._builder.operator_can_be_skipped(reshape):
                continue

            x = tensor_map['x']
            y = tensor_map['y']
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

            to_remove.append(reshape)

        for op in to_remove:
            self._builder.get_operators().remove(op)

        return len(to_remove) != 0
