#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import math
import traceback
from itertools import chain

import numpy as np
import onnx
import onnx.shape_inference
import sympy
from onnx import GraphProto, ModelProto, helper, numpy_helper
from onnxruntime.tools.onnx_model_utils import iterate_graph_per_graph_func, make_dim_param_fixed, \
    make_input_shape_fixed
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference, as_list, as_scalar, get_attribute, get_opset, \
    get_shape_from_sympy_shape, get_shape_from_value_info, handle_negative_axis, is_literal

from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import common, translator
from onnx2tflite.src.logger import Error
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type


def shape_is_well_defined(shape: list) -> bool:
    return all(isinstance(dim, int) and dim >= 0 for dim in shape)


# noinspection PyPep8Naming
class ModelShapeInference(SymbolicShapeInference):
    """ Model shape inference with extended support for quantized operators. """

    def __init__(self, int_max=2 ** 31 - 1, auto_merge=False, guess_output_rank=False, verbose=0):
        super().__init__(int_max, auto_merge, guess_output_rank, verbose)
        self._register_custom_dispatchers()

    def _register_custom_dispatchers(self):
        self.dispatcher_["Cast"] = self._infer_Cast
        self.dispatcher_["Concat"] = self._infer_Concat
        self.dispatcher_["Constant"] = self._infer_Constant
        self.dispatcher_["ConstantOfShape"] = self._infer_ConstantOfShape
        self.dispatcher_["Dropout"] = self._infer_Dropout
        self.dispatcher_["Expand"] = self._infer_Expand
        self.dispatcher_["Flatten"] = self._infer_Flatten
        self.dispatcher_["Identity"] = self._infer_Identity
        self.dispatcher_["OneHot"] = self._infer_OneHot
        self.dispatcher_["QGemm"] = self._infer_QGemm
        self.dispatcher_["QLinearAdd"] = self._infer_QLinearAdd
        self.dispatcher_["QLinearAveragePool"] = self._infer_QLinearAveragePool
        self.dispatcher_["QLinearConcat"] = self._infer_QLinearConcat
        self.dispatcher_["QLinearGlobalAveragePool"] = self._infer_QLinearGlobalAveragePool
        self.dispatcher_["QLinearMul"] = self._infer_QLinearMul
        self.dispatcher_["QLinearSoftmax"] = self._infer_QLinearSoftmax
        self.dispatcher_["QuickGelu"] = self._infer_QuickGelu
        self.dispatcher_["Range"] = self._infer_Range
        self.dispatcher_["ReduceL2"] = self._infer_ReduceX
        self.dispatcher_["ReduceMax"] = self._infer_ReduceX
        self.dispatcher_["ReduceMean"] = self._infer_ReduceX
        self.dispatcher_["ReduceMin"] = self._infer_ReduceX
        self.dispatcher_["ReduceProd"] = self._infer_ReduceX
        self.dispatcher_["ReduceSum"] = self._infer_ReduceSum
        self.dispatcher_["Reshape"] = self._infer_Reshape
        self.dispatcher_["Resize"] = self._infer_Resize
        self.dispatcher_["Slice"] = self._infer_Slice
        self.dispatcher_["Squeeze"] = self._infer_Squeeze
        self.dispatcher_["Tile"] = self._infer_Tile
        self.dispatcher_["Unsqueeze"] = self._infer_Unsqueeze
        self.dispatcher_["Upsample"] = self._infer_Upsample

    def _infer_Expand(self, node):  # noqa: N802
        # Original function only infers the output shape.
        super()._infer_Expand(node)

        # Try to infer the data.
        # noinspection PyBroadException
        try:
            input_data = self._try_get_value(node, 0)
            shape_data = self._try_get_value(node, 1)

            if input_data is not None and shape_data is not None:
                # The output shape has already been inferred. We can use it.
                output_shape = get_shape_from_value_info(self.known_vi_[node.output[0]])
                self.sympy_data_[node.output[0]] = np.broadcast_to(input_data, output_shape)

        except BaseException:
            # Data inference failed. Continue.
            pass

    def _infer_Upsample(self, node):  # noqa: N802
        # Originally there was no dispatcher for this operator, and the inference only worked for v7.

        if get_opset(self.out_mp_) < 9:  # V7
            scales = get_attribute(node, 'scales', None)
            if scales is None:
                logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                         'ONNX `Resize` v7 is missing the required `scales` attribute.')
        else:  # V9
            scales = self._try_get_value(node, 1)
            if scales is None:
                logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                         'Cannot infer the output shape of ONNX `Resize` with a dynamic `scales` input.')

        input_shape = np.array(self._get_sympy_shape(node, 0), np.float32)
        # noinspection PyUnboundLocalVariable
        output_shape = [sympy.simplify(sympy.floor(d * s)) for d, s in zip(input_shape, scales)]
        self._update_computed_dims(output_shape)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(output_shape),
            )
        )

    def _infer_Cast(self, node):
        # Original implementation didn't support data inference for most cases, and the inferred data type was also
        #  not always correct.

        shape = self._get_shape(node, 0)

        if (to_type := get_attribute(node, 'to', None)) is None:
            logger.e(logger.Code.SHAPE_INFERENCE_ERROR, 'ONNX `Cast` is missing the required attribute `to`.')

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], to_type, shape))

        # Try to infer the data.
        # noinspection PyBroadException
        try:
            if (data := self._try_get_value(node, 0)) is not None:
                np_type = to_numpy_type(to_type)
                self.sympy_data_[node.output[0]] = np.asarray(data).astype(np_type).reshape(shape)

        except BaseException:
            # Data inference failed. Continue.
            pass

    def _infer_OneHot(self, node):  # noqa: N802
        sympy_shape = self._get_sympy_shape(node, 0)

        # MODIFIED PART START
        # The original implementation supported only the case when `depth` is a scalar. But the documentation allows
        #  it to be a 1 element tensor as well, which wasn't supported.
        depth = self._try_get_value(node, 1)
        if depth is None:
            # The output shape depends on the value of `depth`. Since the value is not known, shape inference failed.
            logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                     'Failed to infer the output shape of ONNX `OneHot` with a dynamic `depth` input.')

        if isinstance(depth, np.ndarray):
            depth = depth.item()
        elif isinstance(depth, list):
            depth = depth[0]
        # MODIFIED PART END

        axis = get_attribute(node, "axis", -1)
        axis = handle_negative_axis(axis, len(sympy_shape) + 1)
        new_shape = get_shape_from_sympy_shape(
            sympy_shape[:axis]
            + [self._new_symbolic_dim_from_output(node) if not is_literal(depth) else depth]
            + sympy_shape[axis:]
        )
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[2]].type.tensor_type.elem_type,
                new_shape,
            )
        )

    def _infer_Reshape(self, node):  # noqa: N802
        shape_value = self._try_get_value(node, 1)
        vi = self.known_vi_[node.output[0]]
        if shape_value is None:
            shape_shape = self._get_shape(node, 1)
            assert len(shape_shape) == 1
            shape_rank = shape_shape[0]
            assert is_literal(shape_rank)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(self._new_symbolic_shape(shape_rank, node)),
                )
            )
        else:
            shape_value = np.asarray(self._try_get_value(node, 1), np.int64)
            input_sympy_shape = self._get_sympy_shape(node, 0)
            total = 1
            for d in input_sympy_shape:
                total = total * d
            new_sympy_shape = []
            deferred_dim_idx = -1
            non_deferred_size = 1
            for i, d in enumerate(shape_value):
                if type(d) == sympy.Symbol:
                    new_sympy_shape.append(d)
                elif d == 0:
                    new_sympy_shape.append(input_sympy_shape[i])
                    non_deferred_size = non_deferred_size * input_sympy_shape[i]
                else:
                    new_sympy_shape.append(d)
                if d == -1:
                    deferred_dim_idx = i
                elif d != 0:
                    non_deferred_size = non_deferred_size * d

            assert new_sympy_shape.count(-1) < 2
            if -1 in new_sympy_shape:
                new_dim = total // non_deferred_size
                new_sympy_shape[deferred_dim_idx] = new_dim

            self._update_computed_dims(new_sympy_shape)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(new_sympy_shape),
                )
            )

            # MODIFIED PART START
            # noinspection PyBroadException
            try:
                data = self._try_get_value(node, 0)
                if data is not None:
                    np_type = to_numpy_type(vi.type.tensor_type.elem_type)
                    self.sympy_data_[node.output[0]] = np.asarray(data).astype(np_type).reshape(
                        get_shape_from_sympy_shape(new_sympy_shape))

            except Exception:
                # Failed to infer the data (doesn't matter).
                pass
            # MODIFIED PART END

    def _infer_Unsqueeze(self, node):  # noqa: N802
        input_shape = self._get_shape(node, 0)
        op_set = get_opset(self.out_mp_)

        # Depending on op-version 'axes' are provided as attribute or via 2nd input
        if op_set < 13:
            axes = get_attribute(node, "axes")
            assert self._try_get_value(node, 1) is None
        else:
            axes = self._try_get_value(node, 1)
            assert get_attribute(node, "axes") is None

        output_rank = len(input_shape) + len(axes)
        axes = [handle_negative_axis(a, output_rank) for a in axes]

        input_axis = 0
        output_shape = []
        for i in range(output_rank):
            if i in axes:
                output_shape.append(1)
            else:
                output_shape.append(input_shape[input_axis])
                input_axis += 1

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )

        # MODIFIED PART START
        # noinspection PyBroadException
        try:
            data = self._try_get_value(node, 0)
            if data is not None:
                np_type = to_numpy_type(vi.type.tensor_type.elem_type)
                self.sympy_data_[node.output[0]] = np.asarray(data).astype(np_type).reshape(output_shape)
        except Exception:
            # Failed to infer the data (doesn't matter).
            pass
        # MODIFIED PART END

    def _infer_Squeeze(self, node):  # noqa: N802
        input_shape = self._get_shape(node, 0)
        op_set = get_opset(self.out_mp_)

        # Depending on op-version 'axes' are provided as attribute or via 2nd input
        if op_set < 13:
            axes = get_attribute(node, "axes")
            assert self._try_get_value(node, 1) is None
        else:
            axes = self._try_get_value(node, 1)
            assert get_attribute(node, "axes") is None

        if axes is None:
            # No axes have been provided (neither via attribute nor via input).
            # In this case the 'Shape' op should remove all axis with dimension 1.
            # For symbolic dimensions we guess they are !=1.
            output_shape = [s for s in input_shape if s != 1]
            if self.verbose_ > 0:
                symbolic_dimensions = [s for s in input_shape if type(s) != int]  # noqa: E721
                if len(symbolic_dimensions) > 0:
                    logger.debug(
                        f"Symbolic dimensions in input shape of op: '{node.op_type}' node: '{node.name}'. "
                        f"Assuming the following dimensions are never equal to 1: {symbolic_dimensions}"
                    )
        else:
            axes = [handle_negative_axis(a, len(input_shape)) for a in axes]
            output_shape = []
            for i in range(len(input_shape)):
                if i not in axes:
                    output_shape.append(input_shape[i])
                else:
                    assert input_shape[i] == 1 or type(input_shape[i]) != int  # noqa: E721
                    if self.verbose_ > 0 and type(input_shape[i]) != int:  # noqa: E721
                        logger.debug(
                            f"Symbolic dimensions in input shape of op: '{node.op_type}' node: '{node.name}'. "
                            f"Assuming the dimension '{input_shape[i]}' at index {i} of the input to be equal to 1."
                        )

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )

        # MODIFIED PART START
        # noinspection PyBroadException
        try:
            data = self._try_get_value(node, 0)
            if data is not None:
                np_type = to_numpy_type(vi.type.tensor_type.elem_type)
                self.sympy_data_[node.output[0]] = np.asarray(data).astype(np_type).reshape(output_shape)
        except Exception:
            # Failed to infer the data (doesn't matter).
            pass
        # MODIFIED PART END

    def _infer_Transpose(self, node):  # noqa: N802
        # noinspection PyBroadException
        try:
            # Original code only infers the data.
            super()._infer_Transpose(node)
        except Exception:
            # Failed to infer the data. Just continue with shape inference.
            pass

        # Infer the output shape. 6D and larger didn't work with the original code.
        input_shape = list(self._get_shape(node, 0))
        perm = get_attribute(node, "perm", list(reversed(range(len(input_shape)))))

        output_shape = translator.apply_permutation_to(input_shape, perm)

        input_type = self.known_vi_[node.input[0]].type.tensor_type.elem_type

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[0], input_type, get_shape_from_sympy_shape(output_shape))
        )

    def _infer_Slice(self, node):  # noqa: N802
        # The majority of this code was taken from `symbolic_shape_inference.py`.

        # SymPy fails to prove that `x_0 + ... + x_n >= 0` if one of `x_i` is a `sympy.Min(a, b)`,
        # even when the relation holds for both `a` and `b`.
        #
        # When given `expr` of form `min(a, b) + ...`, this function returns `[a + ..., b + ...]`,
        # so that we can prove inequalities for both expressions separately.
        #
        # If the number of `min(...)` subexpressions is not exactly one, this function just returns `[expr]`.
        def flatten_min(expr):
            # MODIFIED PART START
            assert isinstance(expr, sympy.Add), f"`Slice` shape inference: Expected a sum of two arguments, got {expr}"
            # MODIFIED PART END
            min_positions = [idx for idx in range(len(expr.args)) if isinstance(expr.args[idx], sympy.Min)]
            if len(min_positions) == 1:
                min_pos = min_positions[0]

                def replace_min_with_arg(arg_idx):
                    replaced = list(expr.args)
                    # MODIFIED PART START
                    assert isinstance(
                        replaced[min_pos], sympy.Min
                    ), f"`Slice` shape inference: Expected a sympy.Min() at position {min_pos}, got {replaced[min_pos]}"
                    assert (
                            len(replaced[min_pos].args) == 2
                    ), f"`Slice` shape inference: Expected a sympy.Min() with exactly 2 arguments, got {replaced[min_pos]}"
                    # MODIFIED PART END
                    replaced[min_pos] = replaced[min_pos].args[arg_idx]
                    return sympy.Add(*replaced)

                return [
                    replace_min_with_arg(0),
                    replace_min_with_arg(1),
                ]
            return [expr]

        def less_equal(x, y):
            try:
                return bool(x <= y)
            except TypeError:
                pass
            try:
                return bool(y >= x)
            except TypeError:
                pass
            try:
                return bool(-x >= -y)
            except TypeError:
                pass
            try:
                return bool(-y <= -x)
            except TypeError:
                pass
            try:
                return bool(y - x >= 0)
            except TypeError:
                # the last attempt; this may raise TypeError
                return all(bool(d >= 0) for d in flatten_min(y - x))

        def handle_negative_index(index, bound):
            """normalizes a negative index to be in [0, bound)"""
            try:
                if not less_equal(0, index):
                    if is_literal(index) and index <= -self.int_max_:
                        # this case is handled separately
                        return index
                    return bound + index
            except TypeError:
                # MODIFIED PART START
                logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                         'Shape inference of `Slice` failed unexpectedly.')
                # MODIFIED PART END
            return index

        if get_opset(self.out_mp_) <= 9:
            axes = get_attribute(node, "axes")
            starts = get_attribute(node, "starts")
            ends = get_attribute(node, "ends")
            if not axes:
                axes = list(range(len(starts)))
            steps = [1] * len(axes)
        else:
            starts = as_list(self._try_get_value(node, 1), keep_none=True)
            ends = as_list(self._try_get_value(node, 2), keep_none=True)
            axes = self._try_get_value(node, 3)
            steps = self._try_get_value(node, 4)
            if axes is None and not (starts is None and ends is None):
                axes = list(range(0, len(starts if starts is not None else ends)))
            if steps is None and not (starts is None and ends is None):
                steps = [1] * len(starts if starts is not None else ends)
            axes = as_list(axes, keep_none=True)
            steps = as_list(steps, keep_none=True)

        new_sympy_shape = self._get_sympy_shape(node, 0)
        if starts is None or ends is None:
            if axes is None:
                for i in range(len(new_sympy_shape)):
                    new_sympy_shape[i] = self._new_symbolic_dim_from_output(node, 0, i)
            else:
                new_sympy_shape = get_shape_from_sympy_shape(new_sympy_shape)
                for i in axes:
                    new_sympy_shape[i] = self._new_symbolic_dim_from_output(node, 0, i)
        else:
            for i, s, e, t in zip(axes, starts, ends, steps):
                e = handle_negative_index(e, new_sympy_shape[i])  # noqa: PLW2901

                if is_literal(e):
                    if e >= self.int_max_:
                        e = new_sympy_shape[i]  # noqa: PLW2901
                    elif e <= -self.int_max_:
                        e = 0 if s > 0 else -1  # noqa: PLW2901
                    elif is_literal(new_sympy_shape[i]):
                        # MODIFIED PART START
                        if t > 0:
                            e = common.clamp(e, 0, new_sympy_shape[i])
                        else:
                            e = common.clamp(e, -1, new_sympy_shape[i] - 1)
                        # MODIFIED PART END
                    else:
                        if e > 0:
                            e = (  # noqa: PLW2901
                                sympy.Min(e, new_sympy_shape[i]) if e > 1 else e
                            )  # special case for slicing first to make computation easier
                else:
                    if is_literal(new_sympy_shape[i]):
                        e = sympy.Min(e, new_sympy_shape[i])  # noqa: PLW2901
                    else:
                        try:
                            if not less_equal(e, new_sympy_shape[i]):
                                e = new_sympy_shape[i]  # noqa: PLW2901
                        except Exception:
                            # MODIFIED PART START
                            logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                                     'Shape inference of `Slice` failed due to the use of symbolic shapes.')
                            # MODIFIED PART END

                s = handle_negative_index(s, new_sympy_shape[i])  # noqa: PLW2901
                if is_literal(new_sympy_shape[i]) and is_literal(s):
                    # MODIFIED PART START
                    if t > 0:
                        s = common.clamp(s, 0, new_sympy_shape[i])
                    else:
                        s = common.clamp(s, 0, new_sympy_shape[i] - 1)
                    # MODIFIED PART END

                # MODIFIED PART START
                if t > 0:
                    # `e` must be >= `s`.
                    e = max(e, s)

                else:
                    # `e` must be <= `s`.
                    e = min(e, s)
                # MODIFIED PART END

                new_sympy_shape[i] = sympy.simplify((e - s + t + (-1 if t > 0 else 1)) // t)

            self._update_computed_dims(new_sympy_shape)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )

        # handle sympy_data if needed, for slice in shape computation
        if (
                node.input[0] in self.sympy_data_
                and [0] == axes
                and starts is not None
                and len(starts) == 1
                and ends is not None
                and len(ends) == 1
                and steps is not None
                and len(steps) == 1
        ):
            input_sympy_data = self.sympy_data_[node.input[0]]
            # MODIFIED PART START  (used to be `np.array` instead of `np.ndarray` + remove single dimension data check)
            if type(input_sympy_data) == list or type(input_sympy_data) == np.ndarray:  # noqa: E721
                # MODIFIED PART END
                self.sympy_data_[node.output[0]] = input_sympy_data[starts[0]: ends[0]: steps[0]]

    def _infer_Identity(self, node):
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                self._get_shape(node, 0),
            )
        )

        # Try to infer the output data.
        values = self._try_get_value(node, 0)
        if values is not None:
            np_type = to_numpy_type(vi.type.tensor_type.elem_type)
            self.sympy_data_[node.output[0]] = np.array(values).astype(np_type)

    def _infer_Resize(self, node):  # noqa: N802
        vi = self.known_vi_[node.output[0]]
        input_sympy_shape = np.array(self._get_sympy_shape(node, 0), np.float32)

        if get_opset(self.out_mp_) <= 10:  # V10
            scales = self._try_get_value(node, 1)
            if scales is not None:
                new_sympy_shape = [sympy.simplify(sympy.floor(d * s)) for d, s in zip(input_sympy_shape, scales)]
                self._update_computed_dims(new_sympy_shape)
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        get_shape_from_sympy_shape(new_sympy_shape),
                    )
                )
            else:
                logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                         'Cannot infer the output shape of ONNX `Resize` with a dynamic `scales` input.')

        else:  # V11+
            uses_roi_tensor = len(node.input) >= 2 and node.input[1] != ''
            scales = self._try_get_value(node, 2)
            sizes = self._try_get_value(node, 3)

            rank = self._get_shape_rank(node, 0)
            if rank != 4:
                # Shape inference is not implemented for these cases. The conversion to TFLite would be currently
                #  impossible anyway.
                logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                         'The inference of the output shape of ONNX `Resize` with rank != 4 is not implemented.')

            axes = get_attribute(node, 'axes', list(range(rank)))  # Axes specify which dimensions are effected.

            if sizes is not None:
                # The code below does not consider the `roi` input, because the conversion doesn't currently support it.
                if uses_roi_tensor and \
                        get_attribute(node, 'coordinate_transformation_mode', 'half_pixel') == 'tf_crop_and_resize':
                    logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                             'The inference of the output shape of ONNX `Resize` is not implemented when the `roi` '
                             'input is specified.')

                output_shape = list(self._get_sympy_shape(node, 0))  # Copy the input shape.

                # Modify the input shape, where `axes` specify.
                for axis, size in zip(axes, sizes):
                    output_shape[axis] = sympy.simplify(size)
                self._update_computed_dims(output_shape)

            elif scales is not None:
                if uses_roi_tensor and \
                        get_attribute(node, 'coordinate_transformation_mode', 'half_pixel') == 'tf_crop_and_resize':
                    # The conversion doesn't support `roi` input, so it is not implemented for shape inference either.
                    logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                             'The inference of the output shape of ONNX `Resize` is not implemented when the `roi` '
                             'input is specified.')

                output_shape = list(self._get_sympy_shape(node, 0))  # Copy the input shape.

                # Modify the input shape, where `axes` specify.
                for axis, scale in zip(axes, scales):
                    output_shape[axis] = sympy.simplify(round(output_shape[axis] * scale))
                self._update_computed_dims(output_shape)

            else:
                logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                         'Cannot infer the output shape of ONNX `Resize` with dynamic inputs.')

            # noinspection PyUnboundLocalVariable
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(output_shape),
                )
            )

    def _infer_Range(self, node):  # noqa: N802
        vi = self.known_vi_[node.output[0]]
        input_data = self._get_int_or_float_values(node, allow_float_values=True)
        if any([i is None for i in input_data]):
            # Dynamic inputs with no inferred data. Shape inference is not possible.
            logger.e(logger.Code.SHAPE_INFERENCE_ERROR, 'Failed to infer output shape of `Range` with dynamic inputs.')

        start = as_scalar(input_data[0])
        limit = as_scalar(input_data[1])
        delta = as_scalar(input_data[2])
        if delta == 0:
            # https://github.com/microsoft/onnxruntime/blob/d30c81d270894f41ccce7b102b1d4aedd9e628b1/onnxruntime/core/providers/cpu/generator/range.cc#L65
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `Range` has `delta` = 0, which is not allowed.')

        if type(0.1) in {type(start), type(limit), type(delta)}:
            # The `Range` uses float parameters. Computation of the output shape can be incorrect, because of float
            #  errors. For example if start=0.1, limit=12.3 and delta=0.1, '(limit - start) / delta' should be 122, but
            #  because of float errors, it is 122.0000000745058. `sympy.ceiling()` then turns it into 123, which is not
            #  correct.
            # Doing the computation in float32 instead of float64 seems to fix this.

            start = np.float32(start)
            limit = np.float32(limit)
            delta = np.float32(delta)

        # This equation is a combination of what ORT and TFLite do:
        # https://github.com/microsoft/onnxruntime/blob/ee603ee3265dbf6eac112baf273b6b69bf696085/onnxruntime/core/providers/cuda/generator/range.cc#L67-L70
        # https://github.com/tensorflow/tensorflow/blob/07cda44e5192ba122718049a2f9e9e3c65fc2dde/tensorflow/lite/kernels/range.cc#L54C34-L54C55
        new_sympy_shape = [sympy.Max(sympy.ceiling((limit - start) / delta), 0)]

        self._update_computed_dims(new_sympy_shape)
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )

        # Try to infer sympy data.
        # noinspection PyBroadException
        try:
            np_type = to_numpy_type(vi.type.tensor_type.elem_type)
            self.sympy_data_[node.output[0]] = np.arange(start, limit, delta).astype(np_type)

        except Exception as _:
            # This shouldn't happen. In case it somehow does happen, simply continue.
            pass

    def _infer_ReduceX(self, node):  # noqa: N802
        """ Infer the output shape for `Reduce*` operators, which used an `axes` attribute up-to version 18 (excluding),
             and then switched to an `axes` input.
            This includes `ReduceL2`, `ReduceMax`, `ReduceMean`, `ReduceProd` and potentially others.
        """
        keep_dims = get_attribute(node, 'keepdims', 1)
        if get_opset(self.out_mp_) >= 18:
            # ReduceMean/ReduceL2 v18+ uses 'axes' as input.
            axes = self._try_get_value(node, 1)
            if axes is None:
                if len(node.input) <= 1 or self.tensor_is_static(node.input[1]) or node.input[1] == '':
                    # The `axes` is omitted. The default value will be used.
                    pass

                else:
                    # The `axes` is a dynamic tensor and there is no inferred data for it.
                    #  It is impossible to correctly infer the output shape.
                    logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                             f'Failed to infer output shape of `{node.op_type}` with a dynamic `axes` input tensor.')

        else:
            # 'axes' is an attribute of the operator.
            axes = get_attribute(node, 'axes')

        shape = self._get_shape(node, 0)
        if axes is None:
            # The default value is all axes.
            axes = np.arange(len(shape))

        axes = [handle_negative_axis(a, len(shape)) for a in axes]

        # Compute the output shape.
        output_shape = []
        for i, d in enumerate(shape):
            if i in axes:
                if keep_dims:
                    output_shape.append(1)
            else:
                output_shape.append(d)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )

    def _infer_ReduceSum(self, node):  # noqa: N802
        keep_dims = get_attribute(node, 'keepdims', 1)
        if get_opset(self.out_mp_) >= 13:
            # ReduceSum v13+ uses 'axes' as input.
            axes = self._try_get_value(node, 1)
            if axes is None:
                if len(node.input) <= 1 or self.tensor_is_static(node.input[1]) or node.input[1] == '':
                    # The `axes` is omitted. The default value will be used.
                    pass

                else:
                    # The `axes` is a dynamic tensor and there is no inferred data for it.
                    #  It is impossible to correctly infer the output shape.
                    logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                             f'Failed to infer output shape of `{node.op_type}` with a dynamic `axes` input tensor.')

        else:
            # 'axes' is an attribute of the operator.
            axes = get_attribute(node, 'axes')

        shape = self._get_shape(node, 0)
        if axes is None:
            # The default value is all axes.
            axes = np.arange(len(shape))

        axes = [handle_negative_axis(a, len(shape)) for a in axes]

        # Compute the output shape.
        output_shape = []
        for i, d in enumerate(shape):
            if i in axes:
                if keep_dims:
                    output_shape.append(1)
            else:
                output_shape.append(d)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )

    def _infer_ConstantOfShape(self, node):
        # Infer sympy data.
        # noinspection PyBroadException
        try:
            shape = self._try_get_value(node, 0)
            if shape is not None:
                value_tensor: onnx.TensorProto = get_attribute(node, 'value')
                if value_tensor is not None:
                    value = numpy_helper.to_array(value_tensor)
                    assert value.size == 1
                else:
                    value = np.array([0.0], np.float32)  # Default value.

                result = np.tile(value, shape)
                self.sympy_data_[node.output[0]] = result

        except Exception:
            # Failed to infer output data. Continue to shape inference.
            pass

        sympy_shape = self._get_int_or_float_values(node)[0]
        if sympy_shape is None:
            logger.e(logger.Code.SHAPE_INFERENCE_ERROR, 'Inference of the output shape of `ConstantOfShape` with a '
                                                        'dynamic input is not possible.')

        vi = self.known_vi_[node.output[0]]
        if sympy_shape is not None:
            if type(sympy_shape) != list:
                sympy_shape = [sympy_shape]
            self._update_computed_dims(sympy_shape)
        else:
            # create new dynamic shape
            # note input0 is a 1D vector of shape, the new symbolic shape has the rank of the shape vector length
            sympy_shape = self._new_symbolic_shape(self._get_shape(node, 0)[0], node)

        if val := get_attribute(node, 'value'):
            data_type = val.data_type
        else:
            # By default, the `ConstantOfShape` has a float32 value (0.0).
            data_type = onnx.TensorProto.FLOAT

        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                data_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )

    def _infer_QGemm(self, node):  # noqa: N802
        a_shape = self._get_shape(node, 0)
        b_shape = self._get_shape(node, 3)

        if get_attribute(node, "transA", False):
            m = a_shape[1]
        else:
            m = a_shape[0]

        if get_attribute(node, "transB", False):
            n = b_shape[0]
        else:
            n = b_shape[1]

        if len(node.input) <= 7:
            # No output quantization parameters specified -> output tensor should be float32.
            output_type = onnx.TensorProto.FLOAT
        else:
            output_type = self.known_vi_[node.input[0]].type.tensor_type.elem_type

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                output_type,
                [m, n]
            )
        )
        self.known_vi_[vi.name] = vi

    def _infer_Tile(self, node):

        # Try to infer the output data.
        try:
            values = [self._try_get_value(node, i) for i in range(len(node.input))]
            if all([val is not None for val in values]):
                inpt = values[0]
                reps = values[1]
                result = np.tile(inpt, reps)
                self.sympy_data_[node.output[0]] = result
        except Exception:
            # Failed to infer output data. Continue to shape inference.
            pass

        # The following code is taken from 'symbolic_shape_infer.py'.
        repeats_value = self._try_get_value(node, 1)
        new_sympy_shape = []
        if repeats_value is not None:
            input_sympy_shape = self._get_sympy_shape(node, 0)
            for i, d in enumerate(input_sympy_shape):
                new_dim = d * repeats_value[i]
                new_sympy_shape.append(new_dim)
            self._update_computed_dims(new_sympy_shape)
        else:
            new_sympy_shape = self._new_symbolic_shape(self._get_shape_rank(node, 0), node)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )

    def _infer_Concat(self, node):
        # The following code is taken from 'symbolic_shape_infer.py'.
        sympy_shape = self._get_sympy_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis"), len(sympy_shape))
        for i_idx in range(1, len(node.input)):
            input_shape = self._get_sympy_shape(node, i_idx)
            if input_shape:
                sympy_shape[axis] = sympy_shape[axis] + input_shape[axis]
        self._update_computed_dims(sympy_shape)
        # merge symbolic dims for non-concat axes
        for d in range(len(sympy_shape)):
            if d == axis:
                continue
            dims = [self._get_shape(node, i_idx)[d] for i_idx in range(len(node.input)) if self._get_shape(node, i_idx)]
            if all([d == dims[0] for d in dims]):
                continue
            merged = self._merge_symbols(dims)
            if type(merged) == str:
                sympy_shape[d] = self.symbolic_dims_[merged] if merged else None
            else:
                sympy_shape[d] = merged
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )

        # MODIFIED PART START
        # Try to infer the output data.
        # noinspection PyBroadException
        try:
            values = [self._try_get_value(node, i) for i in range(len(node.input))]
            if all([val is not None for val in values]):
                # Make sure all values are numpy arrays. Sometimes there is a combination of scalars and arrays...
                normalized_values = []
                for val in values:
                    if isinstance(val, np.ndarray):
                        normalized_values.append(val)
                    elif isinstance(val, list):
                        normalized_values.append(np.asarray(val))
                    else:
                        normalized_values.append(np.asarray([val]))

                result = np.concatenate(normalized_values, get_attribute(node, "axis"))
                result = result.astype(to_numpy_type(vi.type.tensor_type.elem_type))  # Cast to the real type.

                self.sympy_data_[node.output[0]] = result
        except Exception:
            # Failed to infer output data. Continue to shape inference.
            pass
        # MODIFIED PART END

    def _infer_Split_Common(self, node, make_value_info_func):  # noqa: N802
        input_sympy_shape = self._get_sympy_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis", 0), len(input_sympy_shape))

        split = get_attribute(node, "split")
        if split:
            split = [sympy.Integer(s) for s in split]

        elif self.optional_input_not_given(node, 1):
            # No `split` specified.

            num_outputs = get_attribute(node, "num_outputs")
            if num_outputs is None:
                # Split evenly
                num_outputs = sympy.Integer(len(node.output))
                split = [input_sympy_shape[axis] / num_outputs] * num_outputs
                self._update_computed_dims(split)

            else:
                # Split according to the 'num_outputs' attribute.
                num_outputs = sympy.Integer(num_outputs)
                dim_size = input_sympy_shape[axis]
                if dim_size % num_outputs == 0:
                    split = [input_sympy_shape[axis] / num_outputs] * num_outputs

                else:
                    # The dimension doesn't divide nicely -> The last chunk will be smaller. This is according to the
                    #  ONNX documentation. But it doesn't cover all cases. For example if dim_size=4 and num_outputs=3.
                    #  We cannot have the first two 'split' dimensions be the same, and the last one smaller.
                    #  ONNX Runtime implements this quite literally, so it doesn't support these problematic cases.
                    #  https://github.com/microsoft/onnxruntime/blob/7dade5d05b67f4da8cc9ab949d576159682aff20/onnxruntime/core/providers/cpu/tensor/split.cc#L89
                    chunk = np.ceil(float(dim_size) / num_outputs)
                    reminder = dim_size % chunk
                    split = [chunk] * (num_outputs - 1) + [reminder]
                    # The problem arises when reminder >= chunk, or when reminder == 0.
        else:
            # 'split' passed as input tensor
            split = self._try_get_value(node, 1)
            if split is None:
                # Cannot accurately infer the output shape.
                logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                         "Couldn't infer output shape of 'Split' operator with a dynamic 'split' operand.")

        for i_o in range(len(split)):
            vi = self.known_vi_[node.output[i_o]]
            vi.CopyFrom(
                make_value_info_func(
                    node.output[i_o],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(input_sympy_shape[:axis] + [split[i_o]] + input_sympy_shape[axis + 1:]),
                )
            )
            self.known_vi_[vi.name] = vi

    def _infer_Constant(self, node):
        if get_attribute(node, "value") is not None:
            t = get_attribute(node, "value")
            try:
                self.sympy_data_[node.output[0]] = numpy_helper.to_array(t)
            except TypeError as e:
                # The 'to_array()' function doesn't seem to support complex types. As there are other issues with them,
                #  just exit with error for now.
                logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                         "Couldn't infer output shape of 'Constant' operator with a complex data type.", e)

        if get_attribute(node, "sparse_value") is not None:
            # Sparse tensors are not yet supported
            logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                     "Couldn't infer output shape of 'Constant' operator with a sparse value.")

        simple_values = ['value_int', 'value_float', 'value_string', 'value_ints', 'value_floats', 'value_strings']
        for value in simple_values:
            attr = get_attribute(node, value)
            if attr is not None:
                self.sympy_data_[node.output[0]] = np.asarray(attr)

    def _infer_QLinearAdd(self, node):
        input_1_shape = self._get_shape(node, 0)
        input_2_shape = self._get_shape(node, 3)

        output_shape = super()._broadcast_shapes(input_1_shape, input_2_shape)
        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(onnx.helper.make_tensor_value_info(vi.name, output_dtype, output_shape))

    def _infer_QLinearMul(self, node):
        input_1_shape = self._get_shape(node, 0)
        input_2_shape = self._get_shape(node, 3)

        output_shape = super()._broadcast_shapes(input_1_shape, input_2_shape)
        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(onnx.helper.make_tensor_value_info(vi.name, output_dtype, output_shape))

    def _infer_QLinearGlobalAveragePool(self, node):
        input_shape = self._get_shape(node, 0)
        input_rank = len(input_shape)
        channels_last = get_attribute(node, "channels_last")

        if channels_last:
            output_shape = [input_shape[0]] + [1] * (input_rank - 2) + [input_shape[-1]]
        else:
            output_shape = input_shape[:2] + [1] * (input_rank - 2)

        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(onnx.helper.make_tensor_value_info(vi.name, output_dtype, output_shape))

    def _infer_QLinearSoftmax(self, node):
        # Strategy: same shape as input
        super()._propagate_shape_and_type(node, input_index=0, output_index=0)

    def _infer_Dropout(self, node):
        super()._propagate_shape_and_type(node, input_index=0, output_index=0)

        # Dropout layers might have "mask" output tensor. Infer shape to avoid incomplete inference error.
        if len(node.output) > 1 and node.output[1]:
            mask_shape = self._get_shape(node, 0)
            vi = self.known_vi_[node.output[1]]
            vi.CopyFrom(onnx.helper.make_tensor_value_info(node.output[1], onnx.TensorProto.BOOL, mask_shape))

    def _infer_Flatten(self, node):
        """ Ensure that flatten resolves dynamic shapes """

        input_shape = self._get_shape(node, 0)
        dynamic_dim_index = None
        for idx, dim in enumerate(input_shape):
            if (not isinstance(dim, int)) or (dim < 0):
                dynamic_dim_index = idx
                break
        if dynamic_dim_index is not None:
            tmp_shape = input_shape.copy()
            tmp_shape[dynamic_dim_index] = 1

            axis = get_attribute(node, "axis")
            new_shape = [math.prod(tmp_shape[0:axis]), math.prod(tmp_shape[axis:])]

            if dynamic_dim_index < axis:
                new_shape[0] = -1
            else:
                new_shape[1] = -1

            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(onnx.helper.make_tensor_value_info(vi.name, output_dtype, new_shape))

    def _infer_QLinearConcat(self, node):
        axis = get_attribute(node, "axis")

        aggregated_dimension = 0

        for idx in range(2, len(node.input), 3):
            input_shape = self._get_shape(node, idx)
            aggregated_dimension += input_shape[axis]

        output_shape = self._get_shape(node, 2).copy()
        output_shape[axis] = aggregated_dimension

        vi = self.known_vi_[node.output[0]]
        output_dtype = self.known_vi_[node.input[2]].type.tensor_type.elem_type
        vi.CopyFrom(onnx.helper.make_tensor_value_info(vi.name, output_dtype, output_shape))

    def _infer_QLinearAveragePool(self, node):
        sympy_shape = self._compute_avg_pool_shape(node)
        self._update_computed_dims(sympy_shape)
        output = node.output[0]

        input_vi = self.known_vi_[node.input[0]]
        vi = self.known_vi_[output]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                output,
                input_vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )

    def _compute_avg_pool_shape(self, node):
        # This is simplified code from self._compute_conv_pool_shape()
        sympy_shape = self._get_sympy_shape(node, 0)
        kernel_shape = get_attribute(node, "kernel_shape")
        rank = len(kernel_shape)

        if len(sympy_shape) != rank + 2:
            logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                     'Unexpected error occurred during shape inference of ONNX `AveragePool`.')

        # only need to symbolic shape inference if input has symbolic dims in spatial axes
        spatial_shape = sympy_shape[-rank:]
        is_symbolic_dims = [not is_literal(i) for i in spatial_shape]

        if not any(is_symbolic_dims):
            shape = get_shape_from_value_info(self.known_vi_[node.output[0]])
            if shape is not None and len(shape) > 0:
                if len(sympy_shape) != len(shape):
                    logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                             'Unexpected error occurred during shape inference of ONNX `AveragePool`.')
                sympy_shape[-rank:] = [sympy.Integer(d) for d in shape[-rank:]]
                return sympy_shape

        dilations = get_attribute(node, "dilations", [1] * rank)
        strides = get_attribute(node, "strides", [1] * rank)
        effective_kernel_shape = [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]
        pads = get_attribute(node, "pads")
        if pads is None:
            auto_pad = get_attribute(node, "auto_pad", b"NOTSET").decode("utf-8")
            if auto_pad != "VALID" and auto_pad != "NOTSET":
                try:
                    residual = [sympy.Mod(d, s) for d, s in zip(sympy_shape[-rank:], strides)]
                    total_pads = [
                        max(0, (k - s) if r == 0 else (k - r))
                        for k, s, r in zip(effective_kernel_shape, strides, residual)
                    ]
                except TypeError:  # sympy may throw TypeError: cannot determine truth value of Relational
                    total_pads = [
                        max(0, (k - s)) for k, s in zip(effective_kernel_shape, strides)
                    ]  # assuming no residual if sympy throws error
            elif auto_pad == "VALID":
                total_pads = []
            else:
                total_pads = [0] * rank
        else:
            if len(pads) != 2 * rank:
                logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                         'ONNX `AveragePool` has mismatched `kernel_shape` and `pads` attributes.')

            total_pads = [p1 + p2 for p1, p2 in zip(pads[:rank], pads[rank:])]

        ceil_mode = get_attribute(node, "ceil_mode", 0)
        for i in range(rank):
            effective_input_size = sympy_shape[-rank + i]
            if len(total_pads) > 0:
                effective_input_size = effective_input_size + total_pads[i]
            if ceil_mode:
                strided_kernel_positions = sympy.ceiling(
                    (effective_input_size - effective_kernel_shape[i]) / strides[i]
                )
            else:
                strided_kernel_positions = (effective_input_size - effective_kernel_shape[i]) // strides[i]
            sympy_shape[-rank + i] = strided_kernel_positions + 1
        return sympy_shape

    def _find_undefined_dimensions(self, graph: GraphProto):
        """
        Look for dynamic and symbolic dimensions in model inputs and raise error when
        there are any.

        :param graph: Searched graph.
        """

        symbolic_shapes = set()
        dynamic_inputs = set()

        def iterate_value_infos(value_infos):
            for vi in value_infos:
                if vi.type.HasField("tensor_type"):
                    shape = vi.type.tensor_type.shape
                    if shape:
                        for dim in shape.dim:
                            if dim.HasField("dim_param"):
                                symbolic_shapes.add(dim.dim_param)
                            elif dim.HasField("dim_value"):
                                pass
                            else:
                                dynamic_inputs.add(vi.name)

        iterate_graph_per_graph_func(graph, lambda graph: iterate_value_infos(graph.input))

        symbolic_shape_names = "'" + "', '".join(symbolic_shapes) + "'"
        dynamic_inputs_names = "'" + "', '".join(dynamic_inputs) + "'"

        if symbolic_shapes and dynamic_inputs:
            logger.e(logger.Code.INVALID_INPUT,
                     f"Model inputs are not statically defined. They contain following symbolic dimensions: "
                     f"{symbolic_shape_names}, and following input tensors are dynamic: {dynamic_inputs_names}. Make "
                     f"them static using arguments '--symbolic-dimension-into-static' and/or '--set-input-shape'.")

        elif symbolic_shapes:
            logger.e(logger.Code.INVALID_INPUT,
                     f"Model inputs contain following symbolic dimensions: {symbolic_shape_names}. Make these "
                     f"dimensions static using argument '--symbolic-dimension-into-static'.")

        if dynamic_inputs:
            logger.e(logger.Code.INVALID_INPUT,
                     f"Model has dynamically defined inputs: {dynamic_inputs_names}. Make these inputs "
                     f"static using argument '--set-input-shape'.")

    def _get_symbolic_batch_dim_name_for_single_input_graph(self, graph: GraphProto) -> str | None:
        """
        Get name of symbolic batch dimension if it is present in the graph
        and graph has exactly one input tensor.

        :param graph: Searched graph.
        :return: Name of the symbolic batch dimension or 'None' if not present.
        """
        input_names = set(map(lambda inp: inp.name, graph.input))
        initializer_names = set(map(lambda initializer: initializer.name, graph.initializer))

        dynamic_input_names = list(input_names - initializer_names)

        if len(dynamic_input_names) != 1:
            return None

        for vi in graph.input:
            if vi.name == dynamic_input_names[0] and vi.type.HasField("tensor_type"):
                shape = vi.type.tensor_type.shape
                if shape and len(shape.dim) > 0:
                    batch_dim = shape.dim[0]
                    if batch_dim.HasField("dim_param"):
                        return batch_dim.dim_param

        return None

    def preprocess_model(self, in_mp: ModelProto, symbolic_dimensions_mapping: dict[str, int] | None,
                         input_shapes_mapping: dict[str, tuple] | None):
        if input_shapes_mapping:
            for input_name, input_shape in input_shapes_mapping.items():
                make_input_shape_fixed(in_mp.graph, input_name, input_shape)

        if symbolic_dimensions_mapping:
            for symbolic_shape_name, dimension_size in symbolic_dimensions_mapping.items():
                make_dim_param_fixed(in_mp.graph, symbolic_shape_name, dimension_size)

        if dim_name := self._get_symbolic_batch_dim_name_for_single_input_graph(in_mp.graph):
            logger.w(f"Model has symbolic batch dimension: '{dim_name}'. Changing it to 1.")
            make_dim_param_fixed(in_mp.graph, dim_name, 1)

        self._find_undefined_dimensions(in_mp.graph)

        super()._preprocess(in_mp)

    def tensor_is_static(self, tensor_name: str) -> bool:
        if not hasattr(super(), 'initializers_'):
            return False

        return tensor_name in super().initializers_

    # noinspection PyMethodMayBeStatic
    def optional_input_not_given(self, node: onnx.NodeProto, input_index: int):
        """ Return `True` if `node` doesn't have and optional input tensor on index `input_index` specified. """

        return (len(node.input) <= input_index) or (node.input[input_index] == '')

    def get_shapes_of_all_node_tensors(self, node: onnx.NodeProto) -> list[list | None]:
        shapes = []

        for tensor_name in chain(node.input, node.output):
            if tensor_name == '':
                continue

            if tensor_name in self.known_vi_:
                vi = self.known_vi_[tensor_name]
                shapes.append(get_shape_from_value_info(vi))

            elif tensor_name in self.initializers_:
                shapes.append(list(self.initializers_[tensor_name].dims))

            else:
                shapes.append(None)

        return shapes

    # noinspection PyMethodMayBeStatic
    def check_for_problematic_nodes(self, model: onnx.ModelProto):
        """ Check if the model contains nodes which can never have their shapes statically inferred. If it does, raise
             an error with a corresponding message.
        """
        problematic_op_types = ['NonZero', 'NonMaxSuppression', 'Unique']
        problematic_nodes = [node.op_type for node in model.graph.node if node.op_type in problematic_op_types]

        if len(problematic_nodes) != 0:
            problematic_nodes = set(problematic_nodes)  # Remove duplicates.

            logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                     f'The ONNX model contains the nodes `{problematic_nodes}`. It is impossible to statically '
                     'infer their output shapes, because they depend on runtime data.')

    def try_get_first_node_without_inferred_shapes(self, model: onnx.ModelProto) -> onnx.NodeProto | None:
        """ Return the first node in the model, which doesn't have all its shapes inferred. If a shape is symbolic, it
             is considered as inferred by this function. Only completely missing shapes count as not inferred.
            If all shapes are inferred, return `None`.
        """
        for node in model.graph.node:
            shapes = self.get_shapes_of_all_node_tensors(node)
            if any(shape is None or len(shape) == 0 for shape in shapes):
                # This node doesn't have all its shapes inferred.
                return node

        return None

    @staticmethod
    def infer_shapes(in_mp, int_max=2 ** 31 - 1, auto_merge=False, guess_output_rank=False, verbose=0,
                     symbolic_dimensions_mapping: dict[str, int] | None = None,
                     input_shapes_mapping: dict[str, tuple] | None = None,
                     inferred_tensor_data: dict[str, np.ndarray] | None = None,
                     generate_artifacts_after_failed_shape_inference: bool = True) -> onnx.ModelProto:
        """ Infer the shapes of all tensor in the provided model.

        :param in_mp: ONNX ModelProto for which the shapes wil be inferred.
        :param int_max: Maximum int32 value.
        :param auto_merge: Automatically merge symbolic dimensions when possible.
        :param guess_output_rank: Guess the output rank from the input[0] (only applies for unknown operators).
        :param verbose: Print detailed logs of inference. 0: off, 1: warning, 3: detailed.
        :param symbolic_dimensions_mapping: Dictionary mapping names of symbolic dimensions to integers, which will be
                                             set as the static values of the symbolic dimensions.
        :param input_shapes_mapping: Dictionary mapping input tensor names to new shapes.
        :param inferred_tensor_data: Dict which will be filled with inferred data of tensors.
        :param generate_artifacts_after_failed_shape_inference: If shape inference fails or is incomplete, generate the
                                                                partly inferred ONNX model as sym_shape_infer_temp.onnx.
        :return: ONNX ModelProto, equivalent to `in_mp`, but with defined tensor shapes.
        """

        # noinspection PyBroadException
        try:
            symbolic_shape_inference = ModelShapeInference(int_max, auto_merge, guess_output_rank, verbose)
            symbolic_shape_inference.check_for_problematic_nodes(in_mp)

            all_shapes_inferred = False
            symbolic_shape_inference.preprocess_model(in_mp, symbolic_dimensions_mapping, input_shapes_mapping)
            while symbolic_shape_inference.run_:
                all_shapes_inferred = symbolic_shape_inference._infer_impl()
            symbolic_shape_inference._update_output_from_vi()

            for node in in_mp.graph.node:
                node_tensor_shapes = symbolic_shape_inference.get_shapes_of_all_node_tensors(node)
                for shape in node_tensor_shapes:
                    if shape is None or (not shape_is_well_defined(shape)):
                        # Shape contains symbolic dimensions. This is prohibited for now.
                        all_shapes_inferred = False

            if not all_shapes_inferred:
                if generate_artifacts_after_failed_shape_inference:
                    onnx.save_model(symbolic_shape_inference.out_mp_, "sym_shape_infer_temp.onnx",
                                    save_as_external_data=True)
                    logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                             "The inference of the shapes of internal ONNX tensors was not successful. Best effort "
                             "model is stored in 'sym_shape_infer_temp.onnx'.")

                else:
                    logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                             "The inference of the shapes of internal ONNX tensors was not successful.")

            if inferred_tensor_data is not None:
                inferred_tensor_data.clear()
                for tensor_name, tensor_data in symbolic_shape_inference.sympy_data_.items():
                    inferred_tensor_data[tensor_name] = tensor_data

            return symbolic_shape_inference.out_mp_
        except Error as e:
            # Just propagate the error
            raise e
        except Exception as e:
            # Some unexpected error happened during shape inference.
            logger.d(f"Generic shape inference exception caught ({type(e).__name__}). {traceback.format_exc()}")

            # Try to figure out which operator caused the issue.
            # noinspection PyUnboundLocalVariable
            node = symbolic_shape_inference.try_get_first_node_without_inferred_shapes(symbolic_shape_inference.out_mp_)
            if node is not None:
                logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                         'Unexpected internal error occurred during shape inference. A possible cause might be the'
                         f' `{node.op_type}` operator with input tensors {node.input} and output tensors '
                         f'{node.output}. Please report this issue.')

            else:
                logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                         f"Unexpected internal error during shape inference. Please report this issue.")
