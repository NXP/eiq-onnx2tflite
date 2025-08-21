#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.builder.model_builder import tensor_has_data
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList, try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import resize_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.resize_bilinear_options import ResizeBilinear
from onnx2tflite.src.tflite_generator.builtin_options.resize_nearest_neighbor_options import ResizeNearestNeighbor
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES


# noinspection SpellCheckingInspection,PyMethodMayBeStatic
class ResizeConverter(NodeConverter):
    node = "Resize"

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/resize_bilinear.cc#L118-L152
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/resize_nearest_neighbor.cc#L109-L142
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT16]
    verified_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8]

    def _convert_transformation_mode(self, coordinate_transformation_mode: str, y: tflite_model.Tensor) -> (bool, bool):
        """Convert the `coordinate_transformation_mode` attribute of ONNX `Resize` to the attributes `align_corners`
             and `half_pixel_centers` of the TFLite `Resize*` operators.

        :param coordinate_transformation_mode:Attribute of ONNX `Resize`.
        :param y: Main output of the `Resize*` operator.
        :return: TFLite `align_corners` and `half_pixel_centers` attributes.
        """
        if coordinate_transformation_mode == "align_corners":
            align_corners, half_pixel = True, False

        elif coordinate_transformation_mode == "half_pixel":
            align_corners, half_pixel = False, True

        elif coordinate_transformation_mode == "asymmetric":
            align_corners, half_pixel = False, False

        elif coordinate_transformation_mode == "pytorch_half_pixel":
            # Check if this can be represented using the `half_pixel` mode.
            if y.shape.is_well_defined() and all(dim > 1 for dim in y.shape[1:-1]):
                # The output dimensions H and W are > 1.
                align_corners, half_pixel = False, True
            else:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         f"Conversion of ONNX `Resize` with `coordinate_transformation_mode=pytorch_half_pixel` and "
                         f"output shape `{y.shape}` is not possible.")

        else:
            # Unconvertible `coordinate_transformation_mode`.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"Conversion of ONNX `Resize` with `coordinate_transformation_mode` = "
                     f"`{coordinate_transformation_mode}` is not possible.")

        # noinspection PyUnboundLocalVariable
        return align_corners, half_pixel

    def _down_sampling(self, x: tflite_model.Tensor, y: tflite_model.Tensor) -> bool:
        """Determine if the `Resize` is down-sampling at least in 1 dimension.

        :param x: Main input of the ONNX `Resize`.
        :param y: Main output of the ONNX `Resize`.
        :return: True, if at least 1 dimension is being down-sampled.
        """
        return any(y_dim < x_dim for x_dim, y_dim in zip(x.shape, y.shape, strict=False))

    def _assert_convertible_scales(self, scales: np.ndarray, x: tflite_model.Tensor, y: tflite_model.Tensor,
                                   axes: list[int]) -> None:
        """Make sure that the `scales` input of the ONNX `Resize` is convertible to TFLite.
            The only way to convert it is to compute the resulting shape of the output, after the input is scaled.
             This is already done by the shape inference.
            The problem arises, when the scaled input doesn't result in whole numbers. For example, if the input has
             shape [10], and the scales are [1.05]. The oputput would have shape [11], but the values wouldn't
             correspond between ONNX and TFLite. There would be a relatively large error.

            ONNXRT: Aditionally, the ONNX documentation allows up/down scaling of the channels and batch dimensions. TFLite
             doesn't support this, but neither does ONNX Runtime:.
             https://github.com/microsoft/onnxruntime/blob/da86f6f40832cce75548771f6483a2aa7494bc75/onnxruntime/core/providers/cpu/tensor/upsamplebase.h#L435C23-L435C45

        :param scales: `scales` input of the ONNX `Resize`.
        :param x: Main input of the ONNX `Resize`.
        :param y: Main output of the ONNX `Resize`.
        :param axes: The `axes` attribute of the ONNX `Resize`.
        """
        # Make sure the axes are positive.
        axes = [axis if axis >= 0 else axis + x.rank for axis in axes]

        scales = list(scales)

        # Check that the `batch` and `channels` scales are 1.
        for axis, scale in zip(axes, scales, strict=False):
            if axis in {0, 1} and scale != 1.:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX `Resize` which scales the channels or batch dimension is not possible.")

        if x.tensor_format.is_channels_last():  # Should always be True.
            # The axes are for the channels first (ONNX) format. Permute them for the channels last (TFLite) format.
            perm = translator.create_channels_last_to_channels_first_permutation(x.rank)
            axes = [perm[axis] for axis in axes]

        float_output_shape = x.shape.vector.copy()
        for scale, axis in zip(scales, axes, strict=False):
            float_output_shape[axis] = scale * float_output_shape[axis]

        if not np.allclose(np.asarray(y.shape.vector), float_output_shape):
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE, f"Conversion of ONNX `Resize` with input shape `{x.shape}` "
                                                        f"and scales `{scales}` is not possible.")

    def _get_resize_inputs(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator
                           ) -> (tflite_model.Tensor | None, tflite_model.Tensor | None):
        """Return the input tensors `scales` and `sizes` of the ONNX `Resize`.

        :param node: ONNX `Resize` operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX `Resize`.
        :return: The `scales` and `sizes` input tensors or `None` in place of an omitted input.
        """
        sizes = None
        if node.version < 11:
            # Version 10 only had 2 inputs, `X` and `scales`. Both are required.
            if len(t_op.tmp_inputs) != 2:
                logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX `Resize` has invalid number of input tensors")

            scales = t_op.tmp_inputs[1]

        else:
            # Versions 11+ all use 4 inputs, `X`, `roi`, `scales` and `sizes`. The last 3 are optional.
            if not (1 <= len(t_op.tmp_inputs) <= 4):
                logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX `Resize` has invalid number of input tensors")

            scales = try_get_input(t_op, 2)
            sizes = try_get_input(t_op, 3)

        return scales, sizes

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX `Resize` operator into TFLite `ResizeBilinear` or `ResizeNearestNeighbor`.

            The whole conversion strategy relies on the input and output shape being well-defined. This approach is
             similar to the one used by for example `convert_reshape.py`.

            Sometimes with some `magical` shapes, some output values may be quite incorrect. I don't understand why, or
             when this will happen.
            For example, if mode='nearest', input_shape=[1, 3, 100, 100] and sizes=[1, 3, 213, 200], 99.5% of the
             output values will be correct, and only 600 values will be off by no more than 1. When the sizes are for
             example [1, 3, 215, 200], everything is accurate.

            Some attributes/inputs are not considered during conversion:
                - cubic_coeff_a - Only used with `mode = cubic`, which is not convertible.
                - exclude_outside - Only used with `cubic` mode, or `linear` mode with `antialias = 1` and upsampling.
                                     Both cases are not convertible and coreectly detected.
                - extrapolation_value - Only used with `coordinate_transformation_mode = tf_crop_and_resize`, which is
                                         not convertible. (At least not easily)
                - roi - Only used with `coordinate_transformation_mode = tf_crop_and_resize`. (Not supported)

        :param node: ONNX NodeProto representing the `Resize` operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators to add to the model.
        """
        o_resize = cast(resize_attributes.Resize, node.attributes)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]
        if x.rank != 4:
            # TFLite `ResizeBilinear` and `ResizeNearest` are only implemented for 4D inputs.
            # https://github.com/tensorflow/tensorflow/blob/7ac4dc3ea9e7de8b64580bd06f3d746c4bd3f902/tensorflow/lite/kernels/resize_bilinear.cc#L75C30-L75C43
            # https://github.com/tensorflow/tensorflow/blob/7ac4dc3ea9e7de8b64580bd06f3d746c4bd3f902/tensorflow/lite/kernels/resize_nearest_neighbor.cc#L73
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX `Resize` is only supported for 4D inputs.")

        if o_resize.antialias != 0 and self._down_sampling(x, y):
            # The ONNX `antialias` attribute is only active when downsampling, and cannot be represented in TFLite.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX `Resize` with `antialias` != 0 is not possible.")

        if o_resize.keep_aspect_ratio_policy != "stretch":
            # When testing how ONNX Runtime reacts to the other policies, it seemed buggy (non-sense error messages).
            # Instead of investigating further I decied to raise error for now, until we find a model which uses this.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX `Resize` with `keep_aspect_ratio_policy` == "
                     f"`{o_resize.keep_aspect_ratio_policy}` is not yet implemented.")

        if not t_op.is_qdq_quantized():
            self.assert_type_allowed(x.type)

        if t_op.is_quantized_without_qdq():
            # Propagate the quantization parameters, in case they are ever useful in the future.
            propagate_quantization(x, y)

        ops = OpsList(middle_op=t_op)

        align_corners, half_pixel = self._convert_transformation_mode(o_resize.coordinate_transformation_mode, y)

        if o_resize.mode == "linear":
            t_op.builtin_options = ResizeBilinear(align_corners, half_pixel)

        elif o_resize.mode == "nearest":
            t_op.builtin_options = ResizeNearestNeighbor(align_corners, half_pixel)

            if (o_resize.coordinate_transformation_mode == "asymmetric" and o_resize.nearest_mode == "floor") or (o_resize.coordinate_transformation_mode != "asymmetric" and \
                    o_resize.nearest_mode == "round_prefer_ceil"):
                # TFLite can handle this natively.
                pass
            # I haven't found a way to represent this in TFLite. There will always be a rounding error.
            elif self.context.conversion_config.accept_resize_rounding_error:
                # User has decided to accept the rounding error and to convert the model anyway.
                pass
            else:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         f"Accurate conversion of ONNX `Resize` with `nearest_mode={o_resize.nearest_mode}` is not "
                         f"possible because TFLite uses a different rounding approach. {logger.Style.cyan}If you "
                         "are wiling to accept the error caused by different rounding, run the converter again with"
                         " the flag `--accept-resize-rounding-error`.")

        else:
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"Conversion of ONNX `Resize` with mode = `{o_resize.mode}` is not possible.")

        rank = x.rank
        axes = o_resize.axes or list(range(rank))

        scales: tflite_model.Tensor
        sizes: tflite_model.Tensor
        scales, sizes = self._get_resize_inputs(node, t_op)

        if sizes is not None:
            # The ONNX `sizes` define the entire output shape. TFLite `size` only uses the spatial dimensions H and W.
            #  We can simply take the [H, W] from the output tensor. (same strategy as used by `convert_reshape.py`)

            h, w = y.shape[1:3]  # `y` has shape NHWC.
            sizes = self.builder.create_tensor_for_data(np.array([h, w], np.int32), "sizes")
            t_op.tmp_inputs = [x, sizes]

        elif scales is not None:
            if not tensor_has_data(scales):
                # The shape inference would have already failed.
                # If the users skipps shape inference and specifies the output shape, conversion would be possible by
                #  adding a flag, to let the user guarantee this.
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX `Resize` with dynamic `scales` input is not supported.")

            scales: np.ndarray = scales.tmp_buffer.data

            # TFLite doesn't have an equivalent input. We have to compute the resulting output shape and use the
            #  `size` input.
            self._assert_convertible_scales(scales, x, y, axes)

            # Create the `size` TFLite input based on the output shape.
            h, w = y.shape[1:3]  # `y` has shape NHWC.
            sizes = self.builder.create_tensor_for_data(np.array([h, w], np.int32), "sizes")
            t_op.tmp_inputs = [x, sizes]

        else:
            # Prohibited by the documentation.
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, "ONNX `Resize` has both `sizes` and `scales` specified.")

        return ops.flatten()
