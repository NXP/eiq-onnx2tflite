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
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes.upsample_attributes import Upsample
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.resize_bilinear_options import ResizeBilinear
from onnx2tflite.src.tflite_generator.builtin_options.resize_nearest_neighbor_options import ResizeNearestNeighbor
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES


# noinspection SpellCheckingInspection
class UpsampleConverter(NodeConverter):
    node = "Upsample"

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/resize_bilinear.cc#L118-L152
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/resize_nearest_neighbor.cc#L109-L142
    tflite_supported_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT16]
    verified_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8]

    def _get_scales(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[float]:
        attrs = cast(Upsample, node.attributes)

        if node.version < 9:
            # The `scales` are an attribute.
            scales = attrs.scales
            if scales is None:
                # The attribute is required.
                logger.e(logger.Code.INVALID_ONNX_MODEL,
                         f"ONNX `Upsample` v {node.version} is missing the required attribute `scales`.")

        else:
            # The `scales` are an input tensor.
            if len(t_op.tmp_inputs) != 2:
                logger.e(logger.Code.INVALID_ONNX_MODEL,
                         f"ONNX `Upsample` version {node.version} has {len(t_op.tmp_inputs)} inputs instead of 2.")

            scales_tensor = t_op.tmp_inputs[1]
            if tensor_has_data(scales_tensor):
                scales = scales_tensor.tmp_buffer.data
            elif (scales := self.context.onnx_inspector.try_get_inferred_tensor_data(scales_tensor.name)) is not None:
                logger.i(f"Using inferred data for `Upsample` input `scales` named `{scales_tensor.name}`.")

            if scales is None:
                # The scales are dynamic. Shape inference would have already failed.
                # If shape inference is skipped, conversion may be possible by using the `Shape` and `Mul` operators
                #  to multiply the input shape by the dynamic scales, and use the result as the `size` for the Resize*.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX `Upsample` with a dynamic `scales` input is not supported.")

        return list(scales)

    def _check_types(self, t_op: tflite_model.Operator):
        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        if x.type != y.type:
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX `Upsample` has different input and output type.")

        self.assert_type_allowed(x.type)

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX `Upsample` operator to TFLite `ResizeBilinear` or `ResizeNearestNeighbor`.

        Quantized version of this operator is not supported because it cannot be present in a valid ONNX model.
         `Upsample` was added in opset 7 and removed in opset 10.
         All ONNX quantized operators were introduced in opset 10.
         Trese is no overlap in the opsets, so `Upsample` cannot be in the same model as for example
         `QuantizeLinear`.
        """
        if len(t_op.tmp_inputs) not in {1, 2}:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `Upsample` has unexpected number of inputs. Got {len(t_op.tmp_inputs)}, expected 1 or 2.")

        self._check_types(t_op)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]
        if x.rank != 4:
            # TFLite `ResizeBilinear` and `ResizeNearestNeighbor` are only implemented for 4D inputs.
            # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/resize_bilinear.cc#L75
            # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/resize_nearest_neighbor.cc#L73
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     "Conversion of ONNX `Upsample` is only supported for 4D inputs.")

        if x.quantization is not None or y.quantization is not None:
            # ONNX quantization related operators were introduces in opset 10.
            # ONNX `Upsample` was removed in opset 10 and ONNX Runtime crashes if one is in a model.
            # That's why conversion of quantized `Upsample` is not supported.
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX `Upsample` with quantized data is not supported.")

        scales = self._get_scales(node, t_op)
        if len(scales) != x.rank:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f"ONNX `Upsample` has `scales` with {len(scales)} elements and a {x.rank}D input.")

        # ONNX allows scaling of the batch and channels as well, which isn't supported by TFLite.
        # The `scales` refer to channels first dimensions (N, C, H, W).
        if any(s != 1 for s in scales[:2]):
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     f"Conversion of ONNX `Upsample` with scales = {scales} is not possible.")

        if any(s < 1 for s in scales):
            logger.e(logger.Code.INVALID_ONNX_MODEL, "ONNX `Upsample` has some `scales` < 1.")

        attrs = cast(Upsample, node.attributes)
        if attrs.mode == "nearest":
            # ONNX doens't specify what extrapolation parameters are used. Experiments yield best results with these
            #  arguments.
            t_op.builtin_options = ResizeNearestNeighbor(False, False)

        elif attrs.mode == "linear":
            t_op.builtin_options = ResizeBilinear(False, False)

        else:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX `Upsample` has unexpected `mode` attribute ({attrs.mode}).")

        # Create the `size` input tensor for the TFLite `Resize*` operator.
        if not y.shape.is_well_defined():
            # The shape of `y` contains symbolic dimensions.
            # Conversion may be possible by adding `Shape` and `Mul` operators.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX `Upsample` with unknown tensor shapes is not supported.")

        # Create the `size` TFLite input based on the output shape.
        h, w = y.shape[1:3]  # `y` has shape NHWC.
        sizes = self.builder.create_tensor_for_data(np.array([h, w], np.int32), "sizes")
        t_op.tmp_inputs = [x, sizes]

        # Conversion was successful. However, some `scales` values may cause a difference between the ONNX and TFLite
        #  outputs. I can't say with 100% certainty when this happens, but it seems that when the input shape * scales
        #  would result in non-integer shape, some values will be wrong. But it's not always consistent. This probably
        #  happens because of different rounding methods between TFLite and ONNX. A very similar situation happens when
        #  converting `Resize`.
        # Print an info message to let the user know of this potential source of inaccuracies.
        logger.i("Conversion of ONNX `Upsample` may cause the output model to produce slightly different outputs.")

        return [t_op]
