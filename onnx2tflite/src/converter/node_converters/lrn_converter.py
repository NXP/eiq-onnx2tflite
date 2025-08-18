#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
from typing import cast

import onnx2tflite.src.logger as logger
import onnx2tflite.src.tflite_generator.builtin_options.lrn_options as tfl_lrn_options
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import lrn_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class LRNConverter(NodeConverter):
    node = 'LRN'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/local_response_norm.cc#L73-L94
    tflite_supported_types = [TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX 'LRN' to TFLite 'LocalResponseNormalization'. """

        attrs = cast(lrn_attributes.LRN, node.attributes)

        if attrs.alpha <= 0.0:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX LRN attribute 'alpha' is not positive!")

        if attrs.beta <= 0.0:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX LRN attribute 'beta' is not positive!")

        if attrs.size <= 0:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX LRN attribute 'size' is not positive!")

        if (attrs.size % 2) == 0:
            # ONNXRT: Only odd 'size' is supported by ONNX Runtime (onnxruntime/core/providers/cpu/nn/lrn.h: Line 22)
            # Conversion of even 'size' might be problematic because of different rounding in ONNX (ceil x floor).
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE, "ONNX 'LRN' operator has even 'size'!")

        x = t_op.tmp_inputs[0]

        self.assert_type_allowed(x.type)

        if x.rank != 4:
            # ONNXRT: Only input with rank 4 is supported by ONNX Runtime
            #  (onnxruntime/core/providers/cpu/nn/lrn.cc: Line 61)
            logger.e(logger.Code.INVALID_ONNX_MODEL, f"ONNX 'LRN' input tensor has rank '{x.rank}' instead of '4'. "
                                                     "This is not supported by ONNX Runtime and therefore conversion is"
                                                     " not implemented.")

        t_op.builtin_options = tfl_lrn_options.LRN(
            radius=(attrs.size - 1) // 2,
            bias=attrs.bias,
            alpha=attrs.alpha / attrs.size,
            beta=attrs.beta
        )

        return [t_op]
