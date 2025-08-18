#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import cast

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import mod_attributes
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options.floor_mod_options import FloorMod
from onnx2tflite.src.tflite_generator.meta.types import FLOATS, INTS, UINTS


class ModConverter(NodeConverter):
    node = 'Mod'

    onnx_supported_types = FLOATS + INTS + UINTS
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/kernels/floor_mod.cc#L141-L167
    tflite_supported_types = INTS + [TensorType.FLOAT32]
    verified_types = [TensorType.INT32, TensorType.INT64]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX `Mod` operator into TFLite `FloorMod`. """

        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f'ONNX `Mod` has {len(t_op.tmp_inputs)} inputs instead of 2.')

        attrs = cast(mod_attributes.Mod, node.attributes)

        if attrs.fmod == 1:
            # There is no simple way to represent this in TFLite.
            # If the `FloorMod` operator is used, some output values will not be correct. For example:
            #   in_a    in_b  |   ONNX output     TFLite output
            #   -10      9    |   -1               8
            #    10     -9    |    1              -8
            # Conversion may be possible via multiple operators. (doesn't seem like a common use case)
            logger.e(logger.Code.NOT_IMPLEMENTED, 'Conversion of ONNX `Mod` with `fmod=1` is not supported.')

        type_ = t_op.tmp_inputs[0].type
        if type_ in FLOATS and attrs.fmod != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL, 'ONNX `Mod` uses a float type with `fmod=0`.')
        self.assert_type_allowed(type_)

        t_op.builtin_options = FloorMod()

        ops = OpsList(middle_op=t_op)

        # ONNX `Mod` supports shape broadcasting.
        ops.add_pre(self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0]))

        return [t_op]
