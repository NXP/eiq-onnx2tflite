#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import pow_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class PowConverter(NodeConverter):
    node = 'Pow'

    onnx_supported_types = FLOATS + [TensorType.INT32, TensorType.INT64]
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/pow.cc#L131-L145
    tflite_supported_types = [TensorType.FLOAT32, TensorType.INT32]
    verified_types = [TensorType.FLOAT32, TensorType.INT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """ Convert ONNX `Pow` to TFLite `Pow`. """

        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR, 'ONNX `Pow` has invalid number of inputs.')

        base = t_op.tmp_inputs[0]
        power = t_op.tmp_inputs[1]

        # TFLite requires power and base to have the same type.
        if base.type != power.type:
            # If this is required, prepend a `Cast` operator. Not sure if it is a common use case.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     'Conversion of ONNX `Pow` with non-matching input types is not supported.')

        self.assert_type_allowed(base.type)

        if power.type == TensorType.INT32 and not tensor_has_data(power):
            # TFLite only supports positive powers when the `power` tensor is `int32`.
            # If needed, introduce a flag to guarantee positive power values.
            logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                     'Conversion of ONNX `Pow` with type `int32` and dynamic `power` input is not supported because '
                     'the power values at runtime could be negative, which is not supported by TFLite.')

        ops = OpsList(middle_op=t_op)

        # ONNX Pow supports shape broadcasting.
        ops.add_pre(self.builder.ensure_correct_broadcasting(t_op, t_op.tmp_outputs[0]))

        t_op.builtin_options = pow_options.Pow()

        return ops.flatten()
