#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import tile_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, name_for_type


class TileConverter(NodeConverter):
    node = 'Tile'

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/tile.cc#L235-L260
    tflite_supported_types = [TensorType.INT8, TensorType.UINT8, TensorType.FLOAT32, TensorType.INT32, TensorType.INT64,
                              TensorType.STRING, TensorType.BOOL]

    # noinspection PyMethodMayBeStatic
    def _check_input_types(self, t_op: tflite_model.Operator):
        x = t_op.tmp_inputs[0]
        repeats = t_op.tmp_inputs[1]

        self.assert_type_allowed(x.type)

        if repeats.type != TensorType.INT64:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `Tile` has unexpected `repeats` type {name_for_type(repeats.type)}.')

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX `Tile` operator to TFLite `Tile`. """

        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `Tile` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `2`.')

        self._check_input_types(t_op)

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]
        repeats = t_op.tmp_inputs[1]

        ops = OpsList(middle_op=t_op)

        if x.tensor_format.is_channels_last():
            # The repeats refer to a `channels_first` tensor. Modify them to be compatible with the `channels_last`
            #  input tensor.
            perm = translator.create_channels_first_to_channels_last_permutation(x.rank, True)
            if tensor_has_data(repeats):
                # Permute the `repeats` statically.
                repeats_data = list(repeats.tmp_buffer.data)
                repeats_data = translator.apply_permutation_to(repeats_data, perm)

                # Create a new `repeats` tensor, in case it is also used by some other operator.
                new_repeats = self.builder.create_tensor_for_data(np.array(repeats_data, np.int64), 'repeats')
                t_op.tmp_inputs[1] = new_repeats

            else:
                # Prepend a `Gather` operator, to move the values of `repeats` around to match the channels_last format.
                ops.add_pre(self.builder.create_gather_before(t_op, 1, perm, [x.rank]))

        if t_op.is_quantized_without_qdq():
            # Non-QDQ model -> just propagate
            propagate_quantization(x, y)

        elif t_op.is_qdq_quantized() and x.quantization != y.quantization:
            # I/O q-params doesn't match -> external quantizer was used or removed Clip/Relu modified q-params.
            # We need to re-quantize output because Tile expects shared q-params for input and output.
            logger.w("Requantizing the output of Tile operator. Quantizing with onnx2quant quantizer "
                     "can potentially avoid this.")
            scale = x.quantization.scale.vector
            zp = x.quantization.zero_point.vector
            ops.post_ops.insert(0, self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp))

        t_op.builtin_options = tile_options.Tile()

        return ops.flatten()
