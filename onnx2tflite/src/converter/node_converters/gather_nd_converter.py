#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List, cast

import numpy as np

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.quantization_utils import propagate_quantization
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import gather_nd_attributes
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import gather_nd_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, INTS


class GatherNDConverter(NodeConverter):
    node = 'GatherND'

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/gather_nd.cc#L138-L169
    tflite_supported_types = INTS + [TensorType.FLOAT32, TensorType.UINT8, TensorType.STRING, TensorType.BOOL]
    verified_types = INTS + [TensorType.FLOAT32, TensorType.UINT8, TensorType.STRING, TensorType.BOOL]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert the ONNX `GatherND` operator to TFLite `GatherND`. """

        if len(t_op.tmp_inputs) != 2:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `GatherND` has unexpected number of inputs. Got `{len(t_op.tmp_inputs)}`, expected `2`.')

        x = t_op.tmp_inputs[0]
        indices = t_op.tmp_inputs[1]
        y = t_op.tmp_outputs[0]

        self.assert_type_allowed(x.type)
        if t_op.is_quantized_without_qdq():
            propagate_quantization(x, y)

        ops = OpsList(middle_op=t_op)

        if x.tensor_format.is_channels_last():
            # Prepend a `Transpose` operator to make the main input `channels_first`.
            to_onnx_perm = translator.create_channels_last_to_channels_first_permutation(x.rank)
            ops.add_pre(self.builder.create_transpose_operator_before(t_op, 0, to_onnx_perm))
            t_op.tmp_inputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        if y.tensor_format.is_channels_last():
            # Append a `Transpose` to make the output `channels_last` again.
            to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(y.rank, return_list=True)
            ops.add_post(self.builder.create_transpose_operator_after(t_op, 0, to_tflite_perm))
            t_op.tmp_outputs[0].tensor_format = TensorFormat.CHANNELS_FIRST

        if indices.tensor_format.is_channels_last():
            # The indices are 3+D, and they have been assign the `channels last` format. Permute them back to
            #  `channels first` to match the ONNX model.
            to_onnx_perm = translator.create_channels_last_to_channels_first_permutation(indices.rank)
            if tensor_has_data(indices):
                indices = self.builder.duplicate_tensor(indices)
                translator.permute_static_tensor(indices, to_onnx_perm)
                t_op.tmp_inputs[1] = indices

            else:
                ops.add_pre(self.builder.create_transpose_operator_before(t_op, 1, to_onnx_perm))

            t_op.tmp_inputs[1].tensor_format = TensorFormat.CHANNELS_FIRST

        x = t_op.tmp_inputs[0]
        indices = t_op.tmp_inputs[1]
        attrs = cast(gather_nd_attributes.GatherND, node.attributes)
        if attrs.batch_dims != 0:
            # Doesn't seem like a common case.
            if tensor_has_data(indices):
                # The indices would have to be recomputed to support this. The first `batch_dims` dimensions are
                #  "ignored" and the indexing is effectively done not to `x`, by to `x[batch_dims:]`.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         'Conversion of ONNX `GatherND` with `batch_dims != 0` is not yet supported.')
            else:
                # The preprocessing would probably be too difficult (if possible at all) to implement using TFLite
                #  operators at runtime.
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         'Conversion of ONNX `GatherND` with `batch_dims != 0` is not supported.')

        if tensor_has_data(indices):
            indices_data = indices.tmp_buffer.data
        elif (indices_data := self.inspector.try_get_inferred_tensor_data(indices.name)) is not None:
            logger.i(f'Using inferred data for the ONNX `GatherND` input `indices` named `{indices.name}`.')

        # TFLite only supports non-negative indices. Make sure this requirement is satisfied.
        if indices_data is not None:
            if any(elt < 0 for elt in indices_data.flat):
                innermost_dim = indices.shape[-1]
                indices_data = indices_data.reshape([-1, innermost_dim])  # Flatten all dimensions except the last one.
                if len(indices_data.shape) != 2:
                    # This shouldn't happen.
                    logger.e(logger.Code.NOT_IMPLEMENTED,
                             'Conversion of ONNX `GatherND` with negative `indices` is not yet supported.')

                # `indices_data` is now a 2D matrix where the rows are vectors of indices to `x`.
                #  Iterate over the vectors and for each vector, normalize the index to range [0, dim_size - 1].
                new_indices_data = []
                for vector in indices_data:
                    new_vector = []
                    for i, elt in enumerate(vector):
                        if elt < 0:
                            elt += x.shape[i]
                            if elt < 0:
                                logger.e(logger.Code.INVALID_ONNX_MODEL, 'ONNX `GatherND` has invalid `indices`.')

                        new_vector.append(elt)

                    new_indices_data.append(new_vector)

                indices_data = np.array(new_indices_data, np.int64).reshape(indices.shape)
                indices = self.builder.create_tensor_for_data(indices_data, 'indices')
                t_op.tmp_inputs[1] = indices

        else:
            # Dynamic indices.
            if self.context.conversion_config.non_negative_indices:
                # User guarantees that the indices are not negative.
                pass

            else:
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         'Conversion of ONNX `GatherND` with dynamic `indices` is not possible because they may contain'
                         ' negative values, which is not supported by TFLite. '
                         + logger.Message.GUARANTEE_NON_NEGATIVE_INDICES)

        t_op.builtin_options = gather_nd_options.GatherND()

        return ops.flatten()
