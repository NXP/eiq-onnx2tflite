#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from typing import List, cast

import numpy as np
import onnx

from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion.common import OpsList
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.builtin_attributes import multinomial_attributes
from onnx2tflite.src.onnx_parser.meta.types import name_for_onnx_type
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import multinomial_options
from onnx2tflite.src.tflite_generator.meta.types import FLOATS


class MultinomialConverter(NodeConverter):
    node = 'Multinomial'

    onnx_supported_types = FLOATS
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/random_ops.cc#L206
    tflite_supported_types = [TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32]

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> List[tflite_model.Operator]:
        """ Convert ONNX `Multinomial` operator into TFLite `Multinomial`. """

        ops = OpsList(middle_op=t_op)

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     f'ONNX `Multinomial` has unexpected number of inputs ({len(t_op.tmp_inputs)}).')

        x = t_op.tmp_inputs[0]
        self.assert_type_allowed(x.type)

        attrs = cast(multinomial_attributes.Multinomial, node.attributes)

        if attrs.dtype not in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]:
            # ONNX documentation allows only these 2 output types. Both are also supported by TFLite.
            # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/random_ops.cc#L303-L319
            logger.e(logger.Code.INVALID_ONNX_MODEL, 'ONNX `Multinomial` has unexpected value of `dtype` attribute '
                                                     f'({name_for_onnx_type(attrs.dtype)}).')

        if attrs.seed is None:
            # ONNX would auto-generate a seed in this case. TFLite requires a seed, so we must generate a random one.
            seed_int = np.random.randint(np.iinfo('int32').min, np.iinfo('int32').max)

        else:
            # ONNX `Multinomial` supports a `seed` attribute which is type `float32`. TFLite `Multinomial` also has a
            #  `seed` but it is an `int32` type. Therefore, it is not possible to use the same seed after conversion. So
            #  even if both ONNX Runtime and TFLite inference engines used the same random number generator, the
            #  generated values would be different.
            # The best we can do is to use a deterministic bijection to map the `float32` seeds to corresponding `int32`
            #  seeds. So if multiple `Multinomial` operators are used in the same ONNX model with the same seeds, they
            #  will also have the same seeds in the TFLite model.
            logger.w('ONNX `Multinomial` has a specified `seed` attribute. The converted TFLite `Multinomial` will also'
                     ' use a seed, but the operator will generate different random values.')

            seed_bytes = np.array(attrs.seed, 'float32').tobytes()  # float32 -> raw data
            seed_int = np.frombuffer(seed_bytes, 'int32').item()  # raw data -> int32

        num_samples = self.builder.create_tensor_for_data(np.array(attrs.sample_size, 'int32'), 'num_samples')
        t_op.tmp_inputs.append(num_samples)

        t_op.builtin_options = multinomial_options.Multinomial(seed_int, seed_int)

        return ops.flatten()
