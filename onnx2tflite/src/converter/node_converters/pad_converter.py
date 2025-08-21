#
# Copyright 2023 Martin Pavella
# Copyright 2023-2025 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#


from typing import cast

import numpy as np

import onnx2tflite.src.onnx_parser.builtin_attributes.pad_attributes as onnx_pad_attributes
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.converter.builder import model_builder
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.converter.conversion.common import OpsList, try_get_input
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.converter.quantization_utils import propagate_quantization, quantize_static_float_tensor
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model
from onnx2tflite.src.tflite_generator.builtin_options import mirror_pad_options, pad_options, pad_v2_options
from onnx2tflite.src.tflite_generator.meta.types import ALL_TYPES, INTS, name_for_type


# noinspection PyMethodMayBeStatic
class PadConverter(NodeConverter):
    node = "Pad"

    onnx_supported_types = ALL_TYPES
    # https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/pad.cc#L315-L369
    tflite_supported_types = INTS + [TensorType.UINT8, TensorType.FLOAT32]
    verified_types = [TensorType.FLOAT32, TensorType.UINT8, TensorType.INT8, TensorType.INT32, TensorType.INT64]

    # ONNX 'Pad' in 'reflect' mode doesn't work correctly in some cases. Example:
    # pads = [2, 2, 2, 2]
    # input:
    # [[0. 1.]
    #  [2. 3.]]
    #
    # output:
    # [[0. 0. 0. 0. 0. 0.]
    #  [0. 3. 2. 3. 2. 3.]
    #  [0. 1. 0. 1. 0. 1.]
    #  [0. 3. 2. 3. 2. 3.]
    #  [0. 1. 0. 1. 0. 1.]
    #  [0. 3. 2. 3. 2. 3.]]

    def _reflect_mode_bugged_use_case(self, input_shape: list[int], tflite_pads: list[int]) -> bool:
        """Determine if given scenario described by the input shape of the 'Pad' operator and its 'tflite_pads' operand
             will behave unpredictably due to bugs in the ONNX Runtime and TFLite inference engine.

            This situation seems to happen, when the padding for any dimension is at least as large, as the
             corresponding input dimension.

        :param input_shape: Shape of the main input tensor of the 'Pad' operator.
        :param tflite_pads: The 'pad' operand of the TFLite 'Pad' operator.
        :return: True, if the operators will behave unpredictably.
        """
        for pads, dim in zip(tflite_pads, input_shape, strict=False):
            if any(pad >= dim for pad in pads):
                return True

        return False

    def _create_default_constant_tensor_for_type(self, data_type: TensorType,
                                                 builder: model_builder.ModelBuilder) -> tflite_model.Tensor:
        """Create and return a static TFLite tensor containing a single value, which is the default 'constant_value'
             for the ONNX 'Pad' operator with given 'data_type'.

        :param data_type: The data type of the input tensor of the 'Pad' operator.
        :param builder: ModelBuilder object.
        :return: A static TFLite tensor, holding the default constant value.
        """
        if data_type == TensorType.STRING:
            # Generating string tensors is not yet supported, so this has not been tested.
            data = np.asarray([""], np.bytes_)

        elif data_type == TensorType.BOOL:
            # ONNXRT: ONNX Runtime and TFLite inference engine does not support this case, so it has not been tested.
            data = np.asarray([False], np.bool_)

        else:
            np_type = translator.tf_lite_type_to_numpy(data_type)
            data = np.asarray([0], np_type)

        return builder.create_tensor_for_data(data, "constant_value_")

    def _create_tflite_paddings_from_pads_and_axes(self, pads: list[int], axes: list[int], input_rank: int
                                                   ) -> list[list[int]]:
        """Create the TFLite 'paddings' operand from ONNX 'pads' and 'axes' operands. All operands belong to respective
             TFLite ond ONNX 'Pad' operators.

        :param pads: ONNX 'pads' operand.
        :param axes: ONNX 'axes' operand.
        :param input_rank: Rank of the main input of the ONNX 'Pad' operator.
        :return: TFLite 'paddings' operand.
        """
        start_pads = []
        end_pads = []
        half = len(pads) // 2
        for i in range(input_rank):
            if i in axes:
                # Pads for this axis are specified
                pads_axis = axes.index(i)
                start_pads.append(pads[pads_axis])
                end_pads.append(pads[half + pads_axis])

            else:
                # Pads are not specified. Default is 0.
                start_pads.append(0)
                end_pads.append(0)

        full_onnx_pads = start_pads + end_pads

        return translator.onnx_explicit_padding_to_tflite(full_onnx_pads)

    def _convert_pad_v2(self, o_pad: onnx_pad_attributes.Pad,
                        t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX 'Pad' operator version 2, to TFLite.

        :param o_pad: Attributes of the ONNX Pad v2 operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators to add to the model.
        """
        if o_pad.pads is None:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     "ONNX 'Pad' v2 is missing the required 'pads' attribute!")

        if len(t_op.tmp_inputs) != 1:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"ONNX 'Pad' v2 has unexpected number of inputs. Expected '1', got '{len(t_op.tmp_inputs)}'.")

        if t_op.tmp_inputs[0].type != TensorType.FLOAT32:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     "ONNX 'Pad' v2 uses unsupported input type. Expected 'FLOAT32', "
                     f"got '{name_for_type(t_op.tmp_inputs[0].type)}'.")

        x = t_op.tmp_inputs[0]
        y = t_op.tmp_outputs[0]

        ops = OpsList(middle_op=t_op)

        if x.quantization is not None and y.quantization is None:
            # Non-QDQ model -> just propagate
            propagate_quantization(x, y)
        elif x.quantization is not None and y.quantization is not None:
            if x.quantization != y.quantization:
                # I/O q-params doesn't match -> external quantizer was used or removed Clip/Relu modified q-params.
                # We need to re-quantize output because Reshape expects shared q-params for input and output.
                logger.w("Requantizing output of Pad operator. Internal quantizer can potentially avoid this.")
                scale = x.quantization.scale.vector
                zp = x.quantization.zero_point.vector
                ops.add_post(self.builder.create_quantize_operator_after(t_op, 0, x.type, scale, zp))

        if all(pad == 0 for pad in o_pad.pads) and self.builder.operator_can_be_skipped(t_op, self.inspector):
            # Operator is not adding any padding -> skip it
            self.builder.redirect_tensor(t_op.tmp_outputs[0], t_op.tmp_inputs[0])
            return []

        if any(pad < 0 for pad in o_pad.pads):
            # The operator is also removing data. TFLite 'Pad' doesn't support this.
            # https://github.com/tensorflow/tensorflow/blob/a8d000505e924cac9e8c6bfee544912292957d7e/tensorflow/lite/kernels/pad.cc#L136
            # Conversion may be possible via combination with the 'Slice' operator.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX 'Pad' with negative 'pads' is not yet implemented.")

        rank = t_op.tmp_inputs[0].rank
        tfl_paddings = translator.onnx_explicit_padding_to_tflite(list(o_pad.pads))
        if t_op.tmp_inputs[0].tensor_format.is_channels_last():
            # Make sure the order of the 'pads' matches the tensor format.
            to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(rank)
            tfl_paddings = translator.apply_permutation_to(tfl_paddings, to_tflite_perm)

        pads_tensor = self.builder.create_tensor_for_data(np.asarray(tfl_paddings, np.int32), "pads")
        t_op.tmp_inputs.append(pads_tensor)

        if o_pad.mode == "constant":
            if o_pad.value == 0.0:
                # Convert to 'Pad'
                t_op.builtin_options = pad_options.Pad()

            else:
                # Convert to 'PadV2'
                t_op.builtin_options = pad_v2_options.PadV2()

                # Create a tensor for the constant value
                constant_value_tensor = self.builder.create_tensor_for_data(np.asarray([o_pad.value], np.float32),
                                                                            "value")
                t_op.tmp_inputs.append(constant_value_tensor)

            return ops.flatten()

        if o_pad.mode == "reflect":
            # The ONNX Pad in 'reflect' mode somtimes behaves strangely (see top of this file).
            # The TFLite MirrorPad also seems weird and inconsistent (see mirror_pad_options.py).
            # This behavior seems to arise when the pads in any direction are at least as large as the corresponding
            #  input dimension. When this happens, accurate conversion is not possible right now.

            if self._reflect_mode_bugged_use_case(t_op.tmp_inputs[0].shape.vector, tfl_paddings):
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX 'Pad' version 2 in 'reflect' mode is not supported, due to inconsistent "
                         "behavior of the ONNX 'Pad' and the TFLite 'MirrorPad' operators in 'reflect' mode.")

            t_op.builtin_options = mirror_pad_options.MirrorPad()
            return ops.flatten()

        if o_pad.mode == "edge":
            # Conversion may be possible via other operators. Not sure. I haven't found a reasonable way to represent it
            #  in TFLite.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX 'Pad' version 2 in 'edge' mode is not implemented and may not be possible!")
        else:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f"ONNX 'Pad' version 2 has invalid mode attribute '{o_pad.mode}'!")

    def _prepare_constant_and_paddings_from_onnx_pad_v11_plus(self, t_op: tflite_model.Operator,
                                                              builder: model_builder.ModelBuilder
                                                              ) -> tuple[tflite_model.Tensor, tflite_model.Tensor]:
        """Create the 'paddings' and the 'constant' input tensors of the TFLite 'Pad' operator, from the ONNX 'Pad'
             v11+ operator.

            This function requires that the 'pads' ONNX operand is static. Otherwise, program exits.

        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :param builder: ModelBuilder object.
        :return: A 2 element tuple: First is the 'paddings' input tensor and
                                    Second is the 'constant' input tensor of the TFLite 'Pad' operator.
        """
        x = t_op.tmp_inputs[0]
        pads_tensor = t_op.tmp_inputs[1]
        y = t_op.tmp_outputs[0]

        rank = x.rank

        if not tensor_has_data(pads_tensor):
            # ONNX and TFLite use quite different formats for the 'pads' input. Converting a dynamic 'pads' would
            #  require adding multiple extra operators. (probably 2x Gather, Concat and potentially Transpose).
            # This use-case is probably not very common.
            logger.e(logger.Code.NOT_IMPLEMENTED,
                     "Conversion of ONNX 'Pad' with dynamic 'pads' input is not yet supported.")

        pads = list(pads_tensor.tmp_buffer.data)

        if (constant_tensor := try_get_input(t_op, 2)) is None:
            # Create a tensor with the default value for the input type.
            constant_tensor = self._create_default_constant_tensor_for_type(x.type, builder)

            # In case input is quantized, we need to quantize constant '0' as well, but only for QDQ
            if x.quantization is not None and y.quantization is not None:
                # Set constant tensor value to zp -> effectively zero value when "value == zp".
                # Quantization parameters are set later.
                constant_tensor.tmp_buffer.data[0] = x.quantization.zero_point[0]

        if (axes_tensor := try_get_input(t_op, 3)) is not None:
            # 'axes' is specified
            if not tensor_has_data(axes_tensor):
                # This case would require multiple extra operators and is probably not very common.
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of ONNX 'Pad' with dynamic 'axes' input is not yet supported.")

            axes = list(axes_tensor.tmp_buffer.data)

        else:
            # Create the default 'axes'
            axes = list(range(rank))

        tfl_paddings = self._create_tflite_paddings_from_pads_and_axes(pads, axes, rank)
        if x.tensor_format.is_channels_last():
            # Make sure the order of the paddings matches the tensor format.
            to_tflite_perm = translator.create_channels_first_to_channels_last_permutation(rank)
            tfl_paddings = translator.apply_permutation_to(tfl_paddings, to_tflite_perm)

        tfl_paddings_tensor = builder.create_tensor_for_data(np.asarray(tfl_paddings, np.int32), "paddings_")

        return tfl_paddings_tensor, constant_tensor

    def _convert_pad_v11_plus(self, o_pad: onnx_pad_attributes.Pad,
                              t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert ONNX 'Pad' operator version 11 or newer, to TFLite.

        :param o_pad: Attributes of the ONNX Pad v11+ operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators to add to the model.
        """
        if not (2 <= len(t_op.tmp_inputs) <= 4):
            logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                     f"ONNX 'Pad' version 11+ has '{len(t_op.tmp_inputs)}' inputs! Expected 2 to 4.")

        x = t_op.tmp_inputs[0]
        paddings_tensor, constant_tensor = self._prepare_constant_and_paddings_from_onnx_pad_v11_plus(t_op,
                                                                                                      self.builder)
        y = t_op.tmp_outputs[0]

        # Propagate the quantization parameters to the 'constant_value' tensor.
        if x.quantization is not None and constant_tensor.type in [TensorType.INT8, TensorType.UINT8]:
            # Non-QDQ model or zeros tensor for QDQ model - just propagate quantization
            propagate_quantization(x, constant_tensor)
        elif x.quantization is not None and constant_tensor.type in [TensorType.FLOAT32]:
            if tensor_has_data(constant_tensor):
                # QDQ model with static constant tensor - quantize to the same range as input
                scale, zp = x.quantization.scale.vector, x.quantization.zero_point.vector
                constant_tensor = quantize_static_float_tensor(self.builder, constant_tensor, x.type, scale, zp)
            elif x.quantization == constant_tensor.quantization:
                # QDQ model with dynamic constant tensor and quantized with our quantizer = we're fine
                pass
            else:
                logger.e(logger.Code.NOT_IMPLEMENTED,
                         "Conversion of QDQ quantized Pad operator with dynamic 'constant' "
                         "tensor is not implemented yet.")

        # Propagate quantization parameters to the output
        if x.quantization is not None:
            if y.quantization is None:
                # Non-QDQ model - just propagate q-params
                propagate_quantization(x, y)
            elif y.quantization == x.quantization:
                # QDQ model quantized with our quantizer or q-params match
                pass
            else:
                logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of QDQ quantized Pad operator with non-matching IO "
                                                      "q-params. Use internal quantizer to quantize the model!")

        if o_pad.mode == "constant":
            # Convert to 'PadV2'
            t_op.tmp_inputs = [x, paddings_tensor, constant_tensor]
            t_op.builtin_options = pad_v2_options.PadV2()
            return [t_op]

        if o_pad.mode == "reflect":
            # The ONNX Pad in 'reflect' mode somtimes behaves strangely (see top of this file).
            # The TFLite MirrorPad also seems weird and inconsistent (see mirror_pad_options.py).
            # This behavior seems to arise when the pads in any direction are at least as large as the corresponding
            #  input dimension. When this happens, accurate conversion is not possible right now.

            tfl_paddings = list(paddings_tensor.tmp_buffer.data)
            if self._reflect_mode_bugged_use_case(t_op.tmp_inputs[0].shape.vector, tfl_paddings):
                logger.e(logger.Code.CONVERSION_IMPOSSIBLE,
                         "Conversion of ONNX 'Pad' version 11+ in 'reflect' mode is not supported, due to inconsistent "
                         "behavior of the ONNX 'Pad' and the TFLite 'MirrorPad' operators in 'reflect' mode.")

            t_op.builtin_options = mirror_pad_options.MirrorPad()
            t_op.tmp_inputs = [x, paddings_tensor]
            return [t_op]

        if o_pad.mode == "edge":
            # Conversion may be possible via other operators. Not sure. I haven't found a reasonable way to represent it
            #  in TFLite.
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX 'Pad' version 11 or newer in 'edge' mode is not "
                                                  "implemented and may not be possible!")

        elif o_pad.mode == "wrap":
            # The 'wrap' mode has quite complicated behavior and I have not found reasonable way to represent it in
            #  TFLite. In some cases, it could be represented by TFLite MirrorPad in REFLECT mode, when that is fixed.
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX 'Pad' version 11 or newer in 'wrap' mode is not "
                                                  "implemented and may not be possible!")

        else:
            logger.e(logger.Code.INVALID_ONNX_OPERATOR_ATTRIBUTE,
                     f"ONNX 'Pad' has unexpected 'mode' attribute '{o_pad.mode}'.")

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        """Convert the ONNX 'Pad' operator to TFLite 'Pad', 'PadV2' or 'MirrorPad'.

            ONNX 'Pad' version 2 uses 'pads' and 'constant_value' as operator attributes. Versions 11 and newer use
             input tensors instead. For this reason, the conversion is split into 2 functions.

        :param node: ONNX `Pad` operator.
        :param t_op: TFLite operator with inputs and outputs corresponding to the ONNX operator.
        :return: A list of TFLite operators, to add to the model.
        """
        self.assert_type_allowed(t_op.tmp_inputs[0].type)

        o_pad = cast(onnx_pad_attributes.Pad, node.attributes)

        if node.version <= 1:
            logger.e(logger.Code.NOT_IMPLEMENTED, "Conversion of ONNX 'Pad' version 1 is not supported.")

        elif node.version < 11:  # Pad v2
            return self._convert_pad_v2(o_pad, t_op)

        else:  # Pad v 11/13/18/19
            return self._convert_pad_v11_plus(o_pad, t_op)
