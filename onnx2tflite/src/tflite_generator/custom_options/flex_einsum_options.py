#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src.converter.conversion import translator
from onnx2tflite.src.tflite_generator.meta.meta import CustomOptions


class Einsum(CustomOptions):

    def __init__(self, equation: str, num_operands: int, data_type: TensorType) -> None:
        """ Custom options of the `FlexEinsum` operator.

            The operator attributes are specified in:
            https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/core/ops/compat/ops_history_v2/Einsum.pbtxt

            The `CustomOptions` are in the `FLEXBUFFERS` format as specified by the schema.
            The Python library `flatbuffers.flexbuffers` seems to be incompatible with the C++ version of the library.
            I used `mlir` to generate a .tflite model with a `FlexEinsum` operator and then used the `utils/Makefile` to
             convert it to .json. I extracted the "custom_options" from the operator and tried the following:
              ```
                  from flatbuffers import flexbuffers as fx
                  buf = bytes(custom_options_from_json)
                  vec = fx.GetRoot(buf)  # Returns a `Vector` reference
                  attr_map = vec.AsVector[1]  # Returns a `String` reference instead of a `Map` reference.
              ```
            The last command should return a `Map` reference containing the attributes of the operator. Instead, it
             returns a string, which contains the raw data. Clearly there is a parsing issue.

            The solution in this file involves taking the `custom_options` from the .json, and modifying it manually.
             I identified the meaning of the bytes by testing and comparing with other `custom_options`, so there may be
             mistakes.

        :param equation: The equation to be computed by the `Einsum` operator.
        :param num_operands: Number of the operands in the equation.
        :param data_type: The data type used. Refers to some internal TensorFlow type enum, instead of the TFLite
                           `TensorType` from the schema. The mentioned enum can be found here:
                            https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/core/framework/types.proto#L13-L87
                           The `translator` module provides a function for conversion to this type enum.
        """
        eq_len = len(equation)
        equation_ascii = [ord(char) for char in equation]

        internal_type = translator.tflite_type_to_tensor_flow_data_type(data_type)

        data = [
                   6, 69, 105, 110, 115, 117, 109, 0,  # "Einsum"

                   eq_len + 48, 18,  # Metadata

                   6, 69, 105, 110, 115, 117, 109, 26, 0, 26, 0,  # "Einsum" + metadata

                   42, 7, 10, 1,  # Separator (*)

                   78, 18, 2, 24, num_operands,  # Attribute 'N' (number of operands in the equation)

                   42, 7, 10, 1,  # Separator (*)

                   84, 18, 2, 48, internal_type,  # Attribute 'T' (type).

                   42, eq_len + 14, 10, 8,  # Separator (*) + metadata

                   101, 113, 117, 97, 116, 105, 111, 110,  # "equation"

                   18, eq_len + 2, 18, eq_len  # Metadata

               ] + equation_ascii + [
                   50, 0, 0, 2, eq_len + 58, eq_len + 51, 20, 20, 4, 40, 1  # Metadata
               ]

        super().__init__("FlexEinsum", bytearray(data))
