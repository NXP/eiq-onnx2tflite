#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src.tflite_generator.meta.meta import CustomOptions


class FlexTranspose(CustomOptions):

    def __init__(self) -> None:
        super().__init__("FlexTranspose", bytearray([
            9, 84, 114, 97, 110, 115, 112, 111, 115, 101,
            0, 39, 18,
            9, 84, 114, 97, 110, 115, 112, 111, 115, 101, 26, 0, 26, 0, 42, 11, 10, 5, 84, 112, 101, 114, 109, 18, 2,
            48, 3, 42, 7, 10,
            1, 84, 18, 2, 48, 1, 50, 0, 0,
            2, 52, 42, 20, 20, 4, 40, 1
        ]))
