#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src.tflite_generator.meta.meta import CustomOptions


class Erf(CustomOptions):

    def __init__(self) -> None:
        super().__init__("FlexErf", bytearray([
            3, 69, 114, 102,  # b'<size>Erf'
            0, 18, 18,  # vec metadata
            3, 69, 114, 102, 26, 0, 42, 7, 10,  # vec[0]: b'<size>Erf<metadata>'
            1, 84, 18, 2, 48, 1, 50, 0, 0,  # vec[1]: attr 'T'
            2, 25, 21, 20, 20, 4, 40, 1  # vec metadata
        ]))
