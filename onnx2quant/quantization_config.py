#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnxruntime.quantization import CalibrationDataReader

from onnx2tflite.src import logger

PER_CHANNEL_DEFAULT = False


class QuantizationConfig:

    def __init__(self, calibration_data_reader: CalibrationDataReader, args: dict | None = None):
        """Quantization configuration passed through command line arguments or gathered during
        the quantization process.

        :param calibration_data_reader: Calibration dataset reader. This must not be None.
        :param args: Optional dictionary with quantization arguments. Unknown arguments are ignored.
        """
        self.symbolic_dimensions_mapping: dict[str, int] | None = None
        self.input_shapes_mapping: dict[str, tuple] | None = None
        self.allow_opset_10_and_lower: bool = False
        self.calibration_data_reader = calibration_data_reader
        self.per_channel: bool = PER_CHANNEL_DEFAULT
        self.generate_artifacts_after_failed_shape_inference: bool = True

        # Preprocessing flags.
        self.replace_div_with_mul: bool = True
        self.replace_constant_with_static_tensor: bool = True

        if args is not None:
            for key, value in args.items():
                if key in self.__dict__:
                    setattr(self, key, value)

        logger.internal_assert(self.calibration_data_reader is not None, "Calibration data reader not provided!")

    def __repr__(self):
        attrs = []
        for attr in self.__dict__:
            attrs.append(f"{attr}={getattr(self, attr)}")

        return "ConversionConfig[" + ", ".join(attrs) + "]"
