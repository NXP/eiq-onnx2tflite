#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter.builder.model_builder import ModelBuilder
from onnx2tflite.src.model_inspector import ONNXModelInspector


class ConversionContext:
    tflite_builder: ModelBuilder
    onnx_inspector: ONNXModelInspector
    conversion_config: ConversionConfig

    def __init__(self, tflite_builder: ModelBuilder, onnx_inspector: ONNXModelInspector,
                 conversion_config: ConversionConfig):
        """
        Context with data related to current conversion.

        :param tflite_builder: TFLite model builder.
        :param onnx_inspector: Inspector of converted ONNX model.
        :param conversion_config: Conversion configuration flags and metadata.
        """
        self.tflite_builder = tflite_builder
        self.onnx_inspector = onnx_inspector
        self.conversion_config = conversion_config
