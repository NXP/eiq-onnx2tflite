#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""TensorConverter

Module contains high level functions to convert ONNX tensors to TFLite.
"""

from onnx2tflite.src import logger
from onnx2tflite.src.converter.builder import model_builder
from onnx2tflite.src.onnx_parser import onnx_model, onnx_tensor
from onnx2tflite.src.tflite_generator import tflite_model


class TensorConverter:
    """This class provides methods to convert ONNX tensors to TFLite and create them
    using the provided 'ModelBuilder'.
    """

    _builder: model_builder.ModelBuilder

    def __init__(self, builder: model_builder.ModelBuilder) -> None:
        self._builder = builder

    def convert_internal_tensors(self, o_tensors: onnx_model.RepeatedValueInfoProto):
        """Create 'tensor' tables in the 'tensors' vector of the subGraph for oTensors.
        The 'o_tensors' do NOT contain data. They should be the inputs and outputs of
        operators in the graph. 
        Designed for the 'value_info' field in ONNX 'Graph'.
        """
        for o_tensor in o_tensors:
            if o_tensor.type.tensor_type is None:
                logger.e(logger.Code.UNSUPPORTED_ONNX_TYPE,
                         "ONNX: Only type 'tensor_type' is supported for ValueInfo yet!")

            if self._builder.tensor_exists(o_tensor.name):
                # Tensor was already created using a different function
                continue

            buffer = self._builder.build_empty_buffer()
            self._builder.build_empty_tensor(o_tensor, buffer)

    def convert_constant_tensors(self, o_tensors: onnx_tensor.RepeatedTensorProto):
        """Create 'tensor' and 'buffer' tables for the ONNX 'oTensors'.
        The 'oTensors' should have data in them. 
        Designed for the 'initializer' field of the ONNX 'Graph'.
        """
        for o_tensor in o_tensors:
            buffer = self._builder.build_buffer(o_tensor)
            self._builder.build_constant_tensor(o_tensor, buffer)

    def convert_output_tensors(self, o_outputs: onnx_model.RepeatedValueInfoProto):
        """Create 'tensor' tables in the 'tensors' vector of the subGraph for the 'oOutputs'.
        Also create empty buffers in the 'buffers' vector of the model. 
        SHOULD be called before any other tensor building function!
        Designed for the 'output' field of the ONNX 'Graph'.
        """
        if self._builder.buffers_size() != 0:
            logger.d("'Builder.buildOutputTensors()' should be called before any other Tensor building function!")

        outputs = tflite_model.SubGraphOutputs()
        output_map = {}

        for o_output in o_outputs:
            if o_output.type.tensor_type is None:
                logger.e(logger.Code.UNSUPPORTED_ONNX_TYPE,
                         "ONNX: Only type 'tensor_type' is supported for Outputs yet!")

            if o_output.name in output_map:
                tensor = output_map[o_output.name]

            else:
                buffer = self._builder.build_empty_buffer()
                tensor = self._builder.build_empty_tensor(o_output, buffer)
                output_map[tensor.name] = tensor

            outputs.tmp_outputs.append(tensor)

        self._builder.get_sub_graph().outputs = outputs

    def convert_input_tensors(self, o_inputs: onnx_model.RepeatedValueInfoProto):
        """Create 'tensor' tables in the 'tensors' vector of the subGraph for the 'o_inputs'.
        Also create empty buffers in the 'buffers' vector of the model.
        """
        inputs = tflite_model.SubGraphInputs()
        input_map = {}

        for o_input in o_inputs:
            if o_input.type.tensor_type is None:
                logger.e(logger.Code.UNSUPPORTED_ONNX_TYPE,
                         "ONNX: Only type 'tensor_type' is supported for Inputs yet!")

            if o_input.name in input_map:
                tensor = input_map[o_input.name]

            else:
                buffer = self._builder.build_empty_buffer()
                tensor = self._builder.build_empty_tensor(o_input, buffer)
                input_map[tensor.name] = tensor

            inputs.tmp_inputs.append(tensor)

        self._builder.get_sub_graph().inputs = inputs
