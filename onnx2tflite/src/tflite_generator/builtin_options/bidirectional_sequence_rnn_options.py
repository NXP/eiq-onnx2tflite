#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import flatbuffers as fb

import onnx2tflite.lib.tflite.BidirectionalSequenceRNNOptions as libBRNNOptions
from onnx2tflite.lib.tflite.ActivationFunctionType import ActivationFunctionType
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class BidirectionalSequenceRNN(meta.BuiltinOptions):
    time_major: bool  # If True, the first dimension is sequence, otherwise batch.
    fused_activation_function: ActivationFunctionType
    merge_outputs: bool
    asymmetric_quantize_inputs: bool

    def __init__(self, time_major: bool = True, merge_outputs: bool = True, asymmetric_quantize_inputs: bool = False,
                 fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE) -> None:
        super().__init__(BuiltinOptions.BidirectionalSequenceRNNOptions, BuiltinOperator.BIDIRECTIONAL_SEQUENCE_RNN)

        self.time_major = time_major
        self.fused_activation_function = fused_activation_function
        self.merge_outputs = merge_outputs
        self.asymmetric_quantize_inputs = asymmetric_quantize_inputs

    def gen_tflite(self, builder: fb.Builder):
        libBRNNOptions.Start(builder)

        libBRNNOptions.AddTimeMajor(builder, self.time_major)
        libBRNNOptions.AddFusedActivationFunction(builder, self.fused_activation_function)
        libBRNNOptions.AddMergeOutputs(builder, self.merge_outputs)
        libBRNNOptions.AddAsymmetricQuantizeInputs(builder, self.asymmetric_quantize_inputs)

        return libBRNNOptions.End(builder)
