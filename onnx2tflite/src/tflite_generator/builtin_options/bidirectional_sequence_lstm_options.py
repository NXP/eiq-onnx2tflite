#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import flatbuffers as fb

import onnx2tflite.lib.tflite.BidirectionalSequenceLSTMOptions as libBSLSTMOptions
from onnx2tflite.lib.tflite.ActivationFunctionType import ActivationFunctionType
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class BidirectionalSequenceLSTM(meta.BuiltinOptions):
    fused_activation_function: ActivationFunctionType
    cell_clip: float
    proj_clip: float
    merge_outputs: bool

    # V2+
    time_major: bool  # If True, the first dimension is sequence, otherwise batch.

    # V3+
    asymmetric_quantize_inputs: bool

    def __init__(self, cell_clip: float, proj_clip: float, time_major: bool = True, merge_outputs: bool = True,
                 asymmetric_quantize_inputs: bool = False,
                 fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE) -> None:
        super().__init__(BuiltinOptions.BidirectionalSequenceLSTMOptions, BuiltinOperator.BIDIRECTIONAL_SEQUENCE_LSTM)

        self.fused_activation_function = fused_activation_function
        self.cell_clip = cell_clip
        self.proj_clip = proj_clip
        self.merge_outputs = merge_outputs
        self.time_major = time_major
        self.asymmetric_quantize_inputs = asymmetric_quantize_inputs

    def gen_tflite(self, builder: fb.Builder):
        libBSLSTMOptions.Start(builder)

        libBSLSTMOptions.AddFusedActivationFunction(builder, self.fused_activation_function)
        libBSLSTMOptions.AddCellClip(builder, self.cell_clip)
        libBSLSTMOptions.AddProjClip(builder, self.proj_clip)
        libBSLSTMOptions.AddMergeOutputs(builder, self.merge_outputs)
        libBSLSTMOptions.AddTimeMajor(builder, self.time_major)
        libBSLSTMOptions.AddAsymmetricQuantizeInputs(builder, self.asymmetric_quantize_inputs)

        return libBSLSTMOptions.End(builder)
