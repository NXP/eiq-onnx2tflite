#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import flatbuffers as fb

import onnx2tflite.lib.tflite.UnidirectionalSequenceLSTMOptions as libUSLSTMOptions
from onnx2tflite.lib.tflite.ActivationFunctionType import ActivationFunctionType
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.BuiltinOptions import BuiltinOptions
from onnx2tflite.src.tflite_generator.meta import meta


class UnidirectionalSequenceLSTM(meta.BuiltinOptions):
    fused_activation_function: ActivationFunctionType
    cell_clip: float
    proj_clip: float
    time_major: bool  # If True, the first dimension is sequence, otherwise batch.

    # Unidirectional Sequence LSTM v3+
    asymmetric_quantize_inputs: bool

    # Unidirectional Sequence LSTM v4+
    diagonal_recurrent_tensors: bool

    def __init__(self, cell_clip: float, proj_clip: float, time_major: bool = True,
                 asymmetric_quantize_inputs: bool = False, diagonal_recurrent_tensors: bool = False,
                 fused_activation_function: ActivationFunctionType = ActivationFunctionType.NONE) -> None:
        super().__init__(BuiltinOptions.UnidirectionalSequenceLSTMOptions, BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM)

        self.fused_activation_function = fused_activation_function
        self.cell_clip = cell_clip
        self.proj_clip = proj_clip
        self.time_major = time_major
        self.asymmetric_quantize_inputs = asymmetric_quantize_inputs
        self.diagonal_recurrent_tensors = diagonal_recurrent_tensors

    def gen_tflite(self, builder: fb.Builder) -> int:
        libUSLSTMOptions.Start(builder)

        libUSLSTMOptions.AddFusedActivationFunction(builder, self.fused_activation_function)
        libUSLSTMOptions.AddCellClip(builder, self.cell_clip)
        libUSLSTMOptions.AddProjClip(builder, self.proj_clip)
        libUSLSTMOptions.AddTimeMajor(builder, self.time_major)
        libUSLSTMOptions.AddAsymmetricQuantizeInputs(builder, self.asymmetric_quantize_inputs)
        libUSLSTMOptions.AddDiagonalRecurrentTensors(builder, self.diagonal_recurrent_tensors)

        return libUSLSTMOptions.End(builder)
