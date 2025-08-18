#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from .arg_max_attributes import ArgMax
from .arg_min_attributes import ArgMin
from .average_pool_attributes import AveragePool
from .batch_normalization_attributes import BatchNormalization
from .cast_attributes import Cast
from .clip_attributes import Clip
from .concat_attributes import Concat
from .constant_attributes import Constant
from .constant_of_shape_attributes import ConstantOfShape
from .conv_attributes import Conv
from .conv_transpose_attributes import ConvTranspose
from .cum_sum_attributes import CumSum
from .depth_to_space_attributes import DepthToSpace
from .dequantize_linear_attributes import DequantizeLinear
from .dropout_attributes import Dropout
from .einsum_attributes import Einsum
from .elu_attributes import Elu
from .flatten_attributes import Flatten
from .gather_attributes import Gather
from .gather_nd_attributes import GatherND
from .gelu_attributes import Gelu
from .gemm_attributes import Gemm
from .hard_sigmoid_attributes import HardSigmoid
from .instance_normalization_attributes import InstanceNormalization
from .layer_normalization_attributes import LayerNormalization
from .leaky_relu_attributes import LeakyRelu
from .log_softmax_attributes import LogSoftmax
from .lrn_attributes import LRN
from .lstm_attributes import LSTM
from .mat_mul_attributes import MatMul
from .max_pool_attributes import MaxPool
from .mod_attributes import Mod
from .multinomial_attributes import Multinomial
from .one_hot_attributes import OneHot
from .pad_attributes import Pad
from .q_gemm_attributes import QGemm
from .q_linear_average_pool_attributes import QLinearAveragePool
from .q_linear_concat_attributes import QLinearConcat
from .q_linear_conv_attributes import QLinearConv
from .q_linear_global_average_pool_attributes import QLinearGlobalAveragePool
from .q_linear_softmax_attributes import QLinearSoftmax
from .quantize_linear_attributes import QuantizeLinear
from .quick_gelu_attributes import QuickGelu
from .reduce_l2_attributes import ReduceL2
from .reduce_max_attributes import ReduceMax
from .reduce_mean_attributes import ReduceMean
from .reduce_min_attributes import ReduceMin
from .reduce_prod_attributes import ReduceProd
from .reduce_sum_attributes import ReduceSum
from .relu_attributes import Relu
from .reshape_attributes import Reshape
from .resize_attributes import Resize
from .reverse_sequence_attributes import ReverseSequence
from .rnn_attributes import RNN
from .scatter_nd_attributes import ScatterND
from .shape_attributes import Shape
from .slice_attributes import Slice
from .softmax_attributes import Softmax
from .space_to_depth_attributes import SpaceToDepth
from .split_attributes import Split
from .squeeze_attributes import Squeeze
from .transpose_attributes import Transpose
from .unsqueeze_attributes import Unsqueeze
from .upsample_attributes import Upsample
from .where_attributes import Where

__all__ = [
    ArgMax, ArgMin, AveragePool, BatchNormalization, Cast, Clip, Concat, Constant, ConstantOfShape, Conv, ConvTranspose,
    CumSum, DepthToSpace, DequantizeLinear, Dropout, Einsum, Elu, Flatten, Gather, GatherND, Gelu, Gemm, HardSigmoid,
    InstanceNormalization, LayerNormalization, LeakyRelu, LogSoftmax, LRN, LSTM, MatMul, MaxPool, Mod, Multinomial,
    OneHot, Pad, QGemm, QLinearAveragePool, QLinearConcat, QLinearConv, QLinearGlobalAveragePool, QLinearSoftmax,
    QuantizeLinear, QuickGelu, ReduceL2, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, Relu, Reshape, Resize,
    ReverseSequence, RNN, ScatterND, Shape, Slice, Softmax, SpaceToDepth, Split, Squeeze, Transpose, Unsqueeze,
    Upsample, Where
]
