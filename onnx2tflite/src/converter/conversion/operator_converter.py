#
# Copyright 2023 Martin Pavella
# Copyright 2023-2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

"""OperatorConverter

Module contains high level functions to convert ONNX operators to TFLite.
"""

import numpy as np

from onnx2quant.qdq_quantization import RecognizedQDQOps
from onnx2tflite.src import conversion_context, logger
from onnx2tflite.src.converter import node_converters
from onnx2tflite.src.converter.conversion.translator import tf_lite_type_to_numpy
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tensor_formatting import TensorFormat
from onnx2tflite.src.tflite_generator import tflite_model


class OperatorConverter:
    """This class provides methods to convert ONNX operators to TFLite and
    create them using the provided 'ModelBuilder'.
    """

    _context: conversion_context.ConversionContext
    _recognized_qdq_ops: RecognizedQDQOps | None

    _op_type_to_node_converter_constructor_map: dict[str, type[NodeConverter]] = {
        "Abs": node_converters.AbsConverter,
        "Add": node_converters.AddConverter,
        "And": node_converters.AndConverter,
        "ArgMax": node_converters.ArgMaxConverter,
        "ArgMin": node_converters.ArgMinConverter,
        "AveragePool": node_converters.AveragePoolConverter,
        "BatchNormalization": node_converters.BatchNormalizationConverter,
        "Cast": node_converters.CastConverter,
        "Ceil": node_converters.CeilConverter,
        "Clip": node_converters.ClipConverter,
        "Concat": node_converters.ConcatConverter,
        "Constant": node_converters.ConstantConverter,
        "ConstantOfShape": node_converters.ConstantOfShapeConverter,
        "Conv": node_converters.ConvConverter,
        "ConvTranspose": node_converters.ConvTransposeConverter,
        "Cos": node_converters.CosConverter,
        "CumSum": node_converters.CumSumConverter,
        "DepthToSpace": node_converters.DepthToSpaceConverter,
        "DequantizeLinear": node_converters.DequantizeLinearConverter,
        "Div": node_converters.DivConverter,
        "Dropout": node_converters.DropoutConverter,
        "Einsum": node_converters.EinsumConverter,
        "Elu": node_converters.EluConverter,
        "Equal": node_converters.EqualConverter,
        "Erf": node_converters.ErfConverter,
        "Exp": node_converters.ExpConverter,
        "Expand": node_converters.ExpandConverter,
        "Flatten": node_converters.FlattenConverter,
        "Floor": node_converters.FloorConverter,
        "Gather": node_converters.GatherConverter,
        "GatherND": node_converters.GatherNDConverter,
        "Gelu": node_converters.GeluConverter,
        "Gemm": node_converters.GemmConverter,
        "GlobalAveragePool": node_converters.GlobalAveragePoolConverter,
        "GlobalMaxPool": node_converters.GlobalMaxPoolConverter,
        "Greater": node_converters.GreaterConverter,
        "GreaterOrEqual": node_converters.GreaterOrEqualConverter,
        "HardSigmoid": node_converters.HardSigmoidConverter,
        "HardSwish": node_converters.HardSwishConverter,
        "Identity": node_converters.IdentityConverter,
        "InstanceNormalization": node_converters.InstanceNormalizationConverter,
        "LRN": node_converters.LRNConverter,
        "LSTM": node_converters.LSTMConverter,
        "LayerNormalization": node_converters.LayerNormalizationConverter,
        "LeakyRelu": node_converters.LeakyReluConverter,
        "Less": node_converters.LessConverter,
        "LessOrEqual": node_converters.LessOrEqualConverter,
        "Log": node_converters.LogConverter,
        "LogSoftmax": node_converters.SoftmaxConverter,
        "MatMul": node_converters.MatMulConverter,
        "Max": node_converters.MaxConverter,
        "MaxPool": node_converters.MaxPoolConverter,
        "Min": node_converters.MinConverter,
        "Mod": node_converters.ModConverter,
        "Mul": node_converters.MulConverter,
        "Multinomial": node_converters.MultinomialConverter,
        "Neg": node_converters.NegConverter,
        "Not": node_converters.NotConverter,
        "OneHot": node_converters.OneHotConverter,
        "Or": node_converters.OrConverter,
        "PRelu": node_converters.PReluConverter,
        "Pad": node_converters.PadConverter,
        "Pow": node_converters.PowConverter,
        "QGemm": node_converters.QGemmConverter,
        "QLinearAdd": node_converters.QLinearAddConverter,
        "QLinearAveragePool": node_converters.QLinearAveragePoolConverter,
        "QLinearConcat": node_converters.QLinearConcatConverter,
        "QLinearConv": node_converters.QLinearConvConverter,
        "QLinearGlobalAveragePool": node_converters.QLinearGlobalAveragePoolConverter,
        "QLinearMatMul": node_converters.QLinearMatMulConverter,
        "QLinearMul": node_converters.QLinearMulConverter,
        "QLinearSoftmax": node_converters.QLinearSoftmaxConverter,
        "QuantizeLinear": node_converters.QuantizeLinearConverter,
        "QuickGelu": node_converters.QuickGeluConverter,
        "RNN": node_converters.RNNConverter,
        "Range": node_converters.RangeConverter,
        "Reciprocal": node_converters.ReciprocalConverter,
        "ReduceL2": node_converters.ReduceL2Converter,
        "ReduceMax": node_converters.ReduceMaxConverter,
        "ReduceMean": node_converters.ReduceMeanConverter,
        "ReduceMin": node_converters.ReduceMinConverter,
        "ReduceProd": node_converters.ReduceProdConverter,
        "ReduceSum": node_converters.ReduceSumConverter,
        "Relu": node_converters.ReluConverter,
        "Reshape": node_converters.ReshapeConverter,
        "Resize": node_converters.ResizeConverter,
        "ReverseSequence": node_converters.ReverseSequenceConverter,
        "Round": node_converters.RoundConverter,
        "ScatterND": node_converters.ScatterNDConverter,
        "Shape": node_converters.ShapeConverter,
        "Sigmoid": node_converters.SigmoidConverter,
        "Sign": node_converters.SignConverter,
        "Sin": node_converters.SinConverter,
        "Slice": node_converters.SliceConverter,
        "Softmax": node_converters.SoftmaxConverter,
        "SpaceToDepth": node_converters.SpaceToDepthConverter,
        "Split": node_converters.SplitConverter,
        "Sqrt": node_converters.SqrtConverter,
        "Squeeze": node_converters.SqueezeConverter,
        "Sub": node_converters.SubConverter,
        "Sum": node_converters.SumConverter,
        "Tanh": node_converters.TanhConverter,
        "Tile": node_converters.TileConverter,
        "Transpose": node_converters.TransposeConverter,
        "Unsqueeze": node_converters.UnsqueezeConverter,
        "Upsample": node_converters.UpsampleConverter,
        "Where": node_converters.WhereConverter,
        "Xor": node_converters.XorConverter
    }

    def __init__(self, context: conversion_context.ConversionContext,
                 recognized_qdq_ops: RecognizedQDQOps | None = None):
        self._context = context
        self._recognized_qdq_ops = recognized_qdq_ops

    def _convert_node(self, o_node: onnx_model.NodeProto) -> tflite_model.Operator:
        """Create a TFLite 'Operator' from the ONNX 'Node' with corresponding
        'inputs' and 'outputs'.
        """
        t_operator = tflite_model.Operator()

        # Initialize operator inputs
        t_operator.inputs = tflite_model.OperatorInputs()
        for name in o_node.inputs:
            t_operator.tmp_inputs.append(self._context.tflite_builder.tensor_for_name(name))

        # Initialize operator outputs
        t_operator.outputs = tflite_model.OperatorOutputs()
        for name in o_node.outputs:
            t_operator.tmp_outputs.append(self._context.tflite_builder.tensor_for_name(name))

        return t_operator

    def _is_part_of_qdq_cluster(self, o_node: onnx_model.NodeProto) -> bool:
        if self._recognized_qdq_ops is None:
            return False

        return o_node.unique_name in self._recognized_qdq_ops.qdq_cluster_quantization_ops

    # noinspection PyTypeChecker,SpellCheckingInspection
    def _convert_operator(self, node: onnx_model.NodeProto) -> None:
        """Convert an ONNX operator (node) to 1 or multiple TFLite operators and add them to the model."""
        t_op = self._convert_node(node)  # Operator in the TFLite model. Carries info about input and output tensors.

        # Identify ONNX operator and convert it.
        if node.op_type in ["DequantizeLinear", "QuantizeLinear"] and self._is_part_of_qdq_cluster(node):
            # The operator is removed and its input/output tensor is assigned quantization parameters.
            ops_to_add = []

        else:
            if node.op_type not in self._op_type_to_node_converter_constructor_map:
                logger.e(logger.Code.UNSUPPORTED_OPERATOR,
                         f"Conversion of ONNX operator '{node.op_type}' is not yet supported!")

            node_converter_constructor = self._op_type_to_node_converter_constructor_map[node.op_type]
            ops_to_add = node_converter_constructor(self._context).convert(node, t_op)

        # noinspection PyUnboundLocalVariable
        for op in ops_to_add:
            if op.builtin_options is not None:
                op.opcode_index = self._context.tflite_builder.op_code_index_for_op_type(
                    op.builtin_options.operator_type,
                    op.tmp_version
                )

            elif op.custom_options is not None:
                op.opcode_index = self._context.tflite_builder.op_code_index_for_op_type(
                    op.custom_options.operator_type,
                    op.tmp_version,
                    op.custom_options.custom_code
                )

            self._context.tflite_builder.check_and_append_operator(op)

    def convert_operators(self, o_nodes: onnx_model.RepeatedNodeProto) -> None:
        """Find the best way to convert all operators in the ONNX model and convert them to TFLite."""
        # A list of operators (op_type, name), that have been skipped because their output data was inferred during
        #  shape inference.
        skipped_ops: list[tuple[str, str]] = []

        def run_conversion_safe(conversion_fn) -> None:
            has_inconvertible_node = False

            for idx, node in enumerate(o_nodes):
                with logger.loggingContext(logger.NodeLoggingContext(idx)):
                    try:
                        conversion_fn(node)
                    except logger.Error:
                        has_inconvertible_node = True

            if has_inconvertible_node:
                raise logger.Error(logger.Code.CONVERSION_IMPOSSIBLE,
                                   "Some nodes in the model are not supported or cannot be converted. Please "
                                   "review the console output above for more information and possible solutions.")

        def convert_all_ops(node: onnx_model.NodeProto) -> None:
            def _tensors_are_formatless(onnx_tensors: list[str]) -> bool:
                """Return `True` if all ONNX tensors with given names are formatless."""
                tflite_tensors = [self._context.tflite_builder.tensor_for_name(t) for t in onnx_tensors]
                return all(t.tensor_format is TensorFormat.FORMATLESS for t in tflite_tensors)

            # noinspection PyShadowingNames
            def _inferred_data_is_sufficient(inferred_output_data: dict[str: np.ndarray] | None) -> bool:
                if inferred_output_data is None:
                    return False

                if any(t is None for t in inferred_output_data.values()):
                    return False

                if any(self._context.onnx_inspector.is_output_of_model(t)
                       for t in inferred_output_data):
                    return False

                return True

            if not self._context.conversion_config.dont_skip_nodes_with_known_outputs:
                # Try to get the inferred output data.
                used_outputs = self._context.onnx_inspector.get_used_outputs(node)
                inferred_data = {t: self._context.onnx_inspector.try_get_inferred_tensor_data(t) for t in used_outputs}
            else:
                used_outputs = []
                inferred_data = None

            # `FORMATLESS` tensors must always have the exact same shape and data in ONNX and TFLite. Therefore, the
            #  inferred ONNX data can be immediately used in the TFLite model.
            # For other formats, it would be complicated to guarantee correct behavior. Also, the data inference is most
            #  commonly used for `FORMATLESS` tensors.
            if _tensors_are_formatless(used_outputs) and _inferred_data_is_sufficient(inferred_data):
                # The data of all used output tensors of the current node has been inferred during shape inference.
                skipped_ops.append((node.op_type, node.name))

                # Assign the inferred data statically to the output tensors.
                for name, data in inferred_data.items():
                    t = self._context.tflite_builder.tensor_for_name(name)
                    t.tmp_buffer.data = np.asarray(data, tf_lite_type_to_numpy(t.type)).reshape(t.shape.vector)

            else:
                # Convert the node to tflite.
                self._convert_operator(node)

        def convert_qdq_cluster_nodes(node) -> None:
            if node.op_type == "QuantizeLinear" and self._is_part_of_qdq_cluster(node):
                t_op = self._convert_node(node)
                node_converters.QuantizeLinearConverter(self._context).convert_into_tensor(node, t_op)
            elif node.op_type == "DequantizeLinear" and self._is_part_of_qdq_cluster(node):
                t_op = self._convert_node(node)
                node_converters.DequantizeLinearConverter(self._context).convert_into_tensor(node, t_op)

        if self._recognized_qdq_ops is not None:
            run_conversion_safe(convert_qdq_cluster_nodes)
        run_conversion_safe(convert_all_ops)

        if len(skipped_ops) != 0:
            ops_as_str = "\n".join(f"\t{op_type}({name})" for op_type, name in skipped_ops)
            logger.i("The output data of the following nodes has been statically inferred, so they will not be present "
                     "in the output model.\n" + ops_as_str + "\n" +
                     "If you wish to prohibit this and convert all operators, run the conversion again with the flag "
                     f"{logger.Style.bold + logger.Style.cyan}--dont-skip-nodes-with-known-outputs{logger.Style.end}.")
