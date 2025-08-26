#
# Copyright 2024-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import itertools
import os.path
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime.quantization
from onnx import TensorProto
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization.base_quantizer import QuantizationParams
from onnxruntime.quantization.operators import matmul, norm
from onnxruntime.quantization.operators.direct_q8 import QDQDirect8BitOp
from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer as BaseQDQQuantizer
from onnxruntime.quantization.qdq_quantizer import QDQTensorQuantParams
from onnxruntime.quantization.quant_utils import add_pre_process_metadata
from onnxruntime.quantization.registry import QDQRegistry

from onnx2quant.preprocessor import Preprocessor
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.src import logger
from onnx2tflite.src.logger import loggingContext
from onnx2tflite.src.model_inspector import ONNXModelInspector
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type


def _get_tensor_rank(tensor_name: str, model: onnx.ModelProto) -> int | None:
    for t in model.graph().initializer:
        if tensor_name == t.name:
            return len(t.dims)

    for t in itertools.chain(model.graph().input, model.graph().output, model.graph().value_info):
        if tensor_name == t.name:
            return len(t.type.tensor_type.shape.dim)

    return None


class RecognizedQDQOps:
    """Class represents categorized ops based on their relationship to
    QDQ clusters in the model.
    """

    # Names of 'QuantizeLinear' & 'DequantizeLinear' ops that are NOT part of QDQ cluster
    standalone_quantization_ops: list[str]

    # Names of 'QuantizeLinear' & 'DequantizeLinear' ops that are part of QDQ cluster
    qdq_cluster_quantization_ops: list[str]

    # Names of non 'QuantizeLinear' & 'DequantizeLinear' ops that are surrounded by QDQ cluster q-ops
    quantized_float_ops: list[str]

    # Names of non 'QuantizeLinear' & 'DequantizeLinear' ops that are not part of any QDQ cluster
    non_quantized_float_ops: list[str]

    def __init__(self):
        self.standalone_quantization_ops = []
        self.qdq_cluster_quantization_ops = []
        self.quantized_float_ops = []
        self.non_quantized_float_ops = []


class QDQClustersRecognizer:
    """Class for detection of QDQ clusters. It categorizes ops into groups based on their
    relevance to detected clusters.
    """

    # List of ops where we can guarantee successful QDQ cluster conversion to TFLite
    default_supported_qdq_ops = [
        "Add",
        "ArgMax",
        "ArgMin",
        "AveragePool",
        "Clip",
        "Concat",
        "Conv",
        "ConvTranspose",
        "Equal",
        "Expand",
        "Flatten",
        "Gather",
        "GatherND",
        "Gemm",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "Greater",
        "GreaterOrEqual",
        "HardSwish",
        "LeakyRelu",
        "Less",
        "LessOrEqual",
        "LogSoftmax",
        "MatMul",
        "Max",
        "MaxPool",
        "Min",
        "Mul",
        "Pad",
        "PRelu",
        "Relu",
        "ReduceMean",
        "ReduceProd",
        "ReduceSum",
        "Reshape",
        "Resize",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Squeeze",
        "Sub",
        "Sum",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
    ]

    def __init__(self, model_inspector: ONNXModelInspector, supported_qdq_ops: list[str] | None = None):
        """Create QDQClustersRecognizer instance.

        :param model_inspector: ONNXModelInspector object.
        :param supported_qdq_ops: List of op names that should be recognized. If None or empty,
            default list of ops is used instead.
        """
        self.model_inspector = model_inspector
        self.supported_qdq_ops = supported_qdq_ops or self.default_supported_qdq_ops
        self.op_specific_rules = {}

        self._register_operator_specific_rules()

    def _register_operator_specific_rules(self) -> None:
        self.op_specific_rules["Clip"] = self._validate_clip
        self.op_specific_rules["Concat"] = self._validate_concat
        self.op_specific_rules["Conv"] = self._validate_conv
        self.op_specific_rules["ConvTranspose"] = self._validate_conv_transpose
        self.op_specific_rules["MatMul"] = self._validate_mat_mul
        self.op_specific_rules["Sum"] = self._validate_sum
        self.op_specific_rules["Pad"] = self._validate_pad
        self.op_specific_rules["Resize"] = self._validate_resize

    def _validate_clip(self, node: onnx_model.NodeProto) -> bool:
        input_quantized = self.model_inspector.tensor_originates_in_single_consumer_dequantize_op(node.inputs[0])
        return (input_quantized and
                self._is_node_output_quantized(node) and
                self._surrounding_q_ops_have_same_quant_type(node))

    def _validate_concat(self, node: onnx_model.NodeProto) -> bool:
        inputs_quantized = [
            self.model_inspector.tensor_not_float(tensor_name) or
            self.model_inspector.tensor_originates_in_single_consumer_dequantize_op(tensor_name) or
            self.model_inspector.tensor_is_static(tensor_name)
            for tensor_name in node.inputs
        ]
        return (all(inputs_quantized) and
                self._surrounding_q_ops_have_same_quant_type(node) and
                self._is_node_output_quantized(node))

    def _validate_conv(self, node: onnx_model.NodeProto) -> bool:
        # 3D conv doesn't have a quantized TFLite variant. Don't quantize in that case.
        if self.model_inspector.get_tensor_rank_safe(node.inputs[1]) > 4:
            return False

        # We can deal with Conv that has different quantization types, for example
        # UINT8 weights and INT8 activations, so skip this check.

        return (self._is_node_input_quantized(node) and
                self._is_node_output_quantized(node) and
                self._at_least_one_io_tensor_float(node) and
                self._inputs_are_not_outputs_of_model(node))

    def _validate_conv_transpose(self, node: onnx_model.NodeProto) -> bool:
        if not self._surrounding_q_ops_have_same_quant_type(node):
            logger.w(f"QDQ quantized ONNX operator '{node.op_type}' has it's inputs quantized with different "
                     f"types, for example UINT8 activations and INT8 weights. Such operator will not be "
                     f"converted into quantized TFLite variant. Make sure that model uses same quantization "
                     f"types or use internal quantizer to overcome this.")

        return self._node_io_tensors_quantized(node)

    def _validate_mat_mul(self, node: onnx_model.NodeProto) -> bool:
        # We can deal with MatMul that has different quantization types, for example
        # UINT8 weights and INT8 activations, so skip this check.

        return (self._is_node_input_quantized(node) and
                self._is_node_output_quantized(node) and
                self._at_least_one_io_tensor_float(node) and
                self._inputs_are_not_outputs_of_model(node))

    def _validate_sum(self, node: onnx_model.NodeProto) -> bool:
        # Sum with more than 2 inputs is converted into AddN that
        # doesn't support quantized input
        return len(node.inputs) < 3 and self._node_io_tensors_quantized(node)

    def _validate_pad(self, node: onnx_model.NodeProto) -> bool:
        # First input and output quantized
        # There might be float "constant" tensor (idx=2) that is not quantized
        return (self._is_node_output_quantized(node) and
                self._surrounding_q_ops_have_same_quant_type(node) and
                self.model_inspector.tensor_originates_in_single_consumer_dequantize_op(node.inputs[0]))

    def _validate_resize(self, node: onnx_model.NodeProto) -> bool:
        # First input and output quantized
        # There might be float input tensors ('roi', 'scales ') that are not quantized
        return (self._is_node_output_quantized(node) and
                self._surrounding_q_ops_have_same_quant_type(node) and
                self.model_inspector.tensor_originates_in_single_consumer_dequantize_op(node.inputs[0]))

    def _partition_quantization_ops(self) -> tuple[dict[str, onnx_model.NodeProto], dict[str, onnx_model.NodeProto]]:
        """Split model ops into q-ops ('QuantizeLinear' & 'DequantizeLinear') and others. Ops are
        returned as dictionaries with mapping from unique generated op name to 'NodeProto' object.

        :return: Tuple with dictionaries containing mapping from unique op name to 'NodeProto' object.
        """
        quantize_nodes = {}
        non_quantize_nodes = {}

        for node in self.model_inspector.get_nodes():
            if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                quantize_nodes[node.unique_name] = node
            else:
                non_quantize_nodes[node.unique_name] = node

        return quantize_nodes, non_quantize_nodes

    def _node_io_tensors_quantized(self, node: onnx_model.NodeProto) -> bool:
        """Check if node's input/output tensors are quantized or non-float.

        :param node: Analyzed node.
        :return: True if all input and outputs are considered as quantized.
        """
        return (self._is_node_input_quantized(node) and
                self._is_node_output_quantized(node) and
                self._at_least_one_io_tensor_float(node) and
                self._inputs_are_not_outputs_of_model(node) and
                self._surrounding_q_ops_have_same_quant_type(node))

    def _is_qdq_quantized_node(self, node: onnx_model.NodeProto) -> bool:
        """Check if operator can be considered as QDQ quantized. This usually means that all IO tensors are
        surrounded by q-ops but this could be operator specific.

        :param node: Analyzed node.
        :return: True if node can be considered as QDQ quantized.
        """
        return self.op_specific_rules.get(node.op_type, lambda x: self._node_io_tensors_quantized(x))(node)

    def _is_node_input_quantized(self, node: onnx_model.NodeProto) -> bool:
        inputs_quantized = [
            tensor_name == "" or  # Optional input tensor
            self.model_inspector.tensor_not_float(tensor_name) or
            self.model_inspector.tensor_originates_in_single_consumer_dequantize_op(tensor_name) or
            self.model_inspector.tensor_is_shared_dequantized_bias(tensor_name)
            for tensor_name in node.inputs]

        return all(inputs_quantized)

    def _is_node_output_quantized(self, node: onnx_model.NodeProto) -> bool:
        outputs_quantized = [
            self.model_inspector.tensor_not_float(tensor_name) or (
                    self.model_inspector.tensor_leads_to_quantize_op(tensor_name) and
                    not self.model_inspector.is_output_of_model(tensor_name))
            for tensor_name in node.outputs]

        return all(outputs_quantized)

    def _at_least_one_io_tensor_float(self, node: onnx_model.NodeProto) -> bool:
        io_tensors = list(node.outputs) + list(node.inputs)

        # Ignore optional tensors
        io_tensors = list(filter(lambda x: x != "", io_tensors))

        return any([self.model_inspector.tensor_is_float(tensor_name) for tensor_name in io_tensors])

    def _inputs_are_not_outputs_of_model(self, node: onnx_model.NodeProto) -> bool:
        return all([not self.model_inspector.is_output_of_model(tensor_name) for tensor_name in node.inputs])

    def _surrounding_q_ops_have_same_quant_type(self, node: onnx_model.NodeProto) -> bool:
        """Check whether surrounding q-ops of given node have same quantization type (INT8 or UINT8).
        INT32 type is not considered, because it is used in well-defined situations.

        :param node: Analyzed ONNX node.
        :return: True if surrounding q-ops have same quantized type. False otherwise.
        """
        preceding_ops = [self.model_inspector.get_ops_with_output_tensor(tensor_name) for tensor_name in node.inputs]
        subsequent_ops = [self.model_inspector.get_ops_with_input_tensor(tensor_name) for tensor_name in node.outputs]

        surrounding_ops = list(itertools.chain.from_iterable(preceding_ops + subsequent_ops))
        surrounding_q_ops = [op for op in surrounding_ops if op.op_type in ["QuantizeLinear", "DequantizeLinear"]]

        # Get 'zero_point' tensor types
        surrounding_q_op_types = set([self.model_inspector.get_tensor_type(op.inputs[2]) for op in surrounding_q_ops])
        surrounding_q_op_types.discard(TensorProto.INT32)

        # Single (INT8 or UINT8) or zero types
        return len(surrounding_q_op_types) <= 1

    def get_qdq_cluster_quantization_nodes(self, node) -> list[str]:
        """Collects quantization nodes of QDQ cluster. The QDQ cluster consist of preceding 'DequantizeLinear'
        ops and following 'QuantizeLinear' ops.

        :param node: Analyzed node.
        :return: Unique names of cluster q-op nodes.
        """
        preceding_nodes: list[onnx_model.NodeProto] = []
        for i in node.inputs:
            preceding_nodes.extend(self.model_inspector.get_ops_with_output_tensor(i))

        following_nodes: list[onnx_model.NodeProto] = []
        for o in node.outputs:
            following_nodes.extend(self.model_inspector.get_ops_with_input_tensor(o))

        return ([node.unique_name for node in preceding_nodes if node.op_type == "DequantizeLinear"] +
                [node.unique_name for node in following_nodes if node.op_type == "QuantizeLinear"])

    def recognize_ops(self) -> RecognizedQDQOps | None:
        """Detect all QDQ clusters and categorize model's ops based on relationship to any
        of the clusters.

        :return: RecognizedQDQOps object or None if there isn't any q-op in the model.
        """
        if not self.model_inspector.contains_quantization_nodes():
            return None

        if self.model_inspector.has_int8_and_uint8_q_ops():
            logger.w("Model contains (De)QuantizeLinear nodes with both UINT8 and INT8 types. Model was probably "
                     "quantized twice or different types were used for quantization of different ops. Conversion "
                     "of such models can produce non-optimal results (unnecessary 'Quantize' ops in the model). "
                     "Quantize activations and weights with INT8 to get better results.")

        recognized_ops = RecognizedQDQOps()

        quantization_nodes, analyzed_float_nodes = self._partition_quantization_ops()

        for node_name, node_proto in analyzed_float_nodes.items():
            if self._is_qdq_quantized_node(node_proto) and node_proto.op_type in self.supported_qdq_ops:
                recognized_ops.quantized_float_ops.append(node_name)
                recognized_ops.qdq_cluster_quantization_ops.extend(self.get_qdq_cluster_quantization_nodes(node_proto))
            else:
                recognized_ops.non_quantized_float_ops.append(node_name)

        for quantization_node in quantization_nodes:
            if quantization_node not in recognized_ops.qdq_cluster_quantization_ops:
                recognized_ops.standalone_quantization_ops.append(quantization_node)

        qdq_clusters_count = len(recognized_ops.qdq_cluster_quantization_ops)
        if qdq_clusters_count > 0:
            logger.w(f"Some ops in model will be converted to tensor-based quantization representation. "
                     f"Number of ops: {qdq_clusters_count}.")

        return recognized_ops


@dataclass
class InputSpec:
    shape: list[int]
    type: np.dtype | type = None
    min: float = None
    max: float = None
    custom_data_generator: Callable[[list[int]], np.ndarray] = None


class RandomDataCalibrationDataReader(CalibrationDataReader):

    # noinspection PyMethodMayBeStatic
    def _generate_random_data(self, rng: np.random.Generator, shape: list[int], min_: float, max_: float) -> np.ndarray:
        range_ = max_ - min_
        return (rng.random(shape, "float32") * range_ + min_).astype("float32")

    def __init__(self, inputs: dict[str, InputSpec] = None, num_samples=3, seed=42):
        self.data = []
        self.data_iter = None

        rng = np.random.default_rng(seed)

        for _ in range(num_samples):
            sample = {}

            for input_name, input_metadata in inputs.items():
                input_shape = input_metadata.shape

                if input_metadata.custom_data_generator is not None:
                    sample[input_name] = input_metadata.custom_data_generator(input_shape)
                else:
                    input_type = input_metadata.type

                    if input_type == np.float32:
                        sample[input_name] = self._generate_random_data(
                            rng,
                            input_shape,
                            input_metadata.min or 0.0,
                            input_metadata.max or 1.0
                        ).astype("float32")

                    elif input_type == np.int64:
                        sample[input_name] = self._generate_random_data(
                            rng,
                            input_shape,
                            input_metadata.min or 0.0,
                            input_metadata.max or 1000.0
                        ).astype("int64")

                    elif input_type == np.bool_:
                        sample[input_name] = rng.random(input_shape, dtype=np.float32) < 0.5

                    else:
                        raise NotImplementedError

            self.data.append(sample)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def from_onnx_model(onnx_model: onnx_model.ModelProto, num_samples=3, seed=42):
        """Create RandomDataCalibrationDataReader based passed on ONNX model's inputs.

        :param onnx_model: ONNX model w
        :param num_samples: Number of input samples.
        :param seed: Numpy seed.
        :return: RandomDataCalibrationDataReader instance.
        """
        initializers = list(map(lambda x: x.name, [i for i in onnx_model.graph.initializer]))
        inputs = {}
        for inp in onnx_model.graph.input:
            if inp.name in initializers:
                continue
            shape = str(inp.type.tensor_type.shape.dim)
            parsed_shape = [int(s) for s in shape.split() if s.isdigit()]
            numpy_type = to_numpy_type(inp.type.tensor_type.elem_type)
            inputs[inp.name] = InputSpec(parsed_shape, numpy_type)

        return RandomDataCalibrationDataReader(inputs, num_samples=num_samples, seed=seed)

    def to_config(self) -> QuantizationConfig:
        return QuantizationConfig(self)

    def get_next(self) -> dict:
        if not self.data_iter:
            self.data_iter = iter(self.data)

        return next(self.data_iter, None)


class QDQOperator(QDQOperatorBase):
    quantizer: BaseQDQQuantizer

    def _can_quantize_tensor(self, tensor_name) -> bool:
        if not self.quantizer.is_tensor_quantized(tensor_name):
            # Tensor was not already set for quantization
            return True
        quantize_info = self.quantizer.tensors_to_quantize[tensor_name]

        if quantize_info is not None and not quantize_info.is_shared:
            return True

        return False

    def get_model_opset(self) -> int:
        for opset in self.quantizer.model.opset_import():
            if not hasattr(opset, "domain") or opset.domain == "":
                return opset.version

        logger.e(logger.Code.INVALID_ONNX_MODEL, "Model doesn't specify default opset version.")

    def _per_channel_allowed(self) -> bool:
        return self.quantizer.is_per_channel()


class QDQClip(QDQOperator):
    def quantize(self) -> None:
        node = self.node
        assert node.op_type == "Clip"

        if not self.quantizer.is_tensor_quantized(node.input[0]):
            self.quantizer.quantize_activation_tensor(node.input[0])

        # Quantize min/max tensors if they are dynamic and defined
        if len(node.input) > 1 and node.input[1] != "":
            min_is_quantized = self.quantizer.is_tensor_quantized(node.input[1])
            min_is_initializer = self.quantizer.is_input_a_initializer(node.input[1])
            if not min_is_quantized and not min_is_initializer:
                self.quantizer.quantize_output_same_as_input(node.input[1], node.input[0], node.name)

        if len(node.input) > 2 and node.input[2] != "":
            max_is_quantized = self.quantizer.is_tensor_quantized(node.input[2])
            max_is_initializer = self.quantizer.is_input_a_initializer(node.input[2])
            if not max_is_quantized and not max_is_initializer:
                self.quantizer.quantize_output_same_as_input(node.input[2], node.input[0], node.name)

        if not self.disable_qdq_for_node_output:
            for output in node.output:
                self.quantizer.quantize_activation_tensor(output)


class QDQConcat(QDQOperator):
    def quantize(self) -> None:
        node = self.node
        assert node.op_type == "Concat"

        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_activation_tensor(node.output[0])

        for idx in range(len(node.input)):
            if self.quantizer.is_input_a_initializer(node.input[idx]):
                # ORT QDQQuantizer doesn't allow q-param sharing for initializers. TFLite requires all
                # input/output tensors of Concatenation operator to be the same. For that reason, we
                # skip quantization of constant inputs (initializers) here and quantize them in
                # 'convert_concat.py'.
                pass
            elif not self.quantizer.is_tensor_quantized(node.input[idx]):
                # Intentionally shared q-params from output
                self.quantizer.quantize_output_same_as_input(node.input[idx], node.output[0], node.name)


class QDQMatMul(QDQOperator):
    def quantize(self) -> None:
        node = self.node
        assert node.op_type == "MatMul"

        # Determine if the `MatMul` should be quantized per-channel.
        allow_per_channel = False
        old_per_channel_value = self.quantizer.per_channel

        static_weights = self.quantizer.is_input_a_initializer(node.input[1])
        weights_2d = _get_tensor_rank(node.input[1], self.quantizer.model) == 2
        if static_weights and weights_2d and self._per_channel_allowed():
            # The weight can be quantized per-channel.

            if self.get_model_opset() >= 13:
                # Quantize per-channel.
                allow_per_channel = True

            else:
                # Cannot quantize per-channel due to too low opset version.
                logger.w("Couldn't quantize `MatMul` per channel because the model has opset "
                         f"{self.get_model_opset()}, and 13 is the minimum required.")
                allow_per_channel = False

        # Use existing code to quantize the `MatMul`.
        self.quantizer.per_channel = allow_per_channel
        matmul.QDQMatMul(self.quantizer, self.node).quantize()
        self.quantizer.per_channel = old_per_channel_value


class QDQNormalization(QDQOperator):
    def quantize(self) -> None:
        assert self.node.op_type in ["InstanceNormalization", "LayerNormalization"]

        per_channel = self.quantizer.per_channel
        if per_channel:
            logger.w(f"`{self.node.op_type}` will not be per-channel quantized, because it then couldn't be converted "
                     "to TFLite.")

        # Temporarily disable per-channel quantization, because TFLite doesn't support it for these operators.
        self.quantizer.per_channel = False
        norm.QDQNormalization(self.quantizer, self.node).quantize()
        self.quantizer.per_channel = per_channel


class QDQLogSoftmax(QDQOperator):
    def quantize(self) -> None:
        assert self.node.op_type == "LogSoftmax"

        super().quantize()
        # Spec: github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/activations.cc#L638
        if self.quantizer.activation_qType == TensorProto.INT8:
            scale = np.array(16.0 / 256.0, dtype=np.float32)
            zp = np.array(127, dtype=np.int8)
            quant_type = TensorProto.INT8
        else:  # UINT8
            scale = np.array(16.0 / 256.0, dtype=np.float32)
            zp = np.array(255, dtype=np.uint8)
            quant_type = TensorProto.UINT8

        # noinspection PyTypeChecker
        q_params = QuantizationParams(zero_point=zp, scale=scale, quant_type=quant_type)
        self.quantizer.quantization_params[self.node.output[0]] = QDQTensorQuantParams(q_params, None, None)


class QDQMinMax(QDQOperator):
    def quantize(self) -> None:
        assert self.node.op_type in ["Max", "Min"]

        if len(self.node.input) == 1:
            # Op is removed during conversion. Do nothing.
            return

        super().quantize()


class QDQReduceProd(QDQOperator):
    def quantize(self) -> None:
        assert self.node.op_type == "ReduceProd"

        # Only INT8 supported in TFLite
        if self.quantizer.activation_qType == TensorProto.INT8:
            super().quantize()


class QDQScatterND(QDQOperator):
    def quantize(self) -> None:
        node = self.node
        assert node.op_type == "ScatterND"

        if not self.quantizer.is_tensor_quantized(node.input[0]):
            self.quantizer.quantize_activation_tensor(node.input[0])

        self.quantizer.quantize_output_same_as_input(node.output[0], node.input[0], node.name)

        updates_is_quantized = self.quantizer.is_tensor_quantized(node.input[2])
        updates_is_initializer = self.quantizer.is_input_a_initializer(node.input[2])
        if not updates_is_quantized and not updates_is_initializer:
            self.quantizer.quantize_output_same_as_input(node.input[2], node.input[0], node.name)
        else:
            # Re-quantized in ScatterND conversion -> q-params must match
            self.quantizer.quantize_activation_tensor(node.input[2])


class QDQSoftmax(QDQOperator):
    def quantize(self) -> None:
        assert self.node.op_type == "Softmax"

        super().quantize()
        # https://github.com/tensorflow/tensorflow/blob/7b8461c255ba25a04801eff85e297b2ec730b1c9/tensorflow/lite/kernels/activations.cc#L548-L556
        if self.quantizer.activation_qType == TensorProto.INT8:
            scale = np.array(1.0 / 256.0, dtype=np.float32)
            zp = np.array(-128, dtype=np.int8)
            quant_type = TensorProto.INT8
        else:  # UINT8
            scale = np.array(1.0 / 256.0, dtype=np.float32)
            zp = np.array(0, dtype=np.uint8)
            quant_type = TensorProto.UINT8

        # noinspection PyTypeChecker
        q_params = QuantizationParams(zero_point=zp, scale=scale, quant_type=quant_type)
        self.quantizer.quantization_params[self.node.output[0]] = QDQTensorQuantParams(q_params, None, None)


class QDQTanh(QDQOperator):
    def quantize(self) -> None:
        assert self.node.op_type == "Tanh"
        assert self.quantizer.activation_qType == TensorProto.INT8, "Quantization of Tanh supported only for INT8"

        super().quantize()

        zp = np.array(0, dtype=np.int8)
        scale = np.array(1.0 / 128.0, dtype=np.float32)

        # noinspection PyTypeChecker
        q_params = QuantizationParams(zero_point=zp, scale=scale, quant_type=TensorProto.INT8)
        self.quantizer.quantization_params[self.node.output[0]] = QDQTensorQuantParams(q_params, None, None)


class QDQPad(QDQOperator):
    def quantize(self) -> None:
        node = self.node
        assert node.op_type == "Pad"

        if not self.quantizer.is_tensor_quantized(node.input[0]):
            self.quantizer.quantize_activation_tensor(node.input[0])

        if len(node.input) > 2:
            if self.quantizer.is_input_a_initializer(node.input[2]):
                # Covered in 'convert_pad.py'
                pass
            elif not self.quantizer.is_tensor_quantized(node.input[2]):
                self.quantizer.quantize_output_same_as_input(node.input[2], node.input[0], node.name)

        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_output_same_as_input(self.node.output[0], self.node.input[0], self.node.name)


class QDQSigmoid(QDQOperator):
    def quantize(self) -> None:
        assert self.node.op_type == "Sigmoid"

        super().quantize()
        if self.quantizer.activation_qType == TensorProto.INT8:
            scale = np.array(1.0 / 256.0, dtype=np.float32)
            zp = np.array(-128, dtype=np.int8)
            quant_type = TensorProto.INT8
        else:  # UINT8
            scale = np.array(1.0 / 256.0, dtype=np.float32)
            zp = np.array(0, dtype=np.uint8)
            quant_type = TensorProto.UINT8

        # noinspection PyTypeChecker
        q_params = QuantizationParams(zero_point=zp, scale=scale, quant_type=quant_type)
        self.quantizer.quantization_params[self.node.output[0]] = QDQTensorQuantParams(q_params, None, None)


class QDQWhere(QDQOperator):
    def quantize(self) -> None:
        node = self.node
        assert node.op_type == "Where"

        if self.quantizer.force_quantize_no_input_check:
            if not self.disable_qdq_for_node_output:
                self.quantizer.quantize_activation_tensor(node.output[0])

            # Try to share q-params from output to input tensors. Initializer inputs are quantized as
            # normal tensors and re-quantized in convert function.
            input_1_quantized = self.quantizer.is_tensor_quantized(node.input[1])
            input_1_initializer = self.quantizer.is_input_a_initializer(node.input[1])

            if not input_1_quantized and not input_1_initializer:
                self.quantizer.quantize_output_same_as_input(node.input[1], node.output[0], node.name)
            elif not input_1_quantized:
                # Quantize now without sharing and requantize in convert function
                self.quantizer.quantize_activation_tensor(node.input[1])
            else:
                # Tensor already set for quantization, but we can't replace it
                pass

            input_2_quantized = self.quantizer.is_tensor_quantized(node.input[2])
            input_2_initializer = self.quantizer.is_input_a_initializer(node.input[2])

            if not input_2_quantized and not input_2_initializer:
                self.quantizer.quantize_output_same_as_input(node.input[2], node.output[0], node.name)
            elif not input_2_quantized:
                # Quantize now without sharing and requantize in convert function
                self.quantizer.quantize_activation_tensor(node.input[2])
            else:
                # Tensor already set for quantization, but we can't replace it
                pass


class QDQQuantizer:
    # Ops which we want to be surrounded by QDQ cluster during quantization
    default_op_types_to_quantize = [
        "Add",
        "ArgMax",
        "ArgMin",
        "AveragePool",
        # 'BatchNormalization',  # Represented by multiple operators. Quantization can introduce large errors.
        # "Cast",  # Not supported by the nature of the operator
        "Clip",
        "Concat",
        # "Constant",  # Converted directly to tensor
        # "ConstantOfShape",  # Converted directly to tensor
        "Conv",
        "ConvTranspose",
        # "Cos",  # TFLite supports only float https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/elementwise.cc#L512
        # "CumSum",  # TFLite doesn't support quantized version of this operator, https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/cumsum.cc#L39
        # "DepthToSpace",
        # "Div",  # TFLite supports only quantized uint8 https://github.com/tensorflow/tensorflow/issues/42882
        # "Dropout",  # Not supported by the nature of the operator
        # "Einsum",  # Represented as Flex Operator
        # "Elu",
        "Equal",
        # "Erf",  # TFLite doesn't support quantized version of this operator
        "Expand",
        # "Exp",  # TFLite support this operator from 2.13
        "Flatten",
        "Gather",
        "GatherND",
        # "Gelu",
        "Gemm",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "Greater",
        "GreaterOrEqual",
        # "HardSigmoid",  # Represented by multiple operators. Quantization can introduce large errors.
        "HardSwish",
        # "Identity",  # Represented by single tensor in TFLite.
        # "InstanceNormalization",  # Represented by multiple operators. Quantization can introduce large errors.
        # "LayerNormalization",  # Represented by multiple operators. Quantization can introduce large errors.
        "LeakyRelu",
        "Less",
        "LessOrEqual",
        # "Log",  # TFLite doesn't support quantized version of this operator
        "LogSoftmax",
        # "LRN",  # TFLite doesn't support quantized version of this operator
        # "LSTM",
        "MatMul",
        "Max",
        "MaxPool",
        "Min",
        # "Mod",  # TFLite doesn't support quantized version of this operator
        "Mul",
        # "Neg",  # TFLite doesn't support quantized version of this operator
        # "Not",  # TFLite doesn't support quantized version of this operator, only boolean supported https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/elementwise.cc#L556
        # "OneHot",
        "Pad",
        # "Pow",  # TFLite doesn't support quantized version of this operator
        "PRelu",
        # "QuickGelu",  # Represented by multiple operators. Quantization can introduce large errors.
        # "Range",  # TFLite doesn't support quantized version of this operator
        # "Reciprocal",  # TFLite Div supports only quantized uint8 https://github.com/tensorflow/tensorflow/issues/42882
        # "ReduceL2",  # Represented by multiple operators. Quantization can introduce large errors.
        # "ReduceMax",  # TFLite doesn't support quantized version of this operator
        "ReduceMean",
        "ReduceProd",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        # "ReverseSequence",
        # "RNN",
        "ScatterND",
        # "Shape",  # Shape is always removed during preprocessing
        "Sigmoid",
        # "Sin",  # TFLite doesn't support quantized version of this operator, only float is supported https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/lite/kernels/elementwise.cc#L504
        "Slice",
        "Softmax",
        # "SpaceToDepth",
        "Split",
        # "Sqrt",  # TFLite doesn't support quantized version of this operator
        "Squeeze",
        "Sub",
        "Sum",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
        # "Upsample",  # `Upsample` was deprecated in opset 10. This quantizer requires at least 11.
        "Where",
    ]

    def __init__(self, op_types_to_quantize: list[str] | None = None):
        self.op_types_to_quantize = op_types_to_quantize or self.default_op_types_to_quantize
        self._register_custom_qdq_quantizers()

    # noinspection PyMethodMayBeStatic
    def _register_custom_qdq_quantizers(self) -> None:
        QDQRegistry["ArgMax"] = QDQOperatorBase
        QDQRegistry["ArgMin"] = QDQOperatorBase
        QDQRegistry["Clip"] = QDQClip
        QDQRegistry["Concat"] = QDQConcat
        QDQRegistry["Expand"] = QDQDirect8BitOp
        QDQRegistry["Flatten"] = QDQDirect8BitOp
        QDQRegistry["GatherND"] = QDQDirect8BitOp
        QDQRegistry["GlobalAveragePool"] = QDQDirect8BitOp
        QDQRegistry["GlobalMaxPool"] = QDQDirect8BitOp
        QDQRegistry["Greater"] = QDQOperatorBase
        QDQRegistry["GreaterOrEqual"] = QDQOperatorBase
        QDQRegistry["HardSwish"] = QDQOperatorBase
        QDQRegistry["InstanceNormalization"] = QDQNormalization
        QDQRegistry["LayerNormalization"] = QDQNormalization
        QDQRegistry["Less"] = QDQOperatorBase
        QDQRegistry["LessOrEqual"] = QDQOperatorBase
        QDQRegistry["LogSoftmax"] = QDQLogSoftmax
        QDQRegistry["MatMul"] = QDQMatMul
        QDQRegistry["Max"] = QDQMinMax
        QDQRegistry["Min"] = QDQMinMax
        QDQRegistry["Pad"] = QDQPad
        QDQRegistry["ReduceMean"] = QDQOperatorBase
        QDQRegistry["ReduceProd"] = QDQReduceProd
        QDQRegistry["ReduceSum"] = QDQOperatorBase
        QDQRegistry["Relu"] = QDQOperatorBase
        QDQRegistry["Resize"] = QDQDirect8BitOp
        QDQRegistry["ScatterND"] = QDQScatterND
        QDQRegistry["Sigmoid"] = QDQSigmoid
        QDQRegistry["Softmax"] = QDQSoftmax
        QDQRegistry["Tanh"] = QDQTanh
        QDQRegistry["Tile"] = QDQDirect8BitOp
        QDQRegistry["Where"] = QDQWhere

    # noinspection PyUnresolvedReferences
    def _get_model_opset(self, model_proto: onnx_model.ModelProto) -> int:
        for opset in model_proto.opset_import:
            if not hasattr(opset, "domain") or opset.domain == "":
                return opset.version

        logger.e(logger.Code.INVALID_ONNX_MODEL, "Model doesn't specify default opset version.")

    # noinspection PyPep8Naming,PyMethodMayBeStatic
    def _model_has_QDQ_nodes(self, model) -> bool:  # noqa: N802
        """Detect if model already has QuantizeLinear or DequantizeLinear ops.
        """
        return any(
            node.op_type == "QuantizeLinear" or node.op_type == "DequantizeLinear" for node in model.graph.node
        )

    def quantize_model(self, model_proto: onnx_model.ModelProto, quantization_config: QuantizationConfig,
                       save_model=False, saved_model_name="quantized_model.onnx") -> onnx.ModelProto:
        if self._model_has_QDQ_nodes(model_proto):
            logger.e(logger.Code.INVALID_ONNX_MODEL,
                     "Model is already quantized. Quantization of such model can lead to infinite loop. "
                     "Skipping.")

        with (tempfile.TemporaryDirectory(prefix="onnx2tflite_") as tmp_dir,
              loggingContext(logger.BasicLoggingContext.QDQ_QUANTIZER)):
            input_model_path = os.path.join(tmp_dir, "input_model.onnx")
            if save_model:
                output_model_path = saved_model_name
            else:
                output_model_path = os.path.join(tmp_dir, "quantized_model.onnx")

            inferred_model_proto = ModelShapeInference.infer_shapes(
                model_proto,
                symbolic_dimensions_mapping=quantization_config.symbolic_dimensions_mapping,
                input_shapes_mapping=quantization_config.input_shapes_mapping,
                generate_artifacts_after_failed_shape_inference=quantization_config.generate_artifacts_after_failed_shape_inference
            )

            Preprocessor(inferred_model_proto, quantization_config).preprocess()

            if self._get_model_opset(inferred_model_proto) < 11 and not quantization_config.allow_opset_10_and_lower:
                logger.e(logger.Code.INVALID_ONNX_MODEL,
                         "Quantization of models with opset version smaller than 11 can produce "
                         "invalid models. This applies especially to models with operators: Clip, "
                         "Dropout, BatchNormalization and Split. Use parameter 'allow_opset_10_and_lower' "
                         "to disable this check.")

            # Get rid of preprocessing warning
            add_pre_process_metadata(inferred_model_proto)
            onnx.save_model(inferred_model_proto, input_model_path)

            optimized_model_path = os.path.join(tmp_dir, "optimized_model.onnx")

            try:
                sess_option = onnxruntime.SessionOptions()
                sess_option.optimized_model_filepath = optimized_model_path
                sess_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
                # noinspection PyUnusedLocal
                # Disable ConstantSharing optimization because it sometimes produces invalid models (resnet50-caffe2)
                sess = onnxruntime.InferenceSession(input_model_path, sess_option,
                                                    providers=["CPUExecutionProvider"],
                                                    disabled_optimizers=["ConstantSharing"])
                # ONNXRT: Close the session to avoid the cleanup error on Windows for temp folders
                # https://github.com/microsoft/onnxruntime/issues/17627
                del sess

                unoptimized_model_nodes_count = len(inferred_model_proto.graph.node)

                # Run again shape inference on optimized model
                model_proto = onnx.load_model(optimized_model_path)
                inferred_model_proto = ModelShapeInference.infer_shapes(model_proto)
                onnx.save_model(inferred_model_proto, input_model_path)

                removed_ops = unoptimized_model_nodes_count - len(inferred_model_proto.graph.node)
                if removed_ops > 0:
                    logger.i(f"ORT model optimization step removed {removed_ops} ops.")
                elif removed_ops == 0:
                    logger.i("ORT model optimization step hasn't removed any nodes.")
                else:
                    logger.w(f"ORT model optimization step added {abs(removed_ops)} ops.")
            except BaseException as e: # noqa: BLE001
                # Optimization failed - continue with non-optimized model
                logger.w(f"Optimization step failed with: {type(e).__name__}. Skipping.")
                logger.d(f"Optimization step failure reason: {traceback.format_exc()}")

            extra_options = {
                "DedicatedQDQPair": True,
                "ForceQuantizeNoInputCheck": True,
                "ActivationSymmetric": False,  # False by default. Necessary to have correct q-params for Softmax.
            }

            onnxruntime.quantization.quantize_static(
                input_model_path,
                output_model_path,
                per_channel=quantization_config.per_channel,
                calibration_data_reader=quantization_config.calibration_data_reader,
                op_types_to_quantize=self.op_types_to_quantize,
                extra_options=extra_options)

            model = onnx.load_model(output_model_path)

        return model
