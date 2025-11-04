#
# Copyright 2023 Martin Pavella
# Copyright 2023-2025 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#


import typing

# noinspection PyPackageRequirements
import google.protobuf.message
import onnx

from onnx2tflite.src import logger, tensor_formatting
from onnx2tflite.src.onnx_parser import builtin_attributes, onnx_tensor
from onnx2tflite.src.onnx_parser.meta import meta


class Tensor(meta.ONNXObject):
    _descriptor: onnx.TypeProto.Tensor  # Specify the exact type of '_descriptor' from parent class

    # The onnx schema for some reason specifies 'elem_type' as type int32. But comments in the schema state, that the
    # value of 'elem_type' MUST be a valid value of the 'TensorProto.DataType' enum.
    # Here, the type is 'TensorProto.DataType' in the first place, to reflect the comment.
    elem_type: onnx.TensorProto.DataType
    shape: onnx_tensor.TensorShapeProto

    def __init__(self, descriptor: onnx.TypeProto.Tensor) -> None:
        super().__init__(descriptor)

    def _init_attributes(self) -> None:
        # noinspection PyTypeChecker
        self.elem_type = self._descriptor.elem_type  # Warning is probably OK. See 'elem_type' declaration above.
        self.shape = onnx_tensor.TensorShapeProto(self._descriptor.shape)


class Sequence(meta.ONNXObject):
    _descriptor: onnx.TypeProto.Sequence  # Specify the exact type of '_descriptor' from parent class

    elem_type: meta.ONNXObject  # 'Type' object!

    def __init__(self, descriptor: onnx.TypeProto.Sequence) -> None:
        super().__init__(descriptor)
        self.elem_type = TypeProto(descriptor.elem_type)


class Map(meta.ONNXObject):
    _descriptor: onnx.TypeProto.Map  # Specify the exact type of '_descriptor' from parent class

    # The onnx schema for some reason specifies 'key_type' as type int32. But comments in the schema state, that the
    # value of 'key_type' MUST be a valid value of the 'TensorProto.DataType' enum.
    # Here, the type is 'TensorProto.DataType' in the first place, to reflect the comment.
    key_type: onnx.TensorProto.DataType
    value_type: meta.ONNXObject  # 'Type' object!

    def __init__(self, descriptor: onnx.TypeProto.Map) -> None:
        super().__init__(descriptor)

    def _init_attributes(self) -> None:
        # noinspection PyTypeChecker
        self.key_type = self._descriptor.key_type  # Warning is probably OK. See 'key_type' declaration above.
        self.value_type = TypeProto(self._descriptor.value_type)


class Optional(meta.ONNXObject):
    _descriptor: onnx.TypeProto.Optional  # Specify the exact type of '_descriptor' from parent class

    elem_type: meta.ONNXObject  # 'Type' object!

    def __init__(self, descriptor: onnx.TypeProto.Optional) -> None:
        super().__init__(descriptor)

    def _init_attributes(self) -> None:
        self.elem_type = TypeProto(self._descriptor.elem_type)


class Opaque(meta.ONNXObject):
    _descriptor: onnx.TypeProto.Opaque  # Specify the exact type of '_descriptor' from parent class

    domain: str
    name: str

    def __init__(self, descriptor: onnx.TypeProto.Opaque) -> None:
        super().__init__(descriptor)

    def _init_attributes(self) -> None:
        self.domain = self._descriptor.domain
        self.name = self._descriptor.name


class TypeProto(meta.ONNXObject):
    _descriptor: onnx.TypeProto  # Specify the exact type of '_descriptor' from parent class

    denotation: str

    """ The 'Type' object MUST have exactly 1 of these types.
        All unused types are 'None'. 
    """
    tensor_type: Tensor | None
    sequence_type: Sequence | None
    map_type: Map | None
    optional_type: Optional | None
    sparse_tensor_type: onnx_tensor.SparseTensor | None
    opaque_type: Opaque | None

    def __init__(self, descriptor: onnx.TypeProto) -> None:
        super().__init__(descriptor)

    def _init_attributes(self) -> None:
        """Initialize attributes. Called by parent constructor."""
        self.denotation = self._descriptor.denotation
        # Initialize the types. Only 1 will have a value, others are 'None'.
        self._reset_types()
        self.__init_used_type()

    def __init_used_type(self) -> None:
        """Find out which 'type' field was used in the model and initialize the
        corresponding attribute.
        """
        if self._descriptor.HasField("tensor_type"):
            self.tensor_type = Tensor(self._descriptor.tensor_type)
        elif self._descriptor.HasField("sequence_type"):
            self.sequence_type = Sequence(self._descriptor.sequence_type)
        elif self._descriptor.HasField("map_type"):
            self.map_type = Map(self._descriptor.map_type)
        elif self._descriptor.HasField("optional_type"):
            self.optional_type = Optional(self._descriptor.optional_type)
        elif self._descriptor.HasField("sparse_tensor_type"):
            self.sparse_tensor_type = onnx_tensor.SparseTensor(self._descriptor.sparse_tensor_type)
        elif self._descriptor.HasField("opaque_type"):
            self.opaque_type = Opaque(self._descriptor.opaque_type)

    def _reset_types(self) -> None:
        """Set all 'type' attributes to 'None'."""
        self.tensor_type = None
        self.sequence_type = None
        self.map_type = None
        self.optional_type = None
        self.sparse_tensor_type = None
        self.opaque_type = None


class ValueInfoProto(meta.ONNXObject):
    name: str
    type: TypeProto
    doc_string: str

    tensor_format: tensor_formatting.TensorFormat

    def __init__(self, descriptor: onnx.ValueInfoProto) -> None:
        super().__init__(descriptor)

    def _init_attributes(self) -> None:
        self.name = self._descriptor.name
        self.type = TypeProto(self._descriptor.type)
        self.doc_string = self._descriptor.doc_string

        self.tensor_format = tensor_formatting.TensorFormat.NONE


class RepeatedValueInfoProto(list[ValueInfoProto]):
    def __init__(self, descriptor_iterable: typing.MutableSequence[onnx.ValueInfoProto]):
        super().__init__()
        for descriptor in descriptor_iterable:
            self.append(ValueInfoProto(descriptor))


class OperatorSetIdProto(meta.ONNXObject):
    _descriptor: onnx.OperatorSetIdProto  # Specify parent '_descriptor' type

    domain: str
    version: int

    def __init__(self, descriptor: onnx.OperatorSetIdProto) -> None:
        super().__init__(descriptor)

    def _init_attributes(self) -> None:
        self.domain = self._descriptor.domain
        self.version = self._descriptor.version


class RepeatedOperatorSetIdProto(list[OperatorSetIdProto]):
    def __init__(self, descriptor_iterable: typing.MutableSequence[onnx.OperatorSetIdProto]) -> None:
        super().__init__()
        for descriptor in descriptor_iterable:
            self.append(OperatorSetIdProto(descriptor))


class NodeProto(meta.ONNXObject):
    _descriptor: onnx.NodeProto

    inputs: typing.MutableSequence[str]
    outputs: typing.MutableSequence[str]
    name: str
    op_type: str
    domain: str
    version: int
    attributes: meta.ONNXOperatorAttributes | None
    doc_string: str

    # Dictionary which maps an ONNX operator type to a constructor of an onnx_parser attributes class, or None.
    # noinspection SpellCheckingInspection
    op_type_to_attribute_constructor_map: dict[str, typing.Callable | None] = {
        "Abs": None,
        "Add": None,
        "And": None,
        "ArgMax": builtin_attributes.ArgMax,
        "ArgMin": builtin_attributes.ArgMin,
        "AveragePool": builtin_attributes.AveragePool,
        "BatchNormalization": builtin_attributes.BatchNormalization,
        "Cast": builtin_attributes.Cast,
        "Ceil": None,
        "Clip": builtin_attributes.Clip,
        "Concat": builtin_attributes.Concat,
        "Constant": builtin_attributes.Constant,
        "ConstantOfShape": builtin_attributes.ConstantOfShape,
        "Conv": builtin_attributes.Conv,
        "ConvTranspose": builtin_attributes.ConvTranspose,
        "Cos": None,
        "CumSum": builtin_attributes.CumSum,
        "DepthToSpace": builtin_attributes.DepthToSpace,
        "DequantizeLinear": builtin_attributes.DequantizeLinear,
        "Div": None,
        "Dropout": builtin_attributes.Dropout,
        "Einsum": builtin_attributes.Einsum,
        "Elu": builtin_attributes.Elu,
        "Equal": None,
        "Erf": None,
        "Expand": None,
        "Exp": None,
        "Flatten": builtin_attributes.Flatten,
        "Floor": None,
        "Gather": builtin_attributes.Gather,
        "GatherND": builtin_attributes.GatherND,
        "Gelu": builtin_attributes.Gelu,
        "Gemm": builtin_attributes.Gemm,
        "GlobalAveragePool": None,
        "GlobalMaxPool": None,
        "Greater": None,
        "GreaterOrEqual": None,
        "HardSigmoid": builtin_attributes.HardSigmoid,
        "HardSwish": None,
        "Identity": None,
        "InstanceNormalization": builtin_attributes.InstanceNormalization,
        "LRN": builtin_attributes.LRN,
        "LSTM": builtin_attributes.LSTM,
        "LayerNormalization": builtin_attributes.LayerNormalization,
        "LeakyRelu": builtin_attributes.LeakyRelu,
        "Less": None,
        "LessOrEqual": None,
        "Log": None,
        "LogSoftmax": builtin_attributes.LogSoftmax,
        "MatMul": builtin_attributes.MatMul,
        "Max": None,
        "MaxPool": builtin_attributes.MaxPool,
        "Min": None,
        "Mod": builtin_attributes.Mod,
        "Mul": None,
        "Multinomial": builtin_attributes.Multinomial,
        "Neg": None,
        "Not": None,
        "Or": None,
        "OneHot": builtin_attributes.OneHot,
        "PRelu": None,
        "Pad": builtin_attributes.Pad,
        "Pow": None,
        "QGemm": builtin_attributes.QGemm,
        "QLinearAdd": None,
        "QLinearAveragePool": builtin_attributes.QLinearAveragePool,
        "QLinearConcat": builtin_attributes.QLinearConcat,
        "QLinearConv": builtin_attributes.QLinearConv,
        "QLinearGlobalAveragePool": builtin_attributes.QLinearGlobalAveragePool,
        "QLinearMatMul": None,
        "QLinearMul": None,
        "QLinearSoftmax": builtin_attributes.QLinearSoftmax,
        "QuantizeLinear": builtin_attributes.QuantizeLinear,
        "QuickGelu": builtin_attributes.QuickGelu,
        "RNN": builtin_attributes.RNN,
        "Range": None,
        "Reciprocal": None,
        "ReduceL2": builtin_attributes.ReduceL2,
        "ReduceMax": builtin_attributes.ReduceMax,
        "ReduceMean": builtin_attributes.ReduceMean,
        "ReduceMin": builtin_attributes.ReduceMin,
        "ReduceProd": builtin_attributes.ReduceProd,
        "ReduceSum": builtin_attributes.ReduceSum,
        "Relu": builtin_attributes.Relu,
        "Reshape": builtin_attributes.Reshape,
        "Resize": builtin_attributes.Resize,
        "ReverseSequence": builtin_attributes.ReverseSequence,
        "Round": None,
        "ScatterND": builtin_attributes.ScatterND,
        "Shape": builtin_attributes.Shape,
        "Sigmoid": None,
        "Sign": None,
        "Sin": None,
        "Slice": builtin_attributes.Slice,
        "Softmax": builtin_attributes.Softmax,
        "SpaceToDepth": builtin_attributes.SpaceToDepth,
        "Split": builtin_attributes.Split,
        "Sqrt": None,
        "Squeeze": builtin_attributes.Squeeze,
        "Sub": None,
        "Sum": None,
        "Tan": None,
        "Tanh": None,
        "Tile": None,
        "Transpose": builtin_attributes.Transpose,
        "Unsqueeze": builtin_attributes.Unsqueeze,
        "Upsample": builtin_attributes.Upsample,
        "Where": builtin_attributes.Where,
        "Xor": None
    }

    def __init__(self, descriptor: onnx.NodeProto, version: int, index: int, init_node_attributes) -> None:
        self.index = index
        self._init_node_attributes = init_node_attributes
        super().__init__(descriptor)
        self.version = version

    def _init_attributes(self) -> None:
        """Initialize attributes. Called from parent constructor."""
        self.inputs = self._descriptor.input
        self.outputs = self._descriptor.output
        self.name = self._descriptor.name
        self.op_type = self._descriptor.op_type
        self.domain = self._descriptor.domain
        self.doc_string = self._descriptor.doc_string
        if not self.name:
            self.unique_name = f"{self.op_type}_{self.index}"
        else:
            self.unique_name = f"{self.name}_{self.index}"

        if self._init_node_attributes:
            if self.op_type not in NodeProto.op_type_to_attribute_constructor_map:
                logger.e(logger.Code.UNSUPPORTED_OPERATOR, f"ONNX operator '{self.op_type}' is not yet supported!")

            get_attributes_fn = NodeProto.op_type_to_attribute_constructor_map[self.op_type]
            if get_attributes_fn is None:
                self.attributes = None
            else:
                self.attributes = get_attributes_fn(self._descriptor.attribute)

    def __eq__(self, other):
        return self.unique_name == other.unique_name

    def __hash__(self):
        return hash(self.unique_name)


class RepeatedNodeProto(list[NodeProto]):
    def __init__(self, descriptor_iterable: typing.MutableSequence[onnx.NodeProto],
                 opset_imports: RepeatedOperatorSetIdProto,
                 init_node_attributes: bool):
        super().__init__()
        opset_map = {opset.domain: opset.version for opset in opset_imports}

        unsupported_nodes = set()

        for idx, descriptor in enumerate(descriptor_iterable):
            with logger.loggingContext(logger.NodeLoggingContext(idx)):
                if descriptor.domain not in opset_map:
                    logger.e(logger.Code.INVALID_ONNX_OPERATOR,
                             f"Domain '{descriptor.domain}' not present in model's opset_imports.")
                try:
                    self.append(NodeProto(descriptor, opset_map[descriptor.domain], idx, init_node_attributes))
                except logger.Error:
                    unsupported_nodes.add(descriptor.op_type)

        if len(unsupported_nodes) > 0:
            msg = f"Model contains unsupported nodes: [{', '.join(unsupported_nodes)}]."
            logger.e(logger.Code.UNSUPPORTED_OPERATOR, msg)


class GraphProto(meta.ONNXObject):
    _descriptor: onnx.GraphProto  # Specify parent '_descriptor' type
    _opset_imports: RepeatedOperatorSetIdProto

    nodes: RepeatedNodeProto
    name: str
    initializers: onnx_tensor.RepeatedTensorProto
    # TODO sparse_initializers
    doc_string: str
    inputs: RepeatedValueInfoProto
    outputs: RepeatedValueInfoProto
    value_info: RepeatedValueInfoProto

    # TODO quantization_annotation

    def __init__(self, descriptor: onnx.GraphProto, opset_imports: RepeatedOperatorSetIdProto,
                 init_node_attributes: bool) -> None:
        self._opset_imports = opset_imports
        self._init_node_attributes = init_node_attributes
        super().__init__(descriptor)

    def _init_attributes(self) -> None:
        """Initialize attributes. Called from parent constructor."""
        self.name = self._descriptor.name
        self.nodes = RepeatedNodeProto(self._descriptor.node, self._opset_imports, self._init_node_attributes)
        self.initializers = onnx_tensor.RepeatedTensorProto(self._descriptor.initializer)
        self.doc_string = self._descriptor.doc_string
        self.inputs = RepeatedValueInfoProto(self._descriptor.input)
        self.outputs = RepeatedValueInfoProto(self._descriptor.output)
        self.value_info = RepeatedValueInfoProto(self._descriptor.value_info)


class ModelProto(meta.ONNXObject):
    _descriptor: onnx.ModelProto  # Specify parent '_descriptor' type

    ir_version: int
    opset_imports: RepeatedOperatorSetIdProto
    producer_name: str
    producer_version: str
    domain: str
    model_version: int
    doc_string: str
    graph: GraphProto

    # TODO metadata_props
    # TODO training_info
    # TODO functions

    def __init__(self, source: str | onnx.ModelProto, init_node_attributes: bool = True) -> None:
        """Initialize an internal representation of on ONNX model either from an .onnx file or from an
        onnx.ModelProto object.
        :param source: Either a string name of an .onnx file to parse, or a ModelProto object holding model data.
        :param init_node_attributes: Initialize operator attributes. If False, skips and only loads the model structure.
                                    Useful when need to parse models with unsupported ONNX operators.
        """
        self._init_node_attributes = init_node_attributes

        if isinstance(source, str):
            try:
                model = onnx.load(source)
                super().__init__(model)

            except google.protobuf.message.DecodeError:
                logger.e(logger.Code.INVALID_INPUT, f"Couldn't parse model from file '{source}'!")

        elif isinstance(source, onnx.ModelProto):
            super().__init__(source)

        else:
            logger.e(logger.Code.INVALID_INPUT, f"Cannot initialize ONNX model from object of type '{type(source)}'! "
                                                f"Expected type 'onnx.ModelProto' or 'string'.")

    def _init_attributes(self) -> None:
        """Initialize object attributes from the '_descriptor' attribute of the parent object."""
        self.ir_version = self._descriptor.ir_version
        self.opset_imports = RepeatedOperatorSetIdProto(self._descriptor.opset_import)
        self.producer_name = self._descriptor.producer_name
        self.producer_version = self._descriptor.producer_version
        self.domain = self._descriptor.domain
        self.model_version = self._descriptor.model_version
        self.doc_string = self._descriptor.doc_string
        self.graph = GraphProto(self._descriptor.graph, self.opset_imports, self._init_node_attributes)
