#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import itertools
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.onnx_parser import onnx_model, onnx_tensor


class ONNXModelInspector:
    """Class for easier search and interaction with ONNX model. Provides methods to search for
    nodes/tensors.
    """

    # Mapping a name of a tensor to the number of nodes and outputs, which consume the tensor in the ONNX model.
    # Can be None, but then a call to `get_number_of_onnx_consumers()` will raise an error.
    _tensor_consumers_count: dict[str, int] | None

    # Mapping a name of a dynamic tensor to a numpy array, containing the data the tensor will have at runtime.
    # Can be None (if used in testing modules), but then a call to `try_get_inferred_tensor_data()` will raise an error.
    _inferred_tensor_data: dict[str, np.ndarray] | None

    def __init__(self, model: onnx_model.ModelProto, inferred_tensor_data: dict[str, np.ndarray] | None = None):
        self.model = model
        self._tensor_consumers_count = self._count_tensor_consumers()
        self._inferred_tensor_data = inferred_tensor_data or {}

    def try_get_inferred_tensor_data(self, tensor_name: str) -> np.ndarray | None:
        """For the given tensor, return data which was inferred during shape inference of the ONNX model. May return
             None, if there was no data inferred for the tensor.

            Sometimes dynamic tensors in the ONNX model will always have the same data at runtime, because they are
             the outputs of operators which use static tensors. This function allows access to this inferred data, which
             is originally computed by the sympy model shape inference. Not all operators are supported, so custom
             dispatchers may need to be registered in the 'model_shape_inference.py'.

            Careful: the data will be unchanged from the ONNX model. It may need to be permuted before use in the
             TFLite model.

        :param tensor_name: Name of the tensor to get data for.
        :return: A numpy array with the tensor's inferred data, or None.
        """
        if (tensor_data := self._inferred_tensor_data.get(tensor_name, None)) is not None:
            # Make sure the returned value is a numpy array. The shape inference module contains buggy code, so until we
            #  fix all operators, we cannot be sure the sympy data doesn't contain regular python lists.
            return np.asarray(tensor_data)

        return None

    def get_number_of_onnx_consumers_safe(self, tensor_name: str) -> int | None:
        """Return how many nodes in the ONNX model use a tensor with given name as an input.
        If the tensor with 'tensor_name' not exists in the ONNX model returns 0.

        :param tensor_name: Name of the tensor to count number of consumer nodes for.
        :return: Number of nodes using the tensor.
        """
        if self._tensor_consumers_count is None:
            logger.e(logger.Code.INTERNAL_ERROR, "ONNXModelInspector: _tensor_consumers_count was not initialized!")

        return self._tensor_consumers_count.get(tensor_name, 0)

    def get_used_outputs(self, node: onnx_model.NodeProto) -> list[str]:
        """Get a list of names of output tensors for given `node`, that are used later on in the model.

        :param node: ONNX Node for which to find used outputs.
        :return: List of names of the used outputs.
        """
        return [output for output in node.outputs if self.get_number_of_onnx_consumers_safe(output) != 0]

    def get_ops_with_output_tensor(self, tensor_name: str) -> [onnx_model.NodeProto]:
        """Finds all ops in the model, whose tensor_name is in operators output

        :return: List of operators
        """
        return [node for node in self.model.graph.nodes if tensor_name in node.outputs]

    def get_ops_with_input_tensor(self, tensor_name: str) -> [onnx_model.NodeProto]:
        """Finds ops in the model, where tensor with tensor_name is operator's input tensor.

        :return: List of operators
        """
        return [node for node in self.model.graph.nodes if tensor_name in node.inputs]

    def tensor_originates_in_single_consumer_dequantize_op(self, tensor_name) -> bool:
        """Check if tensor named 'tensor_name' originates in 'DequantizeLinear' operator.
        Input and output tensors of such operator must have only single consumer.

        :return: True if tensor originates in 'DequantizeLinear' and IO tensors have single
                    consumer.
        """
        output_nodes = self.get_ops_with_output_tensor(tensor_name)

        if len(output_nodes) != 1 or output_nodes[0].op_type != "DequantizeLinear":
            return False

        # Dequantize IO tensors must have only one consumer
        dequantize_output_consumers = self.get_ops_with_input_tensor(output_nodes[0].outputs[0])
        dequantize_input_consumers = self.get_ops_with_input_tensor(output_nodes[0].inputs[0])

        return len(dequantize_output_consumers) == 1 and len(dequantize_input_consumers) == 1

    def tensor_is_shared_dequantized_bias(self, tensor_name) -> bool:
        """Check if tensor named 'tensor_name' originates in 'DequantizeLinear' operator with
        input type INT32 and all consumers are Conv operators.

        :return: True if tensor originates in 'DequantizeLinear' with input type is INT32 and consumers are Conv ops.
        """
        output_nodes = self.get_ops_with_output_tensor(tensor_name)

        if len(output_nodes) != 1 or output_nodes[0].op_type != "DequantizeLinear":
            return False

        dequantize_input_tensor_type = self.get_tensor_type(output_nodes[0].inputs[0])
        dequantize_consumers = self.get_ops_with_input_tensor(output_nodes[0].outputs[0])
        consumers_are_conv_ops = map(lambda node: node.op_type == "Conv", dequantize_consumers)

        return dequantize_input_tensor_type == onnx.TensorProto.INT32 and all(consumers_are_conv_ops)

    def tensor_leads_to_quantize_op(self, tensor_name) -> bool:
        """Check if tensor named 'tensor_name' leads only to 'QuantizeLinear' operators.

        Note: we do not support QDQ scheme found in old QDQ quantized models from ONNX model Zoo, e.g.
        https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/mobilenet/model/mobilenetv2-12-qdq.onnx,
        where the QDQ specification was not strictly followed, especially for residual connection only one of the
        consumers was a QuantizeLinear node. For instance in above-mentioned model, the 'Add' output was consumed by
        a "QuantizeLinear" and "Add" op for next residual, like this:
          <Add>
         |     <Quantize Linear>
         |     <Dequantize Linear>
         |     <Conv>
         |     <Quantize Linear>
         |     <DequantizeLinear>
          <Add>

        :param tensor_name:
        :return: True if tensor leads to at least one 'QuantizeLinear' node.
        """
        input_nodes = self.get_ops_with_input_tensor(tensor_name)
        return all(i.op_type == "QuantizeLinear" for i in input_nodes)

    def is_output_of_model(self, tensor_name):
        return any(t.name == tensor_name for t in self.model.graph.outputs)

    def is_input_of_model(self, tensor_name: str) -> bool:
        """Determine whether a tensor with given name is an input to the ONNX graph."""
        return any(t.name == tensor_name for t in self.model.graph.inputs)

    def tensor_is_float(self, tensor_name):
        """Check if tensor's type is float.

        :param tensor_name: Name of the searched tensor.
        :return: True is type of searched tensor is float.
        """
        return self.get_tensor_type(tensor_name) == onnx.TensorProto.FLOAT

    def tensor_not_float(self, tensor_name):
        """Check if tensor's type is NOT float.

        :param tensor_name: Name of the searched tensor.
        :return: True is type of searched tensor is not float.
        """
        return not self.tensor_is_float(tensor_name)

    def get_tensor_type(self, tensor_name: str) -> onnx.TensorProto.DataType:
        """Get tensor's type.

        :param tensor_name: Name of the searched tensor.
        :return: Tensor type in TensorProto format.
        """
        tensor = self.find_tensor(tensor_name)

        if isinstance(tensor, onnx_tensor.TensorProto):
            return tensor.data_type
        if isinstance(tensor, onnx_model.ValueInfoProto):
            return tensor.type.tensor_type.elem_type
        logger.e(logger.Code.INTERNAL_ERROR, "Unexpected type of tensor specification")

    def _get_tensor_data(self, tensor_name: str) -> np.ndarray | None:
        for tensor in self.model.graph.initializers:
            if tensor.name == tensor_name:
                return tensor.data

        return None

    def tensor_is_static(self, tensor_name) -> bool:
        """Check if tensor is static and contains only single (scalar) value.

        :param tensor_name: Name of the searched tensor.
        :return: True is type of searched tensor is static and contains scalar value.
        """
        tensor_data = self._get_tensor_data(tensor_name)

        return tensor_data is not None

    def find_tensor(self, tensor_name: str) -> onnx_tensor.TensorProto | onnx_model.ValueInfoProto:
        """Search for tensor with 'tensor_name' in model. In particular in inputs, initializers and value_info field.

        :return: 'TensorProto' or 'ValueInfoProto' object representing tensor.
        """
        # TODO(Lukas) Check if tensor available in both initializers and value_info
        matching_initializers = [i for i in self.model.graph.initializers if i.name == tensor_name]

        if len(matching_initializers) == 1:
            return matching_initializers[0]
        if len(matching_initializers) > 1:
            logger.e(logger.Code.INVALID_ONNX_MODEL, f'Found multiple initializers for tensor "{tensor_name}".')

        matching_value_infos = ([i for i in self.model.graph.inputs if i.name == tensor_name] +
                                [i for i in self.model.graph.outputs if i.name == tensor_name] +
                                [i for i in self.model.graph.value_info if i.name == tensor_name])

        if len(matching_value_infos) == 1:
            return matching_value_infos[0]
        if len(matching_value_infos) > 1:
            logger.w(f'Found multiple value infos for tensor "{tensor_name}". Returning first one.')
            return matching_value_infos[0]

        logger.e(logger.Code.INVALID_ONNX_MODEL,
                 f'Tensor with name: "{tensor_name}" not found! Did you run model shape inference?')

    def get_all_tensors(self) -> dict[str, onnx_tensor.TensorProto]:
        """Get all tensors in model as a dictionary mapping tensor name to 'TensorProto' object.

        :return: Dictionary with tensor name mapped to 'TensorProto' instance.
        """
        graph = self.model.graph

        return {t.name: t for t in graph.inputs + graph.initializers + graph.value_info + graph.outputs}

    def get_non_initializer_input_names(self):
        initializer_names = [i.name for i in self.model.graph.initializers]
        return [i.name for i in self.model.graph.inputs if i.name not in initializer_names]

    def get_nodes(self):
        return self.model.graph.nodes

    def contains_quantization_nodes(self) -> bool:
        """Check if model contains quantization nodes ('QuantizeLinear' or 'DequantizeLinear').

        :return: True if mode contains at least one quantization node.
        """
        for node in self.model.graph.nodes:
            if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                return True

        return False

    def has_int8_and_uint8_q_ops(self) -> bool:
        """Check whether model contains (De)QuantizeLinear ops with both INT8 and
        UINT8 types.

        :return: True if UINT8 and also INT8 q-ops are present. False otherwise.
        """
        has_uint8_ops = False
        has_int8_ops = False

        for node in self.get_nodes():
            if node.op_type == "QuantizeLinear":
                if self.get_tensor_type(node.outputs[0]) == TensorProto.UINT8:
                    has_uint8_ops = True
                elif self.get_tensor_type(node.outputs[0]) == TensorProto.INT8:
                    has_int8_ops = True
            elif node.op_type == "DequantizeLinear":
                if self.get_tensor_type(node.inputs[0]) == TensorProto.UINT8:
                    has_uint8_ops = True
                elif self.get_tensor_type(node.inputs[0]) == TensorProto.INT8:
                    has_int8_ops = True

        return has_uint8_ops and has_int8_ops

    def _count_tensor_consumers(self) -> dict[str, int]:
        """Count the number of consumers for all tensors in the ONNX model. Return a dictionary mapping tensor names to
             the number of consumers.
            Consumers are either nodes in the model, or graph outputs.

        :return: A dictionary mapping tensor names to the number of consumers.
        """
        # The function explicitly sets the default value for known tensors to 0. But I think a defaultdict should still
        #  be used within this function, to handle some edge cases. For example ONNX allows optional input tensors with
        #  to have the name = "", which indicated that they are omitted. No node produces a tensor with name "", so it
        #  would never be explicitly initialized to 0.
        # The function however returns a regular dict. Setting the initial values to 0 explicitly ensures they will be
        #  present in the returned dict.
        tensor_consumers_count = defaultdict(lambda: 0)

        for input_tensor in self.model.graph.inputs:
            # Explicitly set to 0.
            tensor_consumers_count[input_tensor.name] = 0

        # Count how many nodes consume each tensor.
        for node in self.model.graph.nodes:
            for input_tensor_name in node.inputs:
                tensor_consumers_count[input_tensor_name] += 1

            # Explicitly initialize the counts of the output tensors to 0, if they don't yet have a value.
            for output_tensor_name in node.outputs:
                if output_tensor_name not in tensor_consumers_count:
                    tensor_consumers_count[output_tensor_name] = 0

        # Increment the counts for tensors, which are the output of the graph.
        for output_tensor in self.model.graph.outputs:
            tensor_consumers_count[output_tensor.name] += 1

        return dict(tensor_consumers_count)  # Return a regular dict.

    def get_tensor_rank_safe(self, tensor_name: str) -> int | None:
        for t in self.model.graph.initializers:
            if tensor_name == t.name:
                return len(t.dims)

        for t in itertools.chain(self.model.graph.inputs, self.model.graph.outputs, self.model.graph.value_info):
            if tensor_name == t.name:
                return len(t.type.tensor_type.shape.dims)

        return None
