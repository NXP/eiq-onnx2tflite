#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import itertools
import re
from typing import Iterable

import onnx

from onnx2tflite.src.model_inspector import ONNXModelInspector
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.onnx_parser import onnx_model


class ONNXSurgeon:

    def intermediate_tensors_as_outputs(self, model_path_or_bytes,
                                        extracted_tensors_name_regex: str) -> onnx.ModelProto:
        """
        Create model with exposed intermediate tensors.

        :param model_path_or_bytes: Input ONNX model as path to file or serialized model.
        :param extracted_tensors_name_regex: Regular expressions matching names of the tensors
            that should be exposed. Use ".*" to expose all the tensors.
        :return: Model in onnx.ModelProto format with exposed requested tensors.
        """
        if isinstance(model_path_or_bytes, bytes):
            model = onnx.load_model_from_string(model_path_or_bytes)
        else:
            model = onnx.load_model(model_path_or_bytes)
        model = ModelShapeInference.infer_shapes(model)

        # Remove tensors from the inputs, if they are not actually model inputs.
        internal_names = [t.name for t in
                          itertools.chain(model.graph.output, model.graph.value_info, model.graph.initializer)]
        to_remove = [i for i in model.graph.input if i.name in internal_names]
        for i in to_remove:
            model.graph.input.remove(i)

        old_outputs = [output.name for output in model.graph.output]
        appended_tensors = []
        for idx, value_info in enumerate(model.graph.value_info):
            if value_info.name in old_outputs:
                continue
            if re.match(extracted_tensors_name_regex, value_info.name):
                appended_tensors.append(value_info)
        model.graph.output.extend(appended_tensors)
        return model

    def _get_objects_with_names(self, object_list: Iterable[onnx.ValueInfoProto | onnx.TensorProto],
                                names: set[str]) -> list[onnx.ValueInfoProto | onnx.TensorProto]:
        """ Return objects, which have a string '.name' attribute with a value in 'names'.
            The resulting list won't contain duplicates.

        :param object_list: An iterable of objects with a '.name' attribute.
        :param names:
        :return: List of objects from 'object_list', with '.name' attributes in 'names', without duplicates.
        """
        res = []
        for obj in object_list:
            if obj.name in names and obj not in res:
                res.append(obj)

        return res

    def _get_symbolic_dimensions(self, model: onnx.ModelProto) -> set[str]:
        """ Get a set of strings used as symbolic dimensions in a model.

        :param model: ONNX ModelProto to check.
        :return: A set of symbolic dimensions.
        """
        res = set()
        for vi in model.graph.input:
            for dim in vi.type.tensor_type.shape.dim:
                if hasattr(dim, 'dim_param'):
                    res.update(dim.dim_param)

        return res

    def _build_model_with_nodes(self, nodes_to_keep: [onnx.NodeProto], model: onnx.ModelProto):
        # Compute the input and output tensors of all operators
        operator_outputs = set()
        operator_inputs = set()
        for node in nodes_to_keep:
            operator_inputs.update(set(node.input))
            operator_outputs.update(set(node.output))

        # Set of names of tensors, which should be in the returned model.
        used_tensors = operator_inputs.union(operator_outputs)

        static_tensors = set([t.name for t in model.graph.initializer]).intersection(used_tensors)

        # IO of the new model. Extended by the inputs of the first kept node and outputs of the last kept node.
        input_names = operator_inputs.difference(operator_outputs).difference(static_tensors)
        output_names = operator_outputs.difference(operator_inputs)

        possible_input_value_info = list(model.graph.value_info) + list(model.graph.input)
        inputs = self._get_objects_with_names(possible_input_value_info, input_names)

        possible_output_value_info = list(model.graph.value_info) + list(model.graph.output)
        outputs = self._get_objects_with_names(possible_output_value_info, output_names)

        initializer = self._get_objects_with_names(model.graph.initializer, used_tensors)
        value_info = self._get_objects_with_names(model.graph.value_info, used_tensors)

        return onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes=nodes_to_keep,
                name=model.graph.name,
                inputs=inputs,
                outputs=outputs,
                initializer=initializer,
                value_info=value_info
            ), opset_imports=model.opset_import
        )

    def pick_out_operators(self, source: str | onnx.ModelProto, custom_slice: slice,
                           skip_shape_inference=False) -> onnx.ModelProto:
        """ Create a reduced ONNX model, which only contains the operators resulting from a 'custom_slice' on the 'nodes'
             collection of the model.

            The ONNX model can be passed via a path to a '.onnx' file, or directly as a ModelProto object.

        :param source: A path to a '.onnx' file, or a ModelProto object.
        :param custom_slice: A slice of the 'nodes' of the model, which will be in the returned model.
        :param skip_shape_inference: Skip shape inference and symbolic shape definition.
        :return: A ModelProto object, which contains only operators with indices from 'start_idx' to 'end_idx'.
        """
        if isinstance(source, str):
            original_model = onnx.load(source)

        elif isinstance(source, onnx.ModelProto):
            original_model = source

        else:
            raise TypeError

        # Internal shapes must be known.
        if not skip_shape_inference:
            symbolic_dimensions = self._get_symbolic_dimensions(original_model)
            if len(symbolic_dimensions) != 0:
                print(
                    "Tested model uses symbolic shapes. In order to test accuracy, all symbolic dimensions are set to 1.")
            original_model = ModelShapeInference.infer_shapes(original_model, symbolic_dimensions_mapping={
                dim: 1 for dim in symbolic_dimensions
            })

        nodes_to_keep = original_model.graph.node[custom_slice]
        assert len(nodes_to_keep) > 0, "Specified slice doesn't contain any nodes!"

        model = self._build_model_with_nodes(nodes_to_keep, original_model)

        return model

    def _forward_pass(self, inspector: ONNXModelInspector, input_tensors: list[str]) -> [onnx_model.NodeProto]:
        # Traverse forward through the model. Start in 'input_tensors'.
        tensors_to_process = input_tensors.copy()
        forward_pass_nodes = list()
        processed_tensors = set()

        while tensors_to_process:
            tensor = tensors_to_process.pop()

            if tensor in processed_tensors:
                continue

            output_nodes = inspector.get_ops_with_input_tensor(tensor)
            for node in output_nodes:
                if node not in forward_pass_nodes:
                    forward_pass_nodes.append(node)
                tensors_to_process.extend(list(node.outputs))

            processed_tensors.add(tensor)

        return forward_pass_nodes

    def _backward_pass(self, inspector: ONNXModelInspector, output_tensors: list[str]) -> [onnx_model.NodeProto]:
        # Traverse backwards through the model. Start in 'output_tensors'.
        tensors_to_process = output_tensors.copy()
        backward_pass_nodes = list()
        processed_tensors = set()

        while tensors_to_process:
            tensor = tensors_to_process.pop()

            if tensor in processed_tensors:
                continue

            input_nodes = inspector.get_ops_with_output_tensor(tensor)
            for node in input_nodes:
                if node not in backward_pass_nodes:
                    backward_pass_nodes.append(node)
                tensors_to_process.extend(list(node.inputs))

            processed_tensors.add(tensor)

        return backward_pass_nodes

    def _satisfy_nodes_inputs(self, inspector: ONNXModelInspector, nodes: list[onnx_model.NodeProto]):
        # Go over nodes and check if all inputs are satisfied
        input_dependant_nodes = self._forward_pass(inspector, inspector.get_non_initializer_input_names())
        nodes_to_analyze = list(nodes)

        while nodes_to_analyze:
            node = nodes_to_analyze.pop()

            for i in node.inputs:
                if inspector.tensor_is_static(i):
                    # Tensor is initializer
                    continue

                parent_nodes = inspector.get_ops_with_output_tensor(i)
                if len(parent_nodes) > 0:  # Not input tensor
                    parent_node = parent_nodes[0]
                    if parent_node in nodes:
                        # Parent node is in preserved nodes list
                        continue
                    elif parent_node in input_dependant_nodes:
                        # Parent node depends on model input -> tensor will be
                        # used as input of extracted subgraph
                        continue
                    else:
                        # Parent node depends on statically calculated data ->
                        # preserve node and analyze its parent (its input is also static)
                        nodes.insert(0, parent_node)
                        nodes_to_analyze.append(parent_node)

    def _get_subset_of_nodes_from_original_model(self, original_model: onnx.ModelProto,
                                                 nodes_to_keep: list[onnx_model.NodeProto]) -> [onnx.NodeProto]:
        # Go over model nodes and check if we marked as node to be kept
        sorted_nodes = []
        for node in original_model.graph.node:
            for preserved_node in nodes_to_keep:
                # Nodes usually don't have names - do best effort to check equality
                # We are comparing onnx.NodeProto and onnx_model.NodeProto
                if (node.op_type == preserved_node.op_type and
                        node.input == preserved_node.inputs and
                        node.output == preserved_node.outputs):
                    sorted_nodes.append(node)
        return sorted_nodes

    def extract_subgraph(self, source: str | onnx.ModelProto, input_tensors: list[str] | None = None,
                         output_tensors: list[str] | None = None, skip_shape_inference: bool = False
                         ) -> onnx.ModelProto:
        """
        Create a reduced ONNX model, which only contains the operators between input and output tensors.
        Extracted model could have more inputs than it is defined via "input_tensors" argument,
        because operator dependencies might get cut our during the process.

        The ONNX model can be passed via a path to a '.onnx' file, or directly as a ModelProto object.

        :param source: A path to a '.onnx' file, or a ModelProto object.
        :param input_tensors: List of input tensor names. If empty, all input tensors are used instead.
        :param output_tensors: List of output tensor names. If empty, all output tensors are used instead.
        :param skip_shape_inference: Skip shape inference and symbolic shape definition.
        :return: A ModelProto object, which contains only operators limited by input and output tensors.
        """
        if isinstance(source, str):
            original_model = onnx.load(source)
        elif isinstance(source, onnx.ModelProto):
            original_model = source
        else:
            raise TypeError

        if input_tensors is None and output_tensors is None:
            return original_model

        if input_tensors is None:
            input_tensors = [_input.name for _input in original_model.graph.input]

        if output_tensors is None:
            output_tensors = [_output.name for _output in original_model.graph.output]

        # Internal shapes must be known.
        if not skip_shape_inference:
            symbolic_dimensions = self._get_symbolic_dimensions(original_model)
            if len(symbolic_dimensions) != 0:
                print(
                    "Tested model uses symbolic shapes. In order to test accuracy, all symbolic dimensions are set to 1.")
            original_model = ModelShapeInference.infer_shapes(original_model, symbolic_dimensions_mapping={
                dim: 1 for dim in symbolic_dimensions
            })

        inspector = ONNXModelInspector(onnx_model.ModelProto(original_model, init_node_attributes=False))

        forward_pass_nodes = self._forward_pass(inspector, input_tensors)
        backward_pass_nodes = self._backward_pass(inspector, output_tensors)

        assert len(forward_pass_nodes) != 0, ("Zero nodes gathered during forward pass. "
                                              "Is input tensor present in the model?")
        assert len(backward_pass_nodes) != 0, ("No nodes gathered during backward pass. "
                                               "Is output tensor present in the model?")

        nodes_intersection = [node for node in forward_pass_nodes if node in backward_pass_nodes]

        assert len(nodes_intersection) != 0, ("Zero nodes were marked to be kept in extracted model. "
                                              "Are input tensors present in model before output tensors?")

        # Iterate over nodes intersection and look for non-satisfied inputs
        self._satisfy_nodes_inputs(inspector, nodes_intersection)
        nodes_to_keep = self._get_subset_of_nodes_from_original_model(original_model, nodes_intersection)

        return self._build_model_with_nodes(nodes_to_keep, original_model)
