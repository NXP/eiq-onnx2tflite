#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
from collections import defaultdict

import onnx
from onnx.helper import make_tensor_value_info

from onnx2tflite.src import logger
from onnx2tflite.src.preprocess.base_preprocessing_step import PreprocessingStep


class DuplicateDequantizeLinearForEachConsumer(PreprocessingStep):
    """Duplicate DequantizeLinear nodes for each consumer. This optimization is useful when is output of
    single DequantizeLinear op consumed by multiple nodes. Our QDQ cluster recognition doesn't support
    DQL ops with multiple consumers so we're increasing a chance of cluster detection and number of
    quantized operators.
    """

    def disabling_flag(self) -> str:
        return "--no-duplicate-dequantize-linear"

    def run(self) -> None:
        # Find all DequantizeLinear nodes in the model
        if not self.contains_quantization_nodes():
            return

        # Compute consumer mapping
        tensor_consumers = defaultdict(list)
        for node in self.model.graph.node:
            for _input in node.input:
                tensor_consumers[_input].append(node)

        # Find dequantize ops with multiple consumers
        dequantize_nodes_multiple_consumers = []
        for index, node in enumerate(self.model.graph.node):
            if node.op_type == "DequantizeLinear":
                dequantize_consumers = tensor_consumers[node.output[0]]
                all_inputs_static = all([self.is_initializer(tensor) for tensor in node.input])

                if len(dequantize_consumers) > 1 and all_inputs_static:
                    dequantize_nodes_multiple_consumers.append((node, index, dequantize_consumers))

        # Do not run optimization for IR version lower than 10.
        # Old models tend to have initializers also in inputs collection
        # which causes issues.
        if self.model.ir_version < 10:
            if len(dequantize_nodes_multiple_consumers) > 0:
                logger.w("Model contains DequantizeLinear nodes with multiple consumers but model's IR version < 10. "
                         "This will lead to unnecessary Dequantize ops in produced TFLite model and some operators "
                         "will not run quantized fashion. Use model with IR version >= 10.")
            return

        # Process each DequantizeLinear node
        for (node, index, consumers) in dequantize_nodes_multiple_consumers:
            # Get the output tensor name
            dq_linear_output = node.output[0]
            dq_linear_output_tensor = self.get_value_info(dq_linear_output)

            # Model shapes aren't inferred. Skip
            if dq_linear_output_tensor is None:
                continue

            # Get the scale and zero_point tensor data
            input_data = self.try_get_tensor_data(node.input[0])
            scale_data = self.try_get_tensor_data(node.input[1])
            zero_point_data = self.try_get_tensor_data(node.input[2])

            # Replace only for per-tensor quantized nodes
            if scale_data.size != 1:
                continue

            # For each consumer, create a new DequantizeLinear node with unique tensor names
            for i, consumer in enumerate(consumers):

                # Make sure output of DequantizeLinear op is used only once as input of this specific consumer
                if list(consumer.input).count(dq_linear_output) != 1:
                    continue

                # Create unique names for scale and zero_point tensors
                input_name = self.validate_name(f"{node.input[0]}_duplicated_{i}")
                scale_name = self.validate_name(f"{node.input[1]}_duplicated_{i}")
                zero_point_name = self.validate_name(f"{node.input[2]}_duplicated_{i}")
                output_name = self.validate_name(f"{dq_linear_output}_duplicated_{i}")

                # Create new input tensor
                new_input = onnx.numpy_helper.from_array(input_data, input_name)
                self.add_initializer(new_input)

                # Create new scale tensor
                new_scale = onnx.numpy_helper.from_array(scale_data, scale_name)
                self.add_initializer(new_scale)

                # Create new zero_point tensor
                new_zero_point = onnx.numpy_helper.from_array(zero_point_data, zero_point_name)
                self.add_initializer(new_zero_point)

                # Create output tensor
                output_shape = [dim.dim_value for dim in dq_linear_output_tensor.type.tensor_type.shape.dim]
                output_tensor_value_info = make_tensor_value_info(
                    output_name,
                    dq_linear_output_tensor.type.tensor_type.elem_type,
                    output_shape)
                self.model.graph.value_info.append(output_tensor_value_info)

                # Create new DequantizeLinear node
                new_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs=[input_name, scale_name, zero_point_name],
                    outputs=[output_name]
                )

                self.model.graph.node.insert(index, new_node)

                # Wire new operator to consumer's input
                consumer_input_index = list(consumer.input).index(dq_linear_output)
                consumer.input[consumer_input_index] = output_name

            # Remove node and all tensors
            self.model.graph.node.remove(node)
            self.model.graph.value_info.remove(dq_linear_output_tensor)
            self.remove_initializer(node.input[0])
            self.remove_initializer(node.input[1])
            self.remove_initializer(node.input[2])

        onnx.checker.check_model(self.model)
