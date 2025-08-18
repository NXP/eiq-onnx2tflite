#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""
    get_ops_from_model

    Script to get operators from model.
    Outputs a csv line with model path and operators.
"""

import onnx
import argparse
from onnx2tflite.src.model_shape_inference import ModelShapeInference
import onnx2tflite.src.onnx_parser.onnx_model as onnx_model
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type
import numpy as np
import onnx2tflite.src.logger as logger

def find_ops_with_output_tensor(model: onnx_model.ModelProto, tensor_name: str) -> [onnx_model.NodeProto]:
    """
    Finds all ops in the model, whose tensor_name is in operators output
    :return: List of operators
    """
    return [node for node in model.graph.nodes if tensor_name in node.outputs]



def find_ops_with_input_tensor(model: onnx_model.ModelProto, tensor_name: str) -> [onnx_model.NodeProto]:
    """
    Finds ops in the model, where tensor with tensor_name is operator's input tensor.
    :return: List of operators
    """
    return [node for node in model.graph.nodes if tensor_name in node.inputs]

def find_tensor(model: onnx_model.ModelProto, tensor_name: str):
    """
    Search for tensor with 'tensor_name' in model. In particular in inputs, initializers and value_info field.
    Raises error if no or multiple tensors with same name is found. The later case should not happen.
    :return: Tensor
    """
    tensors = ([i for i in model.graph.inputs       if i.name == tensor_name] +
              #[i for i in model.graph.outputs      if i.name == tensor_name] +  # The output tensors are also part of value_info field?
              [i for i in model.graph.initializers if i.name == tensor_name] +
              [i for i in model.graph.value_info   if i.name == tensor_name])
    if not tensors:
        logger.w('No tensor found')
    if len(tensors) != 1:
        logger.e(logger.Code.INVALID_INPUT, 'Multiple tensors found')
    return list(tensors)[0]


def get_tensor_type(tensor) -> np.ScalarType:
    """
    Get data type from tensor.
    """
    if isinstance(tensor, onnx_model.onnx_tensor.TensorProto):
        return to_numpy_type(tensor.data_type)
    if isinstance(tensor, onnx_model.ValueInfoProto):
        return to_numpy_type(tensor.type.tensor_type.elem_type)
    logger.e(logger.Code.INTERNAL_ERROR, 'Cannot determine tensor datatype')



def tensor_from_dequantize_op(model, tensor_name):
    """
    Check if tensor with tensor_name is outputed by a DequantizeLinear operator.
    :return: True if from DequantizeLinear, False otherwise
    """
    output_nodes = find_ops_with_output_tensor(model, tensor_name)
    return len(output_nodes) == 1 and output_nodes[0].op_type == "DequantizeLinear"


def tensor_from_already_quantized_op(model, quantized_ops, tensor_name):
    """
    Check if tensor is outputed by an operator which is already quantized.
    :param model:
    :param quantized_ops: List of quantized operator.
    :param tensor_name:
    :return:
    """
    prev_nodes = find_ops_with_output_tensor(model, tensor_name)
    return len(prev_nodes) == 1 and prev_nodes[0] in quantized_ops

def tensor_to_quantize_op(tensor_name, model):
    """
    Check if tensor goes to single QuantizeLinear operation.
    :param tensor_name:
    :param model:
    :return:
    """
    input_nodes = find_ops_with_input_tensor(model, tensor_name)
    return len(input_nodes) == 1 and input_nodes[0].op_type == "QuantizeLinear"

def tensor_to_quantize_op2(tensor_name, model):
    """
    Check if tensor goes to at least one QuantizeLinear operator.
    :param tensor_name:
    :param model:
    :return:
    """
    input_nodes = find_ops_with_input_tensor(model, tensor_name)
    return any(i.op_type == "QuantizeLinear" for i in input_nodes)


def is_node_input_quantized(model, node):
    """
    Determines if node/operator predecessors are quantized based on these conditions:
    For all input tensors, each tensor:
    - comes from DequatizeLinearNode OR
    - type is not float
    - special cases for some operators (will identify as implemented into the converter). E.g. for Clip, the 'min' and
      'max' inputs can be float.
    :param model:
    :param node:
    :param quantized_ops:
    :return:
    """
    input_tensors = node.inputs
    if node.op_type == "Clip":
        input_tensors = [node.inputs[0]]

    input_tensors_quantized = [tensor_from_dequantize_op(model, tensor) or
                               get_tensor_type(find_tensor(model, tensor)) != np.float32
                               for tensor in input_tensors]
    return all(input_tensors_quantized)

def is_node_input_quantized_3(model, node, quantized_ops):
    """
    Determines if node/operator predecessors are quantized based on these conditions:
    For all input tensors, each tensor:
    - comes from DequatizeLinearNode OR
    - comes from operator already identified as quantized OR
    - type is not float
        - special cases for some operators (will identify as implemented into the converter). E.g. for Clip, the 'min' and
      'max' inputs can be float.
    :param model:
    :param node:
    :param quantized_ops:
    :return:
    """
    input_tensors = node.inputs
    if node.op_type == "Clip":
        input_tensors = [node.inputs[0]]

    input_tensors_quantized = [tensor_from_dequantize_op(model, tensor) or
                               tensor_from_already_quantized_op(model, quantized_ops, tensor) or
                               get_tensor_type(find_tensor(model, tensor)) != np.float32
                               for tensor in input_tensors]
    return all(input_tensors_quantized)

def is_node_output_quantized(model, node):
    """
    Check if the node (operators) outputs are all quantized. It is determined by this conditions:
    For all output tensors, the tensor:
    - goes to QuantizeLinear Operator OR
    - tensor data type is not float32
    """
    output_tensors = node.outputs
    output_tensors_quantized = [tensor_to_quantize_op(tensor, model) or
                                get_tensor_type(find_tensor(model, tensor)) != np.float32
                                for tensor in output_tensors]
    return all(output_tensors_quantized)

def is_node_quantized(model, node):
    """
    Check if operator can be considered as quantized, i.e. all input and output tensors are quantized.
    :param model:
    :param node:
    :return:
    """
    return is_node_input_quantized(model, node) and is_node_output_quantized(model, node)


def is_node_output_quantized2(model, node):
    """
    Check if node output tensors are quantized. Uses those conditions:
    For all tensors, each tensor:
    - goes to at least 1 QuantizeLinear op OR
    - output is not float
    """
    output_tensors = node.outputs
    output_tensors_quantized = [tensor_to_quantize_op2(tensor, model) or
                                get_tensor_type(find_tensor(model, tensor)) != np.float32
                                for tensor in output_tensors]
    return all(output_tensors_quantized)


def is_node_quantized2(model, node):
    return is_node_input_quantized(model, node) and is_node_output_quantized2(model, node)


def get_node_qdq_cluster(model, node):
    """
    Collects node QDQ Cluster. The QDQ cluster consist of preceeding DequantizeLinear ops and following QuantizeLinear.
    """
    input_tensors = node.inputs
    output_tensors = node.outputs

    dequantize_nodes = []
    quantize_nodes = []

    for i in input_tensors:
        dequantize_nodes.extend(find_ops_with_output_tensor(model, i))
    for o in output_tensors:
        quantize_nodes.extend(find_ops_with_input_tensor(model, o))

    return dequantize_nodes + quantize_nodes


def remove_qdq_scheme_ops(model, maybe_qdq_scheme_ops, quantized_ops, dequant_and_quant_ops):
    """
    Remove QuantizeLinear and DequantizeLinear Operators which are used only in QDQ pattern:
    - QuantizeLinear operator is part of QDQ pattern if its input comes from quantized operator
    - DequantizeLinear operator is part of QDQ patters if its output goes to quantized operator
    :param model: Parsed ONNX model object
    :param maybe_qdq_scheme_ops: Collection of QuantizeLinear and DequantizeLinear operators which are candidates to QDQ pattern
    :param quantized_ops: Collection of quantized operators
    :param dequant_and_quant_ops: Collection of remaining QuantizeLinear and DequantizeLinear ops in the ONNX model.
    :return: Updated collection of remaining QuantizeLinear and DequantizeLinear operators.
    """
    for op in maybe_qdq_scheme_ops:
        if op.op_type == "QuantizeLinear" and all(i in quantized_ops for i in
                                                  find_ops_with_output_tensor(model, op.inputs[0])):
            logger.d(f"REMOVING QuantizeOp: {op.name}")
            dequant_and_quant_ops.pop(op.name)
        elif op.op_type == "DequantizeLinear" and all(i in quantized_ops for i in
                                                      find_ops_with_input_tensor(model, op.outputs[0])):
            logger.d(f"REMOVING DequantizeOp: {op.name}")
            dequant_and_quant_ops.pop(op.name)
        else:
            pass
    return dequant_and_quant_ops

def qdq_init(model, quantized_ops=[], float_ops=[], dequantize_and_quantize_ops={}):
    """
    Initialise the QDQ algorithm variables:
    - collect all QuantizeLinear and DequantizeLinear ops
    - all ops are float initially.
    :return:
    """
    for node in model.graph.nodes:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            dequantize_and_quantize_ops[node.name] = node
        else:
            float_ops.append(node)
    return quantized_ops, float_ops, dequantize_and_quantize_ops

def qdq_strict_pattern(model, quantized_ops, float_ops, dequantize_and_quantize_ops):
    """
    Algorith to find quantized nodes following strictly the QDQ pattern:
    - The node inputs come from DequantizeLinear Operator
    - The node outputs goes to QuantizeLinear Operator
    :param model:
    :param quantized_ops:
    :param float_ops:
    :param dequantize_and_quantize_ops:
    :return:
    """
    maybe_qdq_scheme_ops = set()
    new_quantized_ops = []

    # For all nodes determine if the operation can be treated and quantized operator or float operator
    for node in float_ops:
        # if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
        #     continue
        if is_node_quantized(model, node):
            logger.d(f'FOUND Quantized Operator {node.op_type}\t{node.name}')
            maybe_qdq_scheme_ops.update(get_node_qdq_cluster(model, node))
            #  ******************************* Note ********************************************************************
            # We cannot yet remove operators from 'dequant_and_quant_ops' as this is a dictionary and we want to be safe,
            # hence we require the node removed from the dictionary has to exist there.
            # We assume the Quantize or Dequantize operator might be:
            #  - used in QDQ scheme
            #  - used in QDQ scheme but shared in multiple qdq clusters, e.g. consider a residual pattern:
            #                       <DequantizeLinear_1>
            #                        |          <Conv>
            #                        |     <QuantizeLinear>
            #                        |   <DequantizeLinear_2>
            #                          <Add>
            #                   The 'DequantizeLinear_1 is in qdq clusters for Conv and Add operator, hence would be removed
            #                   twice from the dictionary.
            #  - used in QDQ scheme and also to move from quantized compute to float, e.g.:
            #                       <DequantizeLinear_1>
            #                      <Shape>     <GlobalAveragePool>
            #                      <Gather>    <QuantizeLinear>
            #                        ...       <DequantizeLinear_2>
            #                            <Reshape>
            # **********************************************************************************************************
            new_quantized_ops.append(node)
        # else:
        #     #print(f'Node {node.name}: NOT quantized')
        #     float_ops.append(node)

    quantized_ops.extend(new_quantized_ops)
    for op in new_quantized_ops: float_ops.remove(op)
    dequantize_and_quantize_ops = remove_qdq_scheme_ops(model, maybe_qdq_scheme_ops, quantized_ops, dequantize_and_quantize_ops)

    return quantized_ops, float_ops, dequantize_and_quantize_ops

def qdq_strict_pattern2(model, quantized_ops, float_ops, dequantize_and_quantize_ops):
    """
    Algorithm to find quantized nodes following strictly the QDQ pattern with following difference:
    - The node outputs goes to AT LEAST ONE QuantizeLinear Operator. This convers residual pattern found in mobilenetv2-12-qdq.onnx,
    where the Add ops joining the residual outputs was quantized only once.
    :param model:
    :param quantized_ops:
    :param float_ops:
    :param dequantize_and_quantize_ops:
    :return:
    """
    maybe_qdq_scheme_ops = set()
    new_quantized_ops = []

    # For all nodes determine if the operation can be treated and quantized operator or float operator
    for node in float_ops:
        # if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
        #     continue
        if is_node_quantized2(model, node):
            logger.d(f'FOUND Quantized Operator {node.op_type}\t{node.name}')
            maybe_qdq_scheme_ops.update(get_node_qdq_cluster(model, node))
            #  ******************************* Note ********************************************************************
            # We cannot yet remove operators from 'dequant_and_quant_ops' as this is a dictionary and we want to be safe,
            # hence we require the node removed from the dictionary has to exist there.
            # We assume the Quantize or Dequantize operator might be:
            #  - used in QDQ scheme
            #  - used in QDQ scheme but shared in multiple qdq clusters, e.g. consider a residual pattern:
            #                       <DequantizeLinear_1>
            #                        |          <Conv>
            #                        |     <QuantizeLinear>
            #                        |   <DequantizeLinear_2>
            #                          <Add>
            #                   The 'DequantizeLinear_1 is in qdq clusters for Conv and Add operator, hence would be removed
            #                   twice from the dictionary.
            #  - used in QDQ scheme and also to move from quantized compute to float, e.g.:
            #                       <DequantizeLinear_1>
            #                      <Shape>     <GlobalAveragePool>
            #                      <Gather>    <QuantizeLinear>
            #                        ...       <DequantizeLinear_2>
            #                            <Reshape>
            # **********************************************************************************************************
            new_quantized_ops.append(node)
        # else:
        #     #print(f'Node {node.name}: NOT quantized')
        #     float_ops.append(node)

    quantized_ops.extend(new_quantized_ops)
    for op in new_quantized_ops: float_ops.remove(op)
    dequantize_and_quantize_ops = remove_qdq_scheme_ops(model, maybe_qdq_scheme_ops, quantized_ops, dequantize_and_quantize_ops)

    return quantized_ops, float_ops, dequantize_and_quantize_ops

def qdq_ops_sequence(model, quantized_ops, float_ops, dequantize_and_quantize_ops):
    TENSOR_TRANSPOSITION_OPS = ["Relu", "MaxPool", "Clip"]
    new_quantized_ops = []

    def _find_cluster(start_node,  compute_found=False):
        success = False
        if start_node.op_type == "QuantizeLinear":  # recursion stop. We found a closing QuantizeLinear on path
            return True, []
        elif get_tensor_type(find_tensor(model, start_node.outputs[0])) != np.float32:  # recursion stop. We found a operator whose output is not float32,
                                                                                        # this is considered as quantized op and need to include it in return sequence
            return True, [start_node]
        elif start_node.op_type not in TENSOR_TRANSPOSITION_OPS: # We have a compute op on path.
            if compute_found:  # Recursion stop. There is a second compute operator on path
                return False, []
            else:
                compute_found = True
        if len(start_node.outputs) != 1:  # Only consider sequence where all ops have only single output
            return False, []
        next_ops = find_ops_with_input_tensor(model, start_node.outputs[0])
        sequence = [start_node]
        for i in next_ops:
            ret_success, ret_sequence = _find_cluster(i, compute_found)
            if ret_success:
                sequence.extend(ret_sequence)
                success = True
        return success, sequence

    for op in float_ops:
        if op in new_quantized_ops:
            continue
        if is_node_input_quantized_3(model, op, quantized_ops + new_quantized_ops):  # start of potential sequence:
            sucess, cluster = _find_cluster(op)
            if sucess:
                logger.d("FOUND a cluster:")
                for o in cluster:
                    logger.d(f"OpType: {o.op_type}\tName: {o.name}")
                new_quantized_ops.extend(cluster)

    quantized_ops.extend(new_quantized_ops)
    for i in new_quantized_ops:
        float_ops.remove(i)
    maybe_qdq_scheme_ops = {op for new_q_op in new_quantized_ops for op in get_node_qdq_cluster(model, new_q_op)}
    dequantize_and_quantize_ops = remove_qdq_scheme_ops(model, maybe_qdq_scheme_ops, quantized_ops, dequantize_and_quantize_ops)

    return quantized_ops, float_ops, dequantize_and_quantize_ops



def _all_unique(x):
    return len(x) == len(set(x))


def print_summary(quantized_ops, float_ops, dequantize_and_quantize_ops, algorightm=""):
    print("")
    print("*"*30, "REPORT", "*"*42)
    if algorightm:
        print(f"Algorightm: {algorightm}")
    print("Quantized Ops", "-"*60)
    for i in quantized_ops: print(f"OpType: {i.op_type}\tName:{i.name}")
    print("Float Ops", "-"*60)
    for i in float_ops: print(f"OpType: {i.op_type}\tName:{i.name}")
    print("Remaining Quantize and Dequantize Ops", "-" * 40)
    for i in dequantize_and_quantize_ops.values(): print(f"OpType: {i.op_type}\t\tName:{i.name}")
    print(f"*" * 80)


def onnx_model_sanity_check(onnx_model):
    # Check if all nodes names are unique:
    node_names = [node.name for node in onnx_model.graph.nodes]
    print(f"All Node Names unique: {_all_unique(node_names)}")

    input_names = [i.name for i in onnx_model.graph.inputs]
    print(f"All Input Names unique: {_all_unique(input_names)}")
    output_names = [o.name for o in onnx_model.graph.outputs]
    print(f"All Output Names unique: {_all_unique(output_names)}")
    tensor_names = [t.name for t in onnx_model.graph.value_info]
    print(f"All tensor Names unique: {_all_unique(tensor_names)}")
    static_tensor_names = [t.name for t in onnx_model.graph.initializers]
    print(f"All static tensor Names unique: {_all_unique(static_tensor_names)}")
    for i in input_names:
        print(f"Tensor type: {get_tensor_type(find_tensor(onnx_model, i))}\t{i}")
    for i in output_names:
        print(f"Tensor type: {get_tensor_type(find_tensor(onnx_model, i))}\t{i}")
    for i in tensor_names:
        print(f"Tensor type: {get_tensor_type(find_tensor(onnx_model, i))}\t{i}")
    for i in static_tensor_names:
        print(f"Tensor type: {get_tensor_type(find_tensor(onnx_model, i))}\t{i}")

    return (_all_unique(input_names) and _all_unique(output_names) and
            _all_unique(tensor_names) and _all_unique(static_tensor_names)
            )


if __name__ == "__main__":
    logger.MIN_OUTPUT_IMPORTANCE = logger.MessageImportance.DEBUG
    parser = argparse.ArgumentParser(description='Utility to analyze QDQ Quantized ONNX model.')
    parser.add_argument("onnx_file")

    args = parser.parse_args()

    parsed_onnx_model = onnx.load(args.onnx_file)
    parsed_onnx_model = ModelShapeInference.infer_shapes(parsed_onnx_model)
    internal_onnx_model = onnx_model.ModelProto(parsed_onnx_model, init_node_attributes=False)

    if not onnx_model_sanity_check(internal_onnx_model):
        exit(-1)

    # Get Quantized Ops based on individual algorithms:
    quantized_ops, float_ops, dequant_and_quant_ops = qdq_init(internal_onnx_model)
    print_summary(quantized_ops, float_ops, dequant_and_quant_ops, algorightm="qdq_init")

    quantized_ops, float_ops, dequant_and_quant_ops = qdq_strict_pattern(internal_onnx_model, quantized_ops, float_ops, dequant_and_quant_ops)
    print_summary(quantized_ops, float_ops, dequant_and_quant_ops, algorightm="qdq_strict_pattern")

    quantized_ops, float_ops, dequant_and_quant_ops = qdq_strict_pattern2(internal_onnx_model, quantized_ops, float_ops, dequant_and_quant_ops)
    print_summary(quantized_ops, float_ops, dequant_and_quant_ops, algorightm="qdq_strict_pattern2")

    quantized_ops, float_ops, dequant_and_quant_ops = qdq_ops_sequence(internal_onnx_model, quantized_ops, float_ops, dequant_and_quant_ops)
    print_summary(quantized_ops, float_ops, dequant_and_quant_ops, algorightm="qdq_sequence")
    
