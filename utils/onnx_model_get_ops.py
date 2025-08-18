#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""
    get_ops_from_model

    Script to get operators from model.
    Outputs a csv line with model path and operators.
"""

import argparse

import onnx

import onnx2tflite.src.onnx_parser.onnx_model

SUPPORTED_ONNX_OPERATORS = onnx2tflite.src.onnx_parser.onnx_model.NodeProto.op_type_to_attribute_constructor_map.keys()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility to investigate operators from given model. Generates one '
                                                 'line, comma separated values with model path and requested items.')
    parser.add_argument("onnx_file")
    parser.add_argument("--show", dest="show", required=False, choices=['all', 'unsupported'],
                        default='all', help="What operators to show. Valid arguments are 'all' or 'unsupported'. "
                                            "Default is 'all'. The output is informative only. Only compares "
                                            "the set of implemented operators and the operators present in the model "
                                            "Does not evaluates if the specific variant of the operator is convertible "
                                            "or not")
    parser.add_argument('--all-ops-supported', dest='all_ops_supported', required=False, action='store_true',
                        default=False, help="Print True/False as first item before the model path, informing if all "
                                            "the operators present in the model are supported by the converter")

    args = parser.parse_args()

    onnx_model = onnx.load(args.onnx_file)

    onnx_model_ops = set()
    for node in onnx_model.graph.node:
        onnx_model_ops.add(node.op_type)

    missing_ops = set(list(onnx_model_ops)) - set(SUPPORTED_ONNX_OPERATORS)
    all_ops_supported = not bool(missing_ops)

    if args.all_ops_supported:
        print(f'{all_ops_supported},', end="")

    to_print = onnx_model_ops
    if args.show == 'unsupported':
        to_print = missing_ops
    print(f'{args.onnx_file}', *sorted(list(to_print)), sep=",")
