#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""onnx2tflite

This module provides a CLI for the converter of ONNX models to TFLite.
"""

import argparse
import logging
import ntpath
from argparse import Namespace

import onnx2tflite.src.logger as context_logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert

logger = logging.getLogger("onnx2tflite")
syslog = logging.StreamHandler()
syslog.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logger.addHandler(syslog)


def _get_user_choice_parser() -> argparse.ArgumentParser:
    """Return a parser, which handles options used to let the user provide additional information for the
    conversion. Without this information, the converter wouldn't be able to guarantee accurate conversion. This
    way, the user chooses to convert the model anyway, and the validity of the model is on the user's
    responsibility.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--guarantee-non-negative-indices", action=argparse.BooleanOptionalAction, default=False,
                        help="Guarantee that an 'indices' input tensor will always contain non-negative values. This "
                             "applies to operators: 'Gather', 'GatherND', 'OneHot' and 'ScatterND'.")
    parser.add_argument("--cast-int64-to-int32", action=argparse.BooleanOptionalAction, default=False,
                        help="Cast some nodes with type INT64 to INT32 when TFLite doesn't support INT64. Such nodes "
                             "are often used in ONNX to calculate shapes/indices, so full range of INT64 isn't "
                             "necessary. This applies to operators: `Abs` and `Div`.")
    parser.add_argument("--accept-resize-rounding-error", action=argparse.BooleanOptionalAction, default=False,
                        help="Accept the error caused by a different rounding approach of the ONNX 'Resize' and TFLite "
                             "'ResizeNearestNeighbor' operators, and convert the model anyway.")
    parser.add_argument("--skip-opset-version-check", action=argparse.BooleanOptionalAction, default=False,
                        help="Ignore the checks for supported opset versions of the ONNX model and try to convert it "
                             "anyway. This can result in an invalid output TFLite model.")

    return parser


def _get_conversion_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--allow-inputs-stripping", action=argparse.BooleanOptionalAction, default=True,
                        help="Model inputs will be removed if they are not necessary for inference and "
                             "their values are derived during the conversion.")
    parser.add_argument("--keep-io-tensors-format", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep the format of input and output tensors of the converted model the same, "
                             "as in the original ONNX model (NCHW).")
    parser.add_argument("--skip-shape-inference", action=argparse.BooleanOptionalAction, default=False,
                        help="Shape inference will be skipped before model conversion. This option can "
                             "be used only if model's shapes are fully defined. Defined shapes are necessary for "
                             "successful conversion.")
    parser.add_argument("--qdq-aware-conversion", action=argparse.BooleanOptionalAction, default=True,
                        help="Quantized QDQ model with QDQ pairs (Q-Ops created by QDQ quantizer) will be "
                             "converted into optimized variant with QDQ pairs represented as tensors' "
                             "quantization parameters.")
    parser.add_argument("-s", "--symbolic-dimension-into-static", dest="symbolic_dimensions_mapping",
                        type=str, action="extend", nargs="*",
                        help="Change symbolic dimension in model to static (fixed) value. Provided mapping must "
                             "follow this format '<dim_name>:<dim_size>', for example 'batch:1'. This argument "
                             "can be used multiple times.")
    parser.add_argument("-m", "--set-input-shape", dest="input_shapes_mapping",
                        type=str, action="extend", nargs="*",
                        help="Override model input shape. Provided mapping must follow format '<dim_name>:(<dim_0>,"
                             "<dim_1>,...)', for example 'input_1:(1,3,224,224)'. This argument can be used multiple "
                             "times.")
    parser.add_argument("--dont-skip-nodes-with-known-outputs", action=argparse.BooleanOptionalAction, default=False,
                        help="Sometimes it is possible to statically infer the output data of some nodes. These nodes "
                             "will then not be a part of the output model. This flag will force the converter to keep "
                             "them in anyway.")

    parser.add_argument("--allow-select-ops", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow the converter to use the `SELECT_TF_OPS` operators, which require Flex delegate at "
                             "runtime.")

    parser.add_argument("--generate-artifacts-after-failed-shape-inference", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="If the shape inference fails or is incomplete, generate the partly inferred ONNX model as"
                             " `sym_shape_infer_temp.onnx`.")

    return parser


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        prog="onnx2tflite", parents=[_get_conversion_parser(), _get_user_choice_parser()],
        description="""
            Convert a '.onnx' DNN model to an equivalent '.tflite' model.
            By default the output '.tflite' file will be generated in the current 
            working directory and have the same name as the input '.onnx' file.
        """
    )
    parser.add_argument("onnx_file", help="Path to ONNX (*.onnx) model.")
    parser.add_argument("-o", "--output", type=str, required=False, metavar="out",
                        help="Path to output '.tflite' file.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print detailed information related to conversion process.")

    return parser.parse_args()


def run_conversion() -> None:
    """Create argument parser"""
    args = parse_arguments()
    output_tflite = args.output

    if output_tflite is None:
        # Default output file has the same name as input, but with a different file
        # extension and is in the current working directory

        file_name = ntpath.basename(args.onnx_file)  # Get filename
        file_name = ntpath.splitext(file_name)[0]  # Remove '.onnx' extension
        output_tflite = file_name + ".tflite"

    if args.verbose:
        # Print all logging messages
        context_logger.MIN_OUTPUT_IMPORTANCE = context_logger.MessageImportance.INFO
        logger.setLevel(logging.DEBUG)

    if args.symbolic_dimensions_mapping:
        assert isinstance(args.symbolic_dimensions_mapping, list)
        try:
            args.symbolic_dimensions_mapping = convert.parse_symbolic_dimensions_mapping(
                args.symbolic_dimensions_mapping)
        except Exception as err:  # noqa: BLE001
            context_logger.e(context_logger.Code.INVALID_INPUT, str(err))

    if args.input_shapes_mapping:
        assert isinstance(args.input_shapes_mapping, list)
        try:
            args.input_shapes_mapping = convert.parse_input_shape_mapping(args.input_shapes_mapping)
        except Exception as err:  # noqa: BLE001
            context_logger.e(context_logger.Code.INVALID_INPUT, str(err))

    """ Convert the model """
    binary_tflite_model = convert.convert_model(args.onnx_file, ConversionConfig(vars(args)))

    with open(output_tflite, "wb") as f:
        f.write(binary_tflite_model)

    logger.info(f"Successfully converted '{args.onnx_file}' model to '{output_tflite}'.")


def run_conversion_wrapper() -> None:
    try:
        run_conversion()
    except context_logger.Error as e:
        exit(e.error_code.value)


if __name__ == "__main__":
    run_conversion_wrapper()
