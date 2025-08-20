#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import argparse
import logging
import os.path
import traceback
from functools import cache

# noinspection PyPackageRequirements
import google.protobuf.message
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader

from onnx2quant.qdq_quantization import QDQQuantizer
from onnx2quant.quantization_config import PER_CHANNEL_DEFAULT, QuantizationConfig
from onnx2tflite.src import logger as context_logger
from onnx2tflite.src.converter import convert
from onnx2tflite.src.logger import BasicLoggingContext, Error, loggingContext

logger = logging.getLogger("onnx2tflite")
syslog = logging.StreamHandler()
syslog.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logger.addHandler(syslog)


def _get_preprocessing_parser():
    """Return a parser which handles options used by the pre-processing stage that comes before quantization."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--replace-div-with-mul", action=argparse.BooleanOptionalAction, default=True,
                        help="Replace some `Div` operators with `Mul`. `Div` doesn't support int8 quantization in "
                             "TFLite so this is replacement can avoid having to compute `Div` in float32.")
    parser.add_argument("--replace-constant-with-static-tensor", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Remove `Constant` nodes and directly assign static data to their output tensors.")

    return parser


def _parse_arguments():
    parser = argparse.ArgumentParser(
        prog="onnx2quant",
        description="""
            Quantize ONNX model in 'TFLite conversion optimized way'. This tool
            produces QDQ model with per-tensor quantization and INT8 activations.
            Some operators can be QDQ quantized even if there isn't quantized variant 
            in ONNX but TFLite supports quantized version of this specific operator.
        """,
        parents=[_get_preprocessing_parser()]
    )
    parser.add_argument("onnx_model", help="Path to input ONNX '*.onnx' model.")
    parser.add_argument("-o", "--output", type=str, required=False,
                        help="Path to the resulting quantized ONNX model. (default: '<input_model_name>_quant.onnx')")
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction,
                        required=False, default=PER_CHANNEL_DEFAULT,
                        help="Quantize some weight tensors per-channel instead of per-tensor. This should result in a "
                             "higher accuracy.")
    parser.add_argument("-l", "--allow-opset-10-and-lower", action=argparse.BooleanOptionalAction,
                        required=False, default=False,
                        help="Allow quantization of models with opset version 10 and lower. Quantization of such models "
                             "can produce invalid models because opset is forcefully updated to version 11. This applies "
                             "especially to models with operators: Clip, Dropout, BatchNormalization and Split.")
    parser.add_argument("-c", "--calibration-dataset-mapping", dest="calibration_dataset_mapping",
                        type=str, action="extend", nargs="+", required=True,
                        help="Mapping between model input and calibration dataset directory with *.npy files. Value must "
                             "be in format '<input_name>;<path_to_dir>', for example 'input_1;data_3_224/'. Argument "
                             "can be used multiple times to specify multiple inputs for the model. In case model "
                             "has semicolon in input tensor's name, it has to be renamed.")
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
    parser.add_argument("--generate-artifacts-after-failed-shape-inference", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="If the shape inference fails or is incomplete, generate the partially inferred ONNX model "
                             "as `sym_shape_infer_temp.onnx`.")

    return parser.parse_args()


def _parse_calibration_dataset_mapping(mapped_calibration_dataset: list[str] | str) -> dict[str, str]:
    """Parse calibration dataset as comma separated string with mapping or list of mappings. For both applies
    that mapping must be in format '<input_name>;<path_to_dir>' for example 'input_1;data_3_224/'.

    :param mapped_calibration_dataset: Comma separated string or list of calibration dataset mapping.
    :return: Input name to calibration dataset dir mapping parsed as a dictionary.
    """
    parsed_mapping = {}

    if isinstance(mapped_calibration_dataset, str):
        mapped_calibration_dataset = mapped_calibration_dataset.split(",")

    for mapping in mapped_calibration_dataset:
        mapping_details = mapping.split(";")

        if len(mapping_details) != 2:
            context_logger.e(context_logger.Code.INVALID_INPUT,
                             f"Calibration dataset mapping '{mapping}' in invalid format. Must be "
                             f"'<input_name>;<path_to_dataset>' for example 'input_1;data_3_224/'.")
        parsed_mapping[mapping_details[0]] = mapping_details[1]

    return parsed_mapping


def _quantize_model(onnx_model: onnx.ModelProto, output_onnx_model_path, args: dict):
    """Create QDQ quantized model based on data defined by calibration dataset.

    :param onnx_model: ONNX model in ModelProto format.
    :param output_onnx_model_path: Path where final quantized model should be saved.
    :param args: Quantization arguments as dict provided via CLI.
    """
    calibration_data_reader = NpyCalibrationDataReader(args["calibration_dataset_mapping"])
    quantization_config = QuantizationConfig(calibration_data_reader, args)
    quantized_model = QDQQuantizer().quantize_model(onnx_model, quantization_config=quantization_config)

    onnx.save_model(quantized_model, output_onnx_model_path)


class NpyCalibrationDataReader(CalibrationDataReader):

    def __init__(self, calibration_dataset_mapping: dict[str, str]):
        """CalibrationDataReader for data saved as numpy arrays (*.npy files).

        :param calibration_dataset_mapping: Mapping between model inputs and directories with npy files.
        """
        self.input_data_paths = []
        self.input_names = []
        self.iterator_pos = 0

        for input_name, dir_path in calibration_dataset_mapping.items():
            self.input_names.append(input_name)

            filepaths = self._get_filepaths(dir_path)
            if len(filepaths) == 0:
                context_logger.e(context_logger.Code.INVALID_INPUT,
                                 f"No *.npy files found in directory '{dir_path}'")

            self.input_data_paths.append(filepaths)

        input_lens = [len(dataset) for dataset in self.input_data_paths]

        if len(set(input_lens)) != 1:
            context_logger.e(context_logger.Code.INVALID_INPUT,
                             "Input dataset dirs don't contain same number of *.npy files.")

        self.dataset_length = input_lens[0]

    def _get_filepaths(self, dir_path) -> list[str]:
        """Find all *.npy files in directory and return full file paths as list.

        :param dir_path: Searched directory.
        :return: List of paths to *.npy files.
        """
        if not os.path.isdir(dir_path):
            context_logger.e(context_logger.Code.INVALID_INPUT, f"Path '{dir_path}' is not a directory.")

        return [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(".npy")]

    @staticmethod
    @cache
    def _load_file(npy_file_path) -> np.ndarray:
        """Load *.npy file and cache it till we use all the memory.

        :param npy_file_path: Path to *.npy file.
        :return: Loaded file as numpy array.
        """
        return np.load(npy_file_path)

    def get_next(self) -> dict | None:
        if self.iterator_pos >= self.dataset_length:
            return None

        result = {}
        # Load input sample for every input of the model
        for x in range(len(self.input_names)):
            result[self.input_names[x]] = self._load_file(self.input_data_paths[x][self.iterator_pos])

        self.iterator_pos += 1

        return result

    def rewind(self) -> None:
        self.iterator_pos = 0


def run_quantization() -> None:
    """Create argument parser"""
    args = _parse_arguments()

    output_onnx_model_path = args.output
    if output_onnx_model_path is None:
        model_name = os.path.basename(args.onnx_model)
        output_onnx_model_path = model_name[:-5] + "_quant.onnx"

    with loggingContext(BasicLoggingContext.GLOBAL):
        try:
            if args.symbolic_dimensions_mapping:
                assert isinstance(args.symbolic_dimensions_mapping, list)
                args.symbolic_dimensions_mapping = convert.parse_symbolic_dimensions_mapping(
                    args.symbolic_dimensions_mapping)

            if args.input_shapes_mapping:
                assert isinstance(args.input_shapes_mapping, list)
                args.input_shapes_mapping = convert.parse_input_shape_mapping(args.input_shapes_mapping)

            assert isinstance(args.calibration_dataset_mapping, list)
            args.calibration_dataset_mapping = _parse_calibration_dataset_mapping(args.calibration_dataset_mapping)
        except Exception as e:
            context_logger.e(context_logger.Code.INVALID_INPUT,
                             f"Invalid input error ({type(e).__name__}). {traceback.format_exc()}")

        try:
            onnx_model = onnx.load(args.onnx_model)

            _quantize_model(onnx_model, output_onnx_model_path, vars(args))
        except google.protobuf.message.DecodeError as e:
            context_logger.e(context_logger.Code.INVALID_INPUT, f"Failed to parse file '{args.onnx_model}'!",
                             exception=e)
        except FileNotFoundError as e:
            context_logger.e(context_logger.Code.INVALID_INPUT, f"File '{args.onnx_model}' couldn't be found!",
                             exception=e)
        except Error as e:
            # Just propagate the error
            raise e
        except Exception as e:
            context_logger.e(context_logger.Code.INTERNAL_ERROR,
                             f"Internal error ({type(e).__name__}). {traceback.format_exc()}")


if __name__ == "__main__":
    try:
        run_quantization()
    except context_logger.Error as e:
        exit(e.error_code.value)
