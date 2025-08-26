#
# Copyright 2023 Martin Pavella
# Copyright 2023-2025 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

import traceback

import flatbuffers as fb

# noinspection PyPackageRequirements
import google.protobuf.message
import numpy as np
import onnx.shape_inference

from onnx2quant import qdq_quantization
from onnx2tflite.src import logger, model_inspector, tensor_format_inference
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.conversion_context import ConversionContext
from onnx2tflite.src.converter.builder import model_builder
from onnx2tflite.src.converter.conversion import operator_converter, tensor_converter
from onnx2tflite.src.logger import BasicLoggingContext, Error, loggingContext
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model

# Models with a lower opset will not be converted.
MINIMUM_REQUIRED_OPSET = 7

# A warning message will be printed for models with a higher opset.
MAXIMUM_VERIFIED_OPSET = 22  # Corresponding to ONNX 1.17.


def _assert_supported_opset(model: onnx_model.ModelProto, conversion_context: ConversionContext) -> None:
    """Check that the onnx opset version of `model` is in the supported range."""
    if conversion_context.conversion_config.ignore_opset_version:
        return

    opset_map = {opset.domain: opset.version for opset in model.opset_imports}
    onnx_opset_version = opset_map.get("")
    if onnx_opset_version is None:
        logger.e(logger.Code.INVALID_ONNX_MODEL, "The provided ONNX model doesn't have a specified opset version.")

    if onnx_opset_version < MINIMUM_REQUIRED_OPSET:
        logger.e(logger.Code.NOT_IMPLEMENTED, f"This model uses opset {onnx_opset_version}. Conversion of ONNX models"
                                              f" with opset < {MINIMUM_REQUIRED_OPSET} is not supported. " +
                 logger.Message.IGNORE_OPSET_VERSION)

    if onnx_opset_version > MAXIMUM_VERIFIED_OPSET:
        logger.e(logger.Code.NOT_IMPLEMENTED, f"This model uses opset {onnx_opset_version}. Correct conversion of ONNX "
                                              f"models with opset > {MAXIMUM_VERIFIED_OPSET} is not guaranteed. " +
                 logger.Message.IGNORE_OPSET_VERSION)


def _convert(conversion_context: ConversionContext) -> tflite_model.Model:
    """Convert the ONNX model into an equivalent TFLite model.

    :param conversion_context: Conversion context with references to ONNX model, internal TFLite model and config.
    :return: Internal representation of the converted TFLite model.
    """
    o_model = conversion_context.onnx_inspector.model
    _assert_supported_opset(o_model, conversion_context)

    builder = conversion_context.tflite_builder
    conversion_config = conversion_context.conversion_config

    format_inference = tensor_format_inference.TensorFormatInference(o_model)
    format_inference.identify_tensor_formats()

    tensor_cvt = tensor_converter.TensorConverter(builder)
    tensor_cvt.convert_output_tensors(o_model.graph.outputs)
    tensor_cvt.convert_input_tensors(o_model.graph.inputs)
    tensor_cvt.convert_constant_tensors(o_model.graph.initializers)
    tensor_cvt.convert_internal_tensors(o_model.graph.value_info)

    if conversion_config.qdq_aware_conversion:
        qdq_clusters_recognizer = qdq_quantization.QDQClustersRecognizer(conversion_context.onnx_inspector)
        recognized_qdq_ops = qdq_clusters_recognizer.recognize_ops()
    else:
        recognized_qdq_ops = None

    operator_cvt = operator_converter.OperatorConverter(conversion_context, recognized_qdq_ops)
    operator_cvt.convert_operators(o_model.graph.nodes)

    return builder.finish()


def parse_symbolic_dimensions_mapping(mapped_symbolic_dimensions: list[str] | str) -> dict[str, int]:
    """Parse symbolic dimensions mapping into static as comma separated string with mapping
    or list of mappings. For both applies that mapping must be in format '<dim_name>:<dim_size>'
    for example 'batch:1'.

    :param mapped_symbolic_dimensions: Comma separated string or list of symbolic dimensions mapping.
    :return: Symbolic dimensions mapping parsed as a dictionary.
    """
    parsed_mapping = {}

    if isinstance(mapped_symbolic_dimensions, str):
        mapped_symbolic_dimensions = mapped_symbolic_dimensions.split(",")

    for mapping in mapped_symbolic_dimensions:
        mapping_details = mapping.split(":")

        if len(mapping_details) != 2 or not mapping_details[1].isdigit():
            raise Exception(f"Symbolic dimension mapping '{mapping}' in invalid format. Must be "
                            f"'<dim_name>:<dimension_size>' for example 'batch:1'.")
        parsed_mapping[mapping_details[0]] = int(mapping_details[1])

    return parsed_mapping


def parse_input_shape_mapping(mapped_input_shapes: list[str] | str) -> dict[str, tuple]:
    """Parse dynamic shapes mapping into static as semicolon separated string with mapping
    or list of mappings. For both applies that mapping must be in format
    '<input_name>:(<dim_0>,<dim_1>,...)', for example 'input_1:(1,3,224,224)'.

    :param mapped_input_shapes: Semicolon separated string or list of shape mappings.
    :return: Input shapes mapping parsed as a dictionary.
    """
    parsed_mapping = {}

    if isinstance(mapped_input_shapes, str):
        mapped_input_shapes = mapped_input_shapes.split(";")

    for mapping in mapped_input_shapes:
        mapping_details = mapping.split(":")

        input_shape = (mapping_details[1]
                       .replace("(", "")  # remove ( and )
                       .replace(")", "")
                       .split(","))

        if len(mapping_details) != 2 or not all([dim.isdigit() for dim in input_shape]):
            raise Exception(f"Input shape definition '{mapping}' in invalid format. Must be "
                            f"<dim_name>:(<dim_0>,<dim_1>,...) for example 'input_1:(1,3,224,224)'.")
        parsed_mapping[mapping_details[0]] = tuple([int(dim) for dim in input_shape])

    return parsed_mapping


def build_conversion_context(
        onnx_model: onnx_model.ModelProto,
        conversion_config: ConversionConfig | None = None,
        inferred_tensor_data: dict[str, np.ndarray] | None = None
) -> ConversionContext:
    """Build conversion context for converted ONNX model.

    :param onnx_model: ONNX model in ModelProto format.
    :param conversion_config: ConversionConfig instance with conversion arguments.
    :param inferred_tensor_data: Optional dictionary with tensor data inferred during shape inference.
    :return: Initialized ConversionContext instance.
    """
    if conversion_config is None:
        conversion_config = ConversionConfig()
    description = "doc:'" + onnx_model.doc_string + "' domain:'" + onnx_model.domain \
                  + "' producer:'" + onnx_model.producer_name + " " + onnx_model.producer_version + "'"

    tflite_builder = model_builder.ModelBuilder(3, description, conversion_config)
    onnx_inspector = model_inspector.ONNXModelInspector(onnx_model, inferred_tensor_data)

    context = ConversionContext(tflite_builder, onnx_inspector, conversion_config)

    return context


def convert_model(
        source: str | onnx.ModelProto,
        conversion_config: ConversionConfig | None = None,
) -> bytearray:
    """Convert an ONNX model into TFLite. Model could be provided through path to *.onnx file
    or directly as ModelProto object.

    :param source: Input *.onnx filepath to load the model from or ONNX model
            represented as ModelProto object.
    :param conversion_config: Configuration arguments for the conversion.

    At least one of 'optimization_whitelist' and 'optimization_blacklist' must be 'None'.  If both are 'None', all
     optimizations are applied.


    :return: Binary representation of the converted TFLite model.
    """
    if conversion_config is None:
        conversion_config = ConversionConfig()

    with loggingContext(BasicLoggingContext.GLOBAL):
        try:
            if isinstance(source, str):
                # Parse the ONNX file
                parsed_onnx_model = onnx.load(source)
            elif isinstance(source, onnx.ModelProto):
                # Use model directly
                parsed_onnx_model = source
            else:
                logger.e(logger.Code.INVALID_INPUT,
                         f"Cannot initialize ONNX model from object of type '{type(source)}'! "
                         f"Expected type 'onnx.ModelProto' or 'string'.")

            # Dictionary mapping names of ONNX tensors to their data, inferred by the shape inference.
            inferred_tensor_data = {}

            with loggingContext(BasicLoggingContext.SHAPE_INFERENCE):
                # Infer the shapes of internal tensors
                if conversion_config.skip_shape_inference:
                    try:
                        onnx.checker.check_model(parsed_onnx_model, full_check=True)
                    except Exception as e: # noqa: BLE001
                        logger.e(logger.Code.SHAPE_INFERENCE_ERROR,
                                 f"ONNX model's shapes not completely defined: {e!s}")
                else:
                    parsed_onnx_model = ModelShapeInference.infer_shapes(
                        parsed_onnx_model,
                        symbolic_dimensions_mapping=conversion_config.symbolic_dimensions_mapping,
                        input_shapes_mapping=conversion_config.input_shapes_mapping,
                        inferred_tensor_data=inferred_tensor_data,
                        generate_artifacts_after_failed_shape_inference=conversion_config.generate_artifacts_after_failed_shape_inference
                    )

            with loggingContext(BasicLoggingContext.ONNX_PARSER):
                # Initialize the internal ONNX model representation
                internal_onnx_model = onnx_model.ModelProto(parsed_onnx_model)

            conversion_context = build_conversion_context(internal_onnx_model, conversion_config, inferred_tensor_data)

            with loggingContext(BasicLoggingContext.OPERATOR_CONVERSION):
                # Convert the ONNX model to TFLite
                internal_tflite_model = _convert(conversion_context)

            with loggingContext(BasicLoggingContext.TFLITE_GENERATOR):
                # Serialize the internal TFLite model
                flatbuffer_builder = fb.Builder()
                internal_tflite_model.gen_tflite(flatbuffer_builder)
                flatbuffer_model = flatbuffer_builder.Output()
        except google.protobuf.message.DecodeError as e:
            logger.e(logger.Code.INVALID_INPUT, f"Failed to parse file '{source}'!", exception=e)
        except FileNotFoundError as e:
            logger.e(logger.Code.INVALID_INPUT, f"File '{source}' couldn't be found!", exception=e)
        except Error as e:
            # Just propagate the error
            raise e
        except Exception as e: # noqa: BLE001
            logger.d(f"Generic conversion exception caught ({type(e).__name__}). {traceback.format_exc()}")
            logger.e(logger.Code.INTERNAL_ERROR,
                     f"Unexpected internal error: {type(e).__name__}. Please report this issue.")

        return flatbuffer_model
