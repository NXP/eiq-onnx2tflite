#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import os
import pathlib
import shutil

import numpy as np
import onnx
import onnxruntime
import pandas as pd
import tensorflow as tf
from onnxruntime.quantization import CalibrationDataReader, QuantFormat
from tqdm import tqdm

from onnx2quant.qdq_quantization import QDQQuantizer
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.model_shape_inference import ModelShapeInference
from tests.executors import OnnxExecutor, TFLiteExecutor

_ARTIFACTS_DIR = pathlib.Path(__file__).parent.parent.joinpath("artifacts")
_IMAGENET_SAMPLES = os.path.join(_ARTIFACTS_DIR, "inputs", "imagenet-sample-images")
_PYTORCH_VISION_MODELS = os.path.join(_ARTIFACTS_DIR, "models", "Pytorch_Model_Zoo", "onnx_export", "vision_cnn",
                                      "classification")


def load_image(filename):
    img = tf.io.read_file(str(filename))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 224, 224)
    img = np.array(img, dtype=np.int64)
    img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode='torch')
    img = img.transpose([2, 0, 1])
    img = np.expand_dims(img, 0)

    return img


def load_dataset():
    images = []
    for file in pathlib.Path(_IMAGENET_SAMPLES).glob("*.JPEG"):
        images.append(load_image(file))

    return images


QUANTIZATION_DATASET_SIZE = 100
EVALUATION_DATASET_SIZE = 1000
EXPORTED_TFLITE_QDQ_DIR = "exported_tflite_qdq_models/"
EXPORTED_TFLITE_QOP_DIR = "exported_tflite_qop_models/"
EXPORTED_ONNX_2_QUANT_DIR = "exported_onnx2quant_models/"
EXPORTED_ONNX_DEFAULT_QDQ_DIR = "exported_onnx_default_qdq_quant_models/"
EXPORTED_ONNX_DEFAULT_QOP_DIR = "exported_onnx_default_qop_quant_models/"


class ImagenetCalibrationDataReader(CalibrationDataReader):

    def __init__(self, images, input_name):
        self.data = [{input_name: image} for image in images[:QUANTIZATION_DATASET_SIZE]]
        self.data_iter = None

    def get_next(self) -> dict:
        if not self.data_iter:
            self.data_iter = iter(self.data)

        return next(self.data_iter, None)


enabled_models = [
    ("alexnet.onnx", "input.1"),
    # ("densenet121.onnx", "input.1"),  # Takes ages to run
    ("efficientnet_b0.onnx", "input.1"),
    ("efficientnet_v2_m.onnx", "input.1"),
    ("googlenet.onnx", "x.1"),
    ("inception_v3.onnx", "x.1"),
    ("mnasnet1_0.onnx", "input.1"),
    ("mobilenet_v2.onnx", "input.1"),
    ("mobilenet_v3_small.onnx", "input.1"),
    ("regnet_x_400mf.onnx", "input.1"),
    ("regnet_y_400mf.onnx", "input.1"),
    ("resnet101.onnx", "input.1"),
    ("resnext50_32x4d.onnx", "input.1"),
    ("shufflenet_v2_x1_0.onnx", "input.1"),
    ("squeezenet1_0.onnx", "input.1"),
    ("vgg11.onnx", "input.1"),
    ("vgg19_bn.onnx", "input.1"),
    ("wide_resnet50_2.onnx", "input.1"),
]


def _quantize_default(onnx_model, calibrator, quant_format=QuantFormat.QDQ):
    onnx.save_model(onnx_model, "tmp.onnx")
    onnxruntime.quantization.quant_pre_process(
        "tmp.onnx",
        "tmp_preprocessed.onnx"
    )
    onnxruntime.quantization.quantize_static(
        "tmp_preprocessed.onnx",
        "tmp_quant.onnx",
        quant_format=quant_format,
        calibration_data_reader=calibrator)

    onnx_model = onnx.load_model("tmp_quant.onnx")
    os.remove("tmp.onnx")
    os.remove("tmp_preprocessed.onnx")
    os.remove("tmp_quant.onnx")

    return onnx_model


def benchmark_model(model_path, dataset, input_name, export_models=False):
    model_name = os.path.basename(model_path)
    print(f"Evaluating '{model_name}'")
    onnx_model = onnx.load_model(model_path)

    print("└─ Quantizing")
    qc = QuantizationConfig(ImagenetCalibrationDataReader(dataset, input_name))
    quantized_internal_onnx_model = QDQQuantizer().quantize_model(onnx_model, qc)
    quantized_internal_onnx_model = ModelShapeInference.infer_shapes(quantized_internal_onnx_model)

    calibrator = ImagenetCalibrationDataReader(dataset, input_name)
    quantized_default_qdq_onnx_model = _quantize_default(onnx_model, calibrator, quant_format=QuantFormat.QDQ)
    quantized_default_qdq_onnx_model = ModelShapeInference.infer_shapes(quantized_default_qdq_onnx_model)

    calibrator = ImagenetCalibrationDataReader(dataset, input_name)
    quantized_default_qop_onnx_model = _quantize_default(onnx_model, calibrator, quant_format=QuantFormat.QOperator)
    skip_shape_inference = False
    try:
        quantized_default_qop_onnx_model = ModelShapeInference.infer_shapes(quantized_default_qop_onnx_model)
    except:
        print(f"\tQOP Quantization: Shape inference failed for {model_name} model.")
        skip_shape_inference = True

    cc = ConversionConfig()
    cc.experimental_qdq_aware_conversion = True
    cc.cast_int64_to_int32 = True

    print("└─ Converting to TFLite")
    converted_tflite = bytes(convert.convert_model(quantized_internal_onnx_model, conversion_config=cc))

    converted_tflite_qop = None
    try:
        cc = ConversionConfig()
        cc.experimental_qdq_aware_conversion = True
        cc.cast_int64_to_int32 = True
        cc.skip_shape_inference = skip_shape_inference
        converted_tflite_qop = bytes(convert.convert_model(quantized_default_qop_onnx_model,conversion_config=cc))
    except Exception as e:
        print(f'Converting QOp quantized {model_name} failed')
        print(e)
    onnx_float_executor = OnnxExecutor(onnx_model.SerializeToString())
    exported_onnx_default_qdq_path = os.path.join(EXPORTED_ONNX_DEFAULT_QDQ_DIR, model_name)
    onnx_default_qdq_executor = OnnxExecutor(quantized_default_qdq_onnx_model.SerializeToString(),
                                             save_model=export_models, saved_model_name=exported_onnx_default_qdq_path)
    exported_onnx_default_qop_path = os.path.join(EXPORTED_ONNX_DEFAULT_QOP_DIR, model_name)
    onnx_default_qop_executor = OnnxExecutor(quantized_default_qop_onnx_model.SerializeToString(),
                                             save_model=export_models, saved_model_name=exported_onnx_default_qop_path, skip_shape_inference=True)
    exported_onnx2quant_path = os.path.join(EXPORTED_ONNX_2_QUANT_DIR, model_name)
    onnx_onnx2quant_executor = OnnxExecutor(quantized_internal_onnx_model.SerializeToString(),
                                            save_model=export_models, saved_model_name=exported_onnx2quant_path)
    exported_tflite_path = os.path.join(EXPORTED_TFLITE_QDQ_DIR, model_name + ".tflite")
    tflite_executor = TFLiteExecutor(model_content=converted_tflite, save_model=export_models,
                                     saved_model_name=exported_tflite_path)
    exported_tflite_qop_path = os.path.join(EXPORTED_TFLITE_QOP_DIR, model_name + ".tflite")
    tflite_executor_qop = TFLiteExecutor(model_content=converted_tflite_qop, save_model=export_models,
                                     saved_model_name=exported_tflite_qop_path) if converted_tflite_qop is not None else None

    outputs = []
    idx = 0

    for image in tqdm(dataset[:EVALUATION_DATASET_SIZE], desc=f"{model_name} evaluation"):
        output_onnx_float = onnx_float_executor.inference(image)
        output_default_qdq_onnx = onnx_default_qdq_executor.inference(image)
        output_default_qop_onnx = onnx_default_qop_executor.inference(image)
        output_onnx2quant_onnx = onnx_onnx2quant_executor.inference(image)
        output_tflite = tflite_executor.inference(image)
        output_tflite_qop = tflite_executor_qop.inference(image) if tflite_executor_qop is not None else None

        onnx_float_top1 = 1 if idx == output_onnx_float.argmax() else 0
        onnx_float_top5 = 1 if idx in output_onnx_float.argsort().squeeze()[-5:][::-1] else 0

        onnx_default_qdq_top1 = 1 if idx == output_default_qdq_onnx.argmax() else 0
        onnx_default_qdq_top5 = 1 if idx in output_default_qdq_onnx.argsort().squeeze()[-5:][::-1] else 0

        onnx_default_qop_top1 = 1 if idx == output_default_qop_onnx.argmax() else 0
        onnx_default_qop_top5 = 1 if idx in output_default_qop_onnx.argsort().squeeze()[-5:][::-1] else 0

        onnx_onnx2quant_top1 = 1 if idx == output_onnx2quant_onnx.argmax() else 0
        onnx_onnx2quant_top5 = 1 if idx in output_onnx2quant_onnx.argsort().squeeze()[-5:][::-1] else 0

        tflite_top1 = 1 if idx == output_tflite.argmax() else 0
        tflite_top5 = 1 if idx in output_tflite.argsort().squeeze()[-5:][::-1] else 0

        if output_tflite_qop is not None:
            tflite_top1_qop = 1 if idx == output_tflite_qop.argmax() else 0
            tflite_top5_qop = 1 if idx in output_tflite_qop.argsort().squeeze()[-5:][::-1] else 0
        else:
            tflite_top1_qop = tflite_top5_qop = None

        outputs.append((
            onnx_float_top1, onnx_float_top5,
            onnx_default_qdq_top1, onnx_default_qdq_top5,
            onnx_default_qop_top1, onnx_default_qop_top5,
            onnx_onnx2quant_top1, onnx_onnx2quant_top5,
            tflite_top1, tflite_top5,
            tflite_top1_qop, tflite_top5_qop))

        idx += 1

    df = pd.DataFrame(outputs,
                      columns=["ONNX(float) Top1", "ONNX(float) Top5",
                               "ONNX(quant-default QDQ) Top1", "ONNX(quant-default QDQ) Top5",
                               "ONNX(quant-default QOperator) Top1", "ONNX(quant-default-QOperator) Top5",
                               "ONNX(quant-onnx2quant) Top1", "ONNX(quant-onnx2quant) Top5",
                               "TFLite(quant) Top1", "TFlite(quant) Top5",
                               "TFLite(quant QOP) Top1", "TFlite(quant QOP) Top5"])
    s = df.sum() / len(df)

    print(s.to_string())
    print(f"Evaluation of '{model_name}' done!")

    return s.rename(model_name)


if __name__ == "__main__":
    assert os.path.exists(_IMAGENET_SAMPLES)
    assert os.path.exists(_PYTORCH_VISION_MODELS)

    if os.path.exists(EXPORTED_TFLITE_QDQ_DIR):
        shutil.rmtree(EXPORTED_TFLITE_QDQ_DIR)
    os.makedirs(EXPORTED_TFLITE_QDQ_DIR)
    if os.path.exists(EXPORTED_TFLITE_QOP_DIR):
        shutil.rmtree(EXPORTED_TFLITE_QOP_DIR)
    os.makedirs(EXPORTED_TFLITE_QOP_DIR)
    if os.path.exists(EXPORTED_ONNX_2_QUANT_DIR):
        shutil.rmtree(EXPORTED_ONNX_2_QUANT_DIR)
    os.makedirs(EXPORTED_ONNX_2_QUANT_DIR)
    if os.path.exists(EXPORTED_ONNX_DEFAULT_QDQ_DIR):
        shutil.rmtree(EXPORTED_ONNX_DEFAULT_QDQ_DIR)
    os.makedirs(EXPORTED_ONNX_DEFAULT_QDQ_DIR)
    if os.path.exists(EXPORTED_ONNX_DEFAULT_QOP_DIR):
        shutil.rmtree(EXPORTED_ONNX_DEFAULT_QOP_DIR)
    os.makedirs(EXPORTED_ONNX_DEFAULT_QOP_DIR)

    images = load_dataset()

    result_series = []

    for model_name, input_name in enabled_models:
        model_path = os.path.join(_PYTORCH_VISION_MODELS, model_name)
        result_series.append(benchmark_model(model_path, images, input_name, export_models=True))

    df = pd.concat(result_series, axis=1).transpose()
    print(df.to_string())
    df.to_excel("pytorch_models_results.xlsx")
