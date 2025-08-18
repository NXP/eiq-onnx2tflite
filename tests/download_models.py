#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import os
import pathlib
import shutil
import sys
import tarfile
from itertools import repeat
from multiprocessing.pool import Pool
from typing import Dict

import requests

_CURRENT_DIR = pathlib.Path(__file__).parent

_MODELS = {
    #Vision :
    # "age_googlenet": {
    #     "download_uri": "https://github.com/onnx/models/raw/main/vision/body_analysis/age_gender/models/age_googlenet.tar.gz"
    # },
    "bvlcalexnet-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/alexnet/model/bvlcalexnet-7.tar.gz"
    },
    "bvlcalexnet-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/alexnet/model/bvlcalexnet-8.tar.gz"
    },
    "bvlcalexnet-9": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/alexnet/model/bvlcalexnet-9.tar.gz"
    },
    "bvlcalexnet-12": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/alexnet/model/bvlcalexnet-12.tar.gz",
        "model_name": "bvlcalexnet-12.onnx"
    },
    "caffenet-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/caffenet/model/caffenet-7.tar.gz"
    },
    "caffenet-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/caffenet/model/caffenet-8.tar.gz"
    },
    "caffenet-9": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/caffenet/model/caffenet-9.tar.gz"
    },
    "caffenet-12": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/caffenet/model/caffenet-12.tar.gz",
        "model_name": "caffenet-12.onnx"
    },
    "emotion-ferplus-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-7.tar.gz",
    },
    "emotion-ferplus-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.tar.gz",
    },
    "googlenet-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/googlenet/model/googlenet-7.tar.gz",
    },
    "googlenet-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/googlenet/model/googlenet-8.tar.gz",
    },
    "googlenet-9": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.tar.gz",
    },
    "googlenet-12": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.tar.gz",
        "model_name": "googlenet-12.onnx"
    },
    "inception-v2-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-7.tar.gz",
    },
    "inception-v2-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-8.tar.gz",
    },
    "inception-v2-9": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.tar.gz",
    },
    "mnist-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-7.tar.gz"
    },
    "mnist-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.tar.gz"
    },
    "mnist-12": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.tar.gz",
        "model_name": "mnist-12.onnx"
    },
    "rcnn-ilsvrc13-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-7.tar.gz"
    },
    "rcnn-ilsvrc13-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-8.tar.gz"
    },
    "rcnn-ilsvrc13-9": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.tar.gz"
    },
    "ResNet101-DUC-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/duc/model/ResNet101-DUC-7.tar.gz",
        "model_name": "ResNet101-DUC-7.onnx"
    },
    "ResNet101-DUC-12": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/duc/model/ResNet101-DUC-12.tar.gz",
        "model_name": "ResNet101-DUC-12.onnx"
    },
    "resnet50-caffe2-v1-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-7.tar.gz"
    },
    "resnet50-caffe2-v1-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-8.tar.gz"
    },
    "resnet50-caffe2-v1-9": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.tar.gz"
    },
    "shufflenet-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-7.tar.gz"
    },
    "shufflenet-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-8.tar.gz"
    },
    "shufflenet-9": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-9.tar.gz"
    },
    "super-resolution-10": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.tar.gz",
        "model_name": "super_resolution.onnx"
    },
    "squeezenet1.1-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.tar.gz",
        "model_name": "squeezenet1.1.onnx"
    },
    "tinyyolov2-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.tar.gz"
    },
    "tinyyolov2-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.tar.gz",
        "model_name": "Model.onnx"
    },
    "ultraface-version-RFB-320": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.tar.gz",
        "model_name": "version-RFB-320.onnx"
    },
    "ultraface-version-RFB-640": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-640.tar.gz",
        "model_name": "version-RFB-640.onnx"
    },
    "vgg16-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-7.tar.gz",
        "model_name": "vgg16.onnx"
    },
    "vgg16-12": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-12.tar.gz",
        "model_name": "vgg16-12.onnx"
    },
    "vgg16-bn-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-bn-7.tar.gz",
        "model_name": "vgg16-bn.onnx"
    },
    "vgg19-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg19-7.tar.gz",
        "model_name": "vgg19.onnx"
    },
    "vgg19-bn-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg19-bn-7.tar.gz",
        "model_name": "vgg19-bn-7.onnx"
    },
    "vgg19-caffe2-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg19-caffe2-7.tar.gz"
    },
    "vgg19-caffe2-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg19-caffe2-8.tar.gz"
    },
    "vgg19-caffe2-9": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg19-caffe2-9.tar.gz"
    },
    "zfnet512-7": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/zfnet-512/model/zfnet512-7.tar.gz"
    },
    "zfnet512-8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/zfnet-512/model/zfnet512-8.tar.gz"
    },
    "zfnet512-9": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/zfnet-512/model/zfnet512-9.tar.gz"
    },
    "zfnet512-12": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/zfnet-512/model/zfnet512-12.tar.gz",
        "model_name": "zfnet512-12.onnx"
    },

    # Quantized models
    "resnet50-v1-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12-int8.tar.gz",
        "model_name": "resnet50-v1-12-int8.onnx"
    },
    "caffenet-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/caffenet/model/caffenet-12-int8.tar.gz",
        "model_name": "caffenet-12-int8.onnx"
    },
    "zfnet512-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/zfnet-512/model/zfnet512-12-int8.tar.gz",
        "model_name": "zfnet512-12-int8.onnx"
    },
    "bvlcalexnet-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/alexnet/model/bvlcalexnet-12-int8.tar.gz",
        "model_name": "bvlcalexnet-12-int8.onnx"
    },
    "mnist-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12-int8.tar.gz",
        "model_name": "mnist-12-int8.onnx"
    },
    "mobilenetv2-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12-int8.tar.gz",
        "model_name": "mobilenetv2-12-int8.onnx"
    },
    "vgg16-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-12-int8.tar.gz",
        "model_name": "vgg16-12-int8.onnx"
    },
    "squeezenet1.0-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12-int8.tar.gz",
        "model_name": "squeezenet1.0-12-int8.onnx"
    },
    "ResNet101-DUC-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/duc/model/ResNet101-DUC-12-int8.tar.gz",
        "model_name": "ResNet101-DUC-12-int8.onnx"
    },
    "googlenet-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12-int8.tar.gz",
        "model_name": "googlenet-12-int8.onnx"
    },
    "emotion-ferplus-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-12-int8.tar.gz",
        "model_name": "emotion-ferplus-12-int8.onnx"
    },
    "ultraface-version-RFB-320-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320-int8.tar.gz",
        "model_name": "version-RFB-320-int8.onnx"
    },
    "densenet-12-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-12-int8.tar.gz",
        "model_name": "densenet-12-int8.onnx"
    },
    "efficientnet-lite4-11-int8": {
        "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11-int8.tar.gz",
        "model_name": "efficientnet-lite4-11-int8.onnx"
    },
    # "inception-v1-12-int8": {
    #     "download_uri": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-12-int8.tar.gz",
    #     "model_name": "inception-v1-12-int8.onnx"
    # },
}


def download_model(model_config, download_dir):
    model_name = model_config[0]
    download_uri = model_config[1]["download_uri"]
    filename = os.path.basename(download_uri)
    filepath = os.path.join(download_dir, filename)

    if os.path.exists(filepath):  # already downloaded
        return model_name, filepath

    print(f"Downloading: {model_name}")
    with requests.get(download_uri, stream=True) as r, open(filepath, "wb") as f:
        for chunk in r.iter_content(2048):
            f.write(chunk)
    print(f"Download finished: {model_name}")

    return model_name, filepath


def download_model_wrapper(args):
    return download_model(*args)


def download_models(models_config: Dict, download_dir: str):
    data = list(zip(models_config.items(), repeat(download_dir)))

    with Pool(5) as pool:
        for result in pool.imap_unordered(download_model_wrapper, data):
            models_config[result[0]]["artifact_path"] = result[1]

    return models_config


if __name__ == "__main__":
    # python download-model.py ["model-name" "model-name2" ...]
    models = {}
    for model_name in sys.argv[1:]:
        if model_name in _MODELS:
            models[model_name] = _MODELS[model_name]
        else:
            raise Exception(f"Model '{model_name}' not found in a list of model configurations.")

    if len(models) == 0:
        models = _MODELS

    download_dir = os.path.join(_CURRENT_DIR, "artifacts", "downloaded")
    models_config = download_models(models, download_dir)


    def stripped_members(tar):
        for member in tar.getmembers():
            if '/' not in member.path:  # skip root, works multiplatform
                continue

            member.path = member.path.split('/', 1)[-1]
            yield member


    for artifact_name, artifact_info in models_config.items():
        print(f"Extracting: {artifact_info['artifact_path']}")
        extraction_dir = os.path.join(download_dir, artifact_name)
        shutil.rmtree(extraction_dir, ignore_errors=True)
        os.mkdir(extraction_dir)

        with tarfile.open(artifact_info["artifact_path"]) as tar:
            tar.extractall(path=extraction_dir, members=stripped_members(tar))

        if "model_name" in artifact_info:
            os.rename(os.path.join(extraction_dir, artifact_info["model_name"]),
                      os.path.join(extraction_dir, "model.onnx"))
