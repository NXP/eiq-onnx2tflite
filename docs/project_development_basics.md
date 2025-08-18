# Development

Project contains [ONNX repository](https://github.com/onnx/onnx) as a git submodule with test specifications that has to
be initialized before running any local tests:

```commandline
git submodule update --init --recursive
pip install -e .[dev]
```

## Project structure

The entry point of the program is the **onnx2tflite.py** file in the root directory.

The **data/tflite** directory contains TFLite schema definition (**schema.fbs**) used to generate low level flatbuffer
interface for TFLite model generation. This Python interface is pre-generated in directory **lib/tflite**.

The code in the **onnx2tflite/src** directory is split into multiple subdirectories:

* [**tflite_generator/**](onnx2tflite/src/tflite_generator) - Contains classes used for internal representation of a
  TFLite model and for
  subsequent `*.tflite` file generation.

* [**onnx_parser/**](onnx2tflite/src/onnx_parser) - Contains classes used for internal representation of an ONNX model
  and
  for loading its data from a
  `*.onnx` file.

* [**converter/**](onnx2tflite/src/converter) - Contains files for conversion from the internal ONNX model
  representation
  into the internal TFLite model representation.

## Testing

[pytest](https://docs.pytest.org/) framework is used to run unit/integration tests in this project. Make sure you have
downloaded model artifacts (models & input vectors) with provided script before running any tests:

```commandline
python tests/download_models.py
pytest tests
```
