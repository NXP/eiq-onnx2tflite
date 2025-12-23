# ONNX2TFLite converter & ONNX2Quant quantizer

[![latest-version](https://img.shields.io/badge/eiq--onnx2tflite-0.8.0-brightgreen)](https://eiq.nxp.com/repository/eiq-onnx2tflite/)
![python-badge](https://img.shields.io/badge/Python-3.10.11-green)
![tensorflow-badge](https://img.shields.io/badge/TensorFlow-2.18.1-FF6F00?logo=tensorflow)
![onnx-badge](https://img.shields.io/badge/ONNX-1.17.0-005CED?logo=onnx)
![ORT-badge](https://img.shields.io/badge/ONNX_Runtime-1.21.1-lightgray?logo=onnx)
![onnx-opset](https://img.shields.io/badge/ONNX_Opset-7+-blue?logo=onnx)

ONNX2TFLite converter is CLI tool which allows you to convert ONNX models (`*.onnx`) into the analogous model in
TFLite format (`*.tflite`). ONNX2Quant quantizer performs optimized quantization of ONNX models that results
in models suitable for subsequent conversion into TFLite.

## Installation

```commandline
pip install -r requirements.txt
pip install --index-url https://eiq.nxp.com/repository/ eiq-onnx2tflite
```

## ONNX2TFLite converter

Converter could be used as a standard CLI tool or standalone library.

### CLI usage

```commandline
:~$ onnx2tflite -h
usage: onnx2tflite [-h] [--allow-inputs-stripping | --no-allow-inputs-stripping] [--keep-io-format | --no-keep-io-format]
                   [--skip-shape-inference | --no-skip-shape-inference]
                   [--qdq-aware-conversion | --no-qdq-aware-conversion] [-s [SYMBOLIC_DIMENSIONS_MAPPING ...]]
                   [-m [INPUT_SHAPES_MAPPING ...]]
                   [--dont-skip-nodes-with-known-outputs | --no-dont-skip-nodes-with-known-outputs]
                   [--allow-select-ops | --no-allow-select-ops]
                   [--generate-artifacts-after-failed-shape-inference | --no-generate-artifacts-after-failed-shape-inference]
                   [--guarantee-non-negative-indices | --no-guarantee-non-negative-indices] 
                   [--cast-int64-to-int32 | --no-cast-int64-to-int32]
                   [--accept-resize-rounding-error | --no-accept-resize-rounding-error]
                   [--skip-opset-version-check | --no-skip-opset-version-check] [-o out] [-v]
                   onnx_file

Convert a '.onnx' DNN model to an equivalent '.tflite' model. By default the output '.tflite' file will be generated in the
current working directory and have the same name as the input '.onnx' file.

positional arguments:
  onnx_file             Path to ONNX (*.onnx) model.

options:
  -h, --help            show this help message and exit
  --allow-inputs-stripping, --no-allow-inputs-stripping
                        Model inputs will be removed if they are not necessary for inference and their values are derived
                        during the conversion.
  --keep-io-format, --no-keep-io-format
                        Keep the format of input and output tensors of the converted model the same, as in the original
                        ONNX model (NCHW).
  --skip-shape-inference, --no-skip-shape-inference
                        Shape inference will be skipped before model conversion. This option can be used only if model's
                        shapes are fully defined. Defined shapes are necessary for successful conversion.
  --qdq-aware-conversion, --no-qdq-aware-conversion
                        Quantized QDQ model with QDQ pairs (Q-Ops created by QDQ quantizer) will be converted into
                        optimized variant with QDQ pairs represented as tensors' quantization parameters.
  -s [SYMBOLIC_DIMENSIONS_MAPPING ...], --symbolic-dimension-into-static [SYMBOLIC_DIMENSIONS_MAPPING ...]
                        Change symbolic dimension in model to static (fixed) value. Provided mapping must follow this
                        format '<dim_name>:<dim_size>', for example 'batch:1'. This argument can be used multiple times.
  -m [INPUT_SHAPES_MAPPING ...], --set-input-shape [INPUT_SHAPES_MAPPING ...]
                        Override model input shape. Provided mapping must follow format '<dim_name>:(<dim_0>,<dim_1>,...)',
                        for example 'input_1:(1,3,224,224)'. This argument can be used multiple times.
  --dont-skip-nodes-with-known-outputs, --no-dont-skip-nodes-with-known-outputs
                        Sometimes it is possible to statically infer the output data of some nodes. These nodes will then
                        not be a part of the output model. This flag will force the converter to keep them in anyway.
  --allow-select-ops, --no-allow-select-ops
                        Allow the converter to use the 'SELECT_TF_OPS' operators, which require Flex delegate at runtime.
  --generate-artifacts-after-failed-shape-inference, --no-generate-artifacts-after-failed-shape-inference
                        If the shape inference fails or is incomplete, generate the partly inferred ONNX model as
                        'sym_shape_infer_temp.onnx'.
  --guarantee-non-negative-indices, --no-guarantee-non-negative-indices
                        Guarantee that an 'indices' input tensors will always contain non-negative values. This applies
                        to operators: 'Gather', 'GatherND', 'OneHot' and 'ScatterND'.
  --cast-int64-to-int32, --no-cast-int64-to-int32
                        Cast some nodes with type INT64 to INT32 when TFLite doesn't support INT64. Such nodes are often
                        used in ONNX to calculate shapes/indices, so full range of INT64 isn't necessary. This applies 
                        to operators: 'Abs' and 'Div'."
  --accept-resize-rounding-error, --no-accept-resize-rounding-error
                        Accept the error caused by a different rounding approach of the ONNX 'Resize' and TFLite
                        'ResizeNearestNeighbor' operators, and convert the model anyway.
  --skip-opset-version-check, --no-skip-opset-version-check
                        Ignore the checks for supported opset versions of the ONNX model and try to convert it anyway. This
                        can result in an invalid output TFLite model.
  -o out, --output out  Path to output '.tflite' file.
  -v, --verbose         Print detailed information related to conversion process.
```

### Standalone library

```python
import onnx2tflite.src.converter.convert as convert

binary_tflite_model = convert.convert_model("model.onnx")

with open("model.tflite", "wb") as f:
    f.write(binary_tflite_model)
```

## ONNX2Quant quantizer

ONNX2Quant quantizes ONNX model in 'TFLite conversion optimized way'. This tool produces QDQ model with per-tensor
quantization and INT8 activations/weights. Some operators can be QDQ quantized even if there isn't quantized
variant in ONNX but TFLite supports quantized version of this specific operator.

```commandline
:~$ onnx2quant -h
usage: onnx2quant [-h] [--replace-div-with-mul | --no-replace-div-with-mul]
                  [--replace-constant-with-static-tensor | --no-replace-constant-with-static-tensor] [-o OUTPUT]
                  [--per-channel | --no-per-channel] [-l | --allow-opset-10-and-lower | --no-allow-opset-10-and-lower] 
                  -c CALIBRATION_DATASET_MAPPING [CALIBRATION_DATASET_MAPPING ...] 
                  [-s [SYMBOLIC_DIMENSIONS_MAPPING ...]] [-m [INPUT_SHAPES_MAPPING ...]]
                  [--generate-artifacts-after-failed-shape-inference | --no-generate-artifacts-after-failed-shape-inference]
                  onnx_model

Quantize ONNX model in 'TFLite conversion optimized way'. This tool produces QDQ model with per-tensor/per-channel 
quantization and INT8 activations. Some operators can be QDQ quantized even if there isn't quantized variant in ONNX 
but TFLite supports quantized version of this specific operator.

positional arguments:
  onnx_model            Path to input ONNX '*.onnx' model.

options:
  -h, --help            show this help message and exit
  --replace-div-with-mul, --no-replace-div-with-mul
                        Replace some 'Div' operators with 'Mul'. 'Div' doesn't support int8 quantization in TFLite so this
                        is replacement can avoid having to compute 'Div' in float32.
  --replace-constant-with-static-tensor, --no-replace-constant-with-static-tensor
                        Remove 'Constant' nodes and directly assign static data to their output tensors.
  -o OUTPUT, --output OUTPUT
                        Path to the resulting quantized ONNX model. (default: '<input_model_name>_quant.onnx')
  --per-channel, --no-per-channel
                        Quantize some weight tensors per-channel instead of per-tensor. This should result in a higher
                        accuracy.
  -l, --allow-opset-10-and-lower, --no-allow-opset-10-and-lower
                        Allow quantization of models with opset version 10 and lower. Quantization of such models can
                        produce invalid models because opset is forcefully updated to version 11. This applies especially
                        to models with operators: Clip, Dropout, BatchNormalization and Split.
  -c CALIBRATION_DATASET_MAPPING [CALIBRATION_DATASET_MAPPING ...], --calibration-dataset-mapping CALIBRATION_DATASET_MAPPING [CALIBRATION_DATASET_MAPPING ...]
                        Mapping between model input and calibration dataset directory with *.npy files. Value must be in
                        format '<input_name>;<path_to_dir>', for example 'input_1;data_3_224/'. Argument can be used
                        multiple times to specify multiple inputs for the model. In case modelhas semicolon in input
                        tensor's name, it has to be renamed.
  -s [SYMBOLIC_DIMENSIONS_MAPPING ...], --symbolic-dimension-into-static [SYMBOLIC_DIMENSIONS_MAPPING ...]
                        Change symbolic dimension in model to static (fixed) value. Provided mapping must follow this
                        format '<dim_name>:<dim_size>', for example 'batch:1'. This argument can be used multiple times.
  -m [INPUT_SHAPES_MAPPING ...], --set-input-shape [INPUT_SHAPES_MAPPING ...]
                        Override model input shape. Provided mapping must follow format '<dim_name>:(<dim_0>,<dim_1>,...)',
                        for example 'input_1:(1,3,224,224)'. This argument can be used multiple times.
  --generate-artifacts-after-failed-shape-inference, --no-generate-artifacts-after-failed-shape-inference
                        If the shape inference fails or is incomplete, generate the partly inferred ONNX model as
                        'sym_shape_infer_temp.onnx'.
```

## Development

- [Project structure & testing](docs/project_development_basics.md)
- [How to implement new operator?](docs/new_operator_support.md)
- [What to do when increasing TensorFlow version?](docs/increasing_dependencies_versions.md)
- [ONNXRT code tags](docs/code_tags.md)

### Utilities

Directory **utils/** contains a number of tools to ease development. Make sure you run them from **utils/** directory
and not from project root:

| <div style="width:300px">Command</div>              | Description                                                                                                        |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `ONNX=model.onnx JSON=model.json make onnx-to-json` | Convert ONNX model into JSON representation. This utility simplifies visualisation of model's internal components. |
| `JSON=model.json make json-to-onnx`                 | Convert ONNX model in JSON format back to ONNX (`.onnx`).                                                          |
| `TFL=model.tflite make tflite-to-json`              | Convert TFLite model into its JSON representation.                                                                 |
| `JSON=model.json make json-to-tflite`               | Convert TFLite model in JSON format into TFLite (`.tflite`)                                                        |
| `python onnx_model_get_ops.py model.onnx`           | Print all operators that are present in ONNX model, or check if all ops in the model are supported.                |

## License

- Original license: MIT (see [LICENSE_MIT](licenses/LICENSE_MIT))
- Outgoing license: LA_OPT_Online Code Hosting NXP_Software_License (see
  [LICENSE](LICENSE))
