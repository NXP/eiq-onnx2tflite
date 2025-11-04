# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [onnx2tflite|onnx2quant] Mod op with static inputs is now pre-computed and removed from model
- [onnx2tflite] Conversion of Tan operator

## [0.7.0] - 2025-08-18

### Added

- Initial public release 🎉!

## [0.6.0] - 2025-05-21

### Added

- `[onnx2tflite]` Quantize op that is consumer of Conv2D op is now fused into the Conv2D. This can slightly decrease
  accuracy of improperly quantized models.

### Changed

- `[onnx2tflite|onnx2quant]` Migrated to TensorFlow 2.18.2
- `[onnx2tflite|onnx2quant]` Migrated to ORT 2.21.1
- `[onnx2tflite|onnx2quant]` Migrated to ONNX 1.17.0
- `[onnx2quant]` Updated many operator quantizers: QDQClip, QDQConcat, QDQLogSoftmax, QDQScatterND, QDQSoftmax,
  QDQPad, QDQSigmoid. It is expected that quantized Concat operators will contain more surrounding Quantize ops
  than before, because some functionality was removed from ORT's Quantizer API we're using.
- `[onnx2quant]` Already quantized model (contains De/Quantize ops) cannot be quantized for the second time

## [0.5.1] - 2025-01-23

### Fixed

- `[onnx2tflite]` Improve shape inference of Slice operator for multidimensional input
- `[onnx2tflite]` Pin numpy version to ~1.26 (ORT 1.17 requirement)

## [0.5.0] - 2024-10-04

### Added

- `[onnx2tflite]` Added conversion support for: Multinomial, ReverseSequence
- `[onnx2tflite|onnx2quant]` Added option to not generate a partially inferred ONNX model in case the shape inference
  fails
- `[onnx2tflite]` Added optimization to remove some unnecessary Quantize operators.

### Changed

- `[onnx2tflite|onnx2quant]` Disallowed conversion/quantization of models with dynamic shapes
- `[onnx2tflite]` Operator 'Expand' is skipped when no broadcasting is performed
- `[onnx2tflite]` 'QLinearMatmul' and QDQ quantized 'MatMul' will now be converted to LiteRT 'FullyConnected' when
  possible

### Fixed

- `[onnx2tflite]` Weight tensors are cloned before transposition during Conv/MatMul/QGemm conversion. This avoids TFLite
  runtime errors when weights are shared between multiple nodes.

## [0.4.1] - 2024-08-23

First release of onnx2tflite & onnx2quant 🎉!

### Added

- `[onnx2tflite]` Support for conversion of 108 ONNX operators
- `[onnx2quant]` QDQ quantization support for 50 ONNX operators
