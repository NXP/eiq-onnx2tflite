#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

# Download the TFLite model schema
get-tflite-schema:
	rm -rf ./data/schemas/tflite/schema*
	wget -P ./data/schemas/tflite/ -N https://raw.githubusercontent.com/tensorflow/tensorflow/refs/tags/v2.18.1/tensorflow/compiler/mlir/lite/schema/schema_v3c.fbs

# Compile TFLite schema and generate protobuf library
compile-tflite-schema:
	rm -rf onnx2tflite/lib/tflite/
	flatc -p -o ./onnx2tflite/lib data/schemas/tflite/schema_v3c.fbs

# Download the schema and generate TFLite protobuf library
regenerate-tflite-lib: get-tflite-schema compile-tflite-schema

LB := (
RB := )
# Delete pycache files
clear-pycache:
	find . | grep -E "$(LB)/__pycache__$$|\.pyc$$|\.pyo$$$(RB)" | xargs rm -rf
