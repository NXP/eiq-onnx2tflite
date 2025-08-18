#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#

TFL?=test.tflite
JSON?=test.json
ONNX?=test.onnx

# Convert a .tflite file to json. Some models are too big!
tflite-to-json:
	flatc -t --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs -- $(TFL) --raw-binary

# Convert a json file to .tflite
json-to-tflite:
	flatc -b --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs $(JSON)

# Convert a .onnx file to json. Some models are too big!
onnx-to-json:
	python onnx_to_json.py $(ONNX) > $(JSON)

# Convert a .json to test.onnx
json-to-onnx:
	python json_to_onnx.py $(JSON)
	
# Test if generated .tflite file is smaller than one created by flatc through json
# Does't work with large models!
test-efficiency: 
	flatc -t --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs -- $(TFL) --raw-binary
	mv $(TFL) old.tflite
	flatc -b --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs $(JSON)
	diff $(TFL) old.tflite
	
# Remove all .onnx, .tflite and .json files	
clear:
	rm *.tflite *.onnx *.json 2> /dev/null
	
