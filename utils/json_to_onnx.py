#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    protobufFromJson

Simple script to convert a JSON file to the ONNX (Protocol Buffer) format.
Takes one argument, which is the input JSON file. Output is always 'test.onnx'.
"""

import sys

import onnx
from google.protobuf.json_format import Parse

if len(sys.argv) != 2:
    print("Require 1 .json file!")
    exit(1)

file = sys.argv[1]

with open(file, "r") as f:
    json = f.read()

with open("../thirdparty/onnx/onnx/onnx-ml.proto", "r") as f:
    schema = f.read()

model = Parse(json, onnx.ModelProto())

onnx.save(model, "test.onnx")
